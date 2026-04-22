import cv2
import numpy as np
import requests
import io
import pytoshop
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from segment_anything import sam_model_registry, SamPredictor
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from typing import List

app = FastAPI()

# 1. Khởi tạo SAM
sam = sam_model_registry["vit_h"](checkpoint="/models/sam_vit_h_4b8939.pth")
sam.to(device="cpu")
predictor = SamPredictor(sam)

# 2. Khởi tạo Real-ESRGAN (x2)
model_upscale = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
upsampler = RealESRGANer(
    scale=2,
    model_path='/models/RealESRGAN_x2plus.pth',
    model=model_upscale,
    tile=400, # Chia nhỏ ảnh để tránh tràn RAM
    tile_pad=10,
    pre_pad=0,
    half=False # Chạy trên CPU thì để False
)

class ImageRequest(BaseModel):
    image_url: str
    box_coordinates: List[int]
    do_upscale: bool = False # Mặc định là False (không tăng nét) để an toàn

@app.post("/generate-psd-upscale")
async def generate_psd(req: ImageRequest):
    try:
        # Tải ảnh
        resp = requests.get(req.image_url)
        img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        h, w, _ = img.shape
        
        # Tách Mask
        predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        masks, _, _ = predictor.predict(box=np.array(req.box_coordinates)[None, :], multimask_output=False)
        mask = masks[0]

        # --- XỬ LÝ VẬT THỂ (Layer 1) ---
        mask_uint8 = (mask * 255).astype(np.uint8)
        fg = cv2.merge([img[:,:,0], img[:,:,1], img[:,:,2], mask_uint8])
        
        # Tăng nét vật thể x2
        # Tách RGB và Alpha để tăng nét riêng (Real-ESRGAN không hỗ trợ Alpha trực tiếp tốt)
        fg_rgb = fg[:,:,:3]
        fg_alpha = fg[:,:,3]
        
        upscaled_fg_rgb, _ = upsampler.enhance(fg_rgb, outscale=2)
        upscaled_fg_alpha = cv2.resize(fg_alpha, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
        
        # --- XỬ LÝ NỀN (Layer 2) ---
        mask_inpaint = cv2.dilate(mask_uint8, np.ones((15,15), np.uint8), iterations=1)
        bg_inpainted = cv2.inpaint(img, mask_inpaint, 3, cv2.INPAINT_TELEA)
        # Nền cũng phải x2 để khớp kích thước file PSD
        bg_upscaled = cv2.resize(bg_inpainted, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)

        # Đóng gói PSD
        document = pytoshop.Document(width=w*2, height=h*2)
        
        # Add BG
        bg_layer = pytoshop.LayerRecord(
            channels={i: bg_upscaled[:,:,2-i] for i in range(3)},
            top=0, left=0, bottom=h*2, right=w*2, name='Nen_x2'
        )
        document.layers.append(bg_layer)
        
        # Add FG
        fg_channels = {i: upscaled_fg_rgb[:,:,2-i] for i in range(3)}
        fg_channels[-1] = upscaled_fg_alpha # Alpha channel
        fg_layer = pytoshop.LayerRecord(
            channels=fg_channels,
            top=0, left=0, bottom=h*2, right=w*2, name='VatThe_Net_x2'
        )
        document.layers.append(fg_layer)

        psd_buffer = io.BytesIO()
        pytoshop.write(psd_buffer, document)
        psd_buffer.seek(0)
        
        return StreamingResponse(psd_buffer, media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
