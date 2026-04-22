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

# Biến toàn cục để giữ AI trong RAM
predictor = None
upsampler = None

def load_models():
    global predictor, upsampler
    if predictor is None:
        sam = sam_model_registry["vit_b"](checkpoint="/models/sam_vit_b.pth")
        predictor = SamPredictor(sam)
    if upsampler is None:
        model_upscale = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        upsampler = RealESRGANer(scale=2, model_path='/models/RealESRGAN_x2plus.pth', model=model_upscale, tile=400, half=False)

class ImageRequest(BaseModel):
    image_url: str
    box_coordinates: List[int]
    do_upscale: bool = False

@app.post("/generate-psd-upscale")
async def generate_psd(req: ImageRequest):
    try:
        load_models() # Gọi AI dậy
        resp = requests.get(req.image_url)
        img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        
        predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        masks, _, _ = predictor.predict(box=np.array(req.box_coordinates)[None, :], multimask_output=False)
        mask_uint8 = (masks[0] * 255).astype(np.uint8)

        # Inpaint Nền
        bg_inpainted = cv2.inpaint(img, cv2.dilate(mask_uint8, np.ones((15,15), np.uint8)), 3, cv2.INPAINT_TELEA)

        if req.do_upscale:
            fg_rgb, _ = upsampler.enhance(img, outscale=2)
            fg_alpha = cv2.resize(mask_uint8, (w*2, h*2))
            final_bg = cv2.resize(bg_inpainted, (w*2, h*2))
            fw, fh = w*2, h*2
        else:
            fg_rgb, fg_alpha, final_bg = img, mask_uint8, bg_inpainted
            fw, fh = w, h

        # Đóng gói PSD
        document = pytoshop.Document(width=fw, height=fh)
        for i, name in enumerate(['Layer2_Background', 'Layer1_Object']):
            img_data = final_bg if i==0 else fg_rgb
            channels = {j: img_data[:,:,2-j] for j in range(3)}
            if i==1: channels[-1] = fg_alpha
            layer = pytoshop.LayerRecord(channels=channels, top=0, left=0, bottom=fh, right=fw, name=name)
            document.layers.append(layer)

        psd_out = io.BytesIO()
        pytoshop.write(psd_out, document)
        psd_out.seek(0)
        return StreamingResponse(psd_out, media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
