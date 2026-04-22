import cv2
import numpy as np
import requests
import io
import pytoshop
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from segment_anything import sam_model_registry, SamPredictor
from typing import List

app = FastAPI(title="AI Photoshop Layer Tool")

# 1. Khởi tạo mô hình AI (SAM) lúc server khởi động
device = "cpu"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint="/models/sam_vit_h_4b8939.pth")
sam.to(device=device)
predictor = SamPredictor(sam)

# 2. Định nghĩa cấu trúc dữ liệu JSON nhận tọa độ và link ảnh
class ImageRequest(BaseModel):
    image_url: str
    box_coordinates: List[int] # Nhận [x1, y1, x2, y2]

def download_image_to_cv2(url: str):
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise ValueError("Không thể tải ảnh từ URL")
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# 3. Endpoint chính để xử lý "Cắt & Vẽ bù" thành PSD
@app.post("/generate-psd")
async def generate_psd(req: ImageRequest):
    try:
        # 3.1 Tải ảnh gốc
        img = download_image_to_cv2(req.image_url)
        h, w, _ = img.shape
        # SAM expects RGB input
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3.2 Tách mặt nạ (Mask) vật thể dựa trên tọa độ khoanh vùng
        predictor.set_image(img_rgb)
        input_box = np.array(req.box_coordinates)
        
        masks, scores, logits = predictor.predict(
            box=input_box[None, :],
            multimask_output=False,
        )
        
        if len(masks) == 0:
            raise HTTPException(status_code=400, detail="Không tìm thấy vật thể nào trong vùng khoanh.")
            
        mask = masks[0]
        
        # 3.3 Trích xuất đối tượng (PNG dải trong suốt)
        # Tạo mask chi tiết (Dilation) để khi vẽ bù không bị lem viền
        kernel = np.ones((15, 15), np.uint8)
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        
        # Tách kênh alpha (trong suốt)
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        b, g, r = cv2.split(img)
        rgba = [b, g, r, mask_uint8]
        foreground_cv2 = cv2.merge(rgba)
        
        # 3.4 Chạy AI Inpainting (Vẽ bù nền thông minh)
        # Inpainting in OpenCV is basic, for professional result update to LaMa or other model
        mask_for_inpaint = (dilated_mask * 255).astype(np.uint8)
        background_inpainted_cv2 = cv2.inpaint(img, mask_for_inpaint, 3, cv2.INPAINT_TELEA)
        
        # 3.5 Đóng gói thành file .psd (Photoshop)
        document = pytoshop.Document(width=w, height=h)
        
        # Tạo Layer Nền (phía dưới)
        bg_rgb = cv2.cvtColor(background_inpainted_cv2, cv2.COLOR_BGR2RGB)
        bg_layer_rec = pytoshop.LayerRecord(
             channels = [
                 pytoshop.LayerChannel(pytoshop.ColorId.RED, bg_rgb[:,:,0]),
                 pytoshop.LayerChannel(pytoshop.ColorId.GREEN, bg_rgb[:,:,1]),
                 pytoshop.LayerChannel(pytoshop.ColorId.BLUE, bg_rgb[:,:,2]),
             ],
             top=0, left=0, bottom=h, right=w,
             blend_mode=pytoshop.BlendMode.NORMAL,
             opacity=255,
             visible=True,
             name='Layer2_NenHoanChinh'
        )
        document.layers.append(bg_layer_rec)
        
        # Tạo Layer Vật Thể (phía trên)
        fg_rgb_a = cv2.cvtColor(foreground_cv2, cv2.COLOR_BGRA2RGBA)
        fg_layer_rec = pytoshop.LayerRecord(
             channels = [
                 pytoshop.LayerChannel(pytoshop.ColorId.RED, fg_rgb_a[:,:,0]),
                 pytoshop.LayerChannel(pytoshop.ColorId.GREEN, fg_rgb_a[:,:,1]),
                 pytoshop.LayerChannel(pytoshop.ColorId.BLUE, fg_rgb_a[:,:,2]),
                 pytoshop.LayerChannel(pytoshop.ColorId.ALPHA, fg_rgb_a[:,:,3]),
             ],
             top=0, left=0, bottom=h, right=w,
             blend_mode=pytoshop.BlendMode.NORMAL,
             opacity=255,
             visible=True,
             name='Layer1_VatThe'
        )
        document.layers.append(fg_layer_rec)
        
        # Viết Document ra file-like object
        psd_buffer = io.BytesIO()
        pytoshop.write(psd_buffer, document)
        psd_buffer.seek(0)
        
        return StreamingResponse(
            psd_buffer, 
            media_type="application/octet-stream", # General application type for PSD
            headers={"Content-Disposition": "attachment; filename=AI_Separated_Layers.psd"}
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi AI: {str(e)}")
