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

app = FastAPI(title="AI Photoshop Pro - Upscale & Inpaint")

# ==========================================
# 1. KHỞI TẠO CÁC MÔ HÌNH AI (CHỈ CHẠY 1 LẦN)
# ==========================================
print("Đang tải mô hình SAM...")
sam = sam_model_registry["vit_h"](checkpoint="/models/sam_vit_h_4b8939.pth")
sam.to(device="cpu")
predictor = SamPredictor(sam)

print("Đang tải mô hình Real-ESRGAN...")
model_upscale = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
upsampler = RealESRGANer(
    scale=2,
    model_path='/models/RealESRGAN_x2plus.pth',
    model=model_upscale,
    tile=400, # Chia nhỏ ảnh để tránh tràn RAM
    tile_pad=10,
    pre_pad=0,
    half=False # Chạy trên CPU
)
print("Khởi tạo AI thành công! Sẵn sàng nhận lệnh.")

# ==========================================
# 2. CẤU TRÚC DỮ LIỆU NHẬN TỪ WEB PHP
# ==========================================
class ImageRequest(BaseModel):
    image_url: str
    box_coordinates: List[int]
    do_upscale: bool = False  # Mặc định là False nếu người dùng không tick

def download_image(url: str):
    response = requests.get(url, timeout=15)
    if response.status_code != 200:
        raise ValueError("Không thể tải ảnh từ URL.")
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# ==========================================
# 3. HÀM XỬ LÝ CHÍNH
# ==========================================
@app.post("/generate-psd-upscale")
async def generate_psd(req: ImageRequest):
    try:
        # Bước 1: Tải ảnh gốc
        img = download_image(req.image_url)
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Bước 2: Dùng SAM tách mặt nạ dựa trên tọa độ vẽ
        predictor.set_image(img_rgb)
        input_box = np.array(req.box_coordinates)
        
        masks, _, _ = predictor.predict(
            box=input_box[None, :],
            multimask_output=False,
        )
        
        if len(masks) == 0:
            raise HTTPException(status_code=400, detail="Không tìm thấy vật thể nào trong khung vẽ.")
            
        mask = masks[0]
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Bước 3: Tách lấy Vật thể (Foreground) và vẽ bù Nền (Background)
        # Vật thể
        fg_rgb = img[:, :, :3]
        fg_alpha = mask_uint8
        
        # Nền (Dùng inpaint lấp lỗ thủng)
        kernel = np.ones((15, 15), np.uint8)
        mask_inpaint = cv2.dilate(mask_uint8, kernel, iterations=1)
        bg_inpainted = cv2.inpaint(img, mask_inpaint, 3, cv2.INPAINT_TELEA)

        # Bước 4: Kiểm tra xem khách có yêu cầu Tăng Nét (Upscale) không
        if req.do_upscale:
            print("Đang chạy AI Tăng Nét x2...")
            # Tăng nét vật thể (Chỉ tăng RGB, kênh Alpha dùng nội suy để khớp kích thước)
            upscaled_fg_rgb, _ = upsampler.enhance(fg_rgb, outscale=2)
            upscaled_fg_alpha = cv2.resize(fg_alpha, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
            
            # Phóng to luôn Nền lên x2 để lúc ghép vào PSD không bị lệch
            final_bg = cv2.resize(bg_inpainted, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
            
            final_fg_rgb = upscaled_fg_rgb
            final_fg_alpha = upscaled_fg_alpha
            final_w, final_h = w * 2, h * 2
        else:
            print("Bỏ qua Tăng Nét. Giữ kích thước gốc.")
            # Khách không tick ô Tăng nét -> Giữ nguyên để chạy nhanh
            final_bg = bg_inpainted
            final_fg_rgb = fg_rgb
            final_fg_alpha = fg_alpha
            final_w, final_h = w, h

        # Bước 5: Đóng gói toàn bộ thành file PSD
        document = pytoshop.Document(width=final_w, height=final_h)
        
        # 5.1 Tạo Layer Nền (Nằm dưới cùng)
        bg_rgb_converted = cv2.cvtColor(final_bg, cv2.COLOR_BGR2RGB)
        bg_layer = pytoshop.LayerRecord(
            channels=[
                pytoshop.LayerChannel(pytoshop.ColorId.RED, bg_rgb_converted[:, :, 0]),
                pytoshop.LayerChannel(pytoshop.ColorId.GREEN, bg_rgb_converted[:, :, 1]),
                pytoshop.LayerChannel(pytoshop.ColorId.BLUE, bg_rgb_converted[:, :, 2]),
            ],
            top=0, left=0, bottom=final_h, right=final_w,
            blend_mode=pytoshop.BlendMode.NORMAL,
            opacity=255,
            visible=True,
            name='Layer2_Nen_Hoan_Chinh'
        )
        document.layers.append(bg_layer)
        
        # 5.2 Tạo Layer Vật Thể (Nằm trên)
        fg_rgb_converted = cv2.cvtColor(final_fg_rgb, cv2.COLOR_BGR2RGB)
        fg_layer = pytoshop.LayerRecord(
            channels=[
                pytoshop.LayerChannel(pytoshop.ColorId.RED, fg_rgb_converted[:, :, 0]),
                pytoshop.LayerChannel(pytoshop.ColorId.GREEN, fg_rgb_converted[:, :, 1]),
                pytoshop.LayerChannel(pytoshop.ColorId.BLUE, fg_rgb_converted[:, :, 2]),
                pytoshop.LayerChannel(pytoshop.ColorId.ALPHA, final_fg_alpha),
            ],
            top=0, left=0, bottom=final_h, right=final_w,
            blend_mode=pytoshop.BlendMode.NORMAL,
            opacity=255,
            visible=True,
            name='Layer1_Vat_The_Da_Tach'
        )
        document.layers.append(fg_layer)

        # Bước 6: Trả file PSD về cho trình duyệt
        psd_buffer = io.BytesIO()
        pytoshop.write(psd_buffer, document)
        psd_buffer.seek(0)
        
        return StreamingResponse(
            psd_buffer, 
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=AI_Design_Layers.psd"}
        )

    except Exception as e:
        print(f"LỖI HỆ THỐNG: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý ảnh: {str(e)}")
