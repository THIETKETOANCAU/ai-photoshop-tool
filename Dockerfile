FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 wget \
    && rm -rf /var/lib/apt/lists/*

# Tải sẵn mô hình AI bản Base (vit_b) - Nhẹ hơn, ổn định hơn cho Cloud Run
RUN mkdir -p /models
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O /models/sam_vit_b.pth
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -O /models/RealESRGAN_x2plus.pth

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cấu hình cổng mạng chuẩn Google
ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
