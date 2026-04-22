# Sử dụng base image có sẵn Pytorch cho CPU để tối ưu
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và tạo file
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Tải sẵn mô hình AI SAM của Meta để tool chạy được ngay
RUN mkdir -p /models
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O /models/sam_vit_h_4b8939.pth

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Khởi động server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
