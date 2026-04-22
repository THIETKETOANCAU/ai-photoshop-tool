FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 wget \
    && rm -rf /var/lib/apt/lists/*

# Tải model SAM
RUN mkdir -p /models
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O /models/sam_vit_h_4b8939.pth

# Tải model Real-ESRGAN (phiên bản x2)
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -O /models/RealESRGAN_x2plus.pth

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
