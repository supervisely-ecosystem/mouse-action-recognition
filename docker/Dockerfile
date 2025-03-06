FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    # python3.8 \
    python3-dev \
    python3-pip \
    git \
    curl \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libmagic-dev \
    libexiv2-dev \
    openmpi-bin \
    libopenmpi-dev

RUN ln -s /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==1.11.0+cu115 \
    torchvision==0.12.0+cu115 \
    deepspeed \
    "numpy<1.20" \
    timm==0.4.12 \
    TensorboardX \
    decord \
    einops \
    scipy \
    opencv-python \
    pandas==1.1.3 \
    -f https://download.pytorch.org/whl/torch_stable.html
