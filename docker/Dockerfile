FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3.8 \
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
    libopenmpi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --no-cache-dir \
    torch==1.11.0+cu115 \
    torchvision==0.12.0+cu115 \
    deepspeed==0.16.4 \
    "numpy<1.20" \
    timm==0.4.12 \
    tensorboard \
    TensorboardX \
    decord \
    einops \
    scipy \
    opencv-python \
    pandas==1.1.3 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install Python MPI bindings
RUN pip install --no-cache-dir mpi4py

# For inference
RUN pip install --no-cache-dir numba==0.53.1 numpy==1.21 starlette==0.35.0

# Install Supervisely
RUN pip install --no-cache-dir supervisely[training]==6.73.391
RUN pip install --no-cache-dir supervisely[model-benchmark]==6.73.391

LABEL python_sdk_version=6.73.391
