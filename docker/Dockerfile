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
    deepspeed==0.5.8 \
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

# Install OpenMPI
RUN apt-get update && apt-get install -y --no-install-recommends \
    openmpi-bin \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python MPI bindings
RUN pip install mpi4py

# For inference
RUN pip install numba==0.53.1 numpy==1.21 starlette==0.35.0

# Install Supervisely
RUN pip install supervisely==6.73.366
RUN pip install supervisely[training]==6.73.366
