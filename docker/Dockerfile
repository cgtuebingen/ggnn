FROM nvidia/cuda:10.2-cudnn7-devel

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    rm -rf /var/lib/apt/lists/* \
    /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
    build-essential \
    ca-certificates \
    cmake \
    python3.5 \
    python-dev \
    python3-dev \
    libgflags-dev \
    git

WORKDIR /ggnn
