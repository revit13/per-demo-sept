FROM python:3.9

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    libprotobuf-dev \
    protobuf-compiler \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
apt-get install -y --no-install-recommends \
gdal-bin \ 
libgdal-dev \ 
libgeos-dev 

# Upgrade pip and install Python libraries
RUN pip3 install --upgrade pip && \
    pip3 install -U pip wheel setuptools && \
    pip3 install \
    brevitas \
    onnx \
    networkx \
    osmnx \
    matplotlib \
    torch \
    torchvision \
    networkx


