FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

LABEL maintainer="Hugging Face"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get -y upgrade --only-upgrade systemd openssl cryptsetup \
    && apt-get install -y \
    build-essential \
    bzip2 \
    curl \
    git \
    git-lfs \
    tar \
    gcc \
    g++ \
    cmake \
    libprotobuf-dev \
    protobuf-compiler \
    python3 \
    python3-pip \
    # audio
    libsndfile1-dev \
    ffmpeg \
    && apt-get clean autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

WORKDIR /app

# install dependencies
COPY dockerfiles/pytorch/gpu/requirements.txt /tmp/requirements.txt
RUN pip install  --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# install huggingface inference toolkit
COPY requirements.txt /tmp/requirements.txt
RUN pip install  --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# copy application
COPY src/huggingface_inference_toolkit huggingface_inference_toolkit
COPY src/huggingface_inference_toolkit/webservice_starlette.py webservice_starlette.py

# copy entrypoint and change permissions
COPY scripts/entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

# run app
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]