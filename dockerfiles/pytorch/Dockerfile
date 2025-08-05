ARG BASE_IMAGE=nvidia/cuda:12.1.0-devel-ubuntu22.04

FROM $BASE_IMAGE AS base
SHELL ["/bin/bash", "-c"]

LABEL maintainer="Hugging Face"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && \
    apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y upgrade --only-upgrade systemd openssl cryptsetup && \
    apt-get install -y \
    build-essential \
    bzip2 \
    cmake \
    curl \
    git \
    git-lfs \
    tar \
    gcc \
    g++ \
    libprotobuf-dev \
    protobuf-compiler \
    python3.11 \
    python3.11-dev \
    libsndfile1-dev \
    ffmpeg \
    && apt-get clean autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

# Copying only necessary files as filtered by .dockerignore
COPY . .

# Set Python 3.11 as the default python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip from source
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install wheel and setuptools
RUN pip install --no-cache-dir --upgrade pip ".[torch,st,diffusers]"
RUN pip install flash-attn --no-build-isolation
RUN pip install "mamba-ssm[causal-conv1d]" --no-build-isolation

# Copy application
COPY src/huggingface_inference_toolkit huggingface_inference_toolkit
COPY src/huggingface_inference_toolkit/webservice_starlette.py webservice_starlette.py

# Copy entrypoint and change permissions
COPY --chmod=0755 scripts/entrypoint.sh entrypoint.sh

ENV HF_HUB_USER_AGENT_ORIGIN=azure:maap:gpu-cuda:inference:huggingface-inference-toolkit

ENTRYPOINT ["bash", "-c", "./entrypoint.sh"]
