ARG BASE_IMAGE=nvidia/cuda:12.1.0-devel-ubuntu22.04

FROM $BASE_IMAGE
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
        curl \
        git \
        git-lfs \
        tar \
        gcc \
        g++ \
        cmake \
        libprotobuf-dev \
        protobuf-compiler \
        python3-venv \
        python3-dev \
        python3.11 \
        libsndfile1-dev \
        ffmpeg \
    && apt-get clean autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

COPY . .

# install wheel and setuptools
RUN python3 -m venv .venv && \
    source .venv/bin/activate && \
    pip install -e ".[torch, st, diffusers]"

# copy application
COPY src/huggingface_inference_toolkit huggingface_inference_toolkit
COPY src/huggingface_inference_toolkit/webservice_starlette.py webservice_starlette.py

#unit tests - tmp dir gets removed in entrypoint.sh
COPY . /tmp/hf-inference-test

# copy entrypoint and change permissions
COPY scripts/entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["bash", "-c", "source .venv/bin/activate && ./entrypoint.sh"]