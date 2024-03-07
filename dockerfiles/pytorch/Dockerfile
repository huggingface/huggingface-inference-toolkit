ARG BASE_IMAGE=nvidia/cuda:12.1.0-devel-ubuntu22.04

FROM $BASE_IMAGE

ARG NEURONX=0

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
        python3-dev \
        python3-pip \
        python3.11 \
        libsndfile1-dev \
        ffmpeg \
    && apt-get clean autoremove --yes \
    && rm -rf /var/lib/{apt,cache,log}

# Copying only necessary files as filtered by .dockerignore
COPY . .

RUN if [[ "$NEURONX" == "1" ]];then /bin/bash -c "./dockerfiles/pytorch/neuronx.sh";else pip install --no-cache-dir -U pip ".[torch, st, diffusers]";fi

# copy application
COPY src/huggingface_inference_toolkit huggingface_inference_toolkit
COPY src/huggingface_inference_toolkit/webservice_starlette.py webservice_starlette.py

# copy entrypoint and change permissions
COPY --chmod=0755  scripts/entrypoint.sh entrypoint.sh

ENTRYPOINT ["bash", "-c", "./entrypoint.sh"]

RUN apt-get update && apt-get install -y vim
