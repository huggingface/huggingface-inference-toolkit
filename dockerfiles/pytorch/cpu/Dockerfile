FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]

LABEL maintainer="Hugging Face"

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_USE_CUDA_DSA=1

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

# install dependencies
COPY dockerfiles/pytorch/gpu/requirements.txt requirements-docker.txt
COPY requirements.txt requirements-toolkit.txt

# install wheel and setuptools
RUN python3 -m venv .venv && \
    source .venv/bin/activate && \
    pip install wheel && \
    pip install --no-cache-dir -r requirements-docker.txt && \
    pip install --no-cache-dir -r requirements-toolkit.txt

# copy application
COPY src/huggingface_inference_toolkit huggingface_inference_toolkit
COPY src/huggingface_inference_toolkit/webservice_starlette.py webservice_starlette.py

#unit tests
COPY . /tmp/hf-inference-test

# copy entrypoint and change permissions
COPY scripts/entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["bash", "-c", "source .venv/bin/activate && ./entrypoint.sh"]