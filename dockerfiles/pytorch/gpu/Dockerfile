FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 as builder
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
        python3.11 \
        python3-pip \
        python3.11-venv \
        libsndfile1-dev \
        ffmpeg \
    && apt-get clean autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

# install dependencies
COPY dockerfiles/pytorch/gpu/requirements.txt requirements-docker.txt
COPY requirements.txt requirements-toolkit.txt

# install wheel and setuptools
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    source $HOME/.cargo/env && \
    uv venv && \
    source .venv/bin/activate && \
    uv pip install --no-cache-dir -r requirements-docker.txt && \
    uv pip install --no-cache-dir -r requirements-toolkit.txt

### Runner

FROM nvidia/cuda:12.1.0-base-ubuntu22.04 as runner
SHELL ["/bin/bash", "-c"]

WORKDIR /app

ENV TORCH_USE_CUDA_DSA=1

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3.10-venv \
    curl \
    ffmpeg \
    && apt-get clean autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}
    
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    source $HOME/.cargo/env && \
    source .venv/bin/activate && \
    ls -all

# install dependencies
COPY --from=builder /app .

# copy application
COPY src/huggingface_inference_toolkit huggingface_inference_toolkit
COPY src/huggingface_inference_toolkit/webservice_starlette.py webservice_starlette.py

#unit tests
COPY . /tmp/hf-inference-test

# copy entrypoint and change permissions
COPY scripts/entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["bash", "-c", "source .venv/bin/activate && ./entrypoint.sh"]