FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 as builder
SHELL ["/bin/bash", "-c"]

LABEL maintainer="Hugging Face"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /build

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
    python3.10-venv \
    # audio
    libsndfile1-dev \
    ffmpeg \
    && apt-get clean autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

# install dependencies
COPY dockerfiles/pytorch/gpu/requirements.txt requirements-docker.txt
COPY requirements.txt requirements-toolkit.txt

# install wheel and setuptools
RUN pip install -U pip && \
    pip download --dest ./wheels -r requirements-docker.txt && \
    pip download --dest ./wheels -r requirements-toolkit.txt && \
    pip install --no-index --find-links=./wheels -r requirements-docker.txt && \
    pip install --no-index --find-links=./wheels -r requirements-toolkit.txt

### Runner

FROM nvidia/cuda:12.3.1-base-ubuntu22.04 as runner
SHELL ["/bin/bash", "-c"]

WORKDIR /app
COPY --from=builder /build/wheels /app/wheels

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    python3 \
    python3-pip

# install dependencies
COPY dockerfiles/pytorch/gpu/requirements.txt requirements-docker.txt
COPY requirements.txt requirements-toolkit.txt

RUN pip install -U pip && \
    pip install --no-index --find-links=/app/wheels -r requirements-docker.txt && \
    pip install --no-index --find-links=/app/wheels -r requirements-toolkit.txt

# copy application
COPY src/huggingface_inference_toolkit huggingface_inference_toolkit
COPY src/huggingface_inference_toolkit/webservice_starlette.py webservice_starlette.py

# copy entrypoint and change permissions
COPY scripts/entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["bash", "-c", "./entrypoint.sh"]