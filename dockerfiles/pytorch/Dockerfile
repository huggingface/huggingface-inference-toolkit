ARG BASE_IMAGE=nvidia/cuda:12.1.0-devel-ubuntu22.04
ARG RUNTIME_IMAGE=nvidia/cuda:12.1.0-runtime-ubuntu22.04

# ── builder: compile kenlm and install every dependency into a venv ──────────────────────────────
FROM ${BASE_IMAGE} AS builder
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y upgrade --only-upgrade systemd openssl cryptsetup && \
    apt-get install -y \
      build-essential \
      cmake \
      curl \
      git \
      gcc \
      g++ \
      libprotobuf-dev \
      protobuf-compiler \
      python3.11 \
      python3.11-dev \
      python3.11-venv \
      libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip

# The heavy dependencies first, straight from the files setup.py declares them in, so this layer
# survives a source change. TORCH_REQUIREMENTS picks the flavour: the CPU builds swap in wheels
# without the bundled CUDA runtime.
ARG TORCH_REQUIREMENTS=requirements-torch.txt
COPY requirements.txt ${TORCH_REQUIREMENTS} ./
RUN pip install --no-cache-dir -r requirements.txt -r ${TORCH_REQUIREMENTS}

# Then the toolkit and its two small extras. The core and torch pins above already satisfy what the
# package asks for, so nothing heavy is resolved again here.
# Copying only necessary files as filtered by .dockerignore
COPY . .
RUN pip install --no-cache-dir ".[st,diffusers]"

# ── runtime: no compilers, no dev headers, CUDA runtime instead of devel ─────────────────────────
FROM ${RUNTIME_IMAGE} AS runtime
SHELL ["/bin/bash", "-c"]

LABEL maintainer="Hugging Face"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y upgrade --only-upgrade systemd openssl cryptsetup && \
    apt-get install -y \
      python3.11 \
      bzip2 \
      curl \
      git \
      git-lfs \
      tar \
      libsndfile1 \
      libgomp1 \
      ffmpeg \
    && apt-get clean autoremove --yes \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# The venv is the only thing carried over from the builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application
COPY src/huggingface_inference_toolkit huggingface_inference_toolkit
COPY src/huggingface_inference_toolkit/webservice_starlette.py webservice_starlette.py

# Copy entrypoint and change permissions
COPY --chmod=0755 scripts/entrypoint.sh entrypoint.sh

ENTRYPOINT ["bash", "-c", "./entrypoint.sh"]
