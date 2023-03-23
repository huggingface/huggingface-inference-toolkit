FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

LABEL maintainer="Hugging Face"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get -y upgrade --only-upgrade systemd openssl cryptsetup \
    && apt-get install -y \
    bzip2 \
    curl \
    git \
    git-lfs \
    tar \
    gcc \
    g++ \
    # audio
    libsndfile1-dev \
    ffmpeg \
    && apt-get clean autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

# install micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH}"

RUN curl -L https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xj "bin/micromamba" \
    && touch /root/.bashrc \
    && ./bin/micromamba shell init -s bash -p /opt/conda  \
    && grep -v '[ -z "\$PS1" ] && return' /root/.bashrc  > /opt/conda/bashrc

WORKDIR /app

# install base python dependencies
COPY dockerfiles/pytorch/gpu/environment.yaml /app/environment.yaml
RUN micromamba install -y -n base -f environment.yaml \
    && rm environment.yaml \
    && micromamba clean --all --yes

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
