#!/bin/bash

set -e

# Install system prerequisites
apt-get update -y \
 && apt-get install -y --no-install-recommends \
    gnupg2 \
    wget

echo "deb https://apt.repos.neuron.amazonaws.com jammy main" > /etc/apt/sources.list.d/neuron.list
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -

apt-get update -y \
 && apt-get install -y --no-install-recommends \
    aws-neuronx-dkms=2.* \
    aws-neuronx-collectives=2.* \
    aws-neuronx-runtime-lib=2.* \
    aws-neuronx-tools=2.*

pip install -U pip

pip3 install neuronx-cc==2.12.68.0 \
    torch-neuronx==1.13.1.1.13.1 \
    transformers-neuronx==0.9.474 \
    --extra-index-url=https://pip.repos.neuron.amazonaws.com

pip3 install optimum[neuronx,diffusers]

pip install ".[st,torch1]"
