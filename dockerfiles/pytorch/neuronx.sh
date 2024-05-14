#!/bin/bash

set -e

# Install system prerequisites
apt-get update -y \
 && apt-get install -y --no-install-recommends \
    gnupg2 \
    wget

. /etc/os-release
tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -

apt-get update -y \
 && apt-get install -y --no-install-recommends \
    aws-neuronx-dkms=2.* \
    aws-neuronx-collectives=2.* \
    aws-neuronx-runtime-lib=2.* \
    aws-neuronx-tools=2.*

pip install -U pip

# Taken from optimum neuron, tgi dockerfile
pip3 install \
    neuronx-cc==2.13.66.0 \
    torch-neuronx==2.1.2.2.1.0 \
    transformers-neuronx==0.10.0.21 \
    --extra-index-url=https://pip.repos.neuron.amazonaws.com

pip3 install --extra-index-url=https://pip.repos.neuron.amazonaws.com optimum[neuronx,diffusers]

pip install ".[st,torch-neuronx]"

apt-get clean autoremove --yes

rm -rf /var/lib/{apt,cache,log} fi
