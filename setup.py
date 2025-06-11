from __future__ import absolute_import
import os
from setuptools import find_packages, setup

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirements_path = f"{lib_folder}/requirements.txt"
install_requires = [] # Here we'll add: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

test_requirements_path =  f"{lib_folder}/test-requirements.txt"
if os.path.isfile(test_requirements_path):
    with open(test_requirements_path) as f:
        test_requirements = f.read().splitlines()

# We don't declare our dependency on transformers here because we build with
# different packages for different variants

VERSION = "0.5.5"

# Ubuntu packages
# libsndfile1-dev: torchaudio requires the development version of the libsndfile package which can be installed via a system package manager. On Ubuntu it can be installed as follows: apt install libsndfile1-dev
# ffmpeg: ffmpeg is required for audio processing. On Ubuntu it can be installed as follows: apt install ffmpeg
# libavcodec-extra : libavcodec-extra  includes additional codecs for ffmpeg

extras = {}
extras["st"] = ["sentence_transformers==4.0.2"]
extras["diffusers"] = ["diffusers==0.33.1", "accelerate==1.6.0"]
# Includes `peft` as PEFT requires `torch` so having `peft` as a core dependency
# means that `torch` will be installed even if the `torch` extra is not specified.
extras["torch"] = ["torch==2.5.1", "torchvision", "torchaudio", "peft==0.15.1"]
extras["quality"] = ["isort", "ruff"]
extras["test"] = test_requirements
extras["inf2"] = ["optimum-neuron"]
extras["google"] = ["google-cloud-storage", "crcmod==1.7"]

setup(
    name="huggingface-inference-toolkit",
    version=VERSION,
    author="Hugging Face",
    description="Hugging Face Inference Toolkit is for serving 🤗 Transformers models in containers.",
    url="https://github.com/huggingface/huggingface-inference-toolkit",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=install_requires,
    extras_require=extras,
    entry_points={"console_scripts": "serve=sagemaker_huggingface_inference_toolkit.serving:main"},
    python_requires=">=3.9",
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
