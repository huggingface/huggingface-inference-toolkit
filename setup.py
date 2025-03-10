from __future__ import absolute_import

from setuptools import find_packages, setup

# We don't declare our dependency on transformers here because we build with
# different packages for different variants

VERSION = "0.5.5"

# Ubuntu packages
# libsndfile1-dev: torchaudio requires the development version of the libsndfile package which can be installed via a system package manager. On Ubuntu it can be installed as follows: apt install libsndfile1-dev
# ffmpeg: ffmpeg is required for audio processing. On Ubuntu it can be installed as follows: apt install ffmpeg
# libavcodec-extra : libavcodec-extra  includes additional codecs for ffmpeg

install_requires = [
    "transformers[sklearn,sentencepiece,audio,vision]==4.49.0",
    "huggingface_hub[hf_transfer]==0.27.1",
    # vision
    "Pillow",
    "librosa",
    # speech + torchaudio
    "pyctcdecode>=0.3.0",
    "phonemizer",
    "ffmpeg",
    # web api
    "starlette",
    "uvicorn",
    "pandas",
    "orjson",
]

extras = {}

extras["st"] = ["sentence_transformers==3.3.1"]
extras["diffusers"] = ["diffusers==0.32.1", "accelerate==1.2.1"]
# Includes `peft` as PEFT requires `torch` so having `peft` as a core dependency
# means that `torch` will be installed even if the `torch` extra is not specified.
extras["torch"] = ["torch==2.3.1", "torchvision", "torchaudio", "peft==0.14.0"]
extras["test"] = [
    "pytest==7.2.1",
    "pytest-xdist",
    "parameterized",
    "psutil",
    "datasets",
    "pytest-sugar",
    "mock==2.0.0",
    "docker",
    "requests",
    "tenacity",
]
extras["quality"] = ["isort", "ruff"]
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
    python_requires=">=3.8",
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
