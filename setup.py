from __future__ import absolute_import
import os
from datetime import date
from setuptools import find_packages, setup

# We don't declare our dependency on transformers here because we build with
# different packages for different variants

VERSION = "0.1.0"


# Ubuntu packages
# libsndfile1-dev: torchaudio requires the development version of the libsndfile package which can be installed via a system package manager. On Ubuntu it can be installed as follows: apt install libsndfile1-dev
# ffmpeg: ffmpeg is required for audio processing. On Ubuntu it can be installed as follows: apt install ffmpeg
# libavcodec-extra : libavcodec-extra  inculdes additional codecs for ffmpeg

install_requires = [
    "huggingface_hub>=0.0.8",
    "orjson",
    "robyn",
    # vision
    "Pillow",
    # speech + torchaudio
    "librosa",
    "pyctcdecode>=0.3.0",
    "phonemizer",
]

extras = {}

# Hugging Face specific dependencies
extras["transformers"] = ["transformers[sklearn,sentencepiece]>=4.17.0"]
# framework specific dependencies
extras["torch"] = ["torch>=1.8.0", "torchaudio"]
extras["tensorflow"] = ["tensorflow>=2.4.0"]
# test and quality
extras["test"] = [
    "pytest",
    "pytest-xdist",
    "parameterized",
    "psutil",
    "datasets",
    "pytest-sugar",
    "mock==2.0.0",
]
extras["quality"] = [
    "black",
    "isort",
    "flake8",
]

setup(
    name="huggingface-inference-toolkit",
    version=VERSION,
    author="HuggingFace",
    description=".",
    # long_description=open("README.md", "r", encoding="utf-8").read(),
    # long_description_content_type="text/markdown",
    # keywords="NLP deep-learning transformer pytorch tensorflow BERT GPT GPT-2 AWS Amazon SageMaker Cloud",
    url="",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=install_requires,
    extras_require=extras,
    entry_points={"console_scripts": "serve=sagemaker_huggingface_inference_toolkit.serving:main"},
    python_requires=">=3.9.0",
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
