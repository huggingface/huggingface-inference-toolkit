from __future__ import absolute_import
from datetime import date
from setuptools import find_packages, setup

# We don't declare our dependency on transformers here because we build with
# different packages for different variants

VERSION = "0.2.0"


# Ubuntu packages
# libsndfile1-dev: torchaudio requires the development version of the libsndfile package which can be installed via a system package manager. On Ubuntu it can be installed as follows: apt install libsndfile1-dev
# ffmpeg: ffmpeg is required for audio processing. On Ubuntu it can be installed as follows: apt install ffmpeg
# libavcodec-extra : libavcodec-extra  inculdes additional codecs for ffmpeg

install_requires = [
    # transformers
    "transformers[sklearn,sentencepiece]==4.27.0",
    "huggingface_hub>=0.20.3",
    # api stuff
    "orjson",
    # "robyn",
    # vision
    "Pillow",
    # speech + torchaudio
    "librosa",
    "pyctcdecode>=0.3.0",
    "phonemizer",
    "ffmpeg"
]

extras = {}

extras["st"] = ["sentence_transformers==2.2.1"]
extras["diffusers"] = ["diffusers==0.26.3", "accelerate==0.27.2"]
extras["torch"] = ["torch>=1.8.0", "torchaudio"]
extras["tensorflow"] = ["tensorflow==2.9.3"]
extras["test"] = [
    "pytest",
    "pytest-xdist",
    "parameterized",
    "psutil",
    "datasets",
    "pytest-sugar",
    "mock==2.0.0",
    "docker",
    "requests",
    "tenacity"
]
extras["quality"] = [
    "isort",
    "ruff"
]

setup(
    name="huggingface-inference-toolkit",
    version=VERSION,
    author="HuggingFace",
    description=".",
    url="",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=install_requires,
    extras_require=extras,
    entry_points={"console_scripts": "serve=sagemaker_huggingface_inference_toolkit.serving:main"},
    python_requires=">=3.8.0",
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
