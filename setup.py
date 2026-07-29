from __future__ import absolute_import

from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent


def requirements(filename):
    """
    Read a pinned requirements file.

    Those files are the single home for the pins: the image builder installs them directly, before
    the sources are copied, so the heavy downloads sit in a docker layer that a source change does
    not invalidate. Restating the same pins here would give us two places to keep in sync.
    """
    lines = HERE.joinpath(filename).read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith(("#", "-"))]


# We don't declare our dependency on transformers here because we build with
# different packages for different variants

VERSION = "0.5.6"

# Ubuntu packages
# libsndfile1-dev: torchaudio requires the development version of the libsndfile package which can be installed via a system package manager. On Ubuntu it can be installed as follows: apt install libsndfile1-dev
# ffmpeg: ffmpeg is required for audio processing. On Ubuntu it can be installed as follows: apt install ffmpeg
# libavcodec-extra : libavcodec-extra  includes additional codecs for ffmpeg

install_requires = requirements("requirements.txt")

extras = {}

extras["st"] = ["sentence_transformers==5.6.0"]
extras["diffusers"] = ["diffusers==0.39.0", "accelerate==1.14.0"]
extras["torch"] = requirements("requirements-torch.txt")
extras["test"] = requirements("test-requirements.txt")
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
