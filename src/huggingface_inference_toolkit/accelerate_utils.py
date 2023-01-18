import importlib.util
from typing import Union
from pathlib import Path
from transformers import AutoConfig

_accelerate_available = importlib.util.find_spec("accelerate") is not None


def is_accelerate_available():
    return _accelerate_available


def _get_model_config(path_to_model_artifacts: Union[str, Path]):
    """
    extracts the model config from the model artifacts
    """
    return AutoConfig.from_pretrained(path_to_model_artifacts)


# created based on the following search query: https://github.com/search?q=repo%3Ahuggingface%2Ftransformers+_no_split_modules&type=code&p=1
SUPPORTED_MODEL_PARALLELISM_ARCHITECTURES = [
    "T5",
    "CLIP",
    "Longformer",
    "M2M100",
    "Bart",
    "OPT",
    "GPT",
    "BigBirdPegasus",
    "MT5",
    "CodeGen",
    "LongT5",
    "Bloom",
]


def check_support_for_model_parallelism(model_dir):
    """Check if the model supports model parallelism by the device_map "auto" option from accelerate"""
    # get the model config from the model directory
    config = _get_model_config(model_dir)

    # check if any of the architectures in the config is starts with one of the supported architectures
    # from SUPPORTED_MODEL_PARALLELISM_ARCHITECTURES
    if any(config.architectures[0].startswith(arch) for arch in SUPPORTED_MODEL_PARALLELISM_ARCHITECTURES):
        return True
    # default to false
    return False
