import importlib.util
import math
import os
import sys
from pathlib import Path

from huggingface_inference_toolkit.const import HF_DEFAULT_PIPELINE_NAME, HF_MODULE_NAME
from huggingface_inference_toolkit.logging import logger

_optimum_available = importlib.util.find_spec("optimum") is not None


def is_optimum_available():
    return False
    # TODO: change when supported
    # return _optimum_available


framework2weight = {
    "pytorch": "pytorch*",
    "tensorflow": "tf*",
    "tf": "tf*",
    "pt": "pytorch*",
    "flax": "flax*",
    "rust": "rust*",
    "onnx": "*onnx*",
    "safetensors": "*safetensors",
    "coreml": "*mlmodel",
    "tflite": "*tflite",
    "savedmodel": "*tar.gz",
    "openvino": "*openvino*",
    "ckpt": "*ckpt",
}


def create_artifact_filter(framework):
    """
    Returns a list of regex pattern based on the DL Framework. which will be to used to ignore files when downloading
    """
    ignore_regex_list = list(set(framework2weight.values()))

    pattern = framework2weight.get(framework, None)
    if pattern in ignore_regex_list:
        ignore_regex_list.remove(pattern)
        return ignore_regex_list
    else:
        return []


def check_and_register_custom_pipeline_from_directory(model_dir):
    """
    Checks if a custom pipeline is available and registers it if so.
    """
    # path to custom handler
    custom_module = Path(model_dir).joinpath(HF_DEFAULT_PIPELINE_NAME)
    legacy_module = Path(model_dir).joinpath("pipeline.py")
    if custom_module.is_file():
        logger.info(f"Found custom pipeline at {custom_module}")
        spec = importlib.util.spec_from_file_location(HF_MODULE_NAME, custom_module)
        if spec:
            # add the whole directory to path for submodlues
            sys.path.insert(0, model_dir)
            # import custom handler
            handler = importlib.util.module_from_spec(spec)
            sys.modules[HF_MODULE_NAME] = handler
            spec.loader.exec_module(handler)
            # init custom handler with model_dir
            custom_pipeline = handler.EndpointHandler(model_dir)

    elif legacy_module.is_file():
        logger.warning(
            """You are using a legacy custom pipeline.
            Please update to the new format.
            See documentation for more information."""
        )
        spec = importlib.util.spec_from_file_location("pipeline.PreTrainedPipeline", legacy_module)
        if spec:
            # add the whole directory to path for submodlues
            sys.path.insert(0, model_dir)
            # import custom handler
            pipeline = importlib.util.module_from_spec(spec)
            sys.modules["pipeline.PreTrainedPipeline"] = pipeline
            spec.loader.exec_module(pipeline)
            # init custom handler with model_dir
            custom_pipeline = pipeline.PreTrainedPipeline(model_dir)
    else:
        logger.info(f"No custom pipeline found at {custom_module}")
        custom_pipeline = None
    return custom_pipeline


def _coerce_query_param(value: str):
    """
    Coerce one query-string value to the type a pipeline expects, leaving it a string when it is
    not a number or a boolean.

    `str.isnumeric()` is the wrong test for "is this an int": it is False for `-1` and for any
    float, so those reached the pipeline as strings, and it is True for characters like `²` and
    `½`, which `int()` then refuses. Asking `int()` and `float()` directly answers exactly the
    question we care about.
    """
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        number = float(value)
    except ValueError:
        return value
    # `float` also accepts "nan", "inf" and "infinity". None of them is a meaningful pipeline
    # parameter and JSON cannot represent them, so keep those as strings.
    return number if math.isfinite(number) else value


def convert_params_to_int_or_bool(params):
    """Converts query params to int, float or bool if possible"""
    return {k: _coerce_query_param(v) for k, v in params.items()}


def should_discard_left() -> bool:
    return os.getenv("DISCARD_LEFT", "0").lower() in ["true", "yes", "1"]
