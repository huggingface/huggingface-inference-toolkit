import importlib.util
import sys
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import HfApi, login, snapshot_download
from transformers import WhisperForConditionalGeneration, pipeline
from transformers.file_utils import is_tf_available, is_torch_available
from transformers.pipelines import Pipeline

from huggingface_inference_toolkit.const import HF_DEFAULT_PIPELINE_NAME, HF_MODULE_NAME
from huggingface_inference_toolkit.diffusers_utils import (
    get_diffusers_pipeline,
    is_diffusers_available,
)
from huggingface_inference_toolkit.logging import logger
from huggingface_inference_toolkit.optimum_utils import (
    get_optimum_neuron_pipeline,
    is_optimum_neuron_available,
)
from huggingface_inference_toolkit.sentence_transformers_utils import (
    get_sentence_transformers_pipeline,
    is_sentence_transformers_available,
)

if is_tf_available():
    import tensorflow as tf

if is_torch_available():
    import torch

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


def _is_gpu_available():
    """
    checks if a gpu is available.
    """
    if is_tf_available():
        return True if len(tf.config.list_physical_devices("GPU")) > 0 else False
    elif is_torch_available():
        return torch.cuda.is_available()
    else:
        raise RuntimeError(
            "At least one of TensorFlow 2.0 or PyTorch should be installed. "
            "To install TensorFlow 2.0, read the instructions at https://www.tensorflow.org/install/ "
            "To install PyTorch, read the instructions at https://pytorch.org/."
        )


def _get_framework():
    """
    extracts which DL framework is used for inference, if both are installed use pytorch
    """

    if is_torch_available():
        return "pytorch"
    elif is_tf_available():
        return "tensorflow"
    else:
        raise RuntimeError(
            "At least one of TensorFlow 2.0 or PyTorch should be installed. "
            "To install TensorFlow 2.0, read the instructions at https://www.tensorflow.org/install/ "
            "To install PyTorch, read the instructions at https://pytorch.org/."
        )


def _load_repository_from_hf(
    repository_id: Optional[str] = None,
    target_dir: Optional[Union[str, Path]] = None,
    framework: Optional[str] = None,
    revision: Optional[str] = None,
    hf_hub_token: Optional[str] = None,
):
    """
    Load a model from huggingface hub.
    """

    if hf_hub_token is not None:
        login(token=hf_hub_token)

    if framework is None:
        framework = _get_framework()

    if isinstance(target_dir, str):
        target_dir = Path(target_dir)

    # create workdir
    if not target_dir.exists():
        target_dir.mkdir(parents=True)

    # check if safetensors weights are available
    if framework == "pytorch":
        files = HfApi().model_info(repository_id).siblings
        if any(f.rfilename.endswith("safetensors") for f in files):
            framework = "safetensors"

    # create regex to only include the framework specific weights
    ignore_regex = create_artifact_filter(framework)
    logger.info(
        f"Ignore regex pattern for files, which are not downloaded: { ', '.join(ignore_regex) }"
    )

    # Download the repository to the workdir and filter out non-framework
    # specific weights
    snapshot_download(
        repo_id=repository_id,
        revision=revision,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=ignore_regex,
    )

    return target_dir


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
        spec = importlib.util.spec_from_file_location(
            "pipeline.PreTrainedPipeline", legacy_module
        )
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


def get_device():
    """
    The get device function will return the device for the DL Framework.
    """
    gpu = _is_gpu_available()

    if gpu:
        return 0
    else:
        return -1


def get_pipeline(
    task: str,
    model_dir: Path,
    **kwargs,
) -> Pipeline:
    """
    create pipeline class for a specific task based on local saved model
    """
    device = get_device()
    if is_optimum_neuron_available():
        logger.info("Using device Neuron")
    else:
        logger.info(f"Using device { 'GPU' if device == 0 else 'CPU'}")

    if task is None:
        raise EnvironmentError(
            "The task for this model is not set: Please set one: https://huggingface.co/docs#how-is-a-models-type-of-inference-api-and-widget-determined"
        )
    # define tokenizer or feature extractor as kwargs to load it the pipeline
    # correctly
    if task in {
        "automatic-speech-recognition",
        "image-segmentation",
        "image-classification",
        "audio-classification",
        "object-detection",
        "zero-shot-image-classification",
    }:
        kwargs["feature_extractor"] = model_dir
    elif task in {"image-to-text"}:
        pass
    elif task == "conversational":
        task = "text-generation"
    else:
        kwargs["tokenizer"] = model_dir

    if is_optimum_neuron_available():
        hf_pipeline = get_optimum_neuron_pipeline(task=task, model_dir=model_dir)
    elif is_sentence_transformers_available() and task in [
        "sentence-similarity",
        "sentence-embeddings",
        "sentence-ranking",
    ]:
        hf_pipeline = get_sentence_transformers_pipeline(
            task=task, model_dir=model_dir, device=device, **kwargs
        )
    elif is_diffusers_available() and task == "text-to-image":
        hf_pipeline = get_diffusers_pipeline(
            task=task, model_dir=model_dir, device=device, **kwargs
        )
    else:
        hf_pipeline = pipeline(task=task, model=model_dir, device=device, **kwargs)

    if task == "automatic-speech-recognition" and isinstance(
        hf_pipeline.model, WhisperForConditionalGeneration
    ):
        # set chunk length to 30s for whisper to enable long audio files
        hf_pipeline._preprocess_params["chunk_length_s"] = 30
        hf_pipeline.model.config.forced_decoder_ids = (
            hf_pipeline.tokenizer.get_decoder_prompt_ids(
                language="english", task="transcribe"
            )
        )
    return hf_pipeline


def convert_params_to_int_or_bool(params):
    """Converts query params to int or bool if possible"""
    for k, v in params.items():
        if v.isnumeric():
            params[k] = int(v)
        if v == "false":
            params[k] = False
        if v == "true":
            params[k] = True
    return params


def strtobool(val: str) -> bool:
    """Convert a string representation of truth to True or False booleans.
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.

    Raises:
        ValueError: if 'val' is anything else.

    Note:
        Function `strtobool` copied and adapted from `distutils`, as it's deprected from Python 3.10 onwards.

    References:
        - https://github.com/python/cpython/blob/48f9d3e3faec5faaa4f7c9849fecd27eae4da213/Lib/distutils/util.py#L308-L321
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError(
        f"Invalid truth value, it should be a string but {val} was provided instead."
    )
