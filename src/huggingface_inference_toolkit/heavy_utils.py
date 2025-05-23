# Heavy because they consume a lot of memory and we want to import them as late as possible to reduce the footprint
# Transformers / Sentence transformers utils. This module should be imported as late as possible
# to reduce the memory footprint of a worker: we don't bother handling the uncaching/gc collecting because
# we want to combine it with idle unload: the gunicorn worker will just suppress itself when unused freeing the memory
# as wished
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import HfApi, login, snapshot_download

from transformers import WhisperForConditionalGeneration, pipeline
from transformers.file_utils import is_tf_available, is_torch_available
from transformers.pipelines import Pipeline

from huggingface_inference_toolkit.diffusers_utils import (
    get_diffusers_pipeline,
    is_diffusers_available,
)
from huggingface_inference_toolkit.logging import logger
from huggingface_inference_toolkit.sentence_transformers_utils import (
    get_sentence_transformers_pipeline,
    is_sentence_transformers_available,
)
from huggingface_inference_toolkit.utils import create_artifact_filter
from huggingface_inference_toolkit.optimum_utils import (
    get_optimum_neuron_pipeline,
    is_optimum_neuron_available,
)


def load_repository_from_hf(
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
    logger.info(f"Ignore regex pattern for files, which are not downloaded: { ', '.join(ignore_regex) }")

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


def get_device():
    """
    The get device function will return the device for the DL Framework.
    """
    gpu = _is_gpu_available()

    if gpu:
        return 0
    else:
        return -1


if is_tf_available():
    import tensorflow as tf


if is_torch_available():
    import torch


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


def get_pipeline(
        task: Union[str, None],
        model_dir: Path,
        **kwargs,
) -> Pipeline:
    """
    create pipeline class for a specific task based on local saved model
    """

    # import as late as possible to reduce the footprint

    if task is None:
        raise EnvironmentError(
            "The task for this model is not set: Please set one: https://huggingface.co/docs#how-is-a-models-type-of-inference-api-and-widget-determined"
        )

    if task == "conversational":
        task = "text-generation"

    if is_optimum_neuron_available():
        logger.info("Using device Neuron")
        return get_optimum_neuron_pipeline(task=task, model_dir=model_dir)

    device = get_device()
    logger.info(f"Using device {'GPU' if device == 0 else 'CPU'}")

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
    elif task not in {"image-text-to-text", "image-to-text", "text-to-image"}:
        kwargs["tokenizer"] = model_dir

    if is_sentence_transformers_available() and task in [
        "sentence-similarity",
        "sentence-embeddings",
        "sentence-ranking",
    ]:
        hf_pipeline = get_sentence_transformers_pipeline(task=task, model_dir=model_dir, device=device, **kwargs)
    elif is_diffusers_available() and task == "text-to-image":
        hf_pipeline = get_diffusers_pipeline(task=task, model_dir=model_dir, device=device, **kwargs)
    else:
        hf_pipeline = pipeline(task=task, model=model_dir, device=device, **kwargs)

    if task == "automatic-speech-recognition" and isinstance(hf_pipeline.model, WhisperForConditionalGeneration):
        # set chunk length to 30s for whisper to enable long audio files
        hf_pipeline._preprocess_params["chunk_length_s"] = 30
        hf_pipeline.model.config.forced_decoder_ids = hf_pipeline.tokenizer.get_decoder_prompt_ids(
            language="english", task="transcribe"
        )
    return hf_pipeline  # type: ignore