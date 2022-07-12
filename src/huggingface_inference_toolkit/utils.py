import importlib.util
import logging
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi
from huggingface_hub.file_download import cached_download, hf_hub_url
from transformers import pipeline
from transformers.file_utils import is_tf_available, is_torch_available
from transformers.pipelines import Conversation, Pipeline


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(name)s | %(levelname)s | %(message)s", level=logging.INFO)


if is_tf_available():
    import tensorflow as tf

if is_torch_available():
    import torch

_optimum_available = importlib.util.find_spec("optimum") is not None
_sentence_transformers = importlib.util.find_spec("_sentence-transformers") is not None


def is_optimum_available():
    return _optimum_available


def is_sentence_transformers():
    return _sentence_transformers


def wrap_conversation_pipeline(pipeline):
    def wrapped_pipeline(inputs, *args, **kwargs):
        converted_input = Conversation(
            inputs["text"],
            past_user_inputs=inputs.get("past_user_inputs", []),
            generated_responses=inputs.get("generated_responses", []),
        )
        prediction = pipeline(converted_input, *args, **kwargs)
        return {
            "generated_text": prediction.generated_responses[-1],
            "conversation": {
                "past_user_inputs": prediction.past_user_inputs,
                "generated_responses": prediction.generated_responses,
            },
        }

    return wrapped_pipeline


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


def check_and_register_custom_pipeline_from_directory(model_dir):
    """
    Checks if a custom pipeline is available and registers it if so.
    """
    NotImplementedError("Not implemented yet")


def get_pipeline(task: str, device: int, model_dir: Path, **kwargs) -> Pipeline:
    """
    create pipeline class for a specific task based on local saved model
    """
    if task is None:
        raise EnvironmentError(
            "The task for this model is not set: Please set one: https://huggingface.co/docs#how-is-a-models-type-of-inference-api-and-widget-determined"
        )
    # define tokenizer or feature extractor as kwargs to load it the pipeline correctly
    if task in {
        "automatic-speech-recognition",
        "image-segmentation",
        "image-classification",
        "audio-classification",
        "object-detection",
        "zero-shot-image-classification",
    }:
        kwargs["feature_extractor"] = model_dir
    else:
        kwargs["tokenizer"] = model_dir

    transformers_pipeline = True
    if transformers_pipeline:
        hf_pipeline = pipeline(task=task, model=model_dir, device=device, **kwargs)
    # add check for optimum accelerated pipeline
    elif is_optimum_available():
        pass
        # TODO: add check for optimum accelerated pipeline
    elif is_sentence_transformers():
        pass
    # TODO: add check for sentence transformers pipeline

    # wrapp specific pipeline to support better ux
    if task == "conversational":
        hf_pipeline = wrap_conversation_pipeline(hf_pipeline)

    return hf_pipeline
