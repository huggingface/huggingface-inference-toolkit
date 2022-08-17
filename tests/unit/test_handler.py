import tempfile

from transformers.testing_utils import require_torch, slow, require_tf

import pytest
from huggingface_inference_toolkit.handler import (
    HuggingFaceHandler,
    get_inference_handler_either_custom_or_default_handler,
)

from huggingface_inference_toolkit.utils import _is_gpu_available, _load_repository_from_hf


TASK = "text-classification"
MODEL = "hf-internal-testing/tiny-random-distilbert"
INPUT = {"inputs": "My name is Wolfgang and I live in Berlin"}


@require_torch
def test_pt_get_device():
    import torch

    with tempfile.TemporaryDirectory() as tmpdirname:
        # https://github.com/huggingface/infinity/blob/test-ovh/test/integ/utils.py
        storage_dir = _load_repository_from_hf(MODEL, tmpdirname, framework="pytorch")
        h = HuggingFaceHandler(model_dir=str(storage_dir), task=TASK)
        if torch.cuda.is_available():
            assert h.pipeline.model.device == torch.device(type="cuda", index=0)
        else:
            assert h.pipeline.model.device == torch.device(type="cpu")


@require_torch
def test_pt_predict_call():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # https://github.com/huggingface/infinity/blob/test-ovh/test/integ/utils.py
        storage_dir = _load_repository_from_hf(MODEL, tmpdirname, framework="pytorch")
        h = HuggingFaceHandler(model_dir=str(storage_dir), task=TASK)

        prediction = h(INPUT)
        assert "label" in prediction[0]
        assert "score" in prediction[0]


@require_torch
def test_pt_custom_pipeline():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf(
            "philschmid/custom-pipeline-text-classification", tmpdirname, framework="pytorch"
        )
        h = get_inference_handler_either_custom_or_default_handler(str(storage_dir), task="custom")
        assert h(INPUT) == INPUT


@require_torch
def test_pt_sentence_transformers_pipeline():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf(
            "sentence-transformers/all-MiniLM-L6-v2", tmpdirname, framework="pytorch"
        )
        h = get_inference_handler_either_custom_or_default_handler(str(storage_dir), task="sentence-embeddings")
        pred = h(INPUT)
        assert isinstance(pred["embeddings"], list)


@require_tf
def test_tf_get_device():
    import tensorflow as tf

    with tempfile.TemporaryDirectory() as tmpdirname:
        # https://github.com/huggingface/infinity/blob/test-ovh/test/integ/utils.py
        storage_dir = _load_repository_from_hf(MODEL, tmpdirname, framework="tensorflow")
        h = HuggingFaceHandler(model_dir=str(storage_dir), task=TASK)
        if _is_gpu_available():
            assert h.pipeline.device == 0
        else:
            assert h.pipeline.device == -1


@require_tf
def test_tf_predict_call():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # https://github.com/huggingface/infinity/blob/test-ovh/test/integ/utils.py
        storage_dir = _load_repository_from_hf(MODEL, tmpdirname, framework="tensorflow")
        h = HuggingFaceHandler(model_dir=str(storage_dir), task=TASK)

        prediction = h(INPUT)
        assert "label" in prediction[0]
        assert "score" in prediction[0]


@require_tf
def test_tf_custom_pipeline():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf(
            "philschmid/custom-pipeline-text-classification", tmpdirname, framework="tensorflow"
        )
        h = get_inference_handler_either_custom_or_default_handler(str(storage_dir), task="custom")
        assert h(INPUT) == INPUT


@require_tf
def test_tf_sentence_transformers_pipeline():
    # TODO should fail! because TF is not supported yet
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf(
            "sentence-transformers/all-MiniLM-L6-v2", tmpdirname, framework="tensorflow"
        )
        with pytest.raises(Exception) as exc_info:
            h = get_inference_handler_either_custom_or_default_handler(str(storage_dir), task="sentence-embeddings")

        assert "Unknown task sentence-embeddings" in str(exc_info.value)
