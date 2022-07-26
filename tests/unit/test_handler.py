import tempfile

from transformers.testing_utils import require_torch, slow

import pytest
from huggingface_inference_toolkit.handler import (
    HuggingFaceHandler,
    get_inference_handler_either_custom_or_default_handler,
)

from huggingface_inference_toolkit.utils import _load_repository_from_hf


TASK = "text-classification"
MODEL = "hf-internal-testing/tiny-random-distilbert"
INPUT = {"inputs": "My name is Wolfgang and I live in Berlin"}


def test_get_device_cpu():
    import torch

    with tempfile.TemporaryDirectory() as tmpdirname:
        # https://github.com/huggingface/infinity/blob/test-ovh/test/integ/utils.py
        storage_dir = _load_repository_from_hf(MODEL, tmpdirname, framework="pytorch")
        h = HuggingFaceHandler(model_dir=str(storage_dir), task=TASK)
        assert h.pipeline.model.device == torch.device(type="cpu")


@slow
def test_get_device_gpu():
    import torch

    with tempfile.TemporaryDirectory() as tmpdirname:
        # https://github.com/huggingface/infinity/blob/test-ovh/test/integ/utils.py
        storage_dir = _load_repository_from_hf(MODEL, tmpdirname, framework="pytorch")
        h = HuggingFaceHandler(model_dir=str(storage_dir), task=TASK)
        assert h.pipeline.model.device == torch.device(type="cuda")


@require_torch
def test_predict_call():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # https://github.com/huggingface/infinity/blob/test-ovh/test/integ/utils.py
        storage_dir = _load_repository_from_hf(MODEL, tmpdirname, framework="pytorch")
        h = HuggingFaceHandler(model_dir=str(storage_dir), task=TASK)

        prediction = h(INPUT)
        assert "label" in prediction[0]
        assert "score" in prediction[0]


@require_torch
def test_custom_pipeline():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf(
            "philschmid/custom-pipeline-text-classification", tmpdirname, framework="pytorch"
        )
        h = get_inference_handler_either_custom_or_default_handler(str(storage_dir), task="custom")
        assert h(INPUT) == INPUT


@require_torch
def test_sentence_transformers_pipeline():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf("sentence-transformers/all-MiniLM-L6-v2", tmpdirname, framework="pytorch")
        h = get_inference_handler_either_custom_or_default_handler(str(storage_dir), task="sentence-embeddings")
        pred = h(INPUT)
        assert isinstance(pred["embeddings"], list)
