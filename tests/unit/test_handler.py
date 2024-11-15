import tempfile

import pytest
from typing import Dict
from transformers.testing_utils import require_tf, require_torch

from huggingface_inference_toolkit.handler import (
    HuggingFaceHandler,
    get_inference_handler_either_custom_or_default_handler,
)
from huggingface_inference_toolkit.utils import (
    _is_gpu_available,
    _load_repository_from_hf,
)
from huggingface_inference_toolkit.logging import logger

TASK = "text-classification"
MODEL = "hf-internal-testing/tiny-random-distilbert"


# defined as fixture because it's modified on `pop`
@pytest.fixture
def input_data():
    return {"inputs": "My name is Wolfgang and I live in Berlin"}


@require_torch
def test_pt_get_device() -> None:
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
def test_pt_predict_call(input_data: Dict[str, str]) -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        # https://github.com/huggingface/infinity/blob/test-ovh/test/integ/utils.py
        storage_dir = _load_repository_from_hf(MODEL, tmpdirname, framework="pytorch")
        h = HuggingFaceHandler(model_dir=str(storage_dir), task=TASK)

        prediction = h(input_data)
        assert "label" in prediction[0]
        assert "score" in prediction[0]


@require_torch
def test_pt_custom_pipeline(input_data: Dict[str, str]) -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf(
            "philschmid/custom-pipeline-text-classification",
            tmpdirname,
            framework="pytorch",
        )
        h = get_inference_handler_either_custom_or_default_handler(str(storage_dir), task="custom")
        assert h(input_data) == input_data


@require_torch
def test_pt_sentence_transformers_pipeline(input_data: Dict[str, str]) -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf(
            "sentence-transformers/all-MiniLM-L6-v2", tmpdirname, framework="pytorch"
        )
        h = get_inference_handler_either_custom_or_default_handler(str(storage_dir), task="sentence-embeddings")
        pred = h(input_data)
        assert isinstance(pred["embeddings"], list)


@require_tf
def test_tf_get_device():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # https://github.com/huggingface/infinity/blob/test-ovh/test/integ/utils.py
        storage_dir = _load_repository_from_hf(MODEL, tmpdirname, framework="tensorflow")
        h = HuggingFaceHandler(model_dir=str(storage_dir), task=TASK)
        if _is_gpu_available():
            assert h.pipeline.device == 0
        else:
            assert h.pipeline.device == -1


@require_tf
def test_tf_predict_call(input_data: Dict[str, str]) -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        # https://github.com/huggingface/infinity/blob/test-ovh/test/integ/utils.py
        storage_dir = _load_repository_from_hf(MODEL, tmpdirname, framework="tensorflow")
        handler = HuggingFaceHandler(model_dir=str(storage_dir), task=TASK, framework="tf")

        prediction = handler(input_data)
        assert "label" in prediction[0]
        assert "score" in prediction[0]


@require_tf
def test_tf_custom_pipeline(input_data: Dict[str, str]) -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf(
            "philschmid/custom-pipeline-text-classification",
            tmpdirname,
            framework="tensorflow",
        )
        h = get_inference_handler_either_custom_or_default_handler(str(storage_dir), task="custom")
        assert h(input_data) == input_data


@require_tf
def test_tf_sentence_transformers_pipeline():
    # TODO should fail! because TF is not supported yet
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf(
            "sentence-transformers/all-MiniLM-L6-v2", tmpdirname, framework="tensorflow"
        )
        with pytest.raises(Exception) as _exc_info:
            get_inference_handler_either_custom_or_default_handler(str(storage_dir), task="sentence-embeddings")
