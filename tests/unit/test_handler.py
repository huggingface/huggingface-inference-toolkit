import asyncio
import tempfile
from typing import Dict

import numpy as np
import pytest
from transformers.testing_utils import require_torch

from huggingface_inference_toolkit.handler import (
    HuggingFaceHandler,
    get_inference_handler_either_custom_or_default_handler,
)
from huggingface_inference_toolkit.heavy_utils import (
    load_repository_from_hf,
)

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
        storage_dir = load_repository_from_hf(MODEL, tmpdirname, framework="pytorch")
        h = asyncio.run(HuggingFaceHandler.create(str(storage_dir), task=TASK))
        if torch.cuda.is_available():
            assert h.pipeline.model.device == torch.device(type="cuda", index=0)
        else:
            assert h.pipeline.model.device == torch.device(type="cpu")


@require_torch
def test_pt_predict_call(input_data: Dict[str, str]) -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        # https://github.com/huggingface/infinity/blob/test-ovh/test/integ/utils.py
        storage_dir = load_repository_from_hf(MODEL, tmpdirname, framework="pytorch")
        h = asyncio.run(HuggingFaceHandler.create(str(storage_dir), task=TASK))

        prediction = h(input_data)
        assert "label" in prediction[0]
        assert "score" in prediction[0]


@require_torch
def test_pt_custom_pipeline(input_data: Dict[str, str]) -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = load_repository_from_hf(
            "philschmid/custom-pipeline-text-classification",
            tmpdirname,
            framework="pytorch",
        )
        h = asyncio.run(get_inference_handler_either_custom_or_default_handler(str(storage_dir), task="custom"))
        assert h(input_data) == input_data


@require_torch
def test_pt_sentence_transformers_pipeline(input_data: Dict[str, str]) -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = load_repository_from_hf(
            "sentence-transformers/all-MiniLM-L6-v2", tmpdirname, framework="pytorch"
        )
        h = asyncio.run(get_inference_handler_either_custom_or_default_handler(str(storage_dir), task="sentence-embeddings"))
        pred = h(input_data)
        assert isinstance(pred["embeddings"], np.ndarray)


