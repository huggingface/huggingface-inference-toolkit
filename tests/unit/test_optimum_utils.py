import os
import tempfile

import pytest
from transformers.testing_utils import require_torch

from huggingface_inference_toolkit.optimum_utils import (
    get_input_shapes,
    get_optimum_neuron_pipeline,
    is_optimum_neuron_available,
)
from huggingface_inference_toolkit.utils import _load_repository_from_hf

require_inferentia = pytest.mark.skipif(
    not is_optimum_neuron_available(),
    reason="Skipping tests, since optimum neuron is not available or not running on inf2 instances.",
)


REMOTE_NOT_CONVERTED_MODEL = "hf-internal-testing/tiny-random-BertModel"
REMOTE_CONVERTED_MODEL = "optimum/tiny_random_bert_neuron"
TASK = "text-classification"


@require_torch
@require_inferentia
def test_not_supported_task():
    os.environ["HF_TASK"] = "not-supported-task"
    with pytest.raises(Exception):  # noqa
        get_optimum_neuron_pipeline(task=TASK, target_dir=os.getcwd())


@require_torch
@require_inferentia
def test_get_input_shapes_from_file():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_folder = _load_repository_from_hf(
            repository_id=REMOTE_CONVERTED_MODEL,
            target_dir=tmpdirname,
        )
        input_shapes = get_input_shapes(model_dir=storage_folder)
        assert input_shapes["batch_size"] == 1
        assert input_shapes["sequence_length"] == 32


@require_torch
@require_inferentia
def test_get_input_shapes_from_env():
    os.environ["HF_OPTIMUM_BATCH_SIZE"] = "4"
    os.environ["HF_OPTIMUM_SEQUENCE_LENGTH"] = "32"
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_folder = _load_repository_from_hf(
            repository_id=REMOTE_NOT_CONVERTED_MODEL,
            target_dir=tmpdirname,
        )
        input_shapes = get_input_shapes(model_dir=storage_folder)
        assert input_shapes["batch_size"] == 4
        assert input_shapes["sequence_length"] == 32


@require_torch
@require_inferentia
def test_get_optimum_neuron_pipeline_from_converted_model():
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.system(
            f"optimum-cli export neuron --model philschmid/tiny-distilbert-classification --sequence_length 32 --batch_size 1 {tmpdirname}"
        )
        pipe = get_optimum_neuron_pipeline(task=TASK, model_dir=tmpdirname)
        r = pipe("This is a test")

        assert r[0]["score"] > 0.0
        assert isinstance(r[0]["label"], str)


@require_torch
@require_inferentia
def test_get_optimum_neuron_pipeline_from_non_converted_model():
    os.environ["HF_OPTIMUM_SEQUENCE_LENGTH"] = "32"
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_folder = _load_repository_from_hf(
            repository_id=REMOTE_NOT_CONVERTED_MODEL,
            target_dir=tmpdirname,
        )
        pipe = get_optimum_neuron_pipeline(task=TASK, model_dir=storage_folder)
        r = pipe("This is a test")

        assert r[0]["score"] > 0.0
        assert isinstance(r[0]["label"], str)
