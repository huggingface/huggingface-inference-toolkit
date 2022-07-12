import tempfile

from transformers.testing_utils import require_torch, slow

import pytest


TASK = "text-classification"
MODEL = "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english"
INPUT = {"inputs": "My name is Wolfgang and I live in Berlin"}
OUTPUT = [
    {"word": "Wolfgang", "score": 0.99, "entity": "I-PER", "index": 4, "start": 11, "end": 19},
    {"word": "Berlin", "score": 0.99, "entity": "I-LOC", "index": 9, "start": 34, "end": 40},
]

# TODO: add fixture for temp dir with model weights in it
# @pytest.fixture()
# def inference_handler():
#     return handler_service.HuggingFaceHandlerService()


def test_get_device_cpu():
    pass
    # storage_folder = snapshot_download(
    #     cache_dir=tmpdirname,
    # )
    # h = Handler(storage_folder)
    # assert h.pipeline.model.device === -1


@slow
def test_get_device_gpu():
    pass
    # storage_folder = snapshot_download(
    #     cache_dir=tmpdirname,
    # )
    # h = Handler(storage_folder)
    # assert h.pipeline.model.device === 0


@require_torch
def test_load():
    pass
    # with tempfile.TemporaryDirectory() as tmpdirname:

    # storage_folder = snapshot_download(
    #     cache_dir=tmpdirname,
    # )
    # h = Handler(storage_folder)
    # assert h.task === "text-classification"  # test sentence-transformers as well


@require_torch
def test_predict_call():
    pass
    # with tempfile.TemporaryDirectory() as tmpdirname:
    # load model from hub and save it to tmpdirname

    # storage_folder = snapshot_download(
    #     cache_dir=tmpdirname,
    # )
    # h = Handler(storage_folder)

    # prediction = h(INPUT)
    # assert "label" in prediction[0]
    # assert "score" in prediction[0]


def test_custom_pipeline():

    # load repository from hub
    repo_dir = None

    # handler = inference_handler(repo_dir)
