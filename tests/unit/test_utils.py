import os
import tempfile

from transformers import pipeline
from transformers.file_utils import is_torch_available
from transformers.testing_utils import require_tf, require_torch, slow

from huggingface_inference_toolkit.handler import get_inference_handler_either_custom_or_default_handler
from huggingface_inference_toolkit.utils import (
    _get_framework,
    _is_gpu_available,
    _load_repository_from_hf,
    check_and_register_custom_pipeline_from_directory,
    get_pipeline,
    wrap_conversation_pipeline,
)


MODEL = "lysandre/tiny-bert-random"
TASK = "text-classification"
TASK_MODEL = "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english"


def test_load_revision_repository_from_hf():
    MODEL = "lysandre/tiny-bert-random"
    REVISION = "eb4c77816edd604d0318f8e748a1c606a2888493"
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_folder = _load_repository_from_hf(MODEL, tmpdirname, revision=REVISION)
        # folder contains all config files and pytorch_model.bin
        folder_contents = os.listdir(storage_folder)
        assert "pytorch_model.bin" in folder_contents
        # filter framework
        assert "tf_model.h5" in folder_contents
        # revision doesn't have tokenizer
        assert "tokenizer_config.json" not in folder_contents


@require_tf
def test_load_tensorflow_repository_from_hf():
    MODEL = "lysandre/tiny-bert-random"
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_folder = _load_repository_from_hf(MODEL, tmpdirname, framework="tensorflow")
        # folder contains all config files and pytorch_model.bin
        folder_contents = os.listdir(storage_folder)
        assert "pytorch_model.bin" not in folder_contents
        # custom requirements.txt for custom handler
        assert "requirements.txt" in folder_contents
        # filter framework
        assert "tf_model.h5" in folder_contents
        # revision doesn't have tokenizer
        assert "tokenizer_config.json" in folder_contents


def test_load_onnx_repository_from_hf():
    MODEL = "philschmid/distilbert-onnx-banking77"
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_folder = _load_repository_from_hf(MODEL, tmpdirname, framework="onnx")
        # folder contains all config files and pytorch_model.bin
        folder_contents = os.listdir(storage_folder)
        assert "pytorch_model.bin" not in folder_contents
        # custom requirements.txt for custom handler
        assert "requirements.txt" in folder_contents
        # filter framework
        assert "tf_model.h5" not in folder_contents
        # onnx model
        assert "model.onnx" in folder_contents
        # custom pipeline
        assert "pipeline.py" in folder_contents
        # revision doesn't have tokenizer
        assert "tokenizer_config.json" in folder_contents


@require_torch
def test_load_pytorch_repository_from_hf():
    MODEL = "lysandre/tiny-bert-random"
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_folder = _load_repository_from_hf(MODEL, tmpdirname, framework="pytorch")
        # folder contains all config files and pytorch_model.bin
        folder_contents = os.listdir(storage_folder)
        assert "pytorch_model.bin" in folder_contents
        # custom requirements.txt for custom handler
        assert "requirements.txt" in folder_contents
        # filter framework
        assert "tf_model.h5" not in folder_contents
        # revision doesn't have tokenizer
        assert "tokenizer_config.json" in folder_contents


@slow
def test_gpu_available():
    device = _is_gpu_available()
    assert device is True


@require_torch
def test_get_framework_pytorch():
    framework = _get_framework()
    assert framework == "pytorch"


@require_tf
def test_get_framework_tensorflow():
    framework = _get_framework()
    if is_torch_available():
        assert framework == "pytorch"
    else:
        assert framework == "tensorflow"


def test_get_pipeline():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf(MODEL, tmpdirname, framework="pytorch")
        pipe = get_pipeline(TASK, storage_dir.as_posix())
        res = pipe("Life is good, Life is bad")
        assert "score" in res[0]


@require_torch
def test_wrap_conversation_pipeline():
    init_pipeline = pipeline(
        "conversational",
        model="microsoft/DialoGPT-small",
        tokenizer="microsoft/DialoGPT-small",
        framework="pt",
    )
    conv_pipe = wrap_conversation_pipeline(init_pipeline)
    data = {
        "past_user_inputs": ["Which movie is the best ?"],
        "generated_responses": ["It's Die Hard for sure."],
        "text": "Can you explain why?",
    }
    res = conv_pipe(data)
    assert "conversation" in res
    assert "generated_text" in res


@require_torch
def test_wrapped_pipeline():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf("microsoft/DialoGPT-small", tmpdirname, framework="pytorch")
        conv_pipe = get_pipeline("conversational", storage_dir.as_posix())
        data = {
            "past_user_inputs": ["Which movie is the best ?"],
            "generated_responses": ["It's Die Hard for sure."],
            "text": "Can you explain why?",
        }
        res = conv_pipe(data)
        assert "conversation" in res
        assert "generated_text" in res


def test_local_custom_pipeline():
    model_dir = os.path.join(os.getcwd(), "tests/resources/custom_handler")
    pipeline = check_and_register_custom_pipeline_from_directory(model_dir)
    payload = "test"
    assert pipeline.path == model_dir
    assert pipeline(payload) == payload[::-1]


def test_remote_custom_pipeline():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf(
            "philschmid/custom-pipeline-text-classification", tmpdirname, framework="pytorch"
        )
        pipeline = check_and_register_custom_pipeline_from_directory(str(storage_dir))
        payload = "test"
        assert pipeline.path == str(storage_dir)
        assert pipeline(payload) == payload


def test_get_inference_handler_either_custom_or_default_pipeline():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf(
            "philschmid/custom-pipeline-text-classification", tmpdirname, framework="pytorch"
        )
        pipeline = get_inference_handler_either_custom_or_default_handler(str(storage_dir))
        payload = "test"
        assert pipeline.path == str(storage_dir)
        assert pipeline(payload) == payload

    with tempfile.TemporaryDirectory() as tmpdirname:
        MODEL = "lysandre/tiny-bert-random"
        TASK = "text-classification"
        pipeline = get_inference_handler_either_custom_or_default_handler(MODEL, TASK)
        res = pipeline({"inputs": "Life is good, Life is bad"})
        assert "score" in res[0]
