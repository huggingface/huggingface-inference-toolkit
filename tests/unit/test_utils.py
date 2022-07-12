import os
import tempfile

from transformers import pipeline
from transformers.file_utils import is_torch_available
from transformers.testing_utils import require_tf, require_torch, slow

from huggingface_inference_toolkit.utils import _get_framework, _is_gpu_available, get_pipeline


MODEL = "lysandre/tiny-bert-random"
TASK = "text-classification"
TASK_MODEL = "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english"


def test_gpu_is_not_available():
    device = _is_gpu_available()
    assert device is False


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
    pass
    # with tempfile.TemporaryDirectory() as tmpdirname:
    #     storage_dir = _load_model_from_hub(MODEL, tmpdirname)
    #     pipe = get_pipeline(TASK, -1, storage_dir)
    #     res = pipe("Life is good, Life is bad")
    #     assert "score" in res[0]


@require_torch
def test_wrap_conversation_pipeline():
    pass
    # init_pipeline = pipeline(
    #     "conversational",
    #     model="microsoft/DialoGPT-small",
    #     tokenizer="microsoft/DialoGPT-small",
    #     framework="pt",
    # )
    # conv_pipe = wrap_conversation_pipeline(init_pipeline)
    # data = {
    #     "past_user_inputs": ["Which movie is the best ?"],
    #     "generated_responses": ["It's Die Hard for sure."],
    #     "text": "Can you explain why?",
    # }
    # res = conv_pipe(data)
    # assert "conversation" in res
    # assert "generated_text" in res


@require_torch
def test_wrapped_pipeline():
    pass
    # with tempfile.TemporaryDirectory() as tmpdirname:
    #     storage_dir = _load_model_from_hub("microsoft/DialoGPT-small", tmpdirname)
    #     conv_pipe = get_pipeline("conversational", -1, storage_dir)
    #     data = {
    #         "past_user_inputs": ["Which movie is the best ?"],
    #         "generated_responses": ["It's Die Hard for sure."],
    #         "text": "Can you explain why?",
    #     }
    #     res = conv_pipe(data)
    #     assert "conversation" in res
    #     assert "generated_text" in res
