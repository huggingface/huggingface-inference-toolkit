import asyncio
import importlib.util
import logging
import os
import tempfile
from pathlib import Path

import pytest
from transformers.testing_utils import require_torch, slow

from huggingface_inference_toolkit import heavy_utils as hf_heavy_utils
from huggingface_inference_toolkit.handler import get_inference_handler_either_custom_or_default_handler
from huggingface_inference_toolkit.heavy_utils import (
    _get_framework,
    _is_gpu_available,
    get_pipeline,
    load_repository_from_hf,
)
from huggingface_inference_toolkit.utils import (
    check_and_register_custom_pipeline_from_directory,
    convert_params_to_int_or_bool,
    should_discard_left,
)

TASK_MODEL = "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english"


def test_load_revision_repository_from_hf():
    MODEL = "lysandre/tiny-bert-random"
    REVISION = "eb4c77816edd604d0318f8e748a1c606a2888493"
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_folder = load_repository_from_hf(MODEL, tmpdirname, revision=REVISION)
        # folder contains all config files and pytorch_model.bin
        folder_contents = os.listdir(storage_folder)
        # revision doesn't have tokenizer
        assert "tokenizer_config.json" not in folder_contents


def test_load_onnx_repository_from_hf():
    MODEL = "philschmid/distilbert-onnx-banking77"
    with tempfile.TemporaryDirectory() as tmpdirname:
        ox_tmp = Path(tmpdirname).joinpath("onnx")
        ox_tmp.mkdir(parents=True, exist_ok=True)

        storage_folder = load_repository_from_hf(MODEL, ox_tmp, framework="onnx")
        # folder contains all config files and pytorch_model.bin
        folder_contents = os.listdir(storage_folder)
        assert "pytorch_model.bin" not in folder_contents
        # filter framework
        assert "tf_model.h5" not in folder_contents
        # onnx model
        assert "model.onnx" in folder_contents
        # custom pipeline
        assert "handler.py" in folder_contents
        # revision doesn't have tokenizer
        assert "tokenizer_config.json" in folder_contents


@require_torch
def test_load_pytorch_repository_from_hf():
    MODEL = "lysandre/tiny-bert-random"
    with tempfile.TemporaryDirectory() as tmpdirname:
        pt_tmp = Path(tmpdirname).joinpath("pt")
        pt_tmp.mkdir(parents=True, exist_ok=True)

        storage_folder = load_repository_from_hf(MODEL, pt_tmp, framework="pytorch")
        # folder contains all config files and pytorch_model.bin
        folder_contents = os.listdir(storage_folder)
        assert "pytorch_model.bin" in folder_contents
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


@require_torch
def test_get_pipeline():
    MODEL = "hf-internal-testing/tiny-random-BertForSequenceClassification"
    TASK = "text-classification"
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = load_repository_from_hf(MODEL, tmpdirname, framework="pytorch")
        pipe = get_pipeline(
            task = TASK,
            model_dir = storage_dir.as_posix(),
        )
        res = pipe("Life is good, Life is bad")
        assert "score" in res[0]


@require_torch
def test_whisper_long_audio(cache_test_dir):
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = load_repository_from_hf(
            repository_id = "openai/whisper-tiny",
            target_dir = tmpdirname,
        )
        logging.info(f"Temp dir: {tmpdirname}")
        logging.info(f"POSIX Path: {storage_dir.as_posix()}")
        logging.info(f"Contents: {os.listdir(tmpdirname)}")
        pipe = get_pipeline(
            task = "automatic-speech-recognition",
            model_dir = storage_dir.as_posix(),
        )
        res = pipe(f"{cache_test_dir}/resources/audio/long_sample.mp3")

        assert len(res["text"]) > 700

@require_torch
def test_wrapped_pipeline():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = load_repository_from_hf(
            repository_id = "microsoft/DialoGPT-small",
            target_dir = tmpdirname,
            framework="pytorch"
        )
        conv_pipe = get_pipeline("conversational", storage_dir.as_posix())
        data = [
            {
                "role": "user",
                "content": "Which movie is the best ?"
            },
            {
                "role": "assistant",
                "content": "It's Die Hard for sure."
            },
            {
                "role": "user",
                "content": "Can you explain why?"
            }
        ]
        res = conv_pipe(data, max_new_tokens = 100)
        logging.info(f"Response: {res}")
        message = res[0]["generated_text"][-1]
        assert message["role"] == "assistant"


def test_local_custom_pipeline(cache_test_dir):
    model_dir = f"{cache_test_dir}/resources/custom_handler"
    pipeline = check_and_register_custom_pipeline_from_directory(model_dir)
    payload = "test"
    assert pipeline.path == model_dir
    assert pipeline(payload) == payload[::-1]


def test_remote_custom_pipeline():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = load_repository_from_hf(
            "philschmid/custom-pipeline-text-classification",
            tmpdirname,
            framework="pytorch"
        )
        pipeline = check_and_register_custom_pipeline_from_directory(str(storage_dir))
        payload = "test"
        assert pipeline.path == str(storage_dir)
        assert pipeline(payload) == payload


def test_get_inference_handler_either_custom_or_default_pipeline():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = load_repository_from_hf(
            "philschmid/custom-pipeline-text-classification",
            tmpdirname,
            framework="pytorch"
        )
        pipeline = asyncio.run(get_inference_handler_either_custom_or_default_handler(str(storage_dir)))
        payload = "test"
        assert pipeline.path == str(storage_dir)
        assert pipeline(payload) == payload

    with tempfile.TemporaryDirectory() as tmpdirname:
        MODEL = "lysandre/tiny-bert-random"
        TASK = "text-classification"
        pipeline = asyncio.run(get_inference_handler_either_custom_or_default_handler(MODEL, TASK))
        res = pipeline({"inputs": "Life is good, Life is bad"})
        assert "score" in res[0]


@pytest.mark.parametrize(
    "value,expected",
    [(None, False), ("0", False), ("no", False), ("", False), ("1", True), ("true", True), ("YES", True)],
)
def test_should_discard_left(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("DISCARD_LEFT", raising=False)
    else:
        monkeypatch.setenv("DISCARD_LEFT", value)
    assert should_discard_left() is expected


class _Sibling:
    def __init__(self, rfilename):
        self.rfilename = rfilename


@pytest.mark.parametrize("revision", ["eb4c77816edd604d0318f8e748a1c606a2888493", None])
def test_safetensors_probe_asks_about_the_revision_being_downloaded(monkeypatch, tmp_path, revision):
    # The probe's answer selects the download ignore-patterns, so it has to describe the same
    # commit snapshot_download will fetch. Querying the default branch instead can filter out the
    # only weights the requested revision actually has.
    seen = {}

    class _Info:
        siblings = [_Sibling("config.json"), _Sibling("model.safetensors")]

    def fake_model_info(self, repo_id, **kwargs):
        seen["probe_revision"] = kwargs.get("revision")
        return _Info()

    def fake_snapshot_download(**kwargs):
        seen["download_revision"] = kwargs.get("revision")
        seen["ignore_patterns"] = kwargs.get("ignore_patterns")

    monkeypatch.setattr(hf_heavy_utils.HfApi, "model_info", fake_model_info)
    monkeypatch.setattr(hf_heavy_utils, "snapshot_download", fake_snapshot_download)

    hf_heavy_utils.load_repository_from_hf("some/model", tmp_path, framework="pytorch", revision=revision)

    assert seen["probe_revision"] == revision
    assert seen["probe_revision"] == seen["download_revision"]
    # safetensors found at that revision, so the *safetensors filter must not be applied
    assert "*safetensors" not in seen["ignore_patterns"]
    assert "pytorch*" in seen["ignore_patterns"]


def test_safetensors_probe_keeps_bin_weights_when_the_revision_has_no_safetensors(monkeypatch, tmp_path):
    seen = {}

    class _Info:
        siblings = [_Sibling("config.json"), _Sibling("pytorch_model.bin")]

    monkeypatch.setattr(hf_heavy_utils.HfApi, "model_info", lambda self, repo_id, **kw: _Info())
    monkeypatch.setattr(
        hf_heavy_utils, "snapshot_download", lambda **kw: seen.update(ignore_patterns=kw.get("ignore_patterns"))
    )

    hf_heavy_utils.load_repository_from_hf("some/model", tmp_path, framework="pytorch", revision="abc123")

    # no safetensors at this revision, so the .bin weights are the ones that must survive
    assert "pytorch*" not in seen["ignore_patterns"]
    assert "*safetensors" in seen["ignore_patterns"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        # ints, including the signs `str.isnumeric()` rejects
        ("5", 5),
        ("-1", -1),
        ("+3", 3),
        ("0", 0),
        # floats, none of which `str.isnumeric()` accepts
        ("0.5", 0.5),
        ("-0.5", -0.5),
        ("1e3", 1000.0),
        # booleans
        ("true", True),
        ("false", False),
        # left alone
        ("hello", "hello"),
        ("", ""),
        ("True", "True"),
        ("1,2", "1,2"),
        # `str.isnumeric()` is True for these but `int()` refuses them
        ("\u00b2", "\u00b2"),
        ("\u00bd", "\u00bd"),
        # `float()` accepts these; JSON cannot represent them and no pipeline wants them
        ("nan", "nan"),
        ("inf", "inf"),
        ("-infinity", "-infinity"),
    ],
)
def test_convert_params_coerces_query_values(raw, expected):
    converted = convert_params_to_int_or_bool({"p": raw})["p"]
    assert converted == expected
    assert type(converted) is type(expected)


def test_convert_params_keeps_every_key():
    params = {"top_k": "5", "temperature": "0.5", "seed": "-1", "do_sample": "true", "prompt": "hi"}
    assert convert_params_to_int_or_bool(params) == {
        "top_k": 5,
        "temperature": 0.5,
        "seed": -1,
        "do_sample": True,
        "prompt": "hi",
    }


@pytest.mark.parametrize("handler_file", ["handler.py", "pipeline.py"])
def test_an_unimportable_custom_handler_is_ignored_rather_than_crashing(monkeypatch, tmp_path, handler_file):
    """
    A handler file that exists but yields no import spec used to raise UnboundLocalError on the way
    out of the function, which said nothing about the actual problem. Reachable by pointing
    HF_DEFAULT_PIPELINE_NAME at a file Python cannot build a loader for.
    """
    (tmp_path / handler_file).write_text("class EndpointHandler:\n    def __init__(self, p): pass\n")
    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *a, **k: None)

    assert check_and_register_custom_pipeline_from_directory(str(tmp_path)) is None


def test_no_custom_handler_is_still_none(tmp_path):
    assert check_and_register_custom_pipeline_from_directory(str(tmp_path)) is None
