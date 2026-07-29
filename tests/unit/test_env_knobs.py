from functools import partial
from types import SimpleNamespace

import anyio
import pytest

from huggingface_inference_toolkit import diffusers_utils
from huggingface_inference_toolkit import handler as handler_module
from huggingface_inference_toolkit.diffusers_utils import IEAutoPipelineForText2Image, _generation_default
from huggingface_inference_toolkit.env_utils import ignore_custom_handler
from huggingface_inference_toolkit.handler import get_inference_handler_either_custom_or_default_handler


@pytest.mark.parametrize(
    "value,expected",
    [(None, False), ("false", False), ("0", False), ("true", True), ("1", True), ("YES", True)],
)
def test_ignore_custom_handler_env(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("IGNORE_CUSTOM_HANDLER", raising=False)
    else:
        monkeypatch.setenv("IGNORE_CUSTOM_HANDLER", value)
    assert ignore_custom_handler() is expected


@pytest.fixture
def repo_with_a_custom_handler(monkeypatch):
    """A model directory whose handler.py would be picked up, and a stand-in default handler."""
    looked = []

    def register(model_dir):
        looked.append(model_dir)
        return "custom handler"

    monkeypatch.setattr(handler_module, "check_and_register_custom_pipeline_from_directory", register)
    class FakeDefaultHandler:
        @staticmethod
        async def create(model_dir, task):
            return "default handler"

    monkeypatch.setattr(handler_module, "HuggingFaceHandler", FakeDefaultHandler)
    monkeypatch.delenv("AIP_MODE", raising=False)
    return looked


def test_custom_handler_is_used_by_default(monkeypatch, repo_with_a_custom_handler):
    monkeypatch.delenv("IGNORE_CUSTOM_HANDLER", raising=False)

    resolve = partial(get_inference_handler_either_custom_or_default_handler, "/model", task="text-classification")
    assert anyio.run(resolve) == "custom handler"
    assert repo_with_a_custom_handler == ["/model"]


def test_custom_handler_is_skipped_when_ignored(monkeypatch, repo_with_a_custom_handler):
    monkeypatch.setenv("IGNORE_CUSTOM_HANDLER", "1")

    resolve = partial(get_inference_handler_either_custom_or_default_handler, "/model", task="text-classification")
    assert anyio.run(resolve) == "default handler"
    # not even looked for: importing a repo's handler.py executes its module-level code
    assert repo_with_a_custom_handler == []


class FakeDiffusionPipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=["an image"])


@pytest.fixture
def text2image():
    # bypass __init__, which would load a diffusion model
    pipeline = IEAutoPipelineForText2Image.__new__(IEAutoPipelineForText2Image)
    pipeline.pipeline = FakeDiffusionPipeline()
    return pipeline


def test_generation_defaults_are_applied(monkeypatch, text2image):
    monkeypatch.setattr(diffusers_utils, "DEFAULT_NUM_INFERENCE_STEPS", 12)
    monkeypatch.setattr(diffusers_utils, "DEFAULT_GUIDANCE_SCALE", 3.5)

    assert text2image("a prompt") == "an image"
    assert text2image.pipeline.calls[0]["num_inference_steps"] == 12
    assert text2image.pipeline.calls[0]["guidance_scale"] == 3.5


def test_request_parameters_win_over_the_defaults(monkeypatch, text2image):
    monkeypatch.setattr(diffusers_utils, "DEFAULT_NUM_INFERENCE_STEPS", 12)
    monkeypatch.setattr(diffusers_utils, "DEFAULT_GUIDANCE_SCALE", 3.5)

    text2image("a prompt", num_inference_steps=40, guidance_scale=9.0)

    assert text2image.pipeline.calls[0]["num_inference_steps"] == 40
    assert text2image.pipeline.calls[0]["guidance_scale"] == 9.0


def test_nothing_is_injected_without_the_env(monkeypatch, text2image):
    monkeypatch.setattr(diffusers_utils, "DEFAULT_NUM_INFERENCE_STEPS", None)
    monkeypatch.setattr(diffusers_utils, "DEFAULT_GUIDANCE_SCALE", None)

    text2image("a prompt")

    assert "num_inference_steps" not in text2image.pipeline.calls[0]
    assert "guidance_scale" not in text2image.pipeline.calls[0]


def test_a_zero_guidance_scale_is_honoured(monkeypatch, text2image):
    # 0 disables classifier-free guidance, so it must not be treated as "unset"
    monkeypatch.setattr(diffusers_utils, "DEFAULT_GUIDANCE_SCALE", 0.0)

    text2image("a prompt")

    assert text2image.pipeline.calls[0]["guidance_scale"] == 0.0


@pytest.mark.parametrize("value,expected", [(None, None), ("12", 12), ("0", 0)])
def test_generation_default_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("A_DEFAULT", raising=False)
    else:
        monkeypatch.setenv("A_DEFAULT", value)
    assert _generation_default("A_DEFAULT", int) == expected


@pytest.mark.parametrize("value", ["", "abc"])
def test_a_meaningless_default_fails_at_import_not_per_request(monkeypatch, value):
    # Parsed at module import, so a bad value takes the worker down at startup, where it is
    # attributable to the rollout — instead of 400ing every generation from inside __call__.
    monkeypatch.setenv("A_DEFAULT", value)
    with pytest.raises(ValueError):
        _generation_default("A_DEFAULT", float)
