from functools import partial

import anyio
import pytest

from huggingface_inference_toolkit import webservice_starlette as ws
from huggingface_inference_toolkit.env_utils import task_route_enabled


@pytest.fixture(autouse=True)
def empty_handler_cache(monkeypatch):
    monkeypatch.setattr(ws, "INFERENCE_HANDLERS", {})


@pytest.fixture
def loads(monkeypatch):
    """Record every task a handler is built for, and hand back a stand-in."""
    built = []

    async def fake_resolve(model_dir, task=None):
        built.append(task)
        return f"handler for {task}"

    monkeypatch.setattr(ws, "get_inference_handler_either_custom_or_default_handler", fake_resolve)
    return built


def test_a_handler_is_built_once_per_task(loads):
    assert anyio.run(partial(ws.ensure_handler_loaded, "text-classification")) == (
        "handler for text-classification"
    )
    assert anyio.run(partial(ws.ensure_handler_loaded, "text-classification")) == (
        "handler for text-classification"
    )

    assert loads == ["text-classification"]


def test_each_task_gets_its_own_handler(loads):
    anyio.run(partial(ws.ensure_handler_loaded, "text-classification"))
    anyio.run(partial(ws.ensure_handler_loaded, "feature-extraction"))
    anyio.run(partial(ws.ensure_handler_loaded, "text-classification"))

    assert loads == ["text-classification", "feature-extraction"]
    assert set(ws.INFERENCE_HANDLERS) == {"text-classification", "feature-extraction"}


def test_requests_racing_for_the_same_task_load_it_once(loads):
    async def race():
        async with anyio.create_task_group() as tg:
            for _ in range(5):
                tg.start_soon(ws.ensure_handler_loaded, "text-classification")

    anyio.run(race)

    # the semaphore plus the second check inside it keep five callers to one load
    assert loads == ["text-classification"]


@pytest.mark.parametrize(
    "hf_task,requested,expected",
    [
        # a sentence-transformers repository answers feature-extraction with embeddings
        ("sentence-similarity", "feature-extraction", "sentence-embeddings"),
        ("sentence-embeddings", "feature-extraction", "sentence-embeddings"),
        ("sentence-ranking", "feature-extraction", "sentence-embeddings"),
        # a transformers repository serves feature-extraction as itself
        ("text-classification", "feature-extraction", "feature-extraction"),
        ("feature-extraction", "feature-extraction", "feature-extraction"),
        # anything else passes through untouched
        ("sentence-similarity", "sentence-similarity", "sentence-similarity"),
        ("text-classification", "token-classification", "token-classification"),
    ],
)
def test_resolve_task(monkeypatch, hf_task, requested, expected):
    monkeypatch.setattr(ws, "HF_TASK", hf_task)
    assert ws.resolve_task(requested) == expected


def test_the_route_registration_matches_the_flag():
    # the app is built at import, so this asserts what the module decided rather than re-importing
    paths = {getattr(route, "path", None) for route in ws.app.routes}

    assert ("/pipeline/{task:path}" in paths) is task_route_enabled()


@pytest.mark.parametrize(
    "enable_task_route,compat,expected",
    [
        # unset: follows the compat flag, so the Inference API needs no extra configuration
        (None, "1", True),
        (None, "0", False),
        (None, None, False),
        # set: independent of compat, in both directions
        ("1", "0", True),
        ("0", "1", False),
        ("true", None, True),
    ],
)
def test_task_route_defaults_to_the_compat_flag(monkeypatch, enable_task_route, compat, expected):
    for name, value in (("ENABLE_TASK_ROUTE", enable_task_route), ("API_INFERENCE_COMPAT", compat)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert task_route_enabled() is expected


def test_a_meaningless_value_fails_rather_than_being_guessed(monkeypatch):
    monkeypatch.setenv("ENABLE_TASK_ROUTE", "sometimes")
    with pytest.raises(ValueError):
        task_route_enabled()
