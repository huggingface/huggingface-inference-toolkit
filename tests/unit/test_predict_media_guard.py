import base64
from functools import partial
from io import BytesIO

import anyio
import orjson
import pytest
from PIL import Image
from starlette.requests import Request

from huggingface_inference_toolkit import webservice_starlette as ws


class _Handler:
    """Stands in for the pipeline handler: records the body it is given and answers nothing."""

    def __init__(self):
        self.calls = []

    def __call__(self, body):
        self.calls.append(body)
        return {"ok": True}


@pytest.fixture
def handler(monkeypatch):
    handler = _Handler()

    async def no_download():
        pass

    async def loaded(task):
        return handler

    monkeypatch.setattr(ws, "ensure_model_downloaded", no_download)
    monkeypatch.setattr(ws, "ensure_handler_loaded", loaded)
    monkeypatch.setattr(ws, "HF_TASK", "image-classification")
    return handler


def _post(body: dict):
    payload = orjson.dumps(body)

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
        "path_params": {},
    }
    return anyio.run(partial(ws._predict, Request(scope, receive)))


def _b64_png():
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "blue").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_a_media_string_under_inputs_is_decoded_before_the_handler(handler):
    response = _post({"inputs": {"images": _b64_png()}})
    assert response.status_code == 200
    assert isinstance(handler.calls[0]["inputs"]["images"], Image.Image)


@pytest.mark.parametrize("shape", [{"images": "/var/lib/nonexistent"}, ["/var/lib/nonexistent"]])
def test_a_media_string_under_inputs_never_reaches_the_handler(handler, shape):
    response = _post({"inputs": shape})
    assert response.status_code == 400
    assert b"must be the media content itself" in response.body
    assert handler.calls == []


def test_instances_are_walked_the_same_way(handler):
    # The Vertex AI body carries one input per instance and is not `inputs`: the guard has to
    # run on it too, or a Vertex deployment gets none of the protection.
    response = _post({"instances": [{"images": _b64_png()}, "/var/lib/nonexistent"]})
    assert response.status_code == 400
    assert b"'instances[1]' for task 'image-classification'" in response.body
    assert handler.calls == []


def test_instances_are_decoded_before_the_handler(handler):
    response = _post({"instances": [{"images": _b64_png()}, _b64_png()]})
    assert response.status_code == 200
    instances = handler.calls[0]["instances"]
    assert isinstance(instances[0]["images"], Image.Image)
    assert isinstance(instances[1], Image.Image)
