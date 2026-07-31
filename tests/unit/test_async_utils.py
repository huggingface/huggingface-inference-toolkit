import base64
import threading
from functools import partial

import anyio
import pytest

from huggingface_inference_toolkit.async_utils import MAX_THREADS_GUARD, async_call, offload


class FakeRequest:
    """Only what async_call touches: the disconnect poll."""

    def __init__(self, *verdicts):
        self._verdicts = list(verdicts)
        self.polls = 0

    async def is_disconnected(self):
        self.polls += 1
        return self._verdicts[min(self.polls, len(self._verdicts)) - 1]


def test_runs_the_handler_when_the_caller_is_still_there():
    calls = []
    request = FakeRequest(False, False)

    result = anyio.run(
        partial(async_call, lambda body: calls.append(body) or "prediction", {"inputs": "x"}, request=request)
    )

    assert result == "prediction"
    assert calls == [{"inputs": "x"}]


def test_skips_the_handler_when_the_caller_has_left():
    calls = []
    request = FakeRequest(True)

    result = anyio.run(partial(async_call, lambda body: calls.append(body), {"inputs": "x"}, request=request))

    # None is what makes predict() answer 204
    assert result is None
    assert calls == []


def test_does_not_poll_when_the_feature_is_off():
    # predict() passes request=None when DISCARD_LEFT is not set, so nothing is checked
    calls = []

    result = anyio.run(partial(async_call, lambda body: calls.append(body) or "prediction", {"inputs": "x"}))

    assert result == "prediction"
    assert calls == [{"inputs": "x"}]


def test_detects_a_departed_caller_that_only_shows_on_the_second_poll():
    # is_disconnected() reports only what the event loop has already processed: if the loop was
    # blocked while the caller left, the first poll returns False and its own suspension is what
    # lets the loop catch up. Measured on starlette 0.47.2/uvicorn 0.35.0 and 1.3.1/0.51.0.
    calls = []
    request = FakeRequest(False, True)

    result = anyio.run(partial(async_call, lambda body: calls.append(body), {"inputs": "x"}, request=request))

    assert result is None
    assert calls == []
    assert request.polls == 2


def test_offload_runs_the_codec_off_the_event_loop_thread():
    # The whole point: the loop must stay free to answer /health and to heartbeat to gunicorn
    # while a body is being decoded or a response encoded.
    loop_thread = threading.get_ident()

    codec_thread = anyio.run(partial(offload, threading.get_ident))

    assert codec_thread != loop_thread


def test_offload_does_not_wait_for_the_inference_slot():
    # A codec must not queue behind the model: MAX_THREADS_GUARD rations inference only. Were
    # offload to take it, this would deadlock rather than fail.
    async def scenario():
        async with MAX_THREADS_GUARD:
            with anyio.fail_after(10):
                return await offload(lambda: "decoded")

    assert anyio.run(scenario) == "decoded"


def test_offload_forwards_keyword_arguments():
    # base64.b64decode(validate=True) is one of the calls predict() offloads, and anyio's
    # run_sync takes no kwargs of its own.
    assert anyio.run(partial(offload, base64.b64decode, b"aGk=", validate=True)) == b"hi"


def test_offload_propagates_the_exception():
    # predict() turns a codec failure into a 400, which it can only do if the error surfaces.
    def undecodable():
        raise ValueError("cannot identify image file")

    with pytest.raises(ValueError, match="cannot identify image file"):
        anyio.run(partial(offload, undecodable))
