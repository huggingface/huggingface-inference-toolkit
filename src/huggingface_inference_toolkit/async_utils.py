from functools import partial
from typing import Callable, Optional, TypeVar

import anyio
from anyio import Semaphore
from starlette.requests import Request
from typing_extensions import ParamSpec

from huggingface_inference_toolkit.logging import logger

# To not have too many threads running (which could happen on too many concurrent
# requests, we limit it with a semaphore.
MAX_CONCURRENT_THREADS = 1
MAX_THREADS_GUARD = Semaphore(MAX_CONCURRENT_THREADS)
T = TypeVar("T")
P = ParamSpec("P")


# moves blocking call to asyncio threadpool limited to 1 to not overload the system
# REF: https://stackoverflow.com/a/70929141
async def async_call(
    handler: Callable[P, T], *args, request: Optional[Request] = None, **kwargs
) -> Optional[T]:
    """
    Run `handler` in the threadpool, once a slot is free.

    When `request` is given, the caller is checked for having left just after the slot is
    acquired, and the call is skipped if it has: under a burst, requests queue here while their
    callers time out and walk away, and running the model for them buys nobody anything. The
    check belongs at this exact point — the request has waited its whole queue time by now, so
    this is the latest and most accurate moment to ask.

    A missed detection only costs us an inference we could have skipped; it never discards a
    request whose caller is still waiting.
    """
    logger.info("Setting blocking call to async handler")
    async with MAX_THREADS_GUARD:
        logger.info("Async call semaphore passed")
        if request is not None and await _caller_left(request):
            logger.info("Discarding request as the caller already left")
            return None
        return await anyio.to_thread.run_sync(handler, *args, **kwargs)


async def offload(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """
    Run a CPU-bound codec on the threadpool rather than on the event loop.

    Decoding a request body and encoding a response are not cheap at the sizes we serve: a PNG
    body reaches PIL through `Image.convert("RGB")`, and a generated image goes back out through
    `Image.save`, base64 and orjson. Left on the loop, a single large payload stalls everything
    the loop owes the rest of the process:

    - `/health` stops answering, so the orchestrator can decide a busy worker is a dead one.
    - gunicorn's `--timeout` (30s by default, see `scripts/entrypoint.sh`) kills a worker that
      has not heartbeat, which under plain uvicorn could not happen at all. A request killed
      this way dies mid-flight.
    - `Request.is_disconnected()` only reports a disconnect the loop has already processed, so
      DISCARD_LEFT spots a departed caller sooner on a loop that is free to notice.

    Deliberately outside `MAX_THREADS_GUARD`: that semaphore rations the model, and a codec
    holding an inference slot would queue decoding behind inference for nothing. anyio's own
    thread limiter still bounds how many of these run at once.
    """
    return await anyio.to_thread.run_sync(partial(func, *args, **kwargs))


async def _caller_left(request: Request) -> bool:
    """
    Whether the caller of `request` has given up waiting for its answer.

    Must be called after the request body has been read: `is_disconnected()` polls one ASGI
    message, so before the body is drained it would eat the payload. Once `Request.body()` has
    cached it, the only message left to receive is `http.disconnect`.

    It is polled twice on purpose. `is_disconnected()` polls with an already-cancelled scope, so
    it reports only a disconnect the event loop has already processed. If the loop was blocked
    while the caller went away, uvicorn's `connection_lost` has not run yet and the first poll
    returns False — and that poll's own suspension is what lets the loop catch up, so the second
    one sees it. Measured on starlette 0.47.2 / uvicorn 0.35.0 and 1.3.1 / 0.51.0, with the
    caller aborting mid-request:

        loop blocked during the wait -> [False, True, True]
        loop free during the wait    -> [True,  True,  True]

    Polling twice is what makes this independent of the loop's state, which we should not assume.
    `predict` now hands its codecs to `offload`, so the loop is no longer blocked by the request
    being answered — but the threads that work runs in still hold the GIL between their own
    native calls, and other requests are being served on the same loop.
    """
    return await request.is_disconnected() or await request.is_disconnected()
