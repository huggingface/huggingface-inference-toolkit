import functools
from typing import Any, Callable, Dict, TypeVar

import anyio
from anyio import Semaphore
from typing_extensions import ParamSpec


# To not have too many threads running (which could happen on too many concurrent
# requests, we limit it with a semaphore.
MAX_CONCURRENT_THREADS = 1
MAX_THREADS_GUARD = Semaphore(MAX_CONCURRENT_THREADS)
T = TypeVar("T")
P = ParamSpec("P")


# moves blocking call to asyncio threadpool limited to 1 to not overload the system
# REF: https://stackoverflow.com/a/70929141
async def async_handler_call(handler: Callable[P, T], body: Dict[str, Any]) -> T:
    async with MAX_THREADS_GUARD:
        return await anyio.to_thread.run_sync(functools.partial(handler, body))
