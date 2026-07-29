import asyncio
import contextlib
import logging
import os
import signal
import time

from anyio import Semaphore

LOG = logging.getLogger(__name__)

LAST_START = None
LAST_END = None

UNLOAD_IDLE = os.getenv("UNLOAD_IDLE", "").lower() in ("1", "true")
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", 15))

MAX_REQUESTS = 1000
REQUEST_COUNTER = Semaphore(MAX_REQUESTS)


async def live_check_loop():
    global LAST_START, LAST_END

    pid = os.getpid()

    LOG.info("Starting live check loop")
    sleep_time = max(int(IDLE_TIMEOUT // 5), 1)

    while True:
        await asyncio.sleep(sleep_time)
        LOG.debug("Checking whether we should unload anything from memory")

        last_start = LAST_START
        last_end = LAST_END

        LOG.debug("Checking pid %d activity", pid)
        if not last_start:
            LOG.debug("No request yet, no need to unload")
            continue

        if REQUEST_COUNTER.value < MAX_REQUESTS:
            LOG.info("idle checker: %s requests likely being processed for pid %d, it won't be killed",
                     MAX_REQUESTS - REQUEST_COUNTER.value, pid)
            continue
        if not last_end or last_start >= last_end:
            LOG.warning("This case should not be possible, semaphore unconsistency ? "
                        "Request likely being processed for pid %d", pid)
            continue
        now = time.time()
        last_request_age = now - last_end
        LOG.debug("Pid %d, last request age %s", pid, last_request_age)
        if last_request_age < IDLE_TIMEOUT:
            LOG.debug("Model recently active")
        else:
            LOG.info("Idle checker: worker inactive for too long. Leaving live check loop")
            break
    LOG.info("Aborting this idle worker")
    os.kill(pid, signal.SIGTERM)


@contextlib.asynccontextmanager
async def request_witnesses():
    async with REQUEST_COUNTER:
        LOG.info("Current request count: %s", MAX_REQUESTS - REQUEST_COUNTER.value)
        global LAST_START, LAST_END
        LOG.debug("Last request start was %s", LAST_START)
        LOG.debug("Last request end was %s", LAST_END)
        # Simple assignment, concurrency safe, no need for any lock
        LAST_START = time.time()
        LOG.debug("Current request start timestamp %s", LAST_START)
        try:
            yield
        finally:
            LAST_END = time.time()
            LOG.debug("Current request end timestamp %s", LAST_END)
