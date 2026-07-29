import pytest

from huggingface_inference_toolkit import latency_guard as guard_module
from huggingface_inference_toolkit.latency_guard import LatencyGuard

BASELINE = 0.1  # a normal inference, in seconds
OVERLOADED = 5.0  # what one looks like when the node is thrashing


@pytest.fixture
def clock(monkeypatch):
    """A monotonic clock we drive by hand, so freeze windows don't depend on wall time."""
    now = [1000.0]
    monkeypatch.setattr(guard_module.time, "monotonic", lambda: now[0])
    return now


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(guard_module, "ENABLED", True)


def feed(guard, duration, times=1):
    for _ in range(times):
        guard.record(duration)


def warmed_up(guard):
    """Past the warmup window with a stable baseline, so the guard is armed."""
    feed(guard, BASELINE, guard_module.WARMUP_REQUESTS + 1)
    return guard


def test_disabled_by_default(clock, monkeypatch):
    monkeypatch.setattr(guard_module, "ENABLED", False)
    guard = warmed_up(LatencyGuard())

    feed(guard, OVERLOADED, 5)

    assert guard.accepting is True
    assert guard.auto_frozen is False


def test_does_not_freeze_during_warmup(clock, enabled):
    guard = LatencyGuard()

    # slow from the very first request: there is no baseline to compare against yet
    feed(guard, OVERLOADED, guard_module.WARMUP_REQUESTS)

    assert guard.accepting is True
    assert guard.auto_frozen is False


def test_freezes_when_latency_drifts_above_the_baseline(clock, enabled):
    guard = warmed_up(LatencyGuard())
    assert guard.accepting is True

    feed(guard, OVERLOADED)

    assert guard.auto_frozen is True
    assert guard.accepting is False


def test_half_opens_after_the_freeze_window(clock, enabled):
    guard = warmed_up(LatencyGuard())
    feed(guard, OVERLOADED)
    assert guard.accepting is False

    clock[0] += guard_module.FREEZE_SECONDS - 0.01
    assert guard.accepting is False

    clock[0] += 0.02
    # still considered overloaded, but requests are let through to get fresh samples — without this
    # the guard could never recover: a frozen worker runs no inference, so it records no latency
    assert guard.accepting is True
    assert guard.auto_frozen is True


def test_unfreezes_once_latency_is_back_to_the_baseline(clock, enabled):
    guard = warmed_up(LatencyGuard())
    feed(guard, OVERLOADED)
    clock[0] += guard_module.FREEZE_SECONDS

    feed(guard, BASELINE, 5)

    assert guard.auto_frozen is False
    assert guard.accepting is True


def test_freezes_again_when_the_probe_is_still_slow(clock, enabled):
    guard = warmed_up(LatencyGuard())
    feed(guard, OVERLOADED)
    clock[0] += guard_module.FREEZE_SECONDS
    assert guard.accepting is True  # half open

    feed(guard, OVERLOADED)

    # the window restarts rather than staying open
    assert guard.accepting is False
    clock[0] += guard_module.FREEZE_SECONDS
    assert guard.accepting is True


def test_the_baseline_stops_moving_once_frozen(clock, enabled):
    guard = warmed_up(LatencyGuard())

    # the sample that trips the freeze is recorded while still unfrozen, so it does move the
    # baseline; everything after it must not
    feed(guard, OVERLOADED)
    assert guard.auto_frozen is True
    baseline_while_frozen = guard._slow_ema

    feed(guard, OVERLOADED, 10)

    # otherwise a long overload would teach the guard that slow is normal, and it would stop firing
    assert guard._slow_ema == baseline_while_frozen


class FakePipeline:
    task = "text-classification"

    def __call__(self, inputs, **parameters):
        return [{"label": "A", "score": 1.0}]


def test_the_handler_reports_its_inference_duration(monkeypatch):
    """The wiring: whatever the handler measures is what the guard learns from."""
    from huggingface_inference_toolkit.handler import HuggingFaceHandler

    recorded = []
    monkeypatch.setattr(guard_module.latency_guard, "record", recorded.append)

    handler = HuggingFaceHandler.__new__(HuggingFaceHandler)  # bypass __init__, no model to load
    handler.pipeline = FakePipeline()

    assert handler({"inputs": "x"}) == [{"label": "A", "score": 1.0}]
    assert len(recorded) == 1
    assert recorded[0] >= 0
