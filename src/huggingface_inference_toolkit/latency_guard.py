import os
import time

from huggingface_inference_toolkit.logging import logger

ENABLED = os.getenv("LATENCY_GUARD_ENABLED", "0").lower() in ("1", "true")

# How many inference calls to observe before activating the guard
WARMUP_REQUESTS = int(os.getenv("LATENCY_WARMUP_REQUESTS", "10"))
# EMA alpha for the fast (recent) window — ~3-5 requests
FAST_ALPHA = float(os.getenv("LATENCY_FAST_ALPHA", "0.3"))
# EMA alpha for the slow (baseline) window — ~20 requests
SLOW_ALPHA = float(os.getenv("LATENCY_SLOW_ALPHA", "0.05"))
# Auto-freeze when fast_ema > OVERLOAD_FACTOR * slow_ema
OVERLOAD_FACTOR = float(os.getenv("LATENCY_OVERLOAD_FACTOR", "3.0"))
# Auto-unfreeze when fast_ema < RECOVERY_FACTOR * slow_ema (hysteresis: < OVERLOAD_FACTOR)
RECOVERY_FACTOR = float(os.getenv("LATENCY_RECOVERY_FACTOR", "1.5"))
# How long we stay frozen before letting requests through again to re-measure latency
FREEZE_SECONDS = float(os.getenv("LATENCY_FREEZE_SECONDS", "10"))


class LatencyGuard:
    """
    Tracks pure inference latency via a dual EMA and auto-freezes new request
    acceptance when recent latency drifts too far above the learned baseline.

    Freezing is time bounded: a frozen worker runs no inference, so it records no
    new latency either and could never unfreeze on its own. After FREEZE_SECONDS we
    accept requests again ("half open") to get fresh samples, and either unfreeze for
    good once latency is back to the baseline, or freeze again for another round.
    """

    def __init__(self):
        self._fast_ema = None
        self._slow_ema = None
        self._warmup_count = 0
        self._auto_frozen = False
        self._frozen_at = 0.0

    def record(self, duration_s: float):
        """Called after each inference with its wall-clock duration in seconds. No-op if disabled."""
        if not ENABLED:
            return
        if self._fast_ema is None:
            self._fast_ema = duration_s
            self._slow_ema = duration_s
        else:
            self._fast_ema = FAST_ALPHA * duration_s + (1 - FAST_ALPHA) * self._fast_ema
            # Stop updating the baseline while auto-frozen so it doesn't drift up
            if not self._auto_frozen:
                self._slow_ema = SLOW_ALPHA * duration_s + (1 - SLOW_ALPHA) * self._slow_ema

        self._warmup_count += 1
        if self._warmup_count <= WARMUP_REQUESTS:
            return

        ratio = self._fast_ema / self._slow_ema
        if not self._auto_frozen and ratio > OVERLOAD_FACTOR:
            logger.warning(
                "LatencyGuard: auto-freezing — fast_ema=%.1fms slow_ema=%.1fms ratio=%.2f",
                self._fast_ema * 1000, self._slow_ema * 1000, ratio,
            )
            self._auto_frozen = True
            self._frozen_at = time.monotonic()
        elif self._auto_frozen and ratio < RECOVERY_FACTOR:
            logger.info(
                "LatencyGuard: auto-unfreezing — fast_ema=%.1fms slow_ema=%.1fms ratio=%.2f",
                self._fast_ema * 1000, self._slow_ema * 1000, ratio,
            )
            self._auto_frozen = False
        elif self._auto_frozen and ratio > OVERLOAD_FACTOR:
            # Still overloaded on the half open probe: freeze for another round
            self._frozen_at = time.monotonic()

    @property
    def accepting(self) -> bool:
        if not ENABLED or not self._auto_frozen:
            return True
        # Half open: let requests through again so that fresh latency samples can be recorded
        return time.monotonic() - self._frozen_at >= FREEZE_SECONDS

    @property
    def auto_frozen(self) -> bool:
        return self._auto_frozen


latency_guard = LatencyGuard()
