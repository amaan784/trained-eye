"""Request pacing for the Meta Model API: at most `per_minute` requests start on each
endpoint, shared by every thread that uses the client.

Muse Image allows 150 requests a minute and Muse Spark 3,000
(dev.meta.ai/docs/pricing-rate-limits, read Nov 4); te keeps every endpoint at
config.MAX_RPM, under the smallest of them. Only requests that go out are paced: a cache
hit never reaches the client's transport, so a rerun stays fast.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class Pacer:
    def __init__(self, per_minute: float, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        if not per_minute > 0:
            raise ValueError(f"requests per minute must be above 0, got {per_minute}")
        self.gap = 60.0 / per_minute
        self._next: dict[str, float] = {}
        self._lock = threading.Lock()
        self._clock, self._sleep = clock, sleep

    def wait(self, key: str) -> float:
        """Block until a request on `key` may start; returns how long that took."""
        with self._lock:
            now = self._clock()
            slot = max(now, self._next.get(key, now))
            self._next[key] = slot + self.gap
        delay = slot - now
        if delay > 0:
            self._sleep(delay)
        return delay
