"""A live playground run's cost and state: what it may cost, the metered ledger that
holds each of its Meta calls to the run's reservation, and the Job the page polls."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from backend.config import Settings
from backend.dataengine import labels
from backend.meta.ledger import Entry, Ledger, image_tokens, token_cost, unit_cost
from backend.playground.budget import Budget, Reservation

SKILL = "playground"  # the ledger's skill name for every playground call
POSITIVES = NEGATIVES = 6


def estimate_usd(settings: Settings, count: int = POSITIVES + NEGATIVES, *,
                 ledger=None) -> float:
    """A run before cache hits, every call at its full token budget (as teach.estimate):
    one compile with one keyframe, `count` image edits and `count` labels. Images are
    priced at the ledger's measured tokens per image (ledger.image_tokens)."""
    price, model = settings.price, settings.model
    per_image = image_tokens(settings, ledger).per_image
    label = token_cost(price(model("labeler")), per_image + 250, labels.MAX_TOKENS[0])
    return (token_cost(price(model("compiler")), per_image + 900, 4096)
            + unit_cost(price(model("image")), images=count) + count * label)


class MeteredLedger:
    """One run's view of the client's ledger: a call must fit in the run's reservation
    (with the run's other calls in flight) and pass the ledger's own caps, which count
    every call in flight (of every run, thread and process: the ledger reserves each
    call's estimate until it is recorded). What the ledger records is added up for the
    run. close() when the run ends, for calls that failed before the ledger recorded them."""

    def __init__(self, inner: Ledger, budget: Budget, reservation: Reservation):
        self.inner, self.budget, self.reservation = inner, budget, reservation
        self.spent = 0.0
        self._inflight: dict[int, float] = {}
        self._lock = threading.Lock()

    def __getattr__(self, name):  # path, caps, entries(), totals(): the ledger's own
        return getattr(self.inner, name)

    def check(self, estimate_usd: float, skill: str | None = None) -> None:
        me = threading.get_ident()
        with self._lock:
            others = sum(usd for thread, usd in self._inflight.items() if thread != me)
            self.budget.grow(self.reservation, self.spent + others + estimate_usd)
            self.inner.check(estimate_usd, skill)
            self._inflight[me] = estimate_usd

    def record(self, entry: Entry) -> None:
        me = threading.get_ident()
        with self._lock:
            self.inner.record(entry)
            self.spent += entry.actual_usd
            self._inflight.pop(me, None)

    def release(self, thread: int | None = None) -> None:
        """A call of this thread (or `thread`) that failed before the ledger recorded it."""
        thread = threading.get_ident() if thread is None else thread
        with self._lock:
            self._inflight.pop(thread, None)
            self.inner.release(thread)

    def close(self) -> None:
        with self._lock:
            for thread in self._inflight:
                self.inner.release(thread)
            self._inflight.clear()


@dataclass
class Job:
    id: str
    keys: tuple[str, ...]
    estimate_usd: float
    started: float = field(default_factory=time.monotonic)
    read_back: str | None = None
    refused: bool = False
    spec: dict | None = None
    variations: list[dict] = field(default_factory=list)
    stopped: str | None = None
    cost_usd: float | None = None
    done: bool = False

    def view(self, since: int = 0) -> dict:
        """What the page sees; `since` leaves out the variations it already has."""
        return {"job": self.id, "done": self.done, "read_back": self.read_back,
                "refused": self.refused, "spec": self.spec,
                "planned": POSITIVES + NEGATIVES, "count": len(self.variations),
                "variations": self.variations[max(0, since):], "stopped": self.stopped,
                "estimate_usd": round(self.estimate_usd, 4), "cost_usd": self.cost_usd}
