"""Paid calls several at a time (Muse Image edits, Spark labels).

`run_all` gives each result as it finishes, with its index, so the callers keep their
outputs in plan order. The spend caps still hold: the ledger reserves each call's
estimate under a lock (backend/meta/ledger.py), and the client's Pacer keeps each
endpoint under its requests-per-minute limit.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


def run_all(items: Sequence, fn: Callable[[Any], Any], workers: int = 1
            ) -> Iterator[tuple[int, Any, BaseException | None]]:
    """Yield (index, result, error) for fn(item) over `items`, `workers` at a time, each
    as it finishes. With one worker they run in order in this thread, as a plain loop.

    When the caller stops early (it raises, or closes the generator), the items not
    started are cancelled and the calls in flight finish first, so each paid call is
    recorded in the ledger."""
    if workers <= 1:
        for i, item in enumerate(items):
            try:
                result = fn(item)
            except Exception as err:  # noqa: BLE001 - handed to the caller, which decides
                yield i, None, err
            else:
                yield i, result, None
        return
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for future in as_completed(futures):
            error = future.exception()
            yield futures[future], None if error else future.result(), error
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
