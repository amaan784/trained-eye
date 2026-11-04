"""Paid calls several at a time: the client's Pacer (requests per minute per endpoint) and
dataengine.parallel.run_all, as te teach uses them for image edits and labels. Nothing
reaches the network."""

import json
import random
import threading
import time
import types

import httpx
import pytest

from backend.dataengine.parallel import run_all
from backend.dataengine.synth import generate, plan_variations
from backend.meta.client import MetaAPIError
from backend.meta.pace import Pacer
from tests.test_client import SCHEMA, Recorder, chat_reply, make_client
from tests.test_synth import SPEC


def test_the_pacer_spaces_requests_per_endpoint():
    now, slept = [100.0], []
    pacer = Pacer(120, clock=lambda: now[0], sleep=slept.append)  # one every 0.5 s
    assert [pacer.wait("/images/edits") for _ in range(3)] == [0.0, 0.5, 1.0]
    assert pacer.wait("/chat/completions") == 0.0  # another endpoint has its own pace
    now[0] += 5.0  # long after: no wait
    assert pacer.wait("/images/edits") == 0.0 and slept == [0.5, 1.0]
    with pytest.raises(ValueError, match="above 0"):
        Pacer(0)


def test_the_client_paces_requests_that_go_out_but_not_cache_hits(tmp_path):
    waits = []
    rec = Recorder(chat_reply('{"label": "yes"}'), chat_reply('{"label": "no"}'))
    client, _ = make_client(tmp_path, rec)
    client._pacer = types.SimpleNamespace(wait=waits.append)
    client.structured("muse-spark-1.3", "x", SCHEMA)
    client.structured("muse-spark-1.3", "x", SCHEMA)  # cached: nothing goes out
    client.structured("muse-spark-1.3", "y", SCHEMA)
    assert waits == ["/chat/completions", "/chat/completions"] and len(rec.requests) == 2


def test_run_all_in_order_with_one_worker_and_errors_handed_back():
    def fn(x):
        if x == 2:
            raise MetaAPIError(422, "rejected")
        return x * 10

    out = list(run_all([0, 1, 2, 3], fn, workers=1))
    assert [(i, r) for i, r, e in out] == [(0, 0), (1, 10), (2, None), (3, 30)]
    assert isinstance(out[2][2], MetaAPIError) and out[0][2] is None


def test_run_all_runs_calls_at_once_and_keeps_their_index():
    barrier = threading.Barrier(4, timeout=5)  # only passes with four calls in flight

    def fn(x):
        barrier.wait()
        time.sleep(random.random() / 50)
        return x

    out = list(run_all(list(range(12)), fn, workers=4))
    assert sorted(i for i, _, _ in out) == list(range(12))
    assert all(i == r and e is None for i, r, e in out)


def test_a_caller_that_stops_cancels_the_calls_not_started():
    started = []

    def fn(x):
        started.append(x)
        if x == 0:
            raise MetaAPIError(401, "bad key")
        time.sleep(0.05)
        return x

    with pytest.raises(MetaAPIError, match="bad key"):
        for _, _, err in run_all(list(range(40)), fn, workers=2):
            if err is not None:
                raise err
    assert len(started) < 40  # the rest never ran, so they were never billed


def test_generate_with_workers_writes_the_manifest_in_plan_order(tmp_path):
    png = b"\x89PNG\r\n\x1a\n"

    class SlowEdits:
        def image_edit(self, image, prompt, skill=None):
            time.sleep(random.random() / 50)  # finish out of order
            info = types.SimpleNamespace(model="muse-image-1.0", cache_key=prompt[:8],
                                         billed_usd=0.01, cached=False, cost_usd=0.01)
            return types.SimpleNamespace(images=[png + prompt.encode()], info=info)

    plan = plan_variations(SPEC, ["a" * 64], positives=6, negatives=6)
    rows = generate(SlowEdits(), plan, {"a" * 64: b"jpeg"}, tmp_path, workers=5)
    manifest = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text()
                .splitlines()]
    assert [r["id"] for r in rows] == [r["id"] for r in manifest] == [v.id for v in plan]
    assert all((tmp_path / r["file"]).read_bytes().startswith(png) for r in rows)


def test_te_and_the_phone_use_a_paced_client(monkeypatch):
    from backend import cli, config

    client = cli.make_client(config.load_settings(env={}, env_file=None))
    assert isinstance(client._pacer, Pacer) and client._pacer.gap == 60 / config.MAX_RPM
    assert 10 <= config.PARALLEL_CALLS <= 20 and config.MAX_RPM < 150
    assert isinstance(client._http, httpx.Client)
