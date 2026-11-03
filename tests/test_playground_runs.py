"""A run's estimate, the metered ledger that holds each call to the run's reservation
(a call over a cap is never sent), and the job view the page polls."""

import threading

import httpx
import pytest

from backend.config import load_settings
from backend.meta.client import MetaClient
from backend.meta.ledger import BudgetExceeded, Entry, Ledger
from backend.playground.budget import Budget, CapReached
from backend.playground.runs import Job, MeteredLedger, estimate_usd
from tests.test_teach import FakeMeta, png

KEYS = ("visitor:abc",)


def metered(tmp_path, visitor_cap, fake=None):
    settings = load_settings(env={"TE_DATA_DIR": str(tmp_path), "MODEL_API_KEY": "sk-test"},
                             env_file=None)
    client = MetaClient(settings, transport=httpx.MockTransport(fake or FakeMeta()),
                        sleep=lambda s: None)
    budget = Budget(tmp_path / "budget.json", visitor_cap, 100.0)
    meter = MeteredLedger(client.ledger, budget, budget.reserve(KEYS, 0.0))
    client.ledger = meter
    return client, meter, budget


def test_estimate():
    settings = load_settings(env={}, env_file=None)
    # Muse Wonder 2400 tokens in and 4096 out; Muse Image $0.01 an image; Muse Spark 1750
    # in and 1024 out per label (config.PRICES)
    compile_ = (2400 * 4.00 + 4096 * 20.00) / 1e6
    label = (1750 * 1.25 + 1024 * 4.25) / 1e6
    assert estimate_usd(settings) == pytest.approx(compile_ + 12 * (0.01 + label))
    assert estimate_usd(settings, 0) == pytest.approx(compile_)


def test_a_call_over_the_cap_is_never_sent(tmp_path):
    fake = FakeMeta()
    client, meter, budget = metered(tmp_path, 0.015, fake)
    client.image_edit(png((1, 2, 3)), "one", skill="playground")  # $0.01: fits
    with pytest.raises(CapReached):
        client.image_edit(png((1, 2, 3)), "two", skill="playground")  # $0.01 more: does not
    assert fake.edits == 1 and meter.spent == pytest.approx(0.01)
    assert len(meter.entries()) == 1  # the ledger's own file, through the meter
    budget.settle(meter.reservation, meter.spent)
    assert budget.left(KEYS)["visitor_left_usd"] == pytest.approx(0.005)


def test_calls_in_flight_count_together(tmp_path):
    _, meter, _ = metered(tmp_path, 0.015)
    checked, finish = threading.Event(), threading.Event()

    def worker():  # another worker's call: allowed, sent, not recorded yet
        meter.check(0.01, "playground")
        checked.set()
        finish.wait(5)
        meter.record(Entry("muse-image-1.0", "/images/edits", 0.01, 0.002, skill="playground"))

    thread = threading.Thread(target=worker)
    thread.start()
    checked.wait(5)
    with pytest.raises(CapReached):  # 0.01 in flight + 0.01 more > 0.015
        meter.check(0.01, "playground")
    finish.set()
    thread.join()
    meter.check(0.01, "playground")  # it was recorded at 0.002: now this one fits
    assert meter.spent == 0.002 and meter.reservation.usd == pytest.approx(0.012)
    meter.close()


def test_every_runs_calls_in_flight_count_against_the_ledger(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", 0.015, 100.0)
    budget = Budget(tmp_path / "budget.json", 1.0, 100.0)
    first = MeteredLedger(ledger, budget, budget.reserve(("visitor:a",), 0.0))
    second = MeteredLedger(ledger, budget, budget.reserve(("visitor:b",), 0.0))
    checked, finish = threading.Event(), threading.Event()

    def worker():  # the first run's call, in its own thread: sent, not recorded yet
        first.check(0.01, "playground")
        checked.set()
        finish.wait(5)

    thread = threading.Thread(target=worker)
    thread.start()
    checked.wait(5)
    with pytest.raises(BudgetExceeded) as err:  # 0.01 in flight + 0.01 > the skill cap
        second.check(0.01, "playground")
    assert not isinstance(err.value, CapReached)  # the ledger's cap, not the playground's
    first.close()  # the first run ended without recording (its call failed)
    assert ledger.in_flight_usd() == 0.0
    second.check(0.01, "playground")
    second.close()
    finish.set()
    thread.join()


def test_a_call_that_fails_before_the_ledger_records_it_is_released(tmp_path):
    _, meter, _ = metered(tmp_path, 1.0)
    meter.check(0.01, "playground")
    assert meter.inner.in_flight_usd("playground") == pytest.approx(0.01)
    meter.release()
    assert meter.inner.in_flight_usd() == 0.0 and meter._inflight == {}


def test_job_view():
    job = Job("j1", KEYS, 0.29004, variations=[{"id": i} for i in range(5)])
    view = job.view(since=3)
    assert view["count"] == 5 and view["variations"] == [{"id": 3}, {"id": 4}]
    assert view["planned"] == 12 and view["estimate_usd"] == 0.29 and not view["done"]
    assert job.view(since=-2)["variations"] == job.variations


def test_estimate_prices_images_at_the_tokens_measured_in_the_ledger(tmp_path):
    settings = load_settings(env={"TE_DATA_DIR": str(tmp_path)}, env_file=None)
    ledger = Ledger(settings.ledger_path, 10.0, 120.0)
    ledger.record(Entry("muse-wonder-1.0", "/chat/completions", 0.1, 0.01, input_images=1,
                        input_text_tokens=900,
                        usage={"prompt_tokens": 1700, "completion_tokens": 500}))
    compile_ = (1700 * 4.00 + 4096 * 20.00) / 1e6  # 800 per image, measured
    label = (1050 * 1.25 + 1024 * 4.25) / 1e6
    assert estimate_usd(settings, ledger=ledger) == pytest.approx(compile_ + 12 * (0.01 + label))
    assert estimate_usd(settings) == estimate_usd(settings, ledger=ledger)  # the same file
