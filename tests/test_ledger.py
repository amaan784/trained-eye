import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from backend.cli import main
from backend.config import Price, load_settings
from backend.meta.ledger import (
    IMAGE_TOKENS,
    STALE_S,
    BudgetExceeded,
    Entry,
    ImageTokens,
    Ledger,
    format_report,
    image_tokens,
    token_cost,
    unit_cost,
)

SPARK = Price(input_per_m=1.25, output_per_m=4.25)


def test_costs():
    assert token_cost(SPARK, 1_000_000, 0) == pytest.approx(1.25)
    assert token_cost(SPARK, 2000, 1000) == pytest.approx(0.0025 + 0.00425)
    assert unit_cost(Price(per_image=0.01), images=3) == pytest.approx(0.03)
    assert unit_cost(Price(per_audio_hour=0.18), audio_s=1800) == pytest.approx(0.09)


def test_record_and_totals(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", skill_cap_usd=10, global_cap_usd=120)
    assert ledger.spent() == 0 and ledger.entries() == []
    ledger.record(Entry("muse-spark-1.3", "/responses", 0.02, 0.01, skill="washer_done"))
    ledger.record(Entry("muse-image-1.0", "/images/edits", 0.01, 0.01, skill="washer_done"))
    ledger.record(Entry("sam-3.1", "/responses", 0.0025, 0.0, status="error"))
    t = ledger.totals()
    assert t["calls"] == 3 and t["errors"] == 1
    assert t["actual_usd"] == pytest.approx(0.02)
    assert t["by_skill"]["washer_done"]["calls"] == 2
    assert t["by_model"]["sam-3.1"]["usd"] == 0.0
    assert ledger.spent("washer_done") == pytest.approx(0.02)


def test_caps(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", skill_cap_usd=1.0, global_cap_usd=2.0)
    ledger.record(Entry("muse-image-1.0", "/images/edits", 0.9, 0.9, skill="a"))
    ledger.check(0.1, skill="a")
    with pytest.raises(BudgetExceeded, match="skill 'a'"):
        ledger.check(0.2, skill="a")
    ledger.record(Entry("muse-image-1.0", "/images/edits", 1.0, 1.0, skill="b"))
    with pytest.raises(BudgetExceeded, match="global cap"):
        ledger.check(0.2, skill="c")
    ledger.check(0.05)


def test_report(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", skill_cap_usd=10, global_cap_usd=120)
    assert "(no calls yet)" in format_report(ledger)
    ledger.record(Entry("muse-spark-1.3", "/responses", 0.002, 0.0015, skill="washer_done"))
    report = format_report(ledger)
    assert "calls: 1" in report and "washer_done" in report and "$0.0015" in report


def test_report_columns_line_up_for_long_skill_names(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", skill_cap_usd=10, global_cap_usd=120)
    long_name = "dishwasher_cycle_finished_light_on_front"  # 40 characters, the longest
    for skill in (long_name, "washer_done", "washer_done"):
        ledger.record(Entry("muse-spark-1.3", "/chat/completions", 0.002, 0.0015, skill=skill))
    rows = format_report(ledger).split("by skill:\n")[1].splitlines()
    assert len({row.index("$") for row in rows}) == 1
    assert "    1 call " in rows[0] and "    2 calls " in rows[1]


def test_spend_command(tmp_path, monkeypatch, capsys):
    ledger = Ledger(tmp_path / "ledger.jsonl", skill_cap_usd=10, global_cap_usd=120)
    ledger.record(Entry("muse-spark-1.3", "/responses", 0.002, 0.0015, skill="washer_done"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TE_DATA_DIR", str(tmp_path))
    assert main(["spend"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Meta API spend") and "remaining: $119.9985" in out


# -- calls in flight: check reserves, record releases ----------------------------------

ROOT = Path(__file__).resolve().parents[1]


def spark(usd, skill="a"):
    return Entry("muse-spark-1.3", "/chat/completions", usd, usd, skill=skill)


def test_a_call_in_flight_in_another_thread_counts_against_the_cap(tmp_path):
    path = tmp_path / "ledger.jsonl"
    checked, finish = threading.Event(), threading.Event()

    def phone_teach():  # te serve's teach thread, with its own client and Ledger
        ledger = Ledger(path, skill_cap_usd=1.0, global_cap_usd=100.0)
        ledger.check(0.6, "a")
        checked.set()
        finish.wait(5)
        ledger.record(Entry("muse-wonder-1.0", "/chat/completions", 0.6, 0.5, skill="a"))

    thread = threading.Thread(target=phone_teach)
    thread.start()
    assert checked.wait(5)
    watch = Ledger(path, skill_cap_usd=1.0, global_cap_usd=100.0)  # a watch session's
    assert watch.in_flight_usd("a") == pytest.approx(0.6)
    with pytest.raises(BudgetExceeded, match=r"spent \$0.0000 and \$0.6000 in flight"):
        watch.check(0.6, "a")  # before: both passed, $1.20 against a $1.00 cap
    watch.check(0.6, "b")  # another skill's cap
    watch.release()
    finish.set()
    thread.join(5)
    assert watch.in_flight_usd() == 0.0 and watch.spent("a") == pytest.approx(0.5)
    watch.check(0.5, "a")  # recorded at $0.50: this fits now
    with pytest.raises(BudgetExceeded):
        Ledger(path, 1.0, 100.0).check(0.6, "a")  # the global count holds it too


def test_threads_checking_at_once_cannot_all_pass_the_cap(tmp_path):
    path = tmp_path / "ledger.jsonl"
    start, tried, finish = threading.Barrier(6), threading.Semaphore(0), threading.Event()
    passed, refused = [], []

    def call():
        ledger = Ledger(path, skill_cap_usd=1.0, global_cap_usd=100.0)
        start.wait(5)
        try:
            ledger.check(0.3, "a")
        except BudgetExceeded:
            refused.append(1)
            tried.release()
            return
        passed.append(1)
        tried.release()
        finish.wait(5)  # in flight until every thread has tried
        ledger.record(spark(0.3))

    threads = [threading.Thread(target=call) for _ in range(6)]
    for thread in threads:
        thread.start()
    for _ in threads:
        assert tried.acquire(timeout=5)
    finish.set()
    for thread in threads:
        thread.join(5)
    assert len(passed) == 3 and len(refused) == 3  # before: all six passed
    ledger = Ledger(path, 1.0, 100.0)
    assert ledger.spent("a") == pytest.approx(0.9) and ledger.in_flight_usd() == 0.0


def test_a_threads_next_check_replaces_its_reservation(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", skill_cap_usd=1.0, global_cap_usd=100.0)
    ledger.check(0.4, "a")
    ledger.check(0.8, "a")  # the retry guard, the estimate grown: replaces, does not add
    assert ledger.in_flight_usd("a") == pytest.approx(0.8)
    ledger.record(spark(0.8))
    assert ledger.in_flight_usd() == 0.0


def test_reserve_false_only_asks_and_release_is_free_without_a_reservation(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", skill_cap_usd=1.0, global_cap_usd=100.0)
    ledger.release()
    assert not ledger.lock_path.exists()  # nothing held here: the lock was not touched
    ledger.check(0.9, "a", reserve=False)
    assert ledger.in_flight_usd() == 0.0
    ledger.check(0.9, "a")
    ledger.release()
    assert ledger.in_flight_usd() == 0.0


def test_a_failed_record_still_releases(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", skill_cap_usd=1.0, global_cap_usd=100.0)
    ledger.check(0.5, "a")
    ledger.path.mkdir()  # the ledger cannot be appended to
    with pytest.raises(OSError):
        ledger.record(spark(0.5))
    assert ledger.in_flight_usd() == 0.0


def test_a_thread_that_ended_without_recording_holds_nothing(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", skill_cap_usd=1.0, global_cap_usd=100.0)
    thread = threading.Thread(target=ledger.check, args=(0.9, "a"))
    thread.start()
    thread.join(5)
    assert ledger.in_flight_usd() == 0.0
    ledger.check(0.9, "a")


def test_stale_dead_and_unreadable_reservations_are_dropped(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", skill_cap_usd=1.0, global_cap_usd=100.0)
    alive, now = os.getppid(), time.time()
    ledger.lock_path.write_text(json.dumps({
        f"{alive}:1": {"usd": 0.5, "skill": "a", "ts": now},  # another live process: counts
        f"{alive}:2": {"usd": 0.5, "skill": "a", "ts": now - STALE_S - 1},  # too old
        "999999999:1": {"usd": 0.5, "skill": "a", "ts": now},  # no such process
        f"{alive}:3": {"usd": "lots", "skill": "a", "ts": now},
        "junk": {"usd": 0.5, "skill": "a", "ts": now},
        f"{alive}:4": [0.5],
    }))
    assert ledger.in_flight_usd() == pytest.approx(0.5)
    with pytest.raises(BudgetExceeded, match="in flight"):
        ledger.check(0.6, "a")
    assert list(json.loads(ledger.lock_path.read_text())) == [f"{alive}:1"]
    ledger.lock_path.write_text('{"cut sho')  # a crash mid-write
    assert ledger.in_flight_usd() == 0.0
    ledger.check(0.6, "a")


CHILD = """
import sys
from backend.meta.ledger import Ledger
Ledger(sys.argv[1], 1.0, 100.0).check(0.6, "a")
print("reserved", flush=True)
sys.stdin.readline()
"""


def test_a_call_in_flight_in_another_process_counts_against_the_cap(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", skill_cap_usd=1.0, global_cap_usd=100.0)
    child = subprocess.Popen([sys.executable, "-B", "-c", CHILD, str(ledger.path)], cwd=ROOT,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                             env={**os.environ, "PYTHONPATH": str(ROOT)})
    try:
        assert child.stdout.readline().strip() == "reserved"  # a te teach next to te serve
        with pytest.raises(BudgetExceeded, match="in flight"):
            ledger.check(0.6, "a")
    finally:
        child.communicate("\n", timeout=30)
    assert child.returncode == 0
    ledger.check(0.6, "a")  # it ended without recording: its reservation is gone


# -- image tokens measured from the ledger ---------------------------------------------

class FakeLedger:
    def __init__(self, entries=(), error=None):
        self._entries, self.error = list(entries), error

    def entries(self):
        if self.error:
            raise self.error
        return self._entries


def call(model, images, prompt_tokens, text_tokens, status="ok", **usage):
    return {"model": model, "status": status, "input_images": images,
            "input_text_tokens": text_tokens,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 50, **usage}}


SETTINGS = load_settings(env={}, env_file=None)
WONDER, SPARK_MODEL = SETTINGS.model("compiler"), SETTINGS.model("labeler")


def test_image_tokens_fall_back_to_the_guess_without_data():
    assert image_tokens(SETTINGS, FakeLedger()) == ImageTokens(IMAGE_TOKENS, 0) == \
        ImageTokens(1500)
    assert image_tokens(SETTINGS, FakeLedger(error=ValueError("half a line"))).calls == 0
    assert "a guess" in image_tokens(SETTINGS, FakeLedger()).describe()
    assert image_tokens(SETTINGS).per_image == 1500  # the settings' own ledger: none yet


def test_image_tokens_are_the_mean_over_ok_compile_and_label_calls():
    ledger = FakeLedger([
        call(SPARK_MODEL, 1, 1100, 300),  # 800 for one image
        call(WONDER, 3, 3200, 800),  # 2400 for three
        {"model": SPARK_MODEL, "status": "ok", "input_images": 1, "input_text_tokens": 100,
         "usage": {"input_tokens": 900, "output_tokens": 20}},  # Responses naming: 800
        call(SPARK_MODEL, 1, 9000, 300, status="output_error"),
        call(SPARK_MODEL, 1, 9000, 300, unpriced=True),
        call(SPARK_MODEL, 1, 9000, 300, missing=True),
        call(SPARK_MODEL, 0, 9000, 300),  # no image
        call("muse-image-1.0", 1, 9000, 300),  # another model
        call(SPARK_MODEL, 1, True, 300),
        {"model": SPARK_MODEL, "status": "ok", "usage": {"prompt_tokens": 9000}},  # older
        "not an entry",
    ])
    measured = image_tokens(SETTINGS, ledger)
    assert measured == ImageTokens(800, 3)
    assert measured.describe() == ("image tokens: 800 per image, measured over 3 calls in "
                                   "the ledger")


def test_image_tokens_read_what_the_client_records(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", 10.0, 120.0)
    ledger.record(Entry(SPARK_MODEL, "/chat/completions", 0.01, 0.002,
                        usage={"prompt_tokens": 1450, "completion_tokens": 9},
                        input_images=1, input_text_tokens=250))
    assert image_tokens(SETTINGS, ledger) == ImageTokens(1200, 1)


def test_the_plans_spark_price_uses_the_measured_image_tokens(tmp_path):
    from backend.eval.plan import spark_usd_per_check

    settings = load_settings(env={"TE_DATA_DIR": str(tmp_path)}, env_file=None)
    price = settings.price(SPARK_MODEL)
    assert spark_usd_per_check(settings) == round(token_cost(price, 1500 + 250, 1024), 6)
    Ledger(settings.ledger_path, 10.0, 120.0).record(Entry(
        SPARK_MODEL, "/chat/completions", 0.01, 0.002, input_images=1, input_text_tokens=250,
        usage={"prompt_tokens": 1050, "completion_tokens": 9}))
    assert spark_usd_per_check(settings) == round(token_cost(price, 800 + 250, 1024), 6)
