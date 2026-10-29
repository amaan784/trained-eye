import pytest

from backend.cli import main
from backend.config import Price
from backend.meta.ledger import (
    BudgetExceeded,
    Entry,
    Ledger,
    format_report,
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


def test_spend_command(tmp_path, monkeypatch, capsys):
    ledger = Ledger(tmp_path / "ledger.jsonl", skill_cap_usd=10, global_cap_usd=120)
    ledger.record(Entry("muse-spark-1.3", "/responses", 0.002, 0.0015, skill="washer_done"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TE_DATA_DIR", str(tmp_path))
    assert main(["spend"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Meta API spend") and "remaining: $119.9985" in out
