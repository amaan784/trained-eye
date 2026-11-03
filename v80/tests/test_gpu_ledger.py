"""The Modal GPU ledger (data/modal_ledger.jsonl): what te train, te export and te
serve-student record, and the credit check te train and te export ask with."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from backend.config import MODAL_CREDIT_USD
from backend.training import gpu_ledger

T0 = datetime(2026, 11, 3, 14, 0, tzinfo=UTC)


def test_jobs_are_priced_at_the_config_rate_and_a_key_is_written_once(tmp_path):
    row = gpu_ledger.record(tmp_path, "train", "exp1-mixed", 3600, key="train:exp1-mixed:t1",
                            now=T0)
    assert row == {"t": "2026-11-03T14:00:00Z", "kind": "train", "name": "exp1-mixed",
                   "gpu": "H100", "seconds": 3600.0, "usd": 3.95, "upper_bound": False,
                   "key": "train:exp1-mixed:t1"}
    assert gpu_ledger.record(tmp_path, "train", "exp1-mixed", 3600, key="train:exp1-mixed:t1") \
        is None  # te train --collect after te train saw the same run end
    gpu_ledger.record(tmp_path, "export", "exp1-mixed", 1800)
    assert gpu_ledger.spent_usd(tmp_path) == round(3.95 + 1.975, 2)  # an hour, half an hour
    lines = (tmp_path / "modal_ledger.jsonl").read_text().splitlines()
    assert len(lines) == 2 and all(json.loads(line) for line in lines)
    for kind, seconds in (("deploy", 1), ("train", -1), ("train", float("nan"))):
        with pytest.raises(ValueError):
            gpu_ledger.record(tmp_path, kind, "x", seconds)


def test_a_stop_prices_the_time_since_the_deploy_as_an_upper_bound(tmp_path):
    assert gpu_ledger.record_stop(tmp_path, "trained-eye-base", now=T0) is None  # never deployed
    gpu_ledger.record_deploy(tmp_path, "trained-eye-base", "the base model", now=T0)
    gpu_ledger.record_deploy(tmp_path, "trained-eye-base", "again", now=T0 + timedelta(hours=1))
    gpu_ledger.record_deploy(tmp_path, "trained-eye-student", "runs/exp1-mixed/merged", now=T0)
    assert set(gpu_ledger.open_deploys(tmp_path)) == {"trained-eye-base", "trained-eye-student"}
    stop = gpu_ledger.record_stop(tmp_path, "trained-eye-base", now=T0 + timedelta(hours=2))
    assert (stop["kind"], stop["seconds"], stop["usd"], stop["upper_bound"]) == (
        "serve", 7200.0, 7.9, True)  # from the first deploy since the last stop
    assert set(gpu_ledger.open_deploys(tmp_path)) == {"trained-eye-student"}
    totals = gpu_ledger.totals(tmp_path)
    assert totals["serve"] == {"usd": 7.9, "jobs": 1} and totals["total_usd"] == 7.9


def test_the_credit_check_and_the_report(tmp_path):
    assert gpu_ledger.credit_check(tmp_path, 10.0) is None
    gpu_ledger.record(tmp_path, "train", "exp1-mixed", (MODAL_CREDIT_USD - 5) / 3.95 * 3600)
    assert gpu_ledger.credit_check(tmp_path, 4.0) is None
    why = gpu_ledger.credit_check(tmp_path, 11.85)
    assert "plus this $11.85 would pass the $150 of Modal credit" in why and "--yes" in why
    gpu_ledger.record_deploy(tmp_path, "trained-eye-base", "the base model", now=T0)
    (tmp_path / "modal_ledger.jsonl").open("a").write("not json\n")  # skipped
    report = gpu_ledger.format_report(tmp_path)
    assert report.startswith("Modal GPU spend (data/modal_ledger.jsonl, at config prices): "
                             "$145.00 of $150 credit, $5.00 left")
    assert "training $145.00 (1 runs)" in report
    assert "still deployed: trained-eye-base since 2026-11-03T14:00:00Z" in report
    assert report.isascii()
