"""The W&B report (backend/eval/wandb_report.py). wandb-workspaces is a fake module:
nothing is sent to W&B."""

import sys
import types

import pytest

from backend.eval import wandb_report
from backend.eval.writeup import EXPERIMENTS, render

EVAL_URL = "https://wandb.ai/someone/trained-eye/runs/abc123"


def results(**extra):
    """results.json as te eval --wandb leaves it, with one pending evaluation."""
    return {"plan": "plan.json",
            "on_time": {"burner_left_on": {"window_s": 5.0, "hold_s": []}},
            "evaluations": [{"name": "exp1-mixed-burner_left_on", "experiment": 1,
                             "status": "pending", "missing": ["../data/watch/a.jsonl"]}],
            "charts": {"1": ["exp1_f1.svg"], "4": ["exp4_cost.svg"]}, "wandb": EVAL_URL,
            **extra}


def fake_reports(saved: list, fail: Exception | None = None):
    """wandb_workspaces.reports.v2: every block a dict of what it was given."""
    module = types.ModuleType(wandb_report.MODULE)
    for name in ("H1", "P", "MarkdownBlock", "PanelGrid", "Runset", "LinePlot",
                 "WeavePanelSummaryTable", "MediaBrowser"):
        setattr(module, name, lambda _name=name, **kw: {"block": _name, **kw})

    class Report:
        def __init__(self, **kw):
            self.kw, self.url = kw, None

        def save(self, draft=False):
            if fail:
                raise fail
            saved.append((self.kw, draft))
            self.url = "https://wandb.ai/someone/trained-eye/reports/Trained-Eye--Vmlldzo"
            return self

    module.Report = Report
    return module


def test_one_section_per_experiment_with_the_text_of_results_md():
    res = results()
    spec = wandb_report.plan(res, render(res))
    assert spec["title"] == "Trained Eye: experiments" and spec["entity"] == "someone"
    assert [s["title"] for s in spec["sections"]] == list(EXPERIMENTS.values())
    first, *_, fourth = spec["sections"]
    assert "exp1-mixed-burner_left_on" in first["text"] and "![" not in first["text"]
    assert first["training_tag"] == "exp:exp1" and first["charts"] == ["charts/exp1_f1"]
    assert fourth["training_tag"] is None and fourth["charts"] == ["charts/exp4_cost"]
    assert spec["sections"][1]["text"].startswith("No runs yet")
    assert "Computed by `te eval`" in spec["intro"] and "# Evaluation" not in spec["intro"]
    assert spec["eval_run"]["run"] == "abc123" and EVAL_URL in spec["description"]
    assert wandb_report.plan(res, render(res), "team")["entity"] == "team"


@pytest.mark.parametrize("url", [None, "not logged: the wandb package is not installed",
                                 "https://wandb.ai/someone/other-project/runs/abc123"])
def test_it_needs_the_eval_run_in_the_project(url):
    with pytest.raises(ValueError):
        wandb_report.plan(results(wandb=url), "# Evaluation results\n")


def test_the_report_has_the_loss_panels_the_table_and_the_charts():
    res = results()
    report = wandb_report.build_report(wandb_report.plan(res, render(res)), fake_reports([]))
    blocks = report.kw["blocks"]
    assert report.kw["project"] == "trained-eye" and report.kw["title"] == wandb_report.TITLE
    assert [b["text"] for b in blocks if b["block"] == "H1"] == list(EXPERIMENTS.values())
    grids = [b for b in blocks if b["block"] == "PanelGrid"]
    training = [g for g in grids if g["runsets"][0]["name"].startswith("Training")]
    assert [g["runsets"][0]["filters"] for g in training] == [
        wandb_report.tag_filter(f"exp:exp{n}") for n in (1, 2, 3)]
    assert [p["y"] for p in training[0]["panels"]] == [["train/loss"], ["eval/loss"]]
    evals = [g for g in grids if g["runsets"][0]["name"] == "Evaluation run"]
    assert len(evals) == 4 and all(g["runsets"][0]["filters"] == wandb_report.run_filter(
        "abc123") for g in evals)
    assert evals[0]["panels"] == [
        {"block": "WeavePanelSummaryTable", "table_name": "eval/results"},
        {"block": "MediaBrowser", "media_keys": ["charts/exp1_f1"]}]
    assert len(evals[1]["panels"]) == 1  # Experiment 2 has no chart yet
    assert wandb_report.NO_TRAINING in [b.get("text") for b in blocks if b["block"] == "P"]


def test_saving_needs_a_yes_saves_a_draft_and_never_shows_the_key(monkeypatch):
    res = results()
    spec = wandb_report.plan(res, render(res))
    with pytest.raises(PermissionError):
        wandb_report.save_report(spec)
    monkeypatch.setitem(sys.modules, wandb_report.MODULE, None)
    with pytest.raises(RuntimeError, match="pip install wandb-workspaces"):
        wandb_report.save_report(spec, confirmed=True)
    saved = []
    monkeypatch.setitem(sys.modules, wandb_report.MODULE, fake_reports(saved))
    monkeypatch.setenv("WANDB_API_KEY", "wandb_secret_value")
    url = wandb_report.save_report(spec, confirmed=True)
    assert url.startswith("https://wandb.ai/someone/trained-eye/reports/")
    assert [draft for _, draft in saved] == [True]  # a draft: only its author sees it
    monkeypatch.setitem(sys.modules, wandb_report.MODULE, fake_reports(
        saved, fail=OSError("401 for key wandb_secret_value")))
    with pytest.raises(RuntimeError) as err:
        wandb_report.save_report(spec, confirmed=True)
    assert "wandb_secret_value" not in str(err.value) and "[redacted]" in str(err.value)
