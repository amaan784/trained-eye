"""`te eval` (and `make eval`): score replayed, labelled test clips and write the report.

The plan is a JSON file that lists the evaluations to score:

    {"window_s": 5,
     "evaluations": [
       {"name": "exp1-mixed", "experiment": 1, "skill": "washer_done", "variant": "mixed",
        "clips": [{"log": "../data/watch/replay-20261101-120000-washer1.jsonl",
                   "labels": "../data/labels/washer1.json", "timeline": true}],
        "gpu": "H100", "spark_usd_per_check": 0.003}]}

An evaluation has:
- name: unique; letters, digits, dots, dashes and underscores (it names files and W&B keys).
- experiment: 1 to 4 (writeup.EXPERIMENTS). skill: whose rows are read from each log.
- variant: the label of its bar, point or table row (default: the name).
- clips: one or more pairs of a frame log from `te watch` and the clip's human labels
  {"true": [[start_s, end_s], ...]}; "timeline": true also draws the live timeline of
  that clip. Relative paths are taken from the plan file's folder.
- gpu_usd_per_hour, or gpu (a key of config.MODAL_GPU_USD_PER_HOUR, such as "H100"): the
  student's GPU price. Leave both out for a student on the Mac.
- spark_usd_per_check: one Muse Spark confirmation (default 0).
- scorer_usd_per_frame: a per-frame scorer, for Experiment 4's Spark-on-every-frame run
  (spark_scorer.sent_summary gives the mean price; default 0).
- images and generation_usd: Experiment 3's synthetic images per skill (needed there)
  and what generating them cost.
- highlight: true for the point drawn in the accent colour (Experiment 4: fine-tuned).
- alert_text: the spoken alert, written on the timeline (default "alert").

window_s (default 5) is the event recall window. An evaluation whose log or labels file
does not exist yet is pending, not scored, so the plan can come before the clips.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.config import MODAL_GPU_USD_PER_HOUR
from backend.eval import metrics
from backend.eval.writeup import render, write_charts
from backend.skills.spec import check_name

WANDB_PROJECT = "trained-eye"
PRICES = ("gpu_usd_per_hour", "spark_usd_per_check", "scorer_usd_per_frame")


class Clip(BaseModel):
    model_config = ConfigDict(extra="forbid")
    log: Path
    labels: Path
    timeline: bool = False


class Evaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    experiment: int = Field(ge=1, le=4)
    skill: str
    variant: str | None = None
    clips: list[Clip] = Field(min_length=1)
    gpu: str | None = None
    gpu_usd_per_hour: float | None = Field(None, ge=0)
    spark_usd_per_check: float = Field(0.0, ge=0)
    scorer_usd_per_frame: float = Field(0.0, ge=0)
    images: int | None = Field(None, gt=0)
    generation_usd: float | None = Field(None, ge=0)
    highlight: bool = False
    alert_text: str = "alert"

    @field_validator("skill")
    @classmethod
    def _skill(cls, value: str) -> str:
        return check_name(value)

    @model_validator(mode="after")
    def _check(self) -> Evaluation:
        if self.gpu is not None:
            if self.gpu_usd_per_hour is not None or self.gpu not in MODAL_GPU_USD_PER_HOUR:
                raise ValueError(f"gpu must be one of {sorted(MODAL_GPU_USD_PER_HOUR)}, "
                                 "without gpu_usd_per_hour")
            self.gpu_usd_per_hour = MODAL_GPU_USD_PER_HOUR[self.gpu]
        if self.experiment == 3 and self.images is None:
            raise ValueError(f"{self.name}: Experiment 3 evaluations need images")
        return self


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    window_s: float = Field(metrics.EVENT_WINDOW_S, gt=0)
    evaluations: list[Evaluation]

    @model_validator(mode="after")
    def _unique(self) -> Plan:
        names = [e.name for e in self.evaluations]
        if twice := sorted({n for n in names if names.count(n) > 1}):
            raise ValueError(f"evaluation names must be unique: {', '.join(twice)}")
        return self


def score(ev: Evaluation, base: Path, window_s: float) -> dict:
    """One evaluation's entry in results.json: scored, or pending without its files."""
    entry = {**ev.model_dump(mode="json"), "variant": ev.variant or ev.name}
    missing = [str(p) for c in ev.clips for p in (c.log, c.labels) if not (base / p).is_file()]
    if missing:
        return {**entry, "status": "pending", "missing": missing}
    counts = []
    for c in ev.clips:
        try:
            counts.append(metrics.count_clip(metrics.read_log(base / c.log), ev.skill,
                                             metrics.read_labels(base / c.labels), window_s))
        except (ValueError, KeyError, TypeError) as err:
            raise ValueError(f"{ev.name}, {c.log}: {type(err).__name__} {err}") from err
    summary = metrics.summarize(counts)
    cost = metrics.cost_per_hour(summary, **ev.model_dump(include=set(PRICES)))
    return {**entry, "status": "scored", "metrics": summary, "cost_per_hour": cost}


def log_to_wandb(results: dict, out_dir: Path) -> str:
    """Log every number, the table and the charts to one W&B run. Returns its URL, or why
    nothing was logged. The key comes from the environment or .env and is never shown."""
    try:
        import wandb
    except ImportError:
        return "not logged: the wandb package is not installed (pip install -e \".[train]\")"
    from backend.config import load_settings

    settings = load_settings()
    if settings.has_secret("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_API_KEY", settings.secret("WANDB_API_KEY"))
    scored = [e for e in results["evaluations"] if e["status"] == "scored"]
    keys = list(scored[0]["metrics"]) if scored else []
    run = wandb.init(project=WANDB_PROJECT, name="eval", job_type="eval", tags=["eval", "m3"],
                     config={"plan": results["plan"], "window_s": results["window_s"],
                             "evaluations": [e["name"] for e in results["evaluations"]]})
    for e in scored:
        for key, value in {**e["metrics"], "usd_per_hour": e["cost_per_hour"]["total_usd"]}.items():
            if value is not None:
                run.summary[f"{e['name']}/{key}"] = value
    rows = [[e["name"], e["experiment"], e["skill"], e["variant"],
             *(e["metrics"][k] for k in keys), e["cost_per_hour"]["total_usd"]] for e in scored]
    charts = {f"charts/{Path(n).stem}": wandb.Html((out_dir / n).read_text(encoding="utf-8"))
              for names in results["charts"].values() for n in names}
    run.log({"eval/results": wandb.Table(
        columns=["name", "experiment", "skill", "variant", *keys, "usd_per_hour"], data=rows),
        **charts})
    url = run.url
    run.finish()
    return url


def run_eval(plan_path: Path, out_dir: Path, *, log_wandb: bool = False) -> dict:
    """Score the plan's evaluations; write out_dir/results.json (every number), results.md
    and the SVG charts (writeup.py), and with log_wandb log them to W&B if it is installed."""
    plan_path, out_dir = Path(plan_path), Path(out_dir)
    plan = Plan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    base = plan_path.resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"plan": plan_path.name, "window_s": plan.window_s,
               "evaluations": [score(e, base, plan.window_s) for e in plan.evaluations]}
    results["charts"] = write_charts(results, base, out_dir)
    if log_wandb:
        results["wandb"] = log_to_wandb(results, out_dir)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    (out_dir / "results.md").write_text(render(results), encoding="utf-8")
    return results
