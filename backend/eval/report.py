"""`te eval` (and `make eval`): score replayed, labelled test clips and write the report.

The plan is a JSON file that lists the evaluations to score:

    {"evaluations": [
       {"name": "exp1-mixed", "experiment": 1, "skill": "burner_left_on", "variant": "mixed",
        "clips": [{"log": "../data/watch/replay-20261101-120000-burner1.jsonl",
                   "labels": "../data/labels/burner1.json", "timeline": true}],
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
- run: the model whose replays the logs hold, the folder under data/watch/ (te replay
  --run): a training run, the first run with the same bundle (te experiments says which),
  exp1-mixed for Experiment 4's fine-tuned arm, exp4-base or exp4-spark.
- scorer: "student" (the default) or "spark" (Muse Spark judges every frame).
A clip may also name "clip": the recorded video or frame folder its log replays (te
replay reads it, and Experiment 1's failure galleries take their frames from it).

calibration (optional): what te replay --run replays before the test clips, so each run's
threshold comes from its own validation clips: [{"run", "skill", "clips": [{"clip", "log",
"labels"}]}], calibrated into data/watch/<run>/<skill>/threshold.json (te calibrate --run).

on_time_s (optional): {skill: seconds}, to change a skill's on-time window from its
default (metrics.ON_TIME_S): an event's alert is on time when it comes within that many
seconds after the end of the skill's hold (the hold the logs record). An evaluation whose
log or labels file does not exist yet is pending, not scored, so the plan can come before
the clips. A scored one also records the threshold its logs were scored at, and where it
came from (thresholds_used), and its hold and on-time window ("on_time"), in
results.json, results.md and W&B.

run_eval also writes, next to the charts: Experiment 4's judges pooled over their skills
(the cost chart's three points), Experiment 1's failure galleries (gallery.py, local only:
the frames may show faces), and with data_dir and regression, Spark's error rate and label
agreement per skill (data/skills/<skill>/spark_errors.json, label_stats.json) and te
regress's eval/regression.json, in results.json, results.md and the W&B run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.config import MODAL_GPU_USD_PER_HOUR
from backend.eval import metrics
from backend.eval.regression import markdown as regression_markdown
from backend.eval.writeup import render, write_charts
from backend.skills.spec import check_name

WANDB_PROJECT = "trained-eye"
PRICES = ("gpu_usd_per_hour", "spark_usd_per_check", "scorer_usd_per_frame")
RUN = r"^[a-z0-9][a-z0-9-]{1,39}$"  # training.runs.RUN_NAME, as te train checks it
SPARK_KEYS = ("hand_labelled", "unmatched", "human_unsure", "spark_failed", "spark",
              "confusion", "intended", "kept")  # of spark_errors.json
STATS_KEYS = ("total", "kept", "unsure", "disagree", "label_errors", "agreement")
REGRESSION_KEYS = ("precision", "questions", "base", "finetuned", "changed")
GALLERY_SIZE = 6  # false alarms and misses shown per Experiment 1 evaluation


class Clip(BaseModel):
    model_config = ConfigDict(extra="forbid")
    log: Path
    labels: Path
    timeline: bool = False
    clip: Path | None = None


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
    run: str | None = Field(None, pattern=RUN)
    scorer: Literal["student", "spark"] = "student"

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


class Calibration(BaseModel):
    """Validation clips te replay --run replays, then calibrates the run's threshold on."""

    model_config = ConfigDict(extra="forbid")
    run: str = Field(pattern=RUN)
    skill: str
    clips: list[Clip] = Field(min_length=1)

    @field_validator("skill")
    @classmethod
    def _skill(cls, value: str) -> str:
        return check_name(value)


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    on_time_s: dict[str, float] = {}
    evaluations: list[Evaluation]
    calibration: list[Calibration] = []

    @field_validator("on_time_s")
    @classmethod
    def _windows(cls, value: dict[str, float]) -> dict[str, float]:
        for skill, seconds in value.items():
            check_name(skill)
            if not seconds > 0:
                raise ValueError(f"on_time_s for {skill} must be over 0 seconds")
        return value

    def window(self, skill: str) -> float:
        """The skill's on-time window: this plan's override, else metrics.ON_TIME_S."""
        return metrics.on_time_window(skill, self.on_time_s)

    @model_validator(mode="after")
    def _unique(self) -> Plan:
        names = [e.name for e in self.evaluations]
        if twice := sorted({n for n in names if names.count(n) > 1}):
            raise ValueError(f"evaluation names must be unique: {', '.join(twice)}")
        pairs = [(c.run, c.skill) for c in self.calibration]
        if twice := sorted({f"{r} {s}" for r, s in pairs if pairs.count((r, s)) > 1}):
            raise ValueError(f"one calibration entry per run and skill: {', '.join(twice)}")
        return self


def count(ev: Evaluation, base: Path, window_s: float) -> metrics.ClipCounts | None:
    """The evaluation's counts over its clips, None while a log or labels file is missing."""
    if any(not (base / p).is_file() for c in ev.clips for p in (c.log, c.labels)):
        return None
    total = metrics.ClipCounts()
    for c in ev.clips:
        try:
            total += metrics.count_clip(metrics.read_log(base / c.log), ev.skill,
                                        metrics.read_labels(base / c.labels), window_s)
        except (ValueError, KeyError, TypeError) as err:
            raise ValueError(f"{ev.name}, {c.log}: {type(err).__name__} {err}") from err
    return total


def thresholds_used(ev: Evaluation, base: Path) -> dict:
    """The thresholds the replays scored the skill at, from its frame logs: {"values" (each
    distinct one, rounded to 4 places), "sources" (te watch's threshold_source: flag, run,
    skill, default or spark)}. Each run's threshold comes from its own calibration (te
    replay, te calibrate --run), so this says which one a result is at."""
    values, sources = set(), set()
    for c in ev.clips:
        for row in metrics.read_log(base / c.log):
            if row.get("skill") == ev.skill and row.get("threshold") is not None:
                values.add(round(float(row["threshold"]), 4))
                if row.get("threshold_source"):
                    sources.add(str(row["threshold_source"]))
    return {"values": sorted(values), "sources": sorted(sources)}


def holds_used(ev: Evaluation, base: Path) -> list[float]:
    """The holds the replays fired the skill with, from its frame logs (te watch logs the
    skill's hold_s; a log without it counts as 0)."""
    return sorted({metrics.hold_of(metrics.read_log(base / c.log), ev.skill)
                   for c in ev.clips})


def score(ev: Evaluation, base: Path, window_s: float,
          counts: metrics.ClipCounts | None = None) -> dict:
    """One evaluation's entry in results.json: scored, or pending without its files."""
    entry = {**ev.model_dump(mode="json"), "variant": ev.variant or ev.name}
    if counts is None:
        counts = count(ev, base, window_s)
    if counts is None:
        missing = [str(p) for c in ev.clips for p in (c.log, c.labels)
                   if not (base / p).is_file()]
        return {**entry, "status": "pending", "missing": missing}
    summary = metrics.summarize([counts])
    cost = metrics.cost_per_hour(summary, **ev.model_dump(include=set(PRICES)))
    return {**entry, "status": "scored", "metrics": summary, "cost_per_hour": cost,
            "threshold": thresholds_used(ev, base),
            "on_time": {"hold_s": holds_used(ev, base), "window_s": window_s}}


def on_time(plan: Plan, entries: list[dict]) -> dict:
    """Each plan skill's on-time window and the holds its scored logs used:
    {skill: {"window_s", "hold_s": [...]}} (no holds while nothing is scored)."""
    out = {}
    for skill in sorted({e.skill for e in plan.evaluations}):
        holds = sorted({h for e in entries if e["skill"] == skill and e["status"] == "scored"
                        for h in e["on_time"]["hold_s"]})
        out[skill] = {"window_s": plan.window(skill), "hold_s": holds}
    return out


def pool_judges(entries: list[dict], counts: dict[str, metrics.ClipCounts]) -> list[dict]:
    """Experiment 4's judges (one per variant, in plan order) pooled over their skills for
    the cost chart: the ClipCounts of a judge's scored evaluations added up and summarised,
    and its dollars per hour combined, each evaluation's weighted by its hours of footage
    (None if one has none). `skills` are the scored ones, `of` all the judge has."""
    judges: dict[str, dict] = {}
    for e in entries:
        if e["experiment"] != 4:
            continue
        j = judges.setdefault(e["variant"], {"variant": e["variant"], "skills": [], "of": 0,
                                             "highlight": False, "counts": [], "cost": []})
        j["of"] += 1
        j["highlight"] = j["highlight"] or e["highlight"]
        if e["status"] == "scored":
            j["skills"].append(e["skill"])
            j["counts"].append(counts[e["name"]])
            j["cost"].append((e["cost_per_hour"]["total_usd"], e["metrics"]["hours"]))
    out = []
    for j in judges.values():
        hours = sum(h for _, h in j["cost"])
        usd = (None if not hours or any(t is None for t, _ in j["cost"])
               else round(sum(t * h for t, h in j["cost"]) / hours, 4))
        out.append({"variant": j["variant"], "skills": j["skills"], "of": j["of"],
                    "highlight": j["highlight"], "usd_per_hour": usd,
                    "metrics": metrics.summarize(j["counts"]) if j["counts"] else None})
    return out


def _read(path: Path, keys: tuple[str, ...]) -> dict | None:
    """The given keys of a JSON file, None when there is no file; ValueError if unreadable."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: data[k] for k in keys if k in data}
    except (ValueError, TypeError) as err:  # TypeError: not a JSON object
        raise ValueError(f"{path} cannot be read: {type(err).__name__} {err}") from err


def side_results(skills: list[str], data_dir: Path | None, regression: Path | None) -> dict:
    """Spark's error rate and label agreement per skill, and the regression check: what
    run_eval adds to results.json besides the evaluations (only what it was given)."""
    out: dict = {}
    if data_dir is not None:
        folder = Path(data_dir) / "skills"
        out["spark_errors"] = {s: _read(folder / s / "spark_errors.json", SPARK_KEYS)
                               for s in skills}
        out["label_stats"] = {s: _read(folder / s / "label_stats.json", STATS_KEYS)
                              for s in skills}
    if regression is not None:
        out["regression"] = reg = _read(regression, REGRESSION_KEYS)
        try:
            if reg is not None:
                regression_markdown(reg)
        except (KeyError, TypeError, ValueError) as err:
            raise ValueError(f"{regression} is not te regress's report ({type(err).__name__} "
                             f"{err}): run te regress again") from err
    return out


def side_summary(results: dict) -> dict:
    """The W&B summary keys for Experiment 4's pooled judges, Spark's error rate, the label
    agreement and the regression check (None values left out)."""
    out = {}
    for j in results.get("experiment4") or []:
        for key in ("f1", "accuracy", "frames"):
            out[f"exp4-pooled/{j['variant']}/{key}"] = (j["metrics"] or {}).get(key)
        out[f"exp4-pooled/{j['variant']}/usd_per_hour"] = j["usd_per_hour"]
    for skill, err in (results.get("spark_errors") or {}).items():
        for part in ("spark", "intended", "kept"):
            rate = (err or {}).get(part) or {}
            out[f"spark_errors/{skill}/{part}_rate"] = rate.get("rate")
            out[f"spark_errors/{skill}/{part}_n"] = rate.get("n")
            low, high = rate.get("ci95") or (None, None)
            out[f"spark_errors/{skill}/{part}_ci95_low"] = low
            out[f"spark_errors/{skill}/{part}_ci95_high"] = high
        out[f"spark_errors/{skill}/spark_unsure"] = ((err or {}).get("spark") or {}).get("unsure")
    for skill, stats in (results.get("label_stats") or {}).items():
        for key in ("agreement", "kept", "total"):
            out[f"label_stats/{skill}/{key}"] = (stats or {}).get(key)
    reg = results.get("regression") or {}
    for side in ("base", "finetuned"):
        out[f"regression/{side}_accuracy"] = (reg.get(side) or {}).get("accuracy")
    if reg:
        out["regression/questions"] = reg.get("questions")
        out["regression/changed"] = len(reg.get("changed") or [])
    return {k: v for k, v in out.items() if v is not None}


def log_to_wandb(results: dict, out_dir: Path, entity: str | None = None) -> str:
    """Log every number, the table and the charts to one W&B run (in team `entity`, else the
    key's own account). Returns its URL, or why nothing was logged. The key comes from the
    environment or .env and is never shown."""
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
    run = wandb.init(project=WANDB_PROJECT, entity=entity or None, name="eval",
                     job_type="eval", tags=["eval", "m3"],
                     config={"plan": results["plan"], "on_time": results["on_time"],
                             "evaluations": [e["name"] for e in results["evaluations"]]})
    for e in scored:
        values = (e.get("threshold") or {}).get("values") or []
        for key, value in {**e["metrics"], "usd_per_hour": e["cost_per_hour"]["total_usd"],
                           "threshold": values[0] if len(values) == 1 else None}.items():
            if value is not None:
                run.summary[f"{e['name']}/{key}"] = value
    for key, value in side_summary(results).items():
        run.summary[key] = value
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


def run_eval(plan_path: Path, out_dir: Path, *, log_wandb: bool = False,
             wandb_entity: str | None = None, data_dir: Path | None = None,
             regression: Path | None = None, gallery_size: int = GALLERY_SIZE) -> dict:
    """Score the plan's evaluations; write out_dir/results.json (every number), results.md
    and the SVG charts (writeup.py), Experiment 1's failure galleries (gallery.py, under
    out_dir/galleries/, local only), and with log_wandb log them to W&B if it is installed
    (in team wandb_entity, te eval --wandb-entity). With data_dir (data/), each plan
    skill's spark_errors.json and label_stats.json go in too; with regression, te regress's
    regression.json when it exists."""
    from backend.eval.gallery import write_galleries

    plan_path, out_dir = Path(plan_path), Path(out_dir)
    plan = Plan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    base = plan_path.resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {e.name: count(e, base, plan.window(e.skill)) for e in plan.evaluations}
    entries = [score(e, base, plan.window(e.skill), counts[e.name]) for e in plan.evaluations]
    results = {"plan": plan_path.name, "on_time": on_time(plan, entries),
               "evaluations": entries,
               "experiment4": pool_judges(entries, counts),
               **side_results(sorted({e.skill for e in plan.evaluations}), data_dir,
                              regression)}
    results["charts"] = write_charts(results, base, out_dir)
    results["galleries"] = write_galleries(results, base, out_dir, size=gallery_size)
    if log_wandb:
        results["wandb"] = log_to_wandb(results, out_dir, entity=wandb_entity)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    (out_dir / "results.md").write_text(render(results), encoding="utf-8")
    return results
