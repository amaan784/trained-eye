"""What run_eval writes from the scored evaluations: the SVG charts and results.md.

results.md has one section per experiment, in plain words: its charts, a table of
accuracy and one of speed and cost. Every number comes from report.run_eval's computation
(results.json holds the same ones); n/a means there was nothing to compute it from. An
experiment with no evaluations in the plan says so, and evaluations still waiting for
their clips are listed as pending.
"""

from __future__ import annotations

from pathlib import Path

from backend.eval import metrics
from backend.eval.charts import grouped_bars, line_chart, scatter
from backend.eval.svg import usd
from backend.eval.timeline import timeline

EXPERIMENTS = {
    1: "Experiment 1: real only, synthetic only, mixed",
    2: "Experiment 2: labels filtered by Muse Spark, or not",
    3: "Experiment 3: how many synthetic images",
    4: "Experiment 4: fine-tuned Glimmer, base Glimmer, Muse Spark on every frame",
}
COST_HEADER = ["Run", "Latency median", "Latency p90", "Student errors", "Alerts",
               "Spark checks per hour", "Dollars per hour", "Prices used"]


def _num(value: float | None, spec: str = ".2f", unit: str = "") -> str:
    return "n/a" if value is None else f"{value:{spec}}{unit}"


def _events(caught: int, events: int) -> str:
    return f"{caught} of {events}" if events else "no events"


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(_cell(c) for c in row) + " |" for row in rows)]


def header(experiment: int, window_s: float) -> list[str]:
    extra = {3: ["Synthetic images", "Generation cost"], 4: ["Frame accuracy"]}
    return ["Run", "Variant", "Skill", "Clips", "Footage", "Frames", "Precision", "Recall",
            "F1", f"Events caught within {window_s:g} s", "Events caught at all",
            "False alarms per hour", "Time to alert, median (mean)",
            *extra.get(experiment, [])]


def accuracy_row(e: dict) -> list[str]:
    m = e["metrics"]
    delay = ("n/a" if m["time_to_alert_median_s"] is None else
             f"{m['time_to_alert_median_s']:.1f} s ({m['time_to_alert_mean_s']:.1f} s)")
    row = [e["name"], e["variant"], e["skill"], str(m["clips"]), f"{m['hours'] * 60:.1f} min",
           str(m["frames"]), _num(m["precision"]), _num(m["recall"]), _num(m["f1"]),
           _events(m["caught_in_window"], m["events"]), _events(m["caught"], m["events"]),
           _num(m["false_alarms_per_hour"], ".1f"), delay]
    if e["experiment"] == 3:
        cost = e["generation_usd"]
        row += [str(e["images"]), "n/a" if cost is None else usd(cost)]
    if e["experiment"] == 4:
        row.append(_num(m["accuracy"]))
    return row


def prices(e: dict) -> str:
    gpu = e["gpu_usd_per_hour"]
    parts = ["Mac" if gpu is None else f"GPU {usd(gpu)} per hour"]
    if e["spark_usd_per_check"]:
        parts.append(f"{usd(e['spark_usd_per_check'])} per Spark check")
    elif e["metrics"]["cloud_checks"]:
        parts.append("Spark checks not priced in the plan")
    if e["scorer_usd_per_frame"]:
        parts.append(f"{usd(e['scorer_usd_per_frame'])} per frame scored")
    return ", ".join(parts)


def cost_row(e: dict) -> list[str]:
    m, total = e["metrics"], e["cost_per_hour"]["total_usd"]
    return [e["name"], _num(m["latency_median_s"], unit=" s"), _num(m["latency_p90_s"], unit=" s"),
            str(m["student_errors"]), str(m["alerts"]), _num(m["cloud_checks_per_hour"], ".1f"),
            "n/a" if total is None else usd(total), prices(e)]


def _value(e: dict, key: str) -> float | None:
    return e["metrics"][key] if e["status"] == "scored" else None


def write_charts(results: dict, base: Path, out_dir: Path) -> dict[str, list[str]]:
    """Write the SVG charts into out_dir. Returns their file names by experiment ("1" to
    "4") and "timelines". Clip paths are taken from `base`, the plan's folder."""
    charts: dict[str, list[str]] = {}

    def save(key: str, name: str, svg: str) -> None:
        (out_dir / name).write_text(svg, encoding="utf-8")
        charts.setdefault(key, []).append(name)

    evals, window = results["evaluations"], results["window_s"]
    for exp in (1, 2):
        runs = [e for e in evals if e["experiment"] == exp]
        if any(e["status"] == "scored" for e in runs):
            several = len({e["skill"] for e in runs}) > 1
            groups = [f"{e['variant']} ({e['skill']})" if several else e["variant"] for e in runs]
            recall = f"events caught within {window:g} s"
            save(str(exp), f"exp{exp}_bars.svg", grouped_bars(
                EXPERIMENTS[exp], f"Frame F1 and {recall}, on the labelled test clips", groups,
                [("frame F1", [_value(e, "f1") for e in runs]),
                 (recall, [_value(e, "event_recall_in_window") for e in runs])]))
    runs = [e for e in evals if e["experiment"] == 3 and e["status"] == "scored"]
    for skill in sorted({e["skill"] for e in runs}):
        points = [(e["images"], e["metrics"]["f1"], "cost n/a" if e["generation_usd"] is None
                   else usd(e["generation_usd"])) for e in runs if e["skill"] == skill]
        save("3", f"exp3_{skill}.svg", line_chart(
            f"{EXPERIMENTS[3]}: {skill}", "Frame F1, with the generation cost under each point",
            points, "synthetic images per skill"))
    runs = [e for e in evals if e["experiment"] == 4]
    if any(e["status"] == "scored" for e in runs):
        points = [{"label": e["variant"], "x": e.get("cost_per_hour", {}).get("total_usd"),
                   "y": _value(e, "f1"), "highlight": e["highlight"]} for e in runs]
        save("4", "exp4_cost.svg", scatter(EXPERIMENTS[4], "Frame F1 against dollars per hour "
                                           "of watching", points))
    for e in evals:
        for i, c in enumerate(e["clips"], 1):
            if c.get("timeline") and e["status"] == "scored":
                rows = [r for r in metrics.read_log(base / c["log"]) if r.get("skill") == e["skill"]]
                save("timelines", f"timeline_{e['name']}_{i}.svg", timeline(
                    f"Live timeline: {e['skill']}, {e['variant']}",
                    f"{Path(c['log']).name}, run {e['name']}", rows,
                    metrics.read_labels(base / c["labels"]), e["alert_text"]))
    return charts


def render(results: dict) -> str:
    """The whole results.md for the dict run_eval returns."""
    window = results["window_s"]
    lines = [
        "# Evaluation results", "",
        (f"Computed by `te eval` from the plan {results['plan']}: the frame logs of `te watch` "
         "scored against the human labels of each test clip. Every number below comes from "
         "that computation, and results.json holds all of them. n/a means there was nothing "
         "to compute it from (for example, no alerts give no time to alert)."), "",
        ("How to read the tables: a frame counts as yes when the student's probability is at "
         "or over the skill's threshold. An event is one labelled true range; it is caught "
         f"within {window:g} s when its first alert comes at most {window:g} s after it starts. "
         "False alarms are alerts on frames labelled false. Dollars per hour add the student's "
         "GPU (nothing on the Mac), the Muse Spark checks and any per-frame scorer, at the "
         "prices in the plan."), ""]
    for exp, title in EXPERIMENTS.items():
        runs = [e for e in results["evaluations"] if e["experiment"] == exp]
        scored = [e for e in runs if e["status"] == "scored"]
        pending = [e for e in runs if e["status"] == "pending"]
        lines += [f"## {title}", ""]
        if not runs:
            lines += ["No runs yet: the plan has no evaluations for this experiment.", ""]
            continue
        lines += [f"![{title}]({name})\n" for name in results["charts"].get(str(exp), [])]
        if scored:
            lines += [*table(header(exp, window), [accuracy_row(e) for e in scored]), "",
                      "Speed and cost:", "", *table(COST_HEADER, [cost_row(e) for e in scored]),
                      ""]
        else:
            lines += ["No runs scored yet.", ""]
        if pending:
            lines += ["Pending, waiting for a frame log or labels:", "",
                      *(f"- {e['name']}: " + ", ".join(f"`{m}`" for m in e["missing"])
                        for e in pending), ""]
    timelines = results["charts"].get("timelines", [])
    if timelines:
        lines += ["## Live timelines", "", *(f"![Live timeline]({name})\n" for name in timelines)]
    if results.get("wandb"):
        lines += [f"W&B: {results['wandb']}", ""]
    return "\n".join(lines).rstrip() + "\n"
