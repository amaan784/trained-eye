"""What run_eval writes from the scored evaluations: the SVG charts and results.md.

results.md has one section per experiment, in plain words: its charts, a table of
accuracy and one of speed and cost. Every number comes from report.run_eval's computation
(results.json holds the same ones); n/a means there was nothing to compute it from. An
experiment with no evaluations in the plan says so, and evaluations still waiting for
their clips are listed as pending.

Experiment 2 also gets Muse Spark's error rate against the author's hand labels and the
filter's label agreement, Experiment 4 its judges pooled over the skills (the cost
chart's three points: frame F1 is the accuracy measure plotted, as frame accuracy is
mostly the many frames where nothing happens), and after the experiments come the
regression check and the links to Experiment 1's failure galleries (local only), each
when run_eval had them.
"""

from __future__ import annotations

from pathlib import Path

from backend.eval import metrics
from backend.eval.charts import grouped_bars, line_chart, scatter
from backend.eval.regression import markdown as regression_markdown
from backend.eval.svg import usd
from backend.eval.timeline import timeline

EXPERIMENTS = {
    1: "Experiment 1: real frames only, generated images only, or both",
    2: "Experiment 2: labels filtered by Muse Spark, or not",
    3: "Experiment 3: how many generated images",
    4: "Experiment 4: fine-tuned Glimmer, base Glimmer, Muse Spark on every frame",
}
COST_HEADER = ["Run", "Latency median", "Latency p90", "Student errors", "Alerts",
               "Spark checks per hour", "Dollars per hour", "Prices used"]
SPARK_HEADER = ["Skill", "Hand-labelled", "Spark wrong", "Spark unsure",
                "Generated-for label wrong (unfiltered arm)",
                "Kept by the filter but wrong (filtered arm)", "Spark agreed with the label"]
EXP4_SUBTITLE = "Accuracy as frame F1, pooled over the skills, against dollars per hour"
SPARK_NOTE = ("Muse Spark's labels against your hand labels (te spark-errors, 95% Wilson "
              "intervals), and how often Spark agreed with the label each generated image was "
              "made for (te teach's label_stats.json; the filter keeps those):")
POOLED_NOTE = ("Pooled over the skills (the cost chart's points; dollars per hour weighted by "
               "hours of footage):")
GALLERY_NOTE = ("The worst false alarms and misses of each Experiment 1 run, as frames of "
                "the recorded test clips. They may show faces, so they stay on this Mac: "
                "run te blur on any frame before publishing it.")


def _num(value: float | None, spec: str = ".2f", unit: str = "") -> str:
    return "n/a" if value is None else f"{value:{spec}}{unit}"


def _events(caught: int, events: int) -> str:
    return f"{caught} of {events}" if events else "no events"


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(_cell(c) for c in row) + " |" for row in rows)]


ON_TIME = "Events caught on time"


def header(experiment: int) -> list[str]:
    extra = {3: ["Generated images", "Generation cost"], 4: ["Frame accuracy"]}
    return ["Run", "Variant", "Skill", "Clips", "Footage", "Frames", "Precision", "Recall",
            "F1", ON_TIME, "Events caught at all",
            "False alarms per hour", "Time to alert after the hold, median (mean)", "Threshold (from)",
            *extra.get(experiment, [])]


def threshold_cell(e: dict) -> str:
    """The threshold the logs were scored at and where it came from (te watch's
    threshold_source: run is the run's own calibration), e.g. "0.62 (run)"."""
    used = e.get("threshold") or {}
    values = used.get("values") or []
    if not values:
        return "n/a"
    shown = f"{values[0]:.2f}" if len(values) == 1 else f"{values[0]:.2f} to {values[-1]:.2f}"
    return f"{shown} ({', '.join(used['sources'])})" if used.get("sources") else shown


def accuracy_row(e: dict) -> list[str]:
    m = e["metrics"]
    delay = ("n/a" if m["time_to_alert_median_s"] is None else
             f"{m['time_to_alert_median_s']:.1f} s ({m['time_to_alert_mean_s']:.1f} s)")
    row = [e["name"], e["variant"], e["skill"], str(m["clips"]), f"{m['hours'] * 60:.1f} min",
           str(m["frames"]), _num(m["precision"]), _num(m["recall"]), _num(m["f1"]),
           _events(m["caught_in_window"], m["events"]), _events(m["caught"], m["events"]),
           _num(m["false_alarms_per_hour"], ".1f"), delay, threshold_cell(e)]
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


def judge_label(judge: dict) -> str:
    """An Experiment 4 judge's name, with how many of its skills are in while some wait."""
    done, total = len(judge["skills"]), judge["of"]
    return judge["variant"] if done in (0, total) else f"{judge['variant']} ({done} of " \
        f"{total} skills)"


def pooled_header() -> list[str]:
    return ["Judge", "Skills", "Frames", "F1", "Frame accuracy", ON_TIME,
            "False alarms per hour", "Dollars per hour"]


def pooled_rows(judges: list[dict]) -> list[list[str]]:
    rows = []
    for j in judges:
        m = j["metrics"]
        if not m:
            continue
        rows.append([j["variant"], ", ".join(j["skills"]), str(m["frames"]), _num(m["f1"]),
                     _num(m["accuracy"]), _events(m["caught_in_window"], m["events"]),
                     _num(m["false_alarms_per_hour"], ".1f"),
                     "n/a" if j["usd_per_hour"] is None else usd(j["usd_per_hour"])])
    return rows


def _rate(part: dict | None) -> str:
    if not part or not part.get("n"):
        return "none judged"
    low, high = part["ci95"]
    return (f"{part['errors']} of {part['n']} ({part['rate']:.1%}, 95% CI {low:.1%} to "
            f"{high:.1%})")


def spark_rows(results: dict) -> list[list[str]]:
    """Experiment 2's table of Spark's error rate and the label agreement, per skill."""
    errors, stats = results.get("spark_errors") or {}, results.get("label_stats") or {}
    rows = []
    for skill in sorted(set(errors) | set(stats)):
        err, st = errors.get(skill), stats.get(skill) or {}
        agreement = ("n/a" if st.get("agreement") is None else
                     f"{st['kept']} of {st['total']} kept ({st['agreement']:.1%})")
        if err is None:
            rows.append([skill, "not measured yet (te label --skill, then te spark-errors)",
                         "n/a", "n/a", "n/a", "n/a", agreement])
            continue
        spark = err.get("spark") or {}
        rows.append([skill, str(err.get("hand_labelled", "n/a")), _rate(spark),
                     str(spark.get("unsure", "n/a")), _rate(err.get("intended")),
                     _rate(err.get("kept")), agreement])
    return rows


def write_charts(results: dict, base: Path, out_dir: Path) -> dict[str, list[str]]:
    """Write the SVG charts into out_dir. Returns their file names by experiment ("1" to
    "4") and "timelines". Clip paths are taken from `base`, the plan's folder."""
    charts: dict[str, list[str]] = {}

    def save(key: str, name: str, svg: str) -> None:
        (out_dir / name).write_text(svg, encoding="utf-8")
        charts.setdefault(key, []).append(name)

    evals = results["evaluations"]
    for exp in (1, 2):
        runs = [e for e in evals if e["experiment"] == exp]
        if any(e["status"] == "scored" for e in runs):
            several = len({e["skill"] for e in runs}) > 1
            groups = [f"{e['variant']} ({e['skill']})" if several else e["variant"] for e in runs]
            recall = ON_TIME[0].lower() + ON_TIME[1:]
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
            points, "generated images per skill"))
    judges = results.get("experiment4") or []
    if any(j["metrics"] for j in judges):
        points = [{"label": judge_label(j), "x": j["usd_per_hour"],
                   "y": (j["metrics"] or {}).get("f1"), "highlight": j["highlight"]}
                  for j in judges]
        save("4", "exp4_cost.svg", scatter(EXPERIMENTS[4], EXP4_SUBTITLE, points))
    for e in evals:
        for i, c in enumerate(e["clips"], 1):
            if c.get("timeline") and e["status"] == "scored":
                rows = [r for r in metrics.read_log(base / c["log"])
                        if r.get("skill") == e["skill"]]
                save("timelines", f"timeline_{e['name']}_{i}.svg", timeline(
                    f"Live timeline: {e['skill']}, {e['variant']}",
                    f"{Path(c['log']).name}, run {e['name']}", rows,
                    metrics.read_labels(base / c["labels"]), e["alert_text"]))
    return charts


def extra_lines(exp: int, results: dict) -> list[str]:
    """What an experiment's section adds after its runs: Spark's error rate (2), the
    pooled judges (4)."""
    if exp == 2 and (rows := spark_rows(results)):
        return [SPARK_NOTE, "", *table(SPARK_HEADER, rows), ""]
    if exp == 4:
        rows = pooled_rows(results.get("experiment4") or [])
        return ([POOLED_NOTE, "", *table(pooled_header(), rows), ""] if rows else []) + \
            speed_lines(results.get("speed") or {})
    return []


SPEED_HEADER = ["Model", "Completion tokens a frame (median)", "Frames with reasoning text",
                "Seconds a frame, one at a time (median)"]


def speed_lines(speed: dict) -> list[str]:
    """Why one model answers faster: the tokens each writes per frame (eval/speed.json)."""
    if not speed:
        return []
    first = next(iter(speed.values()))
    clip = Path(first["clip"]).name
    rows = [[name, _num(m["median_completion_tokens"], "g"),
             f"{m['frames_with_reasoning']} of {m['frames']}",
             _num(m["median_seconds"], ".2f", " s")] for name, m in speed.items()]
    return [("Why the speeds differ (`python -m backend.eval.speed`, the student's own request "
             f"sent one frame at a time, on frames of {clip}): every token a model writes "
             "before its yes or no costs time on the GPU."), "",
            *table(SPEED_HEADER, rows), ""]


def _seconds(values: list[float]) -> str:
    return " or ".join(f"{v:g}" for v in values) + " s"


def on_time_text(results: dict) -> str:
    """How each plan skill's events are timed: its hold (from the logs) and window."""
    parts = []
    for skill, t in sorted((results.get("on_time") or {}).items()):
        hold = (f"a {_seconds(t['hold_s'])} hold" if t["hold_s"] else
                "the hold its logs record")
        parts.append(f"{skill}, {hold} and a {t['window_s']:g} s window")
    return "; ".join(parts) + "." if parts else "no skills in the plan yet."


def render(results: dict) -> str:
    """The whole results.md for the dict run_eval returns."""
    footage = results.get("footage")
    lines = [
        f"# Evaluation results: {footage}" if footage else "# Evaluation results", "",
        *([f"Test footage: {footage}. Results on other footage are reported apart.", ""]
          if footage else []),
        *([("Thresholds here were calibrated for frame F1 on the calibration clips, not for "
           "alerts: this footage has too few events that last past the skill's hold to judge "
           "alerts on, so read the frame columns."), ""]
          if results.get("threshold_objective") == "frames" else []),
        (f"Computed by `te eval` from the plan {results['plan']}: the frame logs of `te watch` "
         "scored against the human labels of each test clip. Every number below comes from "
         "that computation, and results.json holds all of them. n/a means there was nothing "
         "to compute it from (for example, no alerts give no time to alert)."), "",
        ("How to read the tables: a frame counts as yes when the student's probability is at "
         "or over the skill's threshold, shown with where it came from (run: the run's own "
         "calibration on its validation clips; skill: the skill-wide one; flag: given by "
         "hand; default: 0.5, not calibrated; spark: only a yes). A skill alerts only once "
         "its condition has stayed true for the skill's hold, so an event is one labelled "
         "true range that lasts at least the hold, and its alert is due when the hold ends. "
         "It is caught on time when its first alert after that comes within the skill's "
         f"on-time window: {on_time_text(results)} False alarms are alerts no event "
         "explains: on frames labelled false, or before the hold ended. Dollars per hour add "
         "the student's GPU (nothing on the Mac), the Muse Spark checks and any per-frame "
         "scorer, at the prices in the plan."), ""]
    for exp, title in EXPERIMENTS.items():
        runs = [e for e in results["evaluations"] if e["experiment"] == exp]
        scored = [e for e in runs if e["status"] == "scored"]
        pending = [e for e in runs if e["status"] == "pending"]
        lines += [f"## {title}", ""]
        if not runs:
            lines += ["No runs yet: the plan has no evaluations for this experiment.", "",
                      *extra_lines(exp, results)]
            continue
        lines += [f"![{title}]({name})\n" for name in results["charts"].get(str(exp), [])]
        if scored:
            lines += [*table(header(exp), [accuracy_row(e) for e in scored]), "",
                      "Speed and cost:", "", *table(COST_HEADER, [cost_row(e) for e in scored]),
                      ""]
        else:
            lines += ["No runs scored yet.", ""]
        if pending:
            lines += ["Pending, waiting for a frame log or labels:", "",
                      *(f"- {e['name']}: " + ", ".join(f"`{m}`" for m in e["missing"])
                        for e in pending), ""]
        lines += extra_lines(exp, results)
    timelines = results["charts"].get("timelines", [])
    if timelines:
        lines += ["## Live timelines", "", *(f"![Live timeline]({name})\n" for name in timelines)]
    if "regression" in results:
        reg = results["regression"]
        lines += ["## Regression check", "",
                  *([regression_markdown(reg), "", "From te regress (eval/regression.json)."]
                    if reg else ["Not run yet: te regress writes eval/regression.json."]), ""]
    if results.get("galleries"):
        lines += ["## Failure galleries (local only)", "", GALLERY_NOTE, "",
                  *(f"- [{run}]({g['page']}): {g['false_alarms']} false alarms and "
                    f"{g['misses']} misses shown" for run, g in results["galleries"].items()),
                  ""]
    if results.get("wandb"):
        lines += [f"W&B: {results['wandb']}", ""]
    return "\n".join(lines).rstrip() + "\n"
