"""te eval --readme <file>: the results block of a README, from results.json only.

The block sits between two marker comments, START and END below; te eval replaces what is
between them (or adds both at the end of the file the first time) and leaves the rest of
the file alone. It holds, all copied from te eval's results.json: a headline table per
experiment with scored evaluations (Experiment 4 pooled over the skills, as its cost
chart), links to the charts, the regression check's line and Spark's error rate per
skill, and the SHA-256 of that results.json as the source, so every number can be traced
to the eval scripts. It refuses when nothing is scored, so a README never shows an empty
or made-up result. The failure galleries are never linked: they are local only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from backend.eval.writeup import EXPERIMENTS, judge_label, on_time_text

START = "<!-- te eval: results start (written by te eval --readme; edits here are lost) -->"
END = "<!-- te eval: results end -->"
CHART_NAMES = {"1": "Experiment 1", "2": "Experiment 2", "3": "Experiment 3",
               "4": "Experiment 4 (cost)"}


def _num(value: float | None, spec: str = ".2f") -> str:
    return "n/a" if value is None else f"{value:{spec}}"


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.2f}"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [_row(header), "|" + "---|" * len(header), *(_row(r) for r in rows)]


def headline(exp: int, results: dict) -> list[str]:
    """One experiment's table: frame F1, events caught on time, false alarms per
    hour and dollars per hour (Experiment 3 adds the images, 4 is pooled over skills)."""
    caught = "Events caught on time"
    if exp == 4 and any(j.get("metrics") for j in results.get("experiment4") or []):
        rows = [[judge_label(j), _num(j["metrics"]["f1"]), _num(j["metrics"]["accuracy"]),
                 _num(j["metrics"]["event_recall_in_window"]),
                 _num(j["metrics"]["false_alarms_per_hour"], ".1f"), _usd(j["usd_per_hour"])]
                for j in results["experiment4"] if j.get("metrics")]
        return _table(["Judge (all skills)", "Frame F1", "Frame accuracy", caught,
                       "False alarms per hour", "Dollars per hour"], rows)
    scored = [e for e in results["evaluations"]
              if e["experiment"] == exp and e["status"] == "scored"]
    extra = ["Synthetic images"] if exp == 3 else []
    rows = [[e["variant"], e["skill"], *([str(e.get("images"))] if exp == 3 else []),
             _num(e["metrics"]["f1"]), _num(e["metrics"]["event_recall_in_window"]),
             _num(e["metrics"]["false_alarms_per_hour"], ".1f"),
             _usd(e["cost_per_hour"]["total_usd"])] for e in scored]
    return _table(["Variant", "Skill", *extra, "Frame F1", caught, "False alarms per hour",
                   "Dollars per hour"], rows) if rows else []


def regression_line(results: dict) -> str | None:
    if "regression" not in results:
        return None
    reg = results["regression"]
    if not reg:
        return "Regression check: not run yet (te regress)."

    def side(part: dict) -> str:
        return ("n/a" if not part.get("n") else
                f"{part['correct']} of {part['n']} ({part['accuracy']:.0%})")
    changed = len(reg.get("changed") or [])
    return (f"Regression check on general questions at {reg.get('precision')}: base "
            f"{side(reg.get('base') or {})}, fine-tuned {side(reg.get('finetuned') or {})}; "
            f"{changed} question{'s' if changed != 1 else ''} changed.")


def spark_lines(results: dict) -> list[str]:
    out = []
    for skill, err in sorted((results.get("spark_errors") or {}).items()):
        spark = (err or {}).get("spark") or {}
        if not spark.get("n"):
            out.append(f"- {skill}: not measured yet (te spark-errors)")
            continue
        low, high = spark["ci95"]
        out.append(f"- {skill}: wrong on {spark['errors']} of {spark['n']} hand-labelled "
                   f"images ({spark['rate']:.1%}, 95% CI {low:.1%} to {high:.1%}), unsure on "
                   f"{spark.get('unsure', 0)}")
    return out


def block(results_path: Path, readme_path: Path) -> str:
    """The text between the markers. ValueError when no evaluation is scored."""
    body = Path(results_path).read_bytes()
    results = json.loads(body)
    evaluations = results.get("evaluations") or []
    scored = [e for e in evaluations if e.get("status") == "scored"]
    if not scored:
        raise ValueError(f"{Path(results_path).name} has no scored evaluation yet, so there "
                         "is nothing to put in the README")
    digest = hashlib.sha256(body).hexdigest()
    source = os.path.relpath(results_path, Path(readme_path).parent).replace(os.sep, "/")
    lines = ["## Results", "",
             (f"Written by `te eval --readme` from `{source}` (SHA-256 `{digest}`): "
              f"{len(scored)} of {len(evaluations)} evaluations scored. Every number here is "
              "copied from that file; n/a means there was nothing to compute it from."), "",
             *([(f"Test footage: {results['footage']}. Results on other footage are reported "
                "apart."), ""] if results.get("footage") else []),
             *([("Thresholds were calibrated for frame F1 on calibration clips (this footage has "
                "too few events past the skill's hold to judge alerts), so read the frame "
                "columns."), ""] if results.get("threshold_objective") == "frames" else []),
             ("An event is caught on time when its alert comes within the skill's window after "
              "the skill's hold ends (the condition must stay true that long before it alerts): "
              f"{on_time_text(results)}"), ""]
    for exp, title in EXPERIMENTS.items():
        table = headline(exp, results)
        if table:
            lines += [f"### {title}", "", *table, ""]
    folder = Path(results_path).parent
    links = []
    for key, name in CHART_NAMES.items():
        for chart in results.get("charts", {}).get(key, []):
            rel = os.path.relpath(folder / chart, Path(readme_path).parent).replace(os.sep, "/")
            label = name if len(results["charts"][key]) == 1 else f"{name}: {Path(chart).stem}"
            links.append(f"[{label}]({rel})")
    if links:
        lines += ["Charts: " + ", ".join(links) + ".", ""]
    if (line := regression_line(results)) is not None:
        lines += [line, ""]
    if spark := spark_lines(results):
        lines += ["Muse Spark's label error rate against hand labels (Experiment 2):", "",
                  *spark, ""]
    return "\n".join(lines).rstrip() + "\n"


def write_readme(readme_path: Path, results_path: Path) -> Path:
    """Put the block between the markers in readme_path (adding them at its end the first
    time). ValueError, leaving the file as it was, when nothing is scored or the markers
    are broken (one without the other, twice, or in the wrong order)."""
    readme_path = Path(readme_path)
    text = readme_path.read_text(encoding="utf-8")
    new = block(results_path, readme_path)
    starts, ends = text.count(START), text.count(END)
    if (starts, ends) == (0, 0):
        text = text.rstrip("\n") + ("\n\n" if text.strip() else "") + f"{START}\n{new}{END}\n"
    elif (starts, ends) != (1, 1) or text.index(START) > text.index(END):
        raise ValueError(f"{readme_path.name}: the results markers are broken; it needs "
                         f"exactly one {START} before one {END}")
    else:
        head, rest = text.split(START, 1)
        text = f"{head}{START}\n{new}{END}{rest.split(END, 1)[1]}"
    readme_path.write_text(text, encoding="utf-8")
    return readme_path
