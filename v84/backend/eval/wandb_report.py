"""The W&B report "Trained Eye: experiments" (`te wandb-report`).

One section per experiment of writeup.EXPERIMENTS, each with:
- the text of that experiment's section of eval/results.md (te eval's tables, so every
  number comes from the eval scripts; the local SVG links are left out),
- the training runs' loss panels, the runs filtered by the tag te train gives them
  (exp:exp1, exp:exp2, exp:exp3; Experiment 4 trains nothing of its own),
- the eval run's results table and that experiment's charts, from the run `te eval --wandb`
  logged (its address is in results.json).

The report is built with the reports API of wandb-workspaces (wandb_workspaces.reports.v2),
imported only when the report is saved: `pip install wandb-workspaces`, not yet on the
approved list. It is saved as a draft in the trained-eye project, which only its author
sees; publishing it is a separate step in the W&B UI, after the author's yes. The panel
classes and the filter expressions below follow the wandb-workspaces documentation and
have not been run against the package yet: check them on the first real run.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import re
from pathlib import Path

from backend.eval.report import WANDB_PROJECT
from backend.eval.writeup import EXPERIMENTS

TITLE = "Trained Eye: experiments"
MODULE = "wandb_workspaces.reports.v2"
INSTALL = "pip install wandb-workspaces"
TRAINING_TAGS = {1: "exp:exp1", 2: "exp:exp2", 3: "exp:exp3"}  # te train: exp:<run prefix>
NO_TRAINING = ("Experiment 4 trains no run of its own: its fine-tuned arm is exp1-mixed "
               "(backend/eval/plan.py), whose loss is in Experiment 1.")
LOSS_KEYS = ("train/loss", "eval/loss")  # what Axolotl's trainer logs to W&B
TABLE_KEY = "eval/results"  # report.log_to_wandb's results table
RUN_URL = re.compile(r"https://[\w.-]+/(?P<entity>[\w.-]+)/(?P<project>[\w.-]+)/runs/"
                     r"(?P<run>[\w-]+)")
IMAGE_LINE = re.compile(r"^!\[[^\]]*\]\([^)]*\)\s*$")


def tag_filter(tag: str) -> str:
    """The runs carrying `tag`, as a run set's filter expression."""
    return f"Metric('Tags') in ['{tag}']"


def run_filter(run_id: str) -> str:
    """One run by its id, as a run set's filter expression."""
    return f"Metric('ID') = '{run_id}'"


def sections(markdown: str) -> dict[str, str]:
    """{"## " heading: its text} of results.md, and "" for the text before the first one.
    The title line and the image lines (local SVG files) are left out."""
    out, heading, lines = {}, "", []
    for line in markdown.splitlines() + ["## "]:
        if line.startswith("## "):
            out[heading] = "\n".join(lines).strip()
            heading, lines = line[3:].strip(), []
        elif not (line.startswith("# ") or IMAGE_LINE.match(line)):
            lines.append(line)
    return out


def eval_run(results: dict) -> dict[str, str]:
    """{"entity", "project", "run", "url"} of the eval run results.json names."""
    url = str(results.get("wandb") or "")
    found = RUN_URL.match(url)
    if not found:
        raise ValueError("results.json names no W&B eval run: run te eval --wandb (or make "
                         "eval with it) first, so the report can show its table and charts")
    if found["project"] != WANDB_PROJECT:
        raise ValueError(f"the eval run is in project {found['project']}, not {WANDB_PROJECT}")
    return {**found.groupdict(), "url": url}


def plan(results: dict, results_md: str, entity: str | None = None) -> dict:
    """The report as plain data: what build_report turns into wandb-workspaces objects."""
    run = eval_run(results)
    text = sections(results_md)
    digest = hashlib.sha256(results_md.encode("utf-8")).hexdigest()
    parts = []
    for exp, title in EXPERIMENTS.items():
        charts = [f"charts/{Path(n).stem}" for n in results.get("charts", {}).get(str(exp), [])]
        parts.append({"experiment": exp, "title": title,
                      "text": text.get(title) or "results.md has no section for this yet.",
                      "training_tag": TRAINING_TAGS.get(exp), "charts": charts})
    return {"title": TITLE, "entity": entity or run["entity"], "project": WANDB_PROJECT,
            "eval_run": run, "intro": text.get("", ""), "sections": parts,
            "description": (f"Built by te wandb-report from eval/results.md (SHA-256 {digest}) "
                            f"and the eval run {run['url']}.")}


def build_report(spec: dict, wr):
    """The wandb-workspaces Report for a plan (`wr`: the reports.v2 module)."""
    entity, project = spec["entity"], spec["project"]
    blocks = [wr.MarkdownBlock(text=spec["intro"])] if spec["intro"] else []
    for part in spec["sections"]:
        blocks += [wr.H1(text=part["title"]), wr.MarkdownBlock(text=part["text"])]
        if part["training_tag"]:
            blocks.append(wr.PanelGrid(
                runsets=[wr.Runset(entity=entity, project=project,
                                   name=f"Training runs ({part['training_tag']})",
                                   filters=tag_filter(part["training_tag"]))],
                panels=[wr.LinePlot(title=key, y=[key]) for key in LOSS_KEYS]))
        else:
            blocks.append(wr.P(text=NO_TRAINING))
        panels = [wr.WeavePanelSummaryTable(table_name=TABLE_KEY)]
        if part["charts"]:
            panels.append(wr.MediaBrowser(media_keys=part["charts"]))
        blocks.append(wr.PanelGrid(
            runsets=[wr.Runset(entity=entity, project=project, name="Evaluation run",
                               filters=run_filter(spec["eval_run"]["run"]))],
            panels=panels))
    return wr.Report(entity=entity, project=project, title=spec["title"],
                     description=spec["description"], blocks=blocks)


def _reports():
    try:
        return importlib.import_module(MODULE)
    except ImportError:
        raise RuntimeError(f"the W&B report needs wandb-workspaces, which is not installed and "
                           f"not on the approved downloads list: ask the author, then "
                           f"{INSTALL}") from None


def save_report(spec: dict, *, confirmed: bool = False) -> str:
    """Save the report as a draft (only its author sees it) and return its address. The
    key comes from the environment or .env and is never shown."""
    if not confirmed:
        raise PermissionError("saving the report needs the author's yes: nothing was saved")
    wr = _reports()
    from backend.config import load_settings

    settings = load_settings()
    if settings.has_secret("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_API_KEY", settings.secret("WANDB_API_KEY"))
    key = os.environ.get("WANDB_API_KEY", "")
    report = build_report(spec, wr)
    try:
        report.save(draft=True)
    except Exception as err:  # noqa: BLE001  (its text must not carry the key)
        text = str(err).replace(key, "[redacted]") if key else str(err)
        raise RuntimeError(f"saving the report failed ({type(err).__name__}): "
                           f"{text[:300]}") from None
    return str(report.url)
