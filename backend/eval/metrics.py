"""Evaluation metrics for watch mode, from `te watch` frame logs and human labels.

A frame log has one JSON line per scored frame and skill (backend/watch/watcher.py):
t, skill, threshold, hold_s, p, over, fires, alert, cloud, latency_s, error. A labels
file says when the condition was true in the clip: {"true": [[start_s, end_s], ...]}.

Each skill has a hold (hold_s in its spec, logged with every row; 0 for a log without it):
watch mode alerts only once the condition has stayed true that long. So an alert is due
at the end of the hold, the start of a labelled range plus hold_s, and each skill has
its own on-time window after that (ON_TIME_S; EVENT_WINDOW_S for any other skill).

Per skill, over one or more clips:
- frame precision, recall, F1 and accuracy: a frame says "yes" when it scored at or over
  the threshold the watcher used (the log's `over`), and it is true when its time is
  inside a labelled range (ends included);
- event recall: an event is a run of the clip's frames inside the labelled ranges (one
  frame outside bridged, as watch mode's 3-of-4 smoothing bridges it), timed by its first
  and last frame, that lasts at least the hold (shorter runs are short_events: nothing
  should fire for them). Its alert is due at its first frame plus the hold. It is caught
  on time ("in window") when its first alert from then comes within the skill's on-time
  window, and caught at all when one fires before its last frame. trigger.evaluate, which
  te calibrate uses, counts events the same way;
- false alarms per hour: alerts that no event explains, per hour of clip: on frames
  labelled false, or before an event's hold ended (early_alerts counts those inside a
  range).
  A clip lasts from its first frame to one frame step after its last: 3600 frames at
  1 fps, one hour;
- time to alert: seconds from the end of an event's hold to its first alert after that,
  over caught events;
- latency per frame: latency_s, median and 90th percentile (not rows latency_measured false);
- scored frames and student errors (rows without a probability).

Dollars per hour add three parts: the student's GPU price per hour (None for the Mac,
which has no hourly price), the Muse Spark checks per hour times the price of one check
(rows with a cloud answer; failed checks are not counted), and a per-frame scorer such as
Spark judging every frame (price per frame times frames per hour).
"""

from __future__ import annotations

import itertools
import json
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from backend.watch.trigger import BRIDGE, EPS, due_spans, explained, runs

EVENT_WINDOW_S = 5.0  # the on-time window for a skill not in ON_TIME_S
# How long after the end of its hold an alert still counts as on time, per skill. The
# burner and the tap are seen at once and matter soonest; a fridge door's view changes as
# the wearer moves, and an egg's foam builds up gradually, so their labels start softer.
ON_TIME_S = {"burner_left_on": 5.0, "tap_left_running": 5.0, "fridge_left_open": 10.0,
             "egg_whisked": 10.0}


def on_time_window(skill: str, overrides: dict | None = None) -> float:
    """The skill's on-time window in seconds: the plan's override, else ON_TIME_S."""
    if overrides and skill in overrides:
        return float(overrides[skill])
    return ON_TIME_S.get(skill, EVENT_WINDOW_S)


def hold_of(rows: Iterable[dict], skill: str) -> float:
    """The hold the watcher used for `skill`, from its frame log rows (0 if none is
    logged). ValueError if the rows disagree: one evaluation is one hold."""
    holds = {float(r["hold_s"]) for r in rows
             if r.get("skill") == skill and r.get("hold_s") is not None}
    if len(holds) > 1:
        raise ValueError(f"the frame log has more than one hold_s for {skill}: "
                         f"{sorted(holds)}")
    return holds.pop() if holds else 0.0


def read_log(path: Path) -> list[dict]:
    """The rows of a frame log (JSON lines; blank lines are skipped)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def read_labels(path: Path) -> list[tuple[float, float]]:
    """The labelled true ranges of a clip from {"true": [[start_s, end_s], ...]}."""
    return check_ranges(json.loads(Path(path).read_text(encoding="utf-8"))["true"])


def check_ranges(ranges: Iterable[Sequence[float]]) -> list[tuple[float, float]]:
    """Sorted ranges, with overlapping or touching ones merged into one event (as the
    labelling tool saves them), so a hand-edited file cannot split an event."""
    out: list[tuple[float, float]] = []
    for a, b in sorted((float(a), float(b)) for a, b in ranges):
        if a > b:
            raise ValueError("each true range must be [start, end] with start <= end")
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def duration_s(times: Iterable[float]) -> float:
    """From the first frame to one frame step (the median gap) after the last."""
    ts = sorted(set(times))
    if len(ts) < 2:
        return 0.0
    return ts[-1] - ts[0] + statistics.median(b - a for a, b in itertools.pairwise(ts))


@dataclass
class ClipCounts:
    """Raw counts for one skill over some clips; add them up with +."""

    clips: int = 0
    frames: int = 0  # scored frames
    errors: int = 0  # frames the student failed on
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    seconds: float = 0.0
    events: int = 0
    short_events: int = 0  # labelled ranges shorter than the hold
    caught_in_window: int = 0
    caught: int = 0
    alerts: int = 0
    false_alarms: int = 0
    early_alerts: int = 0  # false alarms inside a range, before its hold ended
    cloud_checks: int = 0
    delays: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)

    def __add__(self, other: ClipCounts) -> ClipCounts:
        return ClipCounts(**{f.name: getattr(self, f.name) + getattr(other, f.name)
                             for f in fields(self)})


def count_clip(rows: Iterable[dict], skill: str, true_ranges: Iterable[Sequence[float]],
               window_s: float | None = None, hold_s: float | None = None) -> ClipCounts:
    """Counts for one skill in one clip, with the skill's on-time window (on_time_window
    unless given) and hold (the log's, hold_of, unless given). Raises ValueError if the
    log has no rows for it."""
    ranges = check_ranges(true_ranges)
    rows = sorted((r for r in rows if r.get("skill") == skill), key=lambda r: float(r["t"]))
    if not rows:
        raise ValueError(f"the frame log has no rows for {skill}")
    window_s = on_time_window(skill) if window_s is None else float(window_s)
    hold_s = hold_of(rows, skill) if hold_s is None else float(hold_s)
    if window_s <= 0 or hold_s < 0:
        raise ValueError("the on-time window must be over 0 and the hold 0 or more")
    c = ClipCounts(clips=1, seconds=duration_s(float(r["t"]) for r in rows))
    truths = [(float(r["t"]), any(a <= float(r["t"]) <= b for a, b in ranges)) for r in rows]
    # Events as watch mode sees them (trigger.evaluate counts the same way): runs of the
    # frames inside the labelled ranges, a single frame outside one bridged, timed by
    # their first and last frame; a run that lasts the hold is due at its end.
    spans = runs(truths, BRIDGE)
    due = due_spans(spans, hold_s)
    c.short_events = len(spans) - len(due)
    alert_times = []
    for r, (t, truth) in zip(rows, truths, strict=True):
        if r.get("p") is None:
            c.errors += 1
            continue
        c.frames += 1
        over = bool(r["over"]) if "over" in r else float(r["p"]) >= float(r["threshold"])
        c.tp += over and truth
        c.fp += over and not truth
        c.fn += truth and not over
        c.tn += not (truth or over)
        if r.get("latency_s") is not None and r.get("latency_measured", True):
            c.latencies.append(float(r["latency_s"]))
        if r.get("cloud") is not None and not str(r["cloud"]).startswith("error"):
            c.cloud_checks += 1
        if r.get("alert"):
            c.alerts += 1
            ok = explained(t, due)
            c.false_alarms += not ok
            c.early_alerts += truth and not ok
            alert_times.append(t)
    for when, end in due:
        c.events += 1
        hits = [t for t in alert_times if when - EPS <= t <= end]
        if hits:
            delay = max(0.0, hits[0] - when)
            c.caught += 1
            c.delays.append(delay)
            c.caught_in_window += delay <= window_s + EPS
    return c


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def summarize(counts: Iterable[ClipCounts]) -> dict:
    """The metrics over all the given clips, as a plain dict (None where undefined)."""
    c = sum(counts, ClipCounts())
    hours = c.seconds / 3600
    lat = c.latencies
    return {
        "clips": c.clips, "frames": c.frames, "student_errors": c.errors,
        "hours": round(hours, 4),
        "precision": _ratio(c.tp, c.tp + c.fp), "recall": _ratio(c.tp, c.tp + c.fn),
        "f1": _ratio(2 * c.tp, 2 * c.tp + c.fp + c.fn),
        "accuracy": _ratio(c.tp + c.tn, c.frames),
        "events": c.events, "short_events": c.short_events,
        "caught_in_window": c.caught_in_window, "caught": c.caught,
        "event_recall_in_window": _ratio(c.caught_in_window, c.events),
        "event_recall_any": _ratio(c.caught, c.events),
        "alerts": c.alerts, "false_alarms": c.false_alarms, "early_alerts": c.early_alerts,
        "false_alarms_per_hour": _ratio(c.false_alarms, hours),
        "time_to_alert_median_s": _round(statistics.median(c.delays)) if c.delays else None,
        "time_to_alert_mean_s": _round(statistics.mean(c.delays)) if c.delays else None,
        "latency_median_s": _round(statistics.median(lat)) if lat else None,
        "latency_p90_s": _round(np.percentile(lat, 90)) if lat else None,
        "cloud_checks": c.cloud_checks,
        "cloud_checks_per_hour": _ratio(c.cloud_checks, hours),
        "frames_per_hour": _ratio(c.frames, hours),
    }


def cost_per_hour(summary: dict, *, gpu_usd_per_hour: float | None = None,
                  spark_usd_per_check: float = 0.0,
                  scorer_usd_per_frame: float = 0.0) -> dict:
    """Dollars per hour of watching, in parts. None when the clips have no duration."""
    if min(gpu_usd_per_hour or 0.0, spark_usd_per_check, scorer_usd_per_frame) < 0:
        raise ValueError("prices must be 0 or more")
    checks, frames = summary["cloud_checks_per_hour"], summary["frames_per_hour"]
    gpu = float(gpu_usd_per_hour or 0.0)
    spark = None if checks is None else checks * spark_usd_per_check
    scorer = None if frames is None else frames * scorer_usd_per_frame
    total = None if spark is None or scorer is None else gpu + spark + scorer
    return {"student": "Mac" if gpu_usd_per_hour is None else "GPU",
            "gpu_usd": round(gpu, 4), "spark_checks_usd": _round(spark),
            "per_frame_scorer_usd": _round(scorer), "total_usd": _round(total)}
