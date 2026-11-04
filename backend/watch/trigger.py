"""When a skill fires: smoothing, the hold, cooldown and the threshold.

Watch mode scores about one frame a second per skill. A skill's condition is on while at
least 3 of its last 4 frames score at or over its threshold. It fires once the condition
has been on for the skill's hold (hold_s, counted from the first of those frames over
the threshold), then stays quiet for the skill's cooldown. One bright frame (a
reflection, a hand in the way) never fires on its own, a fridge opened for a moment says
nothing when the fridge's hold is 30 s, and a condition that stays true is announced
once per cooldown, not every second. With a hold of 0 it fires as soon as it is on.

The threshold is set on validation footage: the lowest one that gives at most one false
alarm per hour there (`calibrate`). A lower threshold catches the condition sooner, so
the lowest one that keeps false alarms rare is the right one.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

WINDOW = 4
NEEDED = 3
MAX_GAP_S = 10.0  # frames further apart than this do not count together
MAX_FALSE_ALARMS_PER_HOUR = 1.0
BRIDGE = WINDOW - NEEDED  # frames under the threshold a condition that stays on rides over
EPS = 1e-6  # seconds: frame times and holds compared with this much slack


@dataclass
class Trigger:
    """3-of-4 smoothing, a hold and a cooldown for one skill. Times are in seconds."""

    threshold: float
    cooldown_s: float
    hold_s: float = 0.0
    window: int = WINDOW
    needed: int = NEEDED
    max_gap_s: float = MAX_GAP_S
    recent: deque = field(default_factory=deque, repr=False)  # (t, over the threshold)
    quiet_until: float = -math.inf
    last_t: float | None = None
    since: float | None = None  # when the condition came on (None while it is off)

    def __post_init__(self) -> None:
        if not 0 < self.needed <= self.window:
            raise ValueError("need 0 < needed <= window")
        if not self.hold_s >= 0:
            raise ValueError("hold_s must be 0 or more")
        self.recent = deque(self.recent, maxlen=self.window)

    def update(self, p: float, t: float) -> bool:
        """Add the frame scored `p` at time `t`; True if the skill fires now."""
        if self.last_t is not None and t - self.last_t > self.max_gap_s:
            self.clear()  # the stream paused: older frames say nothing about now
        self.last_t = t
        self.recent.append((t, p >= self.threshold))
        if sum(over for _, over in self.recent) < self.needed:
            self.since = None  # off again: the hold starts afresh next time
            return False
        if self.since is None:  # on from the first frame over the threshold in the window
            self.since = next(ft for ft, over in self.recent if over)
        if not self.recent[-1][1]:  # this frame is under: the condition may have just ended
            return False  # (the hold carries on through it, but nothing fires on it)
        if t < self.quiet_until or t - self.since < self.hold_s - EPS:
            return False
        self.quiet_until = t + self.cooldown_s
        self.clear()
        return True

    def clear(self) -> None:
        """Forget the recent frames: the 3-of-4 count and the hold start again."""
        self.recent.clear()
        self.since = None

    def snooze(self, t: float, seconds: float) -> None:
        """Stay quiet for `seconds` from `t` (after the cloud check said no)."""
        self.quiet_until = max(self.quiet_until, t + seconds)


@dataclass
class Frame:
    t: float  # seconds from the start of the clip
    p: float  # the student's probability that the condition is true
    truth: bool  # the human label for this frame


def alerts(frames: Sequence[Frame], threshold: float, cooldown_s: float,
           hold_s: float = 0.0) -> list[Frame]:
    """The frames at which a Trigger with these settings fires."""
    trigger = Trigger(threshold, cooldown_s, hold_s)
    return [f for f in frames if trigger.update(f.p, f.t)]


def runs(points: Iterable[tuple[float, bool]], bridge: int = 0) -> list[tuple[float, float]]:
    """(first, last) times of each run of true points, from (t, truth) pairs in time order.
    Runs apart by at most `bridge` false points are one run, as the 3-of-4 smoothing rides
    over one frame under the threshold (evaluate and the metrics bridge BRIDGE)."""
    out: list[tuple[float, float]] = []
    start = end = None
    gap = 0
    for t, truth in points:
        if truth:
            start = t if start is None else start
            end, gap = t, 0
        elif start is not None:
            gap += 1
            if gap > bridge:
                out.append((start, end))
                start, gap = None, 0
    if start is not None:
        out.append((start, end))
    return out


def events(frames: Sequence[Frame], bridge: int = 0) -> list[tuple[float, float]]:
    """(start, end) times of each run of frames labelled true (see runs)."""
    return runs(((f.t, f.truth) for f in frames), bridge)


def due_spans(spans: Iterable[tuple[float, float]], hold_s: float) -> list[tuple[float, float]]:
    """For each run that lasts at least the hold, (when its alert is due, its end): the
    stretch where an alert is right. A shorter run expects none."""
    return [(a + hold_s, b) for a, b in spans if b >= a + hold_s - EPS]


def explained(t: float, due: Iterable[tuple[float, float]]) -> bool:
    """Whether an alert at `t` falls in one of those stretches (else it is a false alarm)."""
    return any(d - EPS <= t <= b for d, b in due)


def evaluate(clips: Iterable[Sequence[Frame]], threshold: float, cooldown_s: float,
             hold_s: float = 0.0) -> dict:
    """False alarms, caught events and delays for one threshold over labelled clips.

    An event is a run of true frames (one false frame inside it bridged, as the smoothing
    bridges it) that lasts at least the hold; its alert is due when the hold ends (its
    first true frame plus hold_s). An alert from then to the event's last frame catches
    it, and the delay is from when it was due. Any other alert is a false alarm: on a
    frame labelled false, or before the condition had lasted the hold. The evaluation
    (backend/eval/metrics.py) counts the same way."""
    false_alarms = caught = total_events = fired_total = 0
    delays: list[float] = []
    seconds = 0.0
    for frames in clips:
        frames = sorted(frames, key=lambda f: f.t)
        if not frames:
            continue
        seconds += frames[-1].t - frames[0].t
        fired = alerts(frames, threshold, cooldown_s, hold_s)
        fired_total += len(fired)
        due = due_spans(events(frames, BRIDGE), hold_s)  # shorter runs expect nothing
        false_alarms += sum(not explained(f.t, due) for f in fired)
        for when, end in due:
            total_events += 1
            hits = [f.t for f in fired if when - EPS <= f.t <= end]
            if hits:
                caught += 1
                delays.append(max(0.0, hits[0] - when))
    return {"threshold": threshold, "alerts": fired_total, "false_alarms": false_alarms,
            "hours": seconds / 3600,
            "events": total_events, "caught": caught,
            "median_delay_s": sorted(delays)[len(delays) // 2] if delays else None}


def calibrate(clips: Sequence[Sequence[Frame]], cooldown_s: float,
              max_per_hour: float = MAX_FALSE_ALARMS_PER_HOUR, hold_s: float = 0.0) -> dict:
    """The lowest threshold with at most `max_per_hour` false alarms per hour of the
    clips, with its evaluation (with the skill's hold, as watch mode fires). Thresholds
    are tried in steps of 0.001. If even 1.0 gives too many, the result never fires
    (`never_fires`). Raises ValueError without frames."""
    clips = [sorted(c, key=lambda f: f.t) for c in clips if c]
    if not clips:
        raise ValueError("no labelled frames to calibrate on")
    candidates = sorted({min(1.0, math.floor(f.p * 1000) / 1000) for c in clips for f in c}
                        | {1.0})
    for threshold in candidates:  # the cooldown makes the count not monotone: try them all
        result = evaluate(clips, threshold, cooldown_s, hold_s)
        if result["false_alarms"] <= max_per_hour * result["hours"]:
            return result
    return {**evaluate(clips, math.nextafter(1.0, 2.0), cooldown_s, hold_s),
            "never_fires": True}


OBJECTIVES = ("alerts", "frames")


def calibrate_frames(clips: Sequence[Sequence[Frame]], cooldown_s: float,
                     hold_s: float = 0.0) -> dict:
    """The threshold with the best frame F1 over the clips, with the same evaluation as
    calibrate() plus the frame counts. For footage whose events are too few or too short
    to judge alerts on (a frame says yes when it scores at or over the threshold, as the
    evaluation counts it). When a stretch of thresholds ties for the best F1 (a confident
    model scores most frames at 0 or 1), the middle of that stretch is taken: its top
    edge, 1.000 for base Glimmer on Nov 4, failed test frames scored 0.9999. Raises
    ValueError without frames."""
    frames = [f for c in clips for f in c]
    if not frames:
        raise ValueError("no labelled frames to calibrate on")
    positives = sum(f.truth for f in frames)

    def counts(threshold: float) -> tuple[float, int, int]:
        tp = sum(f.truth and f.p >= threshold for f in frames)
        fp = sum((not f.truth) and f.p >= threshold for f in frames)
        f1 = 2 * tp / (2 * tp + fp + (positives - tp)) if positives else 0.0
        return f1, tp, fp

    candidates = sorted({min(1.0, math.floor(f.p * 1000) / 1000) for f in frames} | {1.0})
    scores = [counts(t)[0] for t in candidates]
    top = max(scores)
    # the longest run of neighbouring candidates at the best F1, and the middle of it
    best_run, run = (0, 0), None
    for i, score in enumerate(scores):
        if score >= top - 1e-12:
            run = (run[0], i) if run else (i, i)
            if run[1] - run[0] >= best_run[1] - best_run[0] or scores[best_run[0]] < top - 1e-12:
                best_run = run
        else:
            run = None
    # F1 holds from just above the candidate below the run (0 below the first) to the
    # run's last candidate: no frame scores in between
    low = candidates[best_run[0] - 1] if best_run[0] > 0 else 0.0
    high = candidates[best_run[1]]
    threshold = round((low + high) / 2, 4)
    f1, tp, fp = counts(threshold)
    result = evaluate([sorted(c, key=lambda f: f.t) for c in clips if c], threshold,
                      cooldown_s, hold_s)
    return {**result, "frame_f1": round(f1, 4), "frame_tp": tp, "frame_fp": fp,
            "frame_fn": positives - tp, "tied_above": low, "tied_to": high}


def labelled_frames(rows: Iterable[dict], skill: str,
                    true_ranges: Sequence[Sequence[float]]) -> list[Frame]:
    """Frames of one skill from a watch frame log (te watch), labelled true inside the
    given [start, end] second ranges of the clip. Frames the student failed on are left
    out."""
    ranges = [(float(a), float(b)) for a, b in true_ranges]
    if any(a > b for a, b in ranges):
        raise ValueError("each true range must be [start, end] with start <= end")
    return [Frame(float(r["t"]), float(r["p"]), any(a <= r["t"] <= b for a, b in ranges))
            for r in rows if r.get("skill") == skill and r.get("p") is not None]
