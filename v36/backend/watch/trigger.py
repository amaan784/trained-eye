"""When a skill fires: smoothing, cooldown and the threshold.

Watch mode scores about one frame a second per skill. A skill fires when at least 3 of
its last 4 frames score at or over its threshold, then stays quiet for the skill's
cooldown. One bright frame (a reflection, a hand in the way) never fires on its own,
and a condition that stays true is announced once per cooldown, not every second.

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


@dataclass
class Trigger:
    """3-of-4 smoothing plus a cooldown for one skill. Times are in seconds."""

    threshold: float
    cooldown_s: float
    window: int = WINDOW
    needed: int = NEEDED
    max_gap_s: float = MAX_GAP_S
    recent: deque = field(default_factory=deque, repr=False)
    quiet_until: float = -math.inf
    last_t: float | None = None

    def __post_init__(self) -> None:
        if not 0 < self.needed <= self.window:
            raise ValueError("need 0 < needed <= window")
        self.recent = deque(self.recent, maxlen=self.window)

    def update(self, p: float, t: float) -> bool:
        """Add the frame scored `p` at time `t`; True if the skill fires now."""
        if self.last_t is not None and t - self.last_t > self.max_gap_s:
            self.recent.clear()  # the stream paused: older frames say nothing about now
        self.last_t = t
        self.recent.append(p >= self.threshold)
        if t < self.quiet_until or sum(self.recent) < self.needed:
            return False
        self.quiet_until = t + self.cooldown_s
        self.recent.clear()
        return True

    def snooze(self, t: float, seconds: float) -> None:
        """Stay quiet for `seconds` from `t` (after the cloud check said no)."""
        self.quiet_until = max(self.quiet_until, t + seconds)


@dataclass
class Frame:
    t: float  # seconds from the start of the clip
    p: float  # the student's probability that the condition is true
    truth: bool  # the human label for this frame


def alerts(frames: Sequence[Frame], threshold: float, cooldown_s: float) -> list[Frame]:
    """The frames at which a Trigger with these settings fires."""
    trigger = Trigger(threshold, cooldown_s)
    return [f for f in frames if trigger.update(f.p, f.t)]


def events(frames: Sequence[Frame]) -> list[tuple[float, float]]:
    """(start, end) times of each run of frames labelled true."""
    out, start, prev = [], None, None
    for f in frames:
        if f.truth and start is None:
            start = f.t
        elif not f.truth and start is not None:
            out.append((start, prev.t))
            start = None
        prev = f
    if start is not None:
        out.append((start, prev.t))
    return out


def evaluate(clips: Iterable[Sequence[Frame]], threshold: float, cooldown_s: float) -> dict:
    """False alarms, caught events and delays for one threshold over labelled clips.

    An alert on a frame labelled false is a false alarm. An event (a run of true frames)
    is caught if an alert fires during it; the delay is from its first true frame."""
    false_alarms = caught = total_events = fired_total = 0
    delays: list[float] = []
    seconds = 0.0
    for frames in clips:
        frames = sorted(frames, key=lambda f: f.t)
        if not frames:
            continue
        seconds += frames[-1].t - frames[0].t
        fired = alerts(frames, threshold, cooldown_s)
        fired_total += len(fired)
        false_alarms += sum(not f.truth for f in fired)
        for start, end in events(frames):
            total_events += 1
            hits = [f.t for f in fired if f.truth and start <= f.t <= end]
            if hits:
                caught += 1
                delays.append(hits[0] - start)
    return {"threshold": threshold, "alerts": fired_total, "false_alarms": false_alarms,
            "hours": seconds / 3600,
            "events": total_events, "caught": caught,
            "median_delay_s": sorted(delays)[len(delays) // 2] if delays else None}


def calibrate(clips: Sequence[Sequence[Frame]], cooldown_s: float,
              max_per_hour: float = MAX_FALSE_ALARMS_PER_HOUR) -> dict:
    """The lowest threshold with at most `max_per_hour` false alarms per hour of the
    clips, with its evaluation. Thresholds are tried in steps of 0.001. If even 1.0 gives
    too many, the result never fires (`never_fires`). Raises ValueError without frames."""
    clips = [sorted(c, key=lambda f: f.t) for c in clips if c]
    if not clips:
        raise ValueError("no labelled frames to calibrate on")
    candidates = sorted({min(1.0, math.floor(f.p * 1000) / 1000) for c in clips for f in c}
                        | {1.0})
    for threshold in candidates:  # the cooldown makes the count not monotone: try them all
        result = evaluate(clips, threshold, cooldown_s)
        if result["false_alarms"] <= max_per_hour * result["hours"]:
            return result
    return {**evaluate(clips, math.nextafter(1.0, 2.0), cooldown_s), "never_fires": True}


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
