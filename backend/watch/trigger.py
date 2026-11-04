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
        if t < self.quiet_until or t - self.since < self.hold_s:
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


def evaluate(clips: Iterable[Sequence[Frame]], threshold: float, cooldown_s: float,
             hold_s: float = 0.0) -> dict:
    """False alarms, caught events and delays for one threshold over labelled clips.

    An event is a run of true frames that lasts at least the hold; its alert is due when
    the hold ends (its first true frame plus hold_s). An alert from then to the event's
    end catches it, and the delay is from when it was due. Any other alert is a false
    alarm: on a frame labelled false, or before the condition had lasted the hold."""
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
        runs = events(frames)
        false_alarms += sum(not any(a + hold_s <= f.t <= b for a, b in runs) for f in fired)
        for start, end in runs:
            due = start + hold_s
            if end < due:
                continue  # shorter than the hold: nothing should fire
            total_events += 1
            hits = [f.t for f in fired if due <= f.t <= end]
            if hits:
                caught += 1
                delays.append(hits[0] - due)
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
