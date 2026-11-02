"""Watch mode for a set of taught skills: frames in, alerts out.

For every frame and skill: the student scores the frame, the skill's Trigger decides
whether it fires (3 of the last 4 frames over the threshold, then the cooldown), and if
the skill asks for it (`confirm_with_cloud`), Muse Spark checks the frame before the
alert goes out. If Spark says no, the skill is snoozed for a minute instead of the whole
cooldown. If the check itself fails (network, spend cap), the alert goes out marked as
unconfirmed: missing the washer is worse than one unchecked alert.

Every scored frame is logged as one JSON line (time, probability, decision, the cloud
answer) for the timeline visual and the evaluation. Frames themselves are never written.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from backend.inference.student import StudentError
from backend.skills.spec import SkillSpec
from backend.watch.trigger import Trigger

DEFAULT_THRESHOLD = 0.5  # until calibrate() has set one from validation footage
CLOUD_NO_SNOOZE_S = 60.0


@dataclass
class Alert:
    skill: str
    text: str
    t: float
    p: float
    confirmed: bool | None  # None: no cloud check, or the check failed


@dataclass
class WatchedSkill:
    spec: SkillSpec
    trigger: Trigger
    calibrated: bool


class FrameLog:
    """Appends one JSON line per scored frame and skill; flushed as it goes."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def load_skill(skill_dir: Path) -> WatchedSkill:
    """A taught skill from data/skills/<skill>/: spec.json, and threshold.json if
    calibrate() has written one (else the default threshold)."""
    skill_dir = Path(skill_dir)
    spec = SkillSpec.model_validate_json((skill_dir / "spec.json").read_text())
    if spec.refused:
        raise ValueError(f"{spec.name} was refused when it was taught; it cannot be watched")
    threshold_file = skill_dir / "threshold.json"
    calibrated = threshold_file.is_file()
    threshold = (json.loads(threshold_file.read_text())["threshold"] if calibrated
                 else DEFAULT_THRESHOLD)
    return WatchedSkill(spec, Trigger(float(threshold), float(spec.cooldown_s)), calibrated)


class Watcher:
    """Scores frames for several skills. `scorer.score(image, spec)` returns a Score;
    `confirm(image, spec)` returns a labels.Label (Muse Spark) or raises."""

    def __init__(self, skills: list[WatchedSkill], scorer, *,
                 confirm: Callable[[bytes, SkillSpec], object] | None = None,
                 log: FrameLog | None = None, clock: Callable[[], float] = time.monotonic,
                 cloud_no_snooze_s: float = CLOUD_NO_SNOOZE_S):
        self.skills = {s.spec.name: s for s in skills}
        self.scorer = scorer
        self.confirm = confirm
        self.log = log
        self.clock = clock
        self.cloud_no_snooze_s = cloud_no_snooze_s

    def on_frame(self, image: bytes, t: float | None = None) -> tuple[dict, list[Alert]]:
        """Score one frame for every skill. Returns ({skill: p or None}, alerts)."""
        t = self.clock() if t is None else t
        scores, fired = {}, []
        for name, skill in self.skills.items():
            row = {"t": round(t, 3), "skill": name, "threshold": skill.trigger.threshold}
            try:
                score = self.scorer.score(image, skill.spec)
            except StudentError as err:
                scores[name] = None
                self._log({**row, "p": None, "error": str(err)[:200]})
                continue
            scores[name] = score.p
            row.update(p=round(score.p, 4), over=score.p >= skill.trigger.threshold,
                       latency_s=round(score.latency_s, 3))
            fires = skill.trigger.update(score.p, t)
            row["fires"] = fires
            if fires:
                alert, row["cloud"] = self._check(image, skill, t, score.p)
                row["alert"] = alert is not None
                if alert:
                    fired.append(alert)
            self._log(row)
        return scores, fired

    def _check(self, image: bytes, skill: WatchedSkill, t: float,
               p: float) -> tuple[Alert | None, str | None]:
        spec = skill.spec
        if not (spec.confirm_with_cloud and self.confirm):
            return Alert(spec.name, spec.alert, t, p, None), None
        try:
            label = self.confirm(image, spec)
        except Exception as err:  # noqa: BLE001 - any failure of the check: alert anyway
            return Alert(spec.name, spec.alert, t, p, None), f"error: {type(err).__name__}"
        if getattr(label, "error", False):
            return Alert(spec.name, spec.alert, t, p, None), "error"
        if label.label == "yes":
            return Alert(spec.name, spec.alert, t, p, True), "yes"
        skill.trigger.quiet_until = t  # the cooldown did not start: snooze briefly instead
        skill.trigger.snooze(t, self.cloud_no_snooze_s)
        return None, label.label

    def _log(self, row: dict) -> None:
        if self.log:
            self.log.write(row)


def alert_message(alert: Alert) -> dict:
    """The WebSocket message the phone speaks through the glasses."""
    return {"type": "alert", **{k: v for k, v in asdict(alert).items()}}
