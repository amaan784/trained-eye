"""Watch mode for a set of taught skills: frames in, alerts out.

For every frame and skill: the student scores the frame, the skill's Trigger decides
whether it fires (3 of the last 4 frames over the threshold for the skill's hold_s,
then the cooldown), and if
the skill asks for it (`confirm_with_cloud`), Muse Spark checks the frame before the
alert goes out. If Spark says no, the skill is snoozed for a minute instead of the whole
cooldown. If the check itself fails (network, spend cap), the alert goes out marked as
unconfirmed: missing a burner left on is worse than one unchecked alert.

Every scored frame is logged as one JSON line (time, probability, decision, the cloud
answer, the threshold and where it came from, and the skill's hold_s) for the timeline
visual and the evaluation, which counts an alert as on time from the end of the hold.
Frames themselves are never written.

Thresholds, first found: te watch --threshold ("flag"), the trained run's
data/watch/<run>/<skill>/threshold.json from te calibrate --run ("run"), the skill's
data/skills/<skill>/threshold.json from te calibrate ("skill"), else 0.5 ("default");
te watch --scorer spark uses 1.0 ("spark"), so only a yes counts. Each frame log row says
which, as "threshold_source", and names the run as "run" when there is one (te watch
--run, or the run te serve-student deployed).

In te serve, one CooldownBook holds each skill's quiet time for every session, so a
reconnect, a skills toggle or a second phone does not announce a condition again inside
its cooldown (or the minute after Spark said no). The 3-of-4 count starts afresh with
each session. The book is saved to data/watch/cooldowns.json, so a restart keeps it.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from backend.inference.student import StudentError
from backend.skills.spec import SkillSpec, check_name
from backend.training.runs import check_run
from backend.watch.trigger import Trigger

DEFAULT_THRESHOLD = 0.5  # until calibrate() has set one from validation footage
CLOUD_NO_SNOOZE_S = 60.0
COOLDOWNS_FILE = "cooldowns.json"  # in data/watch/

log = logging.getLogger("trained_eye.watch")


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
    threshold_source: str = "default"  # flag, run, skill, default (or spark: te watch)


class FrameLog:
    """Appends one JSON line per scored frame and skill; flushed as it goes. `fields`
    go into every row (te watch --run puts the run there)."""

    def __init__(self, path: Path, fields: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = dict(fields or {})

    def write(self, row: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({**self.fields, **row}, sort_keys=True) + "\n")


def run_threshold_file(data_dir: Path, run: str, skill: str) -> Path:
    """Where te calibrate --run writes one run's threshold for a skill, next to that
    run's replays: data/watch/<run>/<skill>/threshold.json."""
    return Path(data_dir) / "watch" / check_run(run) / check_name(skill) / "threshold.json"


def read_threshold(path: Path) -> float | None:
    """The threshold in a threshold.json, None if there is no such file. ValueError if it
    cannot be read or holds no finite number."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        value = float(json.loads(path.read_text(encoding="utf-8"))["threshold"])
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError(f"{path} holds no threshold; run te calibrate again") from None
    if not math.isfinite(value):
        raise ValueError(f"{path}: the threshold is {value}; run te calibrate again")
    return value


def skill_threshold(skill_dir: Path, *, run: str | None = None,
                    override: float | None = None) -> tuple[float, str]:
    """(threshold, source) for the skill in data/skills/<skill>/: `override` ("flag"),
    the run's file ("run"), the skill's file ("skill"), else the default ("default")."""
    skill_dir = Path(skill_dir)
    if override is not None:
        return float(override), "flag"
    if run is not None:
        value = read_threshold(run_threshold_file(skill_dir.parent.parent, run,
                                                  skill_dir.name))
        if value is not None:
            return value, "run"
    value = read_threshold(skill_dir / "threshold.json")
    return (value, "skill") if value is not None else (DEFAULT_THRESHOLD, "default")


def load_skill(skill_dir: Path, *, run: str | None = None,
               threshold: float | None = None) -> WatchedSkill:
    """A taught skill from data/skills/<skill>/ (spec.json) with its threshold, as
    skill_threshold picks it for the trained run `run` and an override `threshold`."""
    skill_dir = Path(skill_dir)
    spec = SkillSpec.model_validate_json((skill_dir / "spec.json").read_text())
    if spec.refused:
        raise ValueError(f"{spec.name} was refused when it was taught; it cannot be watched")
    value, source = skill_threshold(skill_dir, run=run, override=threshold)
    return WatchedSkill(spec, Trigger(value, float(spec.cooldown_s), float(spec.hold_s)),
                        source in ("run", "skill"), source)


def system_in_user(skill_dirs: Iterable[Path], *, bundle: dict | None = None,
                   default: bool = False) -> bool:
    """Whether the student was trained with the system prompt folded into the user text,
    so watch mode asks it the same way. A run's bundle manifest says what its training
    saw; else each skill's dataset manifest (data/skills/<skill>/dataset/); skills with
    neither take `default` (TE_SYSTEM_IN_USER). ValueError if the skills disagree."""
    if bundle is not None and isinstance(bundle.get("system_in_user"), bool):
        return bundle["system_in_user"]
    seen: dict[str, bool] = {}
    for skill_dir in map(Path, skill_dirs):
        manifest = skill_dir / "dataset" / f"{skill_dir.name}-teach.manifest.json"
        if not manifest.is_file():
            continue
        try:
            seen[skill_dir.name] = bool(json.loads(manifest.read_text("utf-8"))
                                        .get("system_in_user", False))
        except (OSError, ValueError, AttributeError):
            raise ValueError(f"{manifest} cannot be read; run te teach again") from None
    if len(set(seen.values())) > 1:
        folded = ", ".join(sorted(n for n, v in seen.items() if v))
        raise ValueError(f"the skills' datasets put the system prompt differently (folded "
                         f"into the user text for {folded} only): watch them apart")
    return next(iter(seen.values()), default)


class CooldownBook:
    """Each skill's quiet_until, shared by every Watcher of one te serve process (see the
    module docstring). Times are in the watchers' clock (time.monotonic by default); the
    file, when `path` gives one, holds wall-clock seconds so a restart keeps them.
    `path` may be a function, called on first use (the server's data folder)."""

    def __init__(self, path: Path | Callable[[], Path] | None = None, *,
                 clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time):
        self._path_source = path
        self._path: Path | None = None
        self.clock, self.wall = clock, wall
        self.lock = threading.RLock()  # held across a trigger's read, update and write
        self._until: dict[str, float] | None = None

    def _load(self) -> dict[str, float]:
        if self._until is not None:
            return self._until
        self._until = {}
        try:
            source = self._path_source
            self._path = Path(source() if callable(source) else source) if source else None
            if self._path is None or not self._path.is_file():
                return self._until
            saved = json.loads(self._path.read_text("utf-8"))["quiet_until"]
            offset, now = self.clock() - self.wall(), self.clock()
            for skill, wall_until in saved.items():
                until = float(wall_until) + offset
                if until > now and isinstance(skill, str):
                    self._until[skill] = until
        except Exception as err:  # noqa: BLE001 - a bad file only loses the saved times
            log.warning("cooldowns not loaded from %s: %s", self._path, type(err).__name__)
        return self._until

    def quiet_until(self, skill: str) -> float:
        with self.lock:
            return self._load().get(skill, -math.inf)

    def set(self, skill: str, until: float, *, only_if: float | None = None) -> None:
        """Set the skill's quiet_until; with `only_if`, only while it still holds that
        value (a Spark no shortens this session's cooldown, not another's newer one)."""
        with self.lock:
            until_by_skill = self._load()
            if only_if is not None and until_by_skill.get(skill, -math.inf) != only_if:
                return
            until_by_skill[skill] = until
            self._save()

    def _save(self) -> None:
        if self._path is None:
            return
        now, offset = self.clock(), self.wall() - self.clock()
        data = {"quiet_until": {skill: round(until + offset, 3) for skill, until
                                in sorted(self._until.items()) if until > now}}
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
            tmp.replace(self._path)
        except OSError as err:
            log.warning("cooldowns not saved to %s: %s", self._path, err)


class Watcher:
    """Scores frames for several skills. `scorer.score(image, spec)` returns a Score;
    `confirm(image, spec)` returns a labels.Label (Muse Spark) or raises. With
    `cooldowns`, each skill's quiet time is shared with every Watcher using that book
    (they must share a clock too)."""

    def __init__(self, skills: list[WatchedSkill], scorer, *,
                 confirm: Callable[[bytes, SkillSpec], object] | None = None,
                 log: FrameLog | None = None, clock: Callable[[], float] = time.monotonic,
                 cloud_no_snooze_s: float = CLOUD_NO_SNOOZE_S,
                 cooldowns: CooldownBook | None = None):
        self.skills = {s.spec.name: s for s in skills}
        self.scorer = scorer
        self.confirm = confirm
        self.log = log
        self.clock = clock
        self.cloud_no_snooze_s = cloud_no_snooze_s
        self.cooldowns = cooldowns

    def on_frame(self, image: bytes, t: float | None = None) -> tuple[dict, list[Alert]]:
        """Score one frame for every skill. Returns ({skill: p or None}, alerts)."""
        t = self.clock() if t is None else t
        scores, fired = {}, []
        for name, skill in self.skills.items():
            row = {"t": round(t, 3), "skill": name, "threshold": skill.trigger.threshold,
                   "threshold_source": skill.threshold_source,
                   "hold_s": skill.trigger.hold_s}
            try:
                score = self.scorer.score(image, skill.spec)
            except StudentError as err:
                scores[name] = None
                self._log({**row, "p": None, "error": str(err)[:200]})
                continue
            scores[name] = score.p
            row.update(p=round(score.p, 4), over=score.p >= skill.trigger.threshold,
                       latency_s=(round(score.latency_s, 3)
                                  if getattr(score, "latency_measured", True) else None))
            fires = self._update(skill, score.p, t)
            row["fires"] = fires
            if fires:
                alert, row["cloud"] = self._check(image, skill, t, score.p)
                row["alert"] = alert is not None
                if alert:
                    fired.append(alert)
            self._log(row)
        return scores, fired

    def _update(self, skill: WatchedSkill, p: float, t: float) -> bool:
        book, trigger = self.cooldowns, skill.trigger
        if book is None:
            return trigger.update(p, t)
        with book.lock:  # another session may have fired this skill since our last frame
            trigger.quiet_until = book.quiet_until(skill.spec.name)
            fires = trigger.update(p, t)
            if fires:
                book.set(skill.spec.name, trigger.quiet_until)
        return fires

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
        fired_until = skill.trigger.quiet_until
        skill.trigger.quiet_until = t  # the cooldown did not start: snooze briefly instead
        skill.trigger.snooze(t, self.cloud_no_snooze_s)
        if self.cooldowns is not None:
            self.cooldowns.set(spec.name, skill.trigger.quiet_until, only_if=fired_until)
        return None, label.label

    def _log(self, row: dict) -> None:
        if self.log:
            self.log.write(row)


def alert_message(alert: Alert) -> dict:
    """The WebSocket message the phone speaks through the glasses."""
    return {"type": "alert", **{k: v for k, v in asdict(alert).items()}}
