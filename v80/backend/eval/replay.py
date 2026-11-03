"""te replay: replay a run's pending clips of eval/plan.json into the plan's frame logs.

`te replay --plan eval/plan.json --run <run>` takes the run's clips from the plan: the
validation clips of its calibration entries, then the test clips of every evaluation whose
"run" it is (a log two evaluations share, such as exp1-mixed's and exp4-finetuned's, is
replayed once). A clip is pending while its log does not exist. In order:

1. The endpoint is warmed up first: a cold Modal container loads the model for minutes,
   and OpenAIStudent gives up on a request after 60 s, so one frame is sent again every
   WAIT_S until the endpoint answers or the deadline passes (te replay --warm-up-s).
2. The validation clips are replayed (without Spark confirmations: calibration reads only
   the probabilities). A skill whose validation logs and labels are all there, and that
   has no threshold for the run yet or got new validation logs, is calibrated as te
   calibrate --run does it (data/watch/<run>/<skill>/threshold.json).
3. The test clips are replayed with that threshold. A skill the plan calibrates for the run
   that still has no run threshold keeps its test clips pending (unless --threshold), so no
   test log is written at the 0.5 default.

Each clip is replayed as te watch replays one (a fresh Watcher and trigger per clip, the
same skill loading, prompt placement and confirmations), into <log>.partial, renamed to the
log only when the clip is done with at least one scored frame: an interrupted or failed
replay stays pending. With the spark scorer (exp4-spark) Muse Spark judges every frame:
priced and asked first, every threshold 1.0, no warm-up and no calibration.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from backend.eval.gallery import find_clip
from backend.eval.report import Plan
from backend.inference.student import StudentError
from backend.watch.watcher import FrameLog, WatchedSkill, Watcher

WAIT_S = 15.0  # between warm-up requests
DEFAULT_WARM_UP_S = 900.0  # te replay --warm-up-s; modal_serve.py allows 30 min to start
RETRY_STATUS = (408, 425, 429)  # client errors worth sending again
STATUS = re.compile(r"Client error '(\d{3})")


@dataclass
class Job:
    """One clip to replay: its skill, kind ("validation" or "test"), the plan's paths
    (taken from the plan's folder) and the evaluations whose log it is."""

    skill: str
    kind: str
    log: Path
    labels: Path
    clip: Path | None  # the recorded clip, None while it is not on this Mac
    named: str  # the clip the plan names (or the labels file), for messages
    evaluations: list[str] = field(default_factory=list)


def plan_jobs(plan: Plan, base: Path, run: str) -> list[Job]:
    """The run's validation clips, then its test clips, each log once."""
    jobs: dict[Path, Job] = {}

    def add(skill: str, kind: str, clip, evaluation: str | None) -> None:
        log = (base / clip.log).resolve()
        if log not in jobs:
            named = str(clip.clip) if clip.clip else str(clip.labels)
            jobs[log] = Job(skill, kind, log, (base / clip.labels).resolve(),
                            find_clip(str(clip.clip) if clip.clip else None, clip.labels, base),
                            named)
        if evaluation and evaluation not in jobs[log].evaluations:
            jobs[log].evaluations.append(evaluation)

    for cal in plan.calibration:
        if cal.run == run:
            for clip in cal.clips:
                add(cal.skill, "validation", clip, None)
    for ev in plan.evaluations:
        if ev.run == run:
            for clip in ev.clips:
                add(ev.skill, "test", clip, ev.name)
    return list(jobs.values())


def plan_scorer(plan: Plan, run: str, requested: str | None = None) -> str:
    """Who scores the run's frames, from its evaluations; ValueError when the plan has
    nothing for the run, or `requested` (te replay --scorer) disagrees with it."""
    scorers = {ev.scorer for ev in plan.evaluations if ev.run == run}
    if not scorers and any(c.run == run for c in plan.calibration):
        scorers = {"student"}
    if not scorers:
        reads = sorted({ev.run for ev in plan.evaluations
                        if ev.run and ev.name.startswith(f"{run}-")})
        if reads:  # a run with the same bundle as an earlier one: never replayed apart
            raise ValueError(f"{run}'s evaluations read the replays of {', '.join(reads)} "
                             f"(the same model): te replay --run {reads[0]}")
        known = sorted({ev.run for ev in plan.evaluations if ev.run})
        raise ValueError(f"the plan has no evaluation for run {run}" + (
            f"; its runs: {', '.join(known)}" if known else
            "; its evaluations name no run (te experiments --write-plan writes them)"))
    if len(scorers) > 1:
        raise ValueError(f"the plan scores run {run} with both the student and Muse Spark")
    scorer = scorers.pop()
    if requested and requested != scorer:
        raise ValueError(f"the plan scores run {run} with the {scorer} scorer, not {requested}")
    return scorer


def calibrated_skills(plan: Plan, run: str) -> set[str]:
    return {c.skill for c in plan.calibration if c.run == run}


def answered(err: StudentError) -> bool | None:
    """What a failed warm-up request says: True if the endpoint answered (the reply could
    not be read, but the model ran), False if it never will (a client error such as a
    wrong key or URL, or a stopped app), None if it is not up yet (try again)."""
    text = str(err)
    if not text.startswith("student endpoint failed"):
        return True
    found = STATUS.search(text)
    if found and int(found.group(1)) not in RETRY_STATUS:
        return False
    return None


def warm_up(score: Callable[[], object], deadline_s: float, *, wait_s: float = WAIT_S,
            clock: Callable[[], float] = time.monotonic,
            sleep: Callable[[float], None] = time.sleep,
            say: Callable[[str], None] = print) -> bool:
    """Send one request (`score`) until the endpoint answers or `deadline_s` passes."""
    start, tries = clock(), 0
    while True:
        tries += 1
        try:
            score()
        except StudentError as err:
            verdict = answered(err)
            if verdict is True:
                say(f"the endpoint answered, though not with a readable score: {err}")
                return True
            waited = clock() - start
            if verdict is False:
                say(f"te: the endpoint refused the request: {err}")
                return False
            if waited + wait_s > deadline_s:
                say(f"te: the endpoint did not answer in {waited:.0f} s ({tries} tries): {err}")
                return False
            say(f"waiting for the endpoint to start ({waited:.0f} s, try {tries}): "
                f"{str(err)[:120]}")
            sleep(wait_s)
            continue
        if tries > 1:
            say(f"the endpoint answered after {clock() - start:.0f} s")
        return True


def replay_clip(job: Job, skill: WatchedSkill, scorer, frames: Iterable, *, run: str,
                confirm: Callable | None = None) -> tuple[int, int]:
    """Replay one clip's frames for its skill into job.log (through <log>.partial).
    Returns (frames scored, student errors); the log is written only if one scored."""
    partial = job.log.with_name(job.log.name + ".partial")
    partial.unlink(missing_ok=True)
    watcher = Watcher([skill], scorer, confirm=confirm, log=FrameLog(partial, {"run": run}))
    scored = errors = 0
    for frame in frames:
        scores, _ = watcher.on_frame(frame.data, t=frame.t_s)
        if scores[skill.spec.name] is None:
            errors += 1
        else:
            scored += 1
    if scored:
        os.replace(partial, job.log)
    else:
        partial.unlink(missing_ok=True)
    return scored, errors
