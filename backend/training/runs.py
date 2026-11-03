"""Training runs on this Mac: their names, whether each is running, done or failed, and
which one the student endpoint serves. GET /skills reads these files for the app's Skills
screen (server.taught_skills); nothing here calls Modal.

te train writes data/train/<run>.status.json just before the Modal call and again after
it (also when it is interrupted):

    {"state": "running" | "done" | "failed", "skills": ["washer_done", ...],
     "started": "2026-11-03T14:03:00Z", "finished": null or "<ISO time>",
     "smoke": false, "pid": 4242, "host": "<this Mac's name>"}

`skills` is the list in the run's bundle (data/train/<run>/manifest.json). A run still
marked running whose te train process has gone from this Mac (its terminal was closed)
reads as failed, unless it was launched with te train --detach ("detached": true, no pid):
that one stays running until te train --collect. te serve-student writes
data/train/serving.json after a deploy,

    {"run": "exp1-mixed", "deployed": "<ISO time>", "skills": ["washer_done", ...]}

and removes it after --stop. Times are UTC, ISO 8601 with a Z.
"""

from __future__ import annotations

import json
import os
import re
import socket
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

RUN_NAME = re.compile(r"[a-z0-9][a-z0-9-]{1,39}")  # as te train checks it
STATES = ("running", "done", "failed")
SERVING_FILE = "serving.json"


def check_run(name: str) -> str:
    """`name` if it is a valid run name, else ValueError."""
    if not isinstance(name, str) or not RUN_NAME.fullmatch(name):
        raise ValueError(f"run name {name!r}: lower-case letters, digits and -")
    return name


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def status_path(data_dir: Path, run: str) -> Path:
    return Path(data_dir) / "train" / f"{check_run(run)}.status.json"


def serving_path(data_dir: Path) -> Path:
    return Path(data_dir) / "train" / SERVING_FILE


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)  # a reader never sees half a file


def _read(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_status(data_dir: Path, run: str, state: str, skills: Iterable[str], *,
                 started: str, finished: str | None = None, smoke: bool = False,
                 detached: bool = False) -> dict:
    """Write the run's status file (see the module docstring) and return it."""
    if state not in STATES:
        raise ValueError(f"state must be one of {', '.join(STATES)}")
    status = {"state": state, "skills": sorted(skills), "started": started,
              "finished": finished, "smoke": smoke, "pid": None if detached else os.getpid(),
              "host": socket.gethostname(), **({"detached": True} if detached else {})}
    _write(status_path(data_dir, run), status)
    return status


def _alive(pid) -> bool:
    """Whether process `pid` still runs on this machine (True when it cannot tell)."""
    if os.name == "nt" or not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return True  # on Windows os.kill(pid, 0) would send a Ctrl+C
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # PermissionError: it runs, as another user
        return True
    return True


def read_status(data_dir: Path, run: str) -> dict | None:
    """The run's status with "run" added, or None without a readable status file."""
    status = _read(status_path(data_dir, run))
    if (status is None or status.get("state") not in STATES
            or not isinstance(status.get("skills"), list)):
        return None
    status = {**status, "run": run}
    if (status["state"] == "running" and status.get("host") == socket.gethostname()
            and not _alive(status.get("pid"))):
        status["state"] = "failed"  # te train ended without writing how
    return status


def all_status(data_dir: Path) -> list[dict]:
    """Every readable run status under data/train, by run name."""
    out = []
    for path in sorted((Path(data_dir) / "train").glob("*.status.json")):
        run = path.name.removesuffix(".status.json")
        status = read_status(data_dir, run) if RUN_NAME.fullmatch(run) else None
        if status is not None:
            out.append(status)
    return out


def skill_training(statuses: Iterable[dict], skill: str) -> dict | None:
    """GET /skills' "training" for one skill: {"run", "state", "started", "finished"} of
    the newest run whose bundle lists it, or None. A finished smoke (two steps, no usable
    adapter) does not count; a running one does."""
    mine = [s for s in statuses if skill in s["skills"]
            and not (s.get("smoke") and s["state"] != "running")]
    if not mine:
        return None
    newest = max(mine, key=lambda s: (str(s.get("started") or ""), s["run"]))
    return {key: newest.get(key) for key in ("run", "state", "started", "finished")}


def bundle_manifest(data_dir: Path, run: str) -> dict | None:
    """data/train/<run>/manifest.json (backend/training/prepare.py), or None."""
    return _read(Path(data_dir) / "train" / check_run(run) / "manifest.json")


def run_skills(data_dir: Path, run: str) -> list[str]:
    """The skills a run trained on: its status file's, else its bundle's."""
    status = read_status(data_dir, run)
    if status is not None:
        return sorted(status["skills"])
    skills = (bundle_manifest(data_dir, run) or {}).get("skills")
    return sorted(skills) if isinstance(skills, dict) else []


def write_serving(data_dir: Path, run: str) -> dict:
    """Record that the endpoint now serves `run` (after te serve-student deployed it)."""
    data = {"run": check_run(run), "deployed": now_iso(), "skills": run_skills(data_dir, run)}
    _write(serving_path(data_dir), data)
    return data


def clear_serving(data_dir: Path) -> None:
    """Record that nothing is served now (te serve-student --stop). The file stays, holding
    no run, so GET /skills can tell "stopped" (served: false) from "never recorded"
    (served: null, where the app falls back to calibrated alone)."""
    _write(serving_path(data_dir), {"run": None, "stopped": now_iso()})


def serving_known(data_dir: Path) -> bool:
    """Whether te serve-student has recorded a deploy or a stop on this Mac."""
    return serving_path(data_dir).is_file()


def serving(data_dir: Path) -> dict | None:
    """{"run", "deployed", "skills"} of the run te serve-student last deployed, or None."""
    data = _read(serving_path(data_dir))
    if data is None or not isinstance(data.get("run"), str) or not RUN_NAME.fullmatch(
            data["run"]):
        return None
    skills = data.get("skills")
    if not (isinstance(skills, list) and all(isinstance(s, str) for s in skills)):
        skills = run_skills(data_dir, data["run"])
    return {**data, "skills": skills}
