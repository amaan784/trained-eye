"""Modal GPU spend on this Mac: data/modal_ledger.jsonl, one JSON line per job, so te spend
can print the Modal total and te train and te export can ask before the runs pass the
Modal credit (backend/eval/plan.py's MODAL_CREDIT_USD).

    {"t": "2026-11-03T14:03:00Z", "kind": "train", "name": "exp1-mixed", "gpu": "H100",
     "seconds": 912.4, "usd": 1.0011, "upper_bound": false, "key": "train:exp1-mixed:..."}

Kinds: train (te train: the seconds the training container measured, from its result),
export (te export: the remote call, timed by export.py on this Mac, so the container start
is in it) and serve (te serve-student --stop: from the deploy to the stop). Serving is an
upper bound: a container bills only while it is up and stops after a few idle minutes. A
deploy is a row of kind deploy without seconds; the next stop of that app closes it. A
row with a key is written once (te train --collect after a run te train already saw).

Dollars are seconds at config.MODAL_GPU_USD_PER_HOUR, not checked against Modal's bill
(the Modal dashboard has the real one). CPU jobs, such as the processor preflight, are not
recorded. Times are UTC, ISO 8601 with a Z.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

from backend.config import MODAL_GPU_USD_PER_HOUR

LEDGER_FILE = "modal_ledger.jsonl"
JOBS = ("train", "export", "serve")
TIME = "%Y-%m-%dT%H:%M:%SZ"


def ledger_path(data_dir: Path) -> Path:
    return Path(data_dir) / LEDGER_FILE


def _now() -> datetime:
    return datetime.now(UTC)


def read(data_dir: Path) -> list[dict]:
    """Every readable row, oldest first (a damaged line is skipped)."""
    path = ledger_path(data_dir)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("kind") in (*JOBS, "deploy"):
            rows.append(row)
    return rows


def _append(data_dir: Path, row: dict) -> dict:
    path = ledger_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def record(data_dir: Path, kind: str, name: str, seconds: float, *, gpu: str = "H100",
           upper_bound: bool = False, key: str | None = None,
           now: datetime | None = None) -> dict | None:
    """Append one GPU job and return its row; None (nothing written) when a row with the
    same key is there already."""
    if kind not in JOBS:
        raise ValueError(f"kind must be one of {', '.join(JOBS)}")
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"seconds must be 0 or more, got {seconds}")
    if key is not None and any(r.get("key") == key for r in read(data_dir)):
        return None
    usd = seconds / 3600 * MODAL_GPU_USD_PER_HOUR[gpu]
    row = {"t": (now or _now()).strftime(TIME), "kind": kind, "name": name, "gpu": gpu,
           "seconds": round(seconds, 1), "usd": round(usd, 4), "upper_bound": upper_bound}
    return _append(data_dir, {**row, **({"key": key} if key else {})})


def record_deploy(data_dir: Path, app: str, serves: str, *, gpu: str = "H100",
                  now: datetime | None = None) -> dict:
    """te serve-student deployed `app`; its stop prices the time between."""
    return _append(data_dir, {"t": (now or _now()).strftime(TIME), "kind": "deploy",
                              "name": app, "gpu": gpu, "serves": serves})


def open_deploys(data_dir: Path) -> dict[str, dict]:
    """{app: its first deploy row since that app's last stop}."""
    out: dict[str, dict] = {}
    for row in read(data_dir):
        if row["kind"] == "deploy":
            out.setdefault(row["name"], row)
        elif row["kind"] == "serve":
            out.pop(row["name"], None)
    return out


def record_stop(data_dir: Path, app: str, *, now: datetime | None = None) -> dict | None:
    """te serve-student --stop stopped `app`: a serve row from its open deploy to now (an
    upper bound), or None when no deploy of it is open."""
    deploy = open_deploys(data_dir).get(app)
    if deploy is None:
        return None
    now = now or _now()
    since = datetime.strptime(deploy["t"], TIME).replace(tzinfo=UTC)
    return record(data_dir, "serve", app, max(0.0, (now - since).total_seconds()),
                  gpu=deploy.get("gpu", "H100"), upper_bound=True, now=now)


def totals(data_dir: Path) -> dict:
    """Dollars and jobs by kind, the total, and the deploys still open."""
    by_kind = {kind: {"usd": 0.0, "jobs": 0} for kind in JOBS}
    for row in read(data_dir):
        if row["kind"] in JOBS:
            by_kind[row["kind"]]["usd"] += float(row.get("usd") or 0.0)
            by_kind[row["kind"]]["jobs"] += 1
    total = sum(k["usd"] for k in by_kind.values())
    return {**{k: {"usd": round(v["usd"], 2), "jobs": v["jobs"]} for k, v in by_kind.items()},
            "total_usd": round(total, 2), "open": open_deploys(data_dir)}


def spent_usd(data_dir: Path) -> float:
    return totals(data_dir)["total_usd"]


def credit_usd() -> float:
    from backend.eval.plan import MODAL_CREDIT_USD  # here: plan.py imports backend.cli

    return MODAL_CREDIT_USD


def credit_check(data_dir: Path, estimate_usd: float) -> str | None:
    """Why this job needs a yes even with --yes: what is spent plus its estimate passes
    the Modal credit. None while it stays within."""
    spent, credit = spent_usd(data_dir), credit_usd()
    if spent + estimate_usd <= credit:
        return None
    return (f"Modal spend so far, about ${spent:.2f} (te spend), plus this ${estimate_usd:.2f} "
            f"would pass the ${credit:.0f} of Modal credit, so it needs your yes even with "
            "--yes")


def format_report(data_dir: Path) -> str:
    """The Modal part of te spend."""
    t, credit = totals(data_dir), credit_usd()
    left = max(0.0, credit - t["total_usd"])
    lines = [(f"Modal GPU spend (data/{LEDGER_FILE}, at config prices): ${t['total_usd']:.2f} "
              f"of ${credit:.0f} credit, ${left:.2f} left"),
             (f"  training ${t['train']['usd']:.2f} ({t['train']['jobs']} runs), export "
              f"${t['export']['usd']:.2f} ({t['export']['jobs']}), serving "
              f"${t['serve']['usd']:.2f} ({t['serve']['jobs']}; deploy to stop, an upper "
              "bound)")]
    for app, row in sorted(t["open"].items()):
        lines.append(f"  still deployed: {app} since {row['t']} (not priced until te "
                     "serve-student --stop; it bills only while a container is up)")
    lines.append("  not checked against Modal's bill: the Modal dashboard has the real one")
    return "\n".join(lines)
