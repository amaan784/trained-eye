"""Fetch a run's weights from the Modal volume for `te release --weights`.

Training leaves the adapter in /vol/runs/<run>/out of the volume "trained-eye"
(backend/training/modal_train.py) and `te export` the merged build in /vol/runs/<run>/merged
(backend/training/export.py). `fetch_weights` lists that folder through
modal.Volume.from_name(VOLUME).listdir and downloads with read_file only the top-level
files te release ships (model_card.WEIGHT_SUFFIXES, no README): no checkpoint-* or other
folders, no pickles (.bin, .pt). It checks that the files the release needs are there and
that each download has the size the volume lists, into a hidden folder first, so a folder
at `out_dir` is always complete. Modal is imported only when no volume is given (tests
pass a fake); a real run needs `pip install -e ".[train]"` and `modal token set`.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path, PurePosixPath

from backend.release.model_card import WEIGHT_SUFFIXES
from backend.training.runs import RUN_NAME

VOLUME = "trained-eye"  # modal_train.py's and export.py's VOLUME
FOLDERS = {"adapter": "out", "merged": "merged"}  # under runs/<run>/ on the volume
NEEDED = {"adapter": "adapter_config.json", "merged": "config.json"}  # as upload() checks


def default_dir(data_dir: Path, run: str, kind: str) -> Path:
    """Where te fetch-weights puts a run's files, and where te release looks for them."""
    return Path(data_dir) / "weights" / run / kind


def _volume():
    try:
        import modal
    except ImportError:
        raise RuntimeError('fetching weights needs Modal: pip install -e ".[train]", then '
                           "modal token set") from None
    return modal.Volume.from_name(VOLUME)


def _is_file(entry) -> bool:
    kind = getattr(entry, "type", None)
    return getattr(kind, "name", kind) == "FILE"


def pick(entries) -> tuple[list, list[str]]:
    """(the entries to download, the names left out) of a listdir of the run's folder."""
    keep, skipped = [], []
    for entry in entries:
        name = PurePosixPath(entry.path).name
        if _is_file(entry) and Path(name).suffix in WEIGHT_SUFFIXES and name != "README.md":
            keep.append(entry)
        else:
            skipped.append(name)
    return sorted(keep, key=lambda e: e.path), sorted(skipped)


def fetch_weights(run: str, out_dir: Path, kind: str = "adapter", volume=None) -> dict:
    """Download the run's top-level weight and config files into out_dir (a new or empty
    folder). Returns {"remote", "files": [{"path", "bytes"}], "bytes", "skipped"}."""
    if not RUN_NAME.fullmatch(run or ""):
        raise ValueError(f"{run!r} is not a run name (as te train names them)")
    if kind not in FOLDERS:
        raise ValueError(f"kind must be one of {', '.join(FOLDERS)}")
    out_dir = Path(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f"{out_dir} already has files: remove it or pick another --out")
    volume = _volume() if volume is None else volume
    remote = f"/runs/{run}/{FOLDERS[kind]}"
    try:
        entries = list(volume.listdir(remote))
    except Exception as err:  # noqa: BLE001  (modal's NotFoundError and connection errors)
        raise RuntimeError(f"could not list {remote} on the Modal volume {VOLUME} "
                           f"({type(err).__name__}: {str(err)[:200]})"
                           + (": run te export first" if kind == "merged" else "")) from None
    keep, skipped = pick(entries)
    names = [PurePosixPath(e.path).name for e in keep]
    if NEEDED[kind] not in names or not any(n.endswith(".safetensors") for n in names):
        raise ValueError(f"{remote} on the volume has no {kind} weights ({NEEDED[kind]} and "
                         f".safetensors; it lists {', '.join(names + skipped) or 'nothing'})")
    total = sum(int(e.size) for e in keep)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if (free := shutil.disk_usage(out_dir.parent).free) < total:
        raise ValueError(f"{total / 1e9:.1f} GB to download but {free / 1e9:.1f} GB free at "
                         f"{out_dir.parent}")
    partial = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}-", dir=out_dir.parent))
    try:
        for entry, name in zip(keep, names, strict=True):
            with (partial / name).open("wb") as fh:
                for chunk in volume.read_file(entry.path):
                    fh.write(chunk)
            if (got := (partial / name).stat().st_size) != int(entry.size):
                raise RuntimeError(f"{name}: downloaded {got} bytes, the volume lists "
                                   f"{entry.size}: run the same command again")
        if out_dir.exists():
            out_dir.rmdir()  # empty, checked above
        partial.rename(out_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return {"remote": remote, "bytes": total, "skipped": skipped,
            "files": [{"path": n, "bytes": int(e.size)} for e, n in zip(keep, names,
                                                                         strict=True)]}
