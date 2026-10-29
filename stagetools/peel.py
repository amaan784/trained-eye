"""Build stage folders by peeling layers off the finished tree at the repo root.

Usage: python stagetools/peel.py <repo_root> stagetools/specs/peel_<milestone>.py

A spec module defines:
  BASE   the last existing stage the new ones sit on (0 when building from scratch)
  FIRST, LAST   the stage numbers to build; v<LAST> becomes a copy of the repo root
  PEEL   {k: step}, where step undoes what stage k added on top of stage k-1:
           "remove":  files stage k added
           "restore": files to copy back from v<BASE> (unchanged since BASE)
           "edit":    {file: [(text_in_k, text_in_k_minus_1), ...]}; each text must match
                      exactly once. "CUT:<start>" with new=<end> removes from start up to
                      end; "CUT_TO_END:<marker>" removes from the marker to the end.

Only v<FIRST>..v<LAST> are deleted and rebuilt. When BASE > 0, v<FIRST> is peeled once
more into a temporary folder and must equal v<BASE> byte for byte (README aside), which
proves the new stages sit exactly on the committed ones.
"""

from __future__ import annotations

import filecmp
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

LOCAL_ONLY = {".git", ".claude", ".env", "data", "outputs", "VERSIONS.md", ".venv",
              "stagetools", "__pycache__", ".pytest_cache", ".ruff_cache"}
CACHES = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def copy_root(src: Path, dst: Path) -> None:
    def ignore(dirpath, names):
        skip = {n for n in names if n in CACHES}
        if Path(dirpath) == src:
            skip |= {n for n in names if n in LOCAL_ONLY or (n[:1] == "v" and n[1:].isdigit())}
        return skip
    shutil.copytree(src, dst, ignore=ignore)


def apply(text: str, old: str, new: str, where: str) -> str:
    if old.startswith("CUT_TO_END:"):
        marker = old[len("CUT_TO_END:"):]
        if text.count(marker) != 1:
            raise SystemExit(f"{where}: cut marker found {text.count(marker)} times: {marker!r}")
        return text[: text.index(marker)] + "\n"
    if old.startswith("CUT:"):
        start, end = old[len("CUT:"):], new
        if text.count(start) != 1 or text.count(end) != 1:
            raise SystemExit(f"{where}: cut markers not unique: {start!r} / {end!r}")
        if text.index(end) < text.index(start):
            raise SystemExit(f"{where}: end marker before start")
        return text[: text.index(start)] + text[text.index(end):]
    if text.count(old) != 1:
        raise SystemExit(f"{where}: expected exactly one match, found {text.count(old)}:\n{old}")
    return text.replace(old, new)


def peel(root: Path, base: int, cur: Path, prev: Path, step: dict) -> None:
    shutil.copytree(cur, prev)
    for rel in step.get("remove", []):
        path = prev / rel
        if not path.is_file():
            raise SystemExit(f"{prev.name}: {rel} to remove does not exist")
        path.unlink()
        parent = path.parent
        while parent != prev and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    for rel in step.get("restore", []):
        shutil.copyfile(root / f"v{base}" / rel, prev / rel)
    for rel, edits in step.get("edit", {}).items():
        path = prev / rel
        text = path.read_text(encoding="utf-8")
        for old, new in edits:
            text = apply(text, old, new, f"{prev.name}/{rel}")
        path.write_text(text, encoding="utf-8")


def same_tree(a: Path, b: Path) -> list[str]:
    fa = {p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file()}
    fb = {p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file()}
    return sorted(fa ^ fb) + sorted(r for r in fa & fb if not filecmp.cmp(a / r, b / r, False))


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    spec = importlib.util.spec_from_file_location("peel_spec", sys.argv[2])
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    if S.BASE and not (root / f"v{S.BASE}").is_dir():
        raise SystemExit(f"v{S.BASE} does not exist")
    for k in range(S.FIRST, S.LAST + 1):
        shutil.rmtree(root / f"v{k}", ignore_errors=True)
    copy_root(root, root / f"v{S.LAST}")
    for k in range(S.LAST, S.FIRST, -1):
        peel(root, S.BASE, root / f"v{k}", root / f"v{k - 1}", S.PEEL[k])
        print(f"v{k - 1} built from v{k}")
    if S.BASE:
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe"
            peel(root, S.BASE, root / f"v{S.FIRST}", probe, S.PEEL[S.FIRST])
            shutil.copyfile(root / f"v{S.BASE}" / "README.md", probe / "README.md")
            diff = same_tree(probe, root / f"v{S.BASE}")
        if diff:
            raise SystemExit(f"peeling v{S.FIRST} does not give v{S.BASE}: {diff}")
        print(f"peeling v{S.FIRST} once more gives v{S.BASE} exactly")


if __name__ == "__main__":
    main()
