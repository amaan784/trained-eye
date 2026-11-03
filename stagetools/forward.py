"""Build stage folders forward from a base stage, adding whole final files.

Usage: python stagetools/forward.py <spec.py> [--only K] [--check-base]

The spec module defines
    BASE   the stage the build starts from (its folder must exist and stays as it is)
    FILES  {path: first stage}: each file comes from the repo root, in its final form,
           from its first stage on (before that, the base stage's copy, if any, is kept)
    PIECES {path: [piece, ...]}: parts of a final file that arrive later. A piece is
           ("text", stage, final_text, before_text): until `stage`, final_text (which must
           occur exactly once in the final file) reads as before_text; or
           ("test", stage, name): until `stage`, the top-level test function `name`
           (with its decorators) is left out.
A file with pieces is written from its first stage with every piece of a later stage
undone, so each stage holds the file as it was then. The last stage must equal the root
(apart from README.md, which render_readmes.py writes, and the local-only tools).
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from peel import project_files  # noqa: E402

SKIP = ("__pycache__", ".ruff_cache", ".pytest_cache")


def load(spec_path: str):
    spec = importlib.util.spec_from_file_location("forward_spec", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def drop_test(text: str, name: str) -> str:
    """The text without the top-level function `name` and its decorators."""
    lines = text.splitlines(keepends=True)
    start = next((i for i, ln in enumerate(lines) if re.match(rf"def {name}\(", ln)), None)
    if start is None:
        raise SystemExit(f"test {name} not found")
    end = start + 1  # the body: blank, indented or closing-bracket lines
    above = start  # decorators (and a comment) sit right above the def, with no blank line
    while above > 0 and lines[above - 1].strip():
        above -= 1
    if above < start and lines[above].startswith(("@", "#")):
        start = above
    while end < len(lines) and not re.match(r"[^\s)]", lines[end]):
        end += 1
    text = "".join(lines[:start] + lines[end:])
    return text.rstrip("\n") + "\n" if end == len(lines) else text


def content_at(path: str, k: int, final: str, pieces: list) -> str:
    text = final
    for piece in pieces:
        kind, stage = piece[0], piece[1]
        if stage <= k:
            continue
        if kind == "text":
            _, _, now, before = piece
            if text.count(now) != 1:
                raise SystemExit(f"{path}: piece for v{stage} occurs {text.count(now)} times: "
                                 f"{now[:70]!r}")
            text = text.replace(now, before)
        elif kind == "test":
            text = drop_test(text, piece[2])
        else:
            raise SystemExit(f"{path}: unknown piece {kind}")
    return text


def copy_stage(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*SKIP))


def build(spec, only: int | None = None) -> list[int]:
    stages = sorted(set(spec.FILES.values()) | {p[1] for ps in spec.PIECES.values() for p in ps})
    stages = [k for k in range(spec.BASE + 1, max(stages) + 1)]
    finals = {}
    for path in set(spec.FILES) | set(spec.PIECES):
        if path not in spec.FILES:
            raise SystemExit(f"{path} has pieces but no first stage in FILES")
        finals[path] = (ROOT / path).read_bytes()
    for k in stages:
        if only is not None and k != only:
            continue
        copy_stage(ROOT / f"v{k - 1}", ROOT / f"v{k}")
        for path, first in spec.FILES.items():
            if first > k:
                continue
            dst = ROOT / f"v{k}" / path
            dst.parent.mkdir(parents=True, exist_ok=True)
            pieces = spec.PIECES.get(path, [])
            if pieces:
                dst.write_text(content_at(path, k, finals[path].decode("utf-8"), pieces),
                               encoding="utf-8")
            else:
                dst.write_bytes(finals[path])
            shutil.copymode(ROOT / path, dst)
    return stages


def check_base(spec) -> list[str]:
    """Problems: a file with pieces whose text at BASE differs from the base folder's."""
    problems = []
    for path, first in spec.FILES.items():
        if first > spec.BASE or path not in spec.PIECES:
            continue
        want = content_at(path, spec.BASE, (ROOT / path).read_text(encoding="utf-8"),
                          spec.PIECES[path])
        have = (ROOT / f"v{spec.BASE}" / path).read_text(encoding="utf-8")
        if want != have:
            problems.append(f"{path}: v{spec.BASE} differs from the pieces' text at v{spec.BASE}")
    return problems


def check_last(last: int) -> list[str]:
    folder = ROOT / f"v{last}"
    want = set(project_files(ROOT)) - {"README.md"}
    have = {str(p.relative_to(folder)) for p in folder.rglob("*")
            if p.is_file() and not any(s in p.parts for s in SKIP)} - {"README.md"}
    problems = [f"only at the root: {p}" for p in sorted(want - have)]
    problems += [f"only in v{last}: {p}" for p in sorted(have - want)]
    for rel in sorted(want & have):
        if (ROOT / rel).read_bytes() != (folder / rel).read_bytes():
            problems.append(f"differs from the root: {rel}")
    return problems


def main() -> None:
    spec = load(sys.argv[1])
    only = int(sys.argv[sys.argv.index("--only") + 1]) if "--only" in sys.argv else None
    problems = check_base(spec)
    if problems:
        sys.exit("\n".join(problems))
    if "--check-base" in sys.argv:
        print("base matches")
        return
    stages = build(spec, only)
    print(f"built v{stages[0]} to v{stages[-1]}" if only is None else f"built v{only}")
    if only is None:
        problems = check_last(stages[-1])
        print("\n".join(problems) or f"v{stages[-1]} equals the root")


if __name__ == "__main__":
    main()
