"""Check the vN stage folders against the rules.

Usage: python check_stages.py <repo_root> [--run]

Checks:
  1. v1 has at most 5 files; stages v2 to v49 add or change at most 5 files; from v50 on
     (the author's newer rule) a stage changes 150 to 300 lines against the one before
     it, counting added plus removed lines and leaving out its README.
  2. every stage contains every file of the previous stage (no deletes or renames).
  3. the last stage matches the root byte for byte (local-only paths excluded).
  4. every file path a stage README names exists in that stage.
  5. no caches, secrets or generated data inside any stage.
With --run: runs ruff at the root, and ruff, pytest and the import-location check inside
every stage. Run it with a Python that has only the dependencies installed: an installed
copy of the project (pip install -e of the root or of a v folder) adds its folder to
backend's namespace path, and every stage then fails the import-location check.
"""

from __future__ import annotations

import difflib
import filecmp
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from peel import project_files

FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".env", "data", "outputs",
                   ".venv", "wandb", "build", "dist"}
STAGE_RE = re.compile(r"^v(\d+)$")
CACHE_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
VENV_PY = Path(sys.executable)  # run this script with the Python that has the deps
RUFF = Path(shutil.which("ruff") or Path(sys.executable).with_name("ruff"))


def stage_dirs(root: Path) -> list[Path]:
    dirs = [p for p in root.iterdir() if p.is_dir() and STAGE_RE.match(p.name)]
    return sorted(dirs, key=lambda p: int(STAGE_RE.match(p.name).group(1)))


def files_of(folder: Path, exclude_top: set[str] = frozenset()) -> dict[str, Path]:
    out = {}
    for dirpath, dirnames, filenames in os.walk(folder):
        rel_dir = Path(dirpath).relative_to(folder)
        if exclude_top:
            dirnames[:] = [d for d in dirnames if d not in CACHE_DIRS]
        if rel_dir == Path("."):
            dirnames[:] = [d for d in dirnames if d not in exclude_top and not STAGE_RE.match(d)]
        for name in filenames:
            rel = (rel_dir / name).as_posix()
            if rel_dir == Path(".") and name in exclude_top:
                continue
            out[rel.removeprefix("./")] = Path(dirpath) / name
    return out


ISOLATION_PROBE = ("import json, backend, backend.config as c; "
                   "print(json.dumps([c.__file__, list(backend.__path__)]))")


def isolation_problem(stage: Path, python, env: dict) -> str | None:
    """None if backend, and every part of its namespace path, comes from inside `stage`."""
    loc = subprocess.run([str(python), "-c", ISOLATION_PROBE], cwd=stage, env=env,
                         capture_output=True, text=True)
    stage = stage.resolve()

    def inside(p: str) -> bool:
        return os.path.isabs(p) and Path(p).resolve().is_relative_to(stage)

    if loc.returncode == 0:
        cfg, paths = json.loads(loc.stdout)
        if inside(cfg) and all(inside(p) for p in paths):
            return None
    return (f"{stage.name}: backend imports from outside the folder (use a Python with only "
            f"the dependencies, no pip install of the project): {loc.stdout}{loc.stderr}")


def same(a: Path, b: Path) -> bool:
    return filecmp.cmp(a, b, shallow=False)


LINE_RULE_FROM = 50  # stages before this were built under the 5-file rule
MIN_LINES, MAX_LINES = 150, 300


def lines_of(path: Path | None) -> list[str]:
    if path is None:
        return []
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:  # a binary file counts as one changed line
        return [path.read_bytes().hex()[:64]]


def changed_lines(prev: dict[str, Path], cur: dict[str, Path], changed: list[str]) -> int:
    """Added plus removed lines over the changed files, README.md left out."""
    total = 0
    for rel in changed:
        if rel == "README.md":
            continue
        diff = difflib.unified_diff(lines_of(prev.get(rel)), lines_of(cur.get(rel)), lineterm="",
                                    n=0)
        total += sum(1 for line in diff if line[:1] in "+-" and line[:3] not in ("+++", "---"))
    return total


PATH_TOKEN = re.compile(r"`([A-Za-z0-9_.\-/]+\.[A-Za-z0-9]+)`")


def readme_paths(readme: Path) -> set[str]:
    """Backticked tokens that look like repo file paths (contain a slash or a known name)."""
    names = set()
    for token in PATH_TOKEN.findall(readme.read_text(encoding="utf-8")):
        if token.startswith(("http", "ws:", "-")) or token.count(".") > 3:
            continue
        if "/" in token or token in {"README.md", "pyproject.toml", "CLAUDE.md", "PROGRESS.md",
                                     "LICENSE", ".gitignore", ".gitattributes", ".env.example",
                                     "CONTRIBUTIONS.md"}:
            names.add(token)
    return names


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    run = "--run" in sys.argv
    stages = stage_dirs(root)
    problems: list[str] = []
    global ALL_BASENAMES
    ALL_BASENAMES = {Path(r).name for r in project_files(root)}
    ALL_BASENAMES |= {"VERSIONS.md", "SPEC.md"}
    ALL_BASENAMES -= {"README.md"}
    prev: dict[str, Path] | None = None
    print(f"{len(stages)} stages: {', '.join(s.name for s in stages)}")
    for stage in stages:
        for rel in files_of(stage):  # everything on disk, ignored files included
            parts = set(Path(rel).parts)
            if parts & FORBIDDEN_PARTS or rel.endswith((".pyc", ".egg-info")) or ".egg-info/" in rel:
                problems.append(f"{stage.name}: forbidden file {rel}")
        cur = {rel: stage / rel for rel in project_files(root, stage.name)}
        if prev is None:
            changed = sorted(cur)
            if len(cur) > 5:
                problems.append(f"{stage.name}: has {len(cur)} files (max 5)")
        else:
            missing = sorted(set(prev) - set(cur))
            if missing:
                problems.append(f"{stage.name}: drops files from previous stage: {missing}")
            changed = sorted(r for r in cur if r not in prev or not same(cur[r], prev[r]))
            number = int(STAGE_RE.match(stage.name).group(1))
            if number >= LINE_RULE_FROM:
                count = changed_lines(prev, cur, changed)
                print(f"{stage.name}: {count} lines changed (README left out)")
                if not MIN_LINES <= count <= MAX_LINES:
                    problems.append(f"{stage.name}: changes {count} lines "
                                    f"({MIN_LINES} to {MAX_LINES} allowed)")
            elif len(changed) > 5:
                problems.append(f"{stage.name}: adds or changes {len(changed)} files (max 5)")
        added = [r for r in changed if prev is None or r not in prev]
        modified = [r for r in changed if prev is not None and r in prev]
        print(f"{stage.name}: {len(cur)} files; added {added}; changed {modified}")
        readme = stage / "README.md"
        if not readme.is_file():
            problems.append(f"{stage.name}: no README.md")
        else:
            for name in sorted(readme_paths(readme)):
                if not (stage / name).exists():
                    problems.append(f"{stage.name}: README names {name}, which is not in the folder")
            text = readme.read_text(encoding="utf-8")
            have = {Path(r).name for r in cur}
            for base in sorted(ALL_BASENAMES - have):
                if re.search(rf"(?<![\w/.-]){re.escape(base)}(?![\w-])", text):
                    problems.append(f"{stage.name}: README mentions {base}, which is not in the folder")
        prev = cur

    if stages:
        last = {rel: stages[-1] / rel for rel in project_files(root, stages[-1].name)}
        rootfiles = {rel: root / rel for rel in project_files(root)}
        only_root = sorted(set(rootfiles) - set(last))
        only_last = sorted(set(last) - set(rootfiles))
        differ = sorted(r for r in set(last) & set(rootfiles) if not same(last[r], rootfiles[r]))
        for label, items in (("only in root", only_root), ("only in last stage", only_last),
                             ("differ", differ)):
            if items:
                problems.append(f"last stage vs root, {label}: {items}")
        if not (only_root or only_last or differ):
            print(f"{stages[-1].name} matches the root byte for byte ({len(last)} files)")

    if run:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        root_lint = subprocess.run([str(RUFF), "check", "--no-cache", "."], cwd=root, env=env,
                                   capture_output=True, text=True)
        print(f"root: ruff {'ok' if root_lint.returncode == 0 else 'FAIL'}")
        if root_lint.returncode != 0:
            problems.append(f"root: ruff failed\n{root_lint.stdout}")
        for stage in stages:
            problem = isolation_problem(stage, VENV_PY, env)
            if problem:
                problems.append(problem)
            lint = subprocess.run([str(RUFF), "check", "--no-cache", "."], cwd=stage, env=env,
                                  capture_output=True, text=True)
            tests = subprocess.run([str(VENV_PY), "-m", "pytest", "-p", "no:cacheprovider"],
                                   cwd=stage, env=env, capture_output=True, text=True)
            summary = tests.stdout.strip().splitlines()[-1] if tests.stdout.strip() else tests.stderr
            print(f"{stage.name}: ruff {'ok' if lint.returncode == 0 else 'FAIL'}; pytest: {summary}")
            if lint.returncode != 0:
                problems.append(f"{stage.name}: ruff failed\n{lint.stdout}")
            if tests.returncode != 0:
                problems.append(f"{stage.name}: pytest failed\n{tests.stdout[-3000:]}")
            subprocess.run(["find", str(stage), "-name", "__pycache__", "-type", "d", "-prune",
                            "-exec", "rm", "-rf", "{}", "+"], check=True)

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(" -", p)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
