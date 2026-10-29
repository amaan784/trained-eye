"""Check the vN stage folders against the rules.

Usage: python check_stages.py <repo_root> [--run]

Checks:
  1. v1 has at most 5 files; every later stage adds or changes at most 5 files.
  2. every stage contains every file of the previous stage (no deletes or renames).
  3. the last stage matches the root byte for byte (local-only paths excluded).
  4. every file path a stage README names exists in that stage.
  5. no caches, secrets or generated data inside any stage.
With --run: runs ruff, pytest and the import-location check inside every stage.
"""

from __future__ import annotations

import filecmp
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

LOCAL_ONLY_TOP = {".git", ".claude", ".env", "data", "outputs", "VERSIONS.md", ".venv",
                  "stagetools"}
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


def same(a: Path, b: Path) -> bool:
    return filecmp.cmp(a, b, shallow=False)


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
    ALL_BASENAMES = {Path(r).name for r in files_of(root, exclude_top=LOCAL_ONLY_TOP)}
    ALL_BASENAMES |= {"VERSIONS.md", "SPEC.md"}
    ALL_BASENAMES -= {"README.md"}
    prev: dict[str, Path] | None = None
    print(f"{len(stages)} stages: {', '.join(s.name for s in stages)}")
    for stage in stages:
        cur = files_of(stage)
        for rel in cur:
            parts = set(Path(rel).parts)
            if parts & FORBIDDEN_PARTS or rel.endswith((".pyc", ".egg-info")) or ".egg-info/" in rel:
                problems.append(f"{stage.name}: forbidden file {rel}")
        if prev is None:
            changed = sorted(cur)
            if len(cur) > 5:
                problems.append(f"{stage.name}: has {len(cur)} files (max 5)")
        else:
            missing = sorted(set(prev) - set(cur))
            if missing:
                problems.append(f"{stage.name}: drops files from previous stage: {missing}")
            changed = sorted(r for r in cur if r not in prev or not same(cur[r], prev[r]))
            if len(changed) > 5:
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
        last = files_of(stages[-1])
        rootfiles = files_of(root, exclude_top=LOCAL_ONLY_TOP)
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
        for stage in stages:
            loc = subprocess.run(
                [str(VENV_PY), "-c",
                 "import backend.config as c, backend; print(c.__file__); print(list(backend.__path__))"],
                cwd=stage, env=env, capture_output=True, text=True,
            )
            if loc.returncode != 0 or str(stage) not in loc.stdout or str(root / "backend") + "'" in loc.stdout:
                problems.append(f"{stage.name}: backend imports from outside the folder: {loc.stdout}{loc.stderr}")
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
