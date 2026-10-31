"""Run every stage's README commands inside that stage and write its README.

Usage: python stagetools/render_readmes.py <repo_root> <stages module> <milestone label>

Only the stages defined in the stages module are rendered; other folders are left alone.
The README of the last stage is also written to the repo root (they must match).
Measured outputs are merged into stagetools/measured.json for VERSIONS.md.
"""

from __future__ import annotations

import filecmp
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_stages import isolation_problem  # noqa: E402
from peel import project_files  # noqa: E402

ROOT = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("stages", sys.argv[2])
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
MILESTONE = sys.argv[3]
TOOLS = Path(__file__).resolve().parent
VENV_BIN = Path(sys.executable).parent  # commands run with this Python first on PATH
STAGE_RE = re.compile(r"^v(\d+)$")
CACHES = ("__pycache__", ".pytest_cache", ".ruff_cache")


def clean_env() -> dict:
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("TE_") or k in ("MODEL_API_KEY", "HF_TOKEN", "WANDB_API_KEY",
                                                  "PYTHONPATH", "WANDB_ENTITY"))}
    env["PATH"] = f"{VENV_BIN}:{env['PATH']}"
    env["NO_COLOR"] = "1"
    return env


def files_of(folder: Path) -> dict[str, Path]:
    """The stage's project files: what git tracks or would add there (nothing ignored)."""
    return {rel: folder / rel for rel in project_files(ROOT, folder.name)}


def scrub(folder: Path) -> None:
    for name in CACHES:
        for d in folder.rglob(name):
            shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(folder / "data", ignore_errors=True)


def run(cmd: str, cwd: Path) -> tuple[str, int]:
    out = subprocess.run(cmd, shell=True, cwd=cwd, env=clean_env(), capture_output=True,
                         text=True, timeout=300)
    return (out.stdout + out.stderr).rstrip("\n"), out.returncode


def serve_and_curl(cwd: Path) -> list[tuple[str, str, int]]:
    log = cwd / ".serve.log"
    with log.open("w") as fh:
        proc = subprocess.Popen("exec python -m backend.cli serve --port 8765", shell=True,
                                cwd=cwd, env=clean_env(), stdout=fh, stderr=subprocess.STDOUT)
    try:
        body = ""
        for _ in range(50):
            time.sleep(0.2)
            body, code = run("curl -s http://127.0.0.1:8765/health", cwd)
            if code == 0 and body:
                break
    finally:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
    first = log.read_text().splitlines()[0]
    log.unlink()
    return [("python -m backend.cli serve", first, 0),
            ("curl -s http://127.0.0.1:8765/health    # in a second terminal", body, code)]


def measure(folder: Path, commands: list[str]) -> list[tuple[str, str, int]]:
    problem = isolation_problem(folder, sys.executable, clean_env())
    if problem:
        sys.exit(problem)
    results = []
    for cmd in commands:
        if cmd == "SERVE_AND_CURL":
            results += serve_and_curl(folder)
        elif cmd == "MODAL_NOT_RUN":
            results.append(("modal run backend/training/wandb_smoke.py", None, None))
        else:
            out, code = run(cmd, folder)
            results.append((cmd, out, code))
    scrub(folder)
    return results


def describe_change(rel: str, added: bool) -> str:
    for path, text in S.COMPONENTS:
        if path == rel:
            return text
    if rel == "README.md":
        return "this file: what the stage adds and how to try it"
    if rel.startswith("tests/"):
        return "tests"
    return {
        "pyproject.toml": "requirements and tool settings",
        ".github/workflows/ci.yml": "CI: ruff and pytest on every push",
        ".gitignore": "keeps secrets, local data and outputs out of git",
        ".env.example": "template for the keys (copy to .env)",
        "LICENSE": "from the initial commit, unchanged",
        ".gitattributes": "from the initial commit, unchanged",
    }.get(rel, "")


def block(lines: list[str]) -> str:
    return "\n".join("    " + ln if ln else "" for ln in lines)


def render(k: int, n: int, folder: Path, changed: list[tuple[str, bool]],
           results: list, tests_count: str) -> str:
    st = S.STAGES[k]
    present = files_of(folder)
    out = ["# Trained Eye", "", "Teach your glasses to notice anything.", "", S.INTRO, ""]
    out += [f"## Build stage v{k} ({MILESTONE}): {st['layer']}", "", wrap(st["adds"]), ""]
    out += ["Files added or changed in this stage:", ""]
    for rel, added in changed:
        out.append(wrap_item(f"- `{rel}` ({'new' if added else 'changed'}): "
                             f"{describe_change(rel, added)}"))
    out += ["", "## Setup", "", "Run everything from inside this folder.", ""]
    extra = '".[dev,train]"' if "backend/training/wandb_smoke.py" in present else '".[dev]"'
    venv = getattr(S, "VENV", "python3 -m venv .venv && source .venv/bin/activate")
    setup = [venv, f"pip install -e {extra}"]
    if ".env.example" in present:
        setup.append("cp .env.example .env    # then fill in the keys")
    out += [block(setup), ""]
    out += ["## Try it", "",
            wrap(f"These commands were run inside this folder on {st.get('date', '2026-10-29')} "
                 "(Python 3.13, Linux, no keys set and no network to Meta); the output under "
                 "each is what they printed."), ""]
    for cmd, text, code in results:
        if text is None:
            out += [block([f"$ {cmd}"]), "",
                    "Not run here: it needs your Modal token (`modal token set`) and a Modal "
                    "Secret named wandb holding WANDB_API_KEY. It prints the URL of the logged "
                    "run.", ""]
            continue
        lines = [f"$ {cmd}"] + text.splitlines()
        if code not in (0, None):
            lines.append(f"(exit code {code})")
        out += [block(lines), ""]
    if st.get("note"):
        out += [wrap(st["note"]), ""]
    if "backend/cli.py" in present:
        out += ["After `pip install -e .`, `te <command>` is the same as "
                "`python -m backend.cli <command>`.", ""]
    out += ["## What is in the project so far", ""]
    for path, text in S.COMPONENTS:
        if path in present:
            out.append(wrap_item(f"- `{path}`: {text}."))
    out.append(f"- `tests/`: {tests_count}, all offline; none calls a paid API.")
    out += ["", "## Next", ""]
    if k + 1 in S.STAGES:  # the last stage of a module gets the module's closing text
        nxt = S.STAGES[k + 1]
        out.append(wrap(f"Stage v{k + 1} adds {nxt['next']}"))
    else:
        out.append(wrap(S.NEXT_M0_DONE))
    out += ["", "## Privacy and policy", "", S.POLICY, ""]
    return "\n".join(out)


def first_sentence(text: str) -> str:
    text = re.sub(r"\b(te [a-z-]+)\b", r"\1", text)
    return text.split(". ")[0].rstrip(".") + "."


def wrap_item(text: str) -> str:
    return wrap(text).replace("\n", "\n  ")


def wrap(text: str, width: int = 86) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def main() -> None:
    stages = sorted((p for p in ROOT.iterdir() if p.is_dir() and STAGE_RE.match(p.name)),
                    key=lambda p: int(p.name[1:]))
    first_k = int(stages[0].name[1:])
    n = int(stages[-1].name[1:])
    measured = {}
    prev = None
    if first_k > 1:
        prev = files_of(ROOT / f"v{first_k - 1}")
    for folder in stages:
        k = int(folder.name[1:])
        if k not in S.STAGES:
            prev = files_of(folder)
            continue
        cur = files_of(folder)
        changed = []
        for rel in sorted(cur):
            if rel == "README.md":
                continue
            if prev is None or rel not in prev:
                changed.append((rel, True))
            elif not filecmp.cmp(cur[rel], prev[rel], shallow=False):
                changed.append((rel, False))
        changed.append(("README.md", prev is None))
        results = measure(folder, S.STAGES[k]["commands"])
        pytest_out = next(t for c, t, _ in results if c == S.PYTEST)
        count = re.search(r"(\d+) passed", pytest_out)
        tests_count = f"{count.group(1)} tests" if count else "tests"
        text = render(k, n, folder, changed, results, tests_count)
        (folder / "README.md").write_text(text, encoding="utf-8")
        measured[folder.name] = {"layer": S.STAGES[k]["layer"], "changed": changed,
                                 "results": results, "tests": tests_count}
        print(f"{folder.name}: README written ({len(changed)} files added or changed)")
        prev = files_of(folder)
    shutil.copyfile(stages[-1] / "README.md", ROOT / "README.md")
    old = {}
    path = TOOLS / "measured.json"
    if path.exists():
        old = json.loads(path.read_text())
    old.update(measured)
    path.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
