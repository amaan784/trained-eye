"""Write VERSIONS.md at the repo root from the stage metadata and measured.json."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
TOOLS = Path(__file__).resolve().parent
MODULES = sys.argv[2:]  # stage modules, in milestone order: path:label


def load(path: str):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def short(stage: dict, k: int) -> str:
    if k == 1:
        return "Project skeleton and the settings module."
    text = stage["next"].rstrip(".")
    if text.startswith("te "):  # a command name keeps its case
        return text + "."
    return text[0].upper() + text[1:] + "."


def try_cmd(stage: dict) -> str:
    cmds = [c for c in stage["commands"] if c not in ("python -m pytest", "ruff check .")]
    if not cmds:
        return "`python -m pytest`"
    c = cmds[0]
    if c == "SERVE_AND_CURL":
        return "`python -m backend.cli serve`"
    if c == "MODAL_NOT_RUN":
        return "`modal run backend/training/wandb_smoke.py` (needs Modal)"
    if c.startswith("python -c"):
        return "`python -m pytest`, then the `python -c` line in its README"
    return f"`{c}`"


def main() -> None:
    measured = json.loads((TOOLS / "measured.json").read_text())
    rows, commits = [], []
    all_stages = []
    for spec in MODULES:
        path, label = spec.split(":")
        mod = load(path)
        for k, st in sorted(mod.STAGES.items()):
            all_stages.append((k, label, st))
    last = max(k for k, _, _ in all_stages)
    files = [p for p in (ROOT / f"v{last}").rglob("*") if p.is_file()]
    largest = max(files, key=lambda p: p.stat().st_size)
    total = sum(p.stat().st_size for p in files)
    for k, label, st in all_stages:
        m = measured[f"v{k}"]
        files = ", ".join(f"`{rel}`" + ("" if added else " (changed)") for rel, added in m["changed"])
        tests = m["tests"].replace(" tests", " passed")
        rows.append(f"| v{k} | {label}: {st['layer']} | {short(st, k)} | {files} | {try_cmd(st)} | {tests} |")
        commits.append(f"**v{k}.** `{st['summary']}`\n\n{st['paragraph']}\n")

    text = f"""# Version stages

The finished project split into stages v1 to v{last}, in the order a developer would build
it, so it can be reviewed and committed one layer at a time. These folders and this file
live only on the `build-stages` branch; they are not part of the project itself.

## Ground rules

- Each stage is one commit. Up to v30 a stage adds or changes at most 5 files compared
  with the stage before it (v1 has 5 files in total; the README counts). From v31 on, the
  rule is by size instead: a stage changes 200 to 400 lines of code compared with the
  stage before it, counting added plus removed lines. Its tests, docs, license texts and
  binary files (fonts, the mock video) ride along with the code and are not counted, nor
  is its README.
- Stages are cumulative: vN holds everything in v(N-1) plus what stage N adds.
- No stage deletes or renames a file the previous stage had, so each folder can be copied
  over the repository and committed.
- Every folder runs on its own: it has its own README, `pyproject.toml` (requirements and
  tool settings), tests and CI workflow, and nothing in it points into another folder.
- The last folder, v{last}, is the finished project: byte for byte the same as the
  repository root, apart from local-only things (this file, the v folders,
  `stagetools/`, caches, `.env`, `data/`).
- Folders hold code and docs only: no caches, generated data or secrets.

## Stages

| Stage | Layer | What it adds | Files added or changed | How to try it | Tests |
|---|---|---|---|---|---|
{chr(10).join(rows)}

Every README lists the exact commands and the output they printed when run inside that
folder.

## Why this order

Settings come first because every other module reads them. The cache and the spend
ledger come next, before anything can call the API, so no paid call can ever be made
without them. The `te` command arrives with the ledger report so there is something to
run early. The API client then builds on all three, core calls before the media calls,
whose request shapes are the least certain. The commands that use a real key follow the
client they need. The WebSocket server is independent of the API, so it comes after the
API layers rather than interrupting them. The Modal and W&B script needs nothing else in
the project. Docs come last: the glasses guide, then the project notes that describe the
finished milestone.

M1 (v12 to v21) follows the teach flow from its foundations up. The skill spec and the
policy check come first because everything else consumes a spec, and refusals must
exist before any model is asked. The compiler then fills a spec from a request. The
building blocks that need no API (keyframes, the dataset writer) come before the paid
steps that use them; the paid steps follow the order they run in (SAM, Muse Image, Spark
labels). The pipeline that chains them comes next, then the command that runs it, and
the recording protocol and notes last.

v22 comes after M1 and holds no code: the handover for the next session and the setup
guide, written once M1 was finished so they describe the whole project.

v23 to v30 hold the fixes from the review after M1 (bugs, version compatibility,
functionality against the spec, the UI). They follow the same order as the layers they
fix: settings, the API client, the policy check and compiler, keyframes, the data
engine, the teach flow, te teach, then the docs. Each changes only files the earlier
stages already have, so `git diff --no-index` against the stage before shows exactly
one fix.

From v31 on each stage adds finished files whole, 200 to 400 lines of code, so a file
is written once in its final form, with the later fixes to it already in. v31 to v40
come first because the rest stands on them: the settings and the spend ledger every
later stage reads, the policy check before anything that compiles a skill, the Meta
client the data engine calls, SAM's boxes, then the watch trigger and the student
scorer, the training run records, the watcher, and the training bundle, preflight and
Modal app in the order a run takes.

The rest of M2 (v41 to v63) follows: the teach flow and te teach, te train, then
teaching from the phone from the inside out (the session with its two questions, the
upload and the routes, the server that mounts them), te watch, export and serving on
Modal with their commands. The iOS app (v49 to v63) goes from the data up: settings and
the protocol, the network clients, Meta's copied decoder, the glasses and their camera,
the iPhone camera and speech, the request recorder, the teach client and flow, the app
model, then the look, the app entry and the screens, and the clips with the build
instructions last. It is never compiled here, so its stages have no tests of their own.

M3 (v64 to v77) starts with the labelling tools, since every metric needs labels, then
the metrics and chart pieces, the charts, Spark as a scorer and the regression check,
the write-up, the failure galleries and the runner that joins them, te eval, then each
experiment's own code: Spark's error rate with the session clips, the experiment plan,
the plan file with te experiments, the replay engine with the W&B report, and te replay.
M4 (v78 to v92) follows the release: face blurring before anything that publishes, the
model card, the release folder, regeneration and the weights, the test set and the
release commands, then the playground from its data and caps through the live runs and
the page to the app, its commands, and the notes and credits last.

A file whose later parts need a later module arrives in parts: backend/cli.py gets each
command module's registration with the stage that adds that module, and a few tests
arrive with the stage whose module they need (each stage's README says which).

## How to review a stage

Clone the branch next to your repository, then compare any two neighbouring folders:

    git clone --branch build-stages --single-branch https://github.com/amaan784/trained-eye ~/te-stages
    cd ~/te-stages
    git diff --no-index v4 v5            # everything stage 5 adds
    git diff --no-index --stat v4 v5     # just the list of files

Inside a folder, the README says what the stage adds and how to try it. To run it:

    cd v5
    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    python -m pytest

(Remove the `.venv` folder afterwards, or create it outside the stage folder, so it is not
copied into the repository.)

## How to commit the stages

Work in your own clone of the repository on `main`, with a clean working tree. For each
stage in order, copy the folder over the repository and commit it with the summary and
description below:

    cd ~/trained-eye                       # your repository, on main
    rsync -a ~/te-stages/v1/ ./            # copy v1 over the repo; nothing is deleted
    git add -A
    git commit -m "<summary>" -m "<description>"

Then the same for v2, v3 and so on. Earlier stages never change when later ones are
added, so carry on from the first stage you have not committed yet. `git status` before each commit should show only the files
listed for that stage. LICENSE and .gitattributes already exist in the repository
from its first commit, and the stage copies are identical, so v10 changes only its own
files. After the last stage, `main` matches the root of the `build-stages` branch, and
you can push it.

{chr(10).join(commits)}
## Notes

- **Reconstructed stages.** Each milestone was written as one finished tree, and its
  stages were then made from it by removing layers with exact edits (whole files, or
  exact blocks such as the media part of the client), working backwards. No code outside
  a layer was rewritten. M1's stages were peeled from the finished M1 tree, and peeling
  v12 once more gives v11 byte for byte (README aside), so M1 sits exactly on M0; the
  same holds for v22 on v21 and v23 on v22. From v31 on the stages were built forward
  instead, from v30: each adds finished files whole, and a file that arrives in parts is
  the finished file with its later parts left out (`stagetools/forward.py` with
  `stagetools/specs/forward_v41.py` from v41; `stagetools/specs/pushed_main.md` lists
  what v31 to v40 hold). The last folder was checked to equal the repository root. Each
  folder was then tested as its own project. The READMEs were written per stage, from
  commands run inside each folder.
- **Files from the first commit.** `.gitignore` enters at v2 as the original Python
  template plus Trained Eye's own entries. `LICENSE` (MIT, the original) and
  `.gitattributes` enter unchanged at v10, so the last folder matches the repository.
- **Branch-only tools.** `stagetools/` holds the scripts that built and checked these
  folders (see `stagetools/README.md`). Like this file, it lives only on this branch.
- **Large files.** None. The largest file is `{largest.relative_to(ROOT / f"v{last}")}` at
  about {largest.stat().st_size // 1024} KB; the whole of v{last} is about {total // 1024} KB.
- **Untested paths.** The build machine had no API keys, no network access to Meta, and
  no PyPI, so these were not run:
  - live Meta API calls (`te models`, `te api-smoke`); all client tests use
    `httpx.MockTransport`. The SAM, Muse Image and Voice Transcribe request shapes come
    from the build spec and still need a live call to confirm.
  - `modal run backend/training/wandb_smoke.py` (Modal and W&B are not installed there);
    its tests only parse the script.
  - the WebSocket under `te serve` (uvicorn there had no websockets package); `/health`
    was checked under `te serve`, and the WebSocket in process (`te ws-selftest`, tests).
  - `pip install -e .`: the build machine's system setuptools (a Debian build) fails on
    every package with `AttributeError: install_layout`. The `te` entry point was checked
    instead by calling `backend.cli:main` from each stage's `[project.scripts]`.
  - the GitHub Actions workflow itself; the same `ruff check .` and `pytest` were run in
    every folder (GitHub ran it on the branch root after each push).
  - `te serve` on a real network: its phone address and log setup were tested with
    uvicorn's run replaced.
  - training on Modal (M2): the Modal app was read by its tests, not run, and `te train`
    was tested with the modal command replaced. The CUDA image tag and the H100 price
    are unchecked.
  - the student endpoint: no trained model exists yet, so the scorer, `te watch` and
    watch mode were tested against fake endpoints.
  - Python 3.11 and 3.12, and macOS: everything ran on Python 3.13 on Linux. For the
    review after M1 the code was also compiled with Python 3.11 and 3.12 and type-checked
    against the 3.11 standard library, but not run there.
  - the glasses guide: written from Meta's SDK docs and samples; nobody has followed it
    on real glasses yet.
  - the teach flow against the real API (M1): every test uses a fake Meta API, so the
    first real washer run on the Mac is the first time Muse Wonder, SAM, Muse Image and
    Spark see these prompts. SAM's text is read for its boxes (masks are not decoded).
  - the iPhone app: never compiled (no Swift toolchain on the build machine); it needs
    Xcode and the glasses (ios/README.md lists the checks).
  - ffmpeg on the Mac: video decoding was tested with ffmpeg on Linux.

## How the folders were checked

A script (kept outside the repository) checked every folder after it was built:

1. v1 has at most 5 files; v2 to v30 each add or change at most 5 files compared with
   the one before (README included); from v31 each changes 200 to 400 lines of code
   (tests, docs, license texts, binaries and the README left out).
2. Every folder contains every file of the folder before it.
3. v{last} matches the repository root byte for byte (local-only things excluded).
4. Every file a README names, in backticks or as a bare file name, exists in that folder.
5. No folder holds caches, `.env`, `data/`, build output or secrets.
6. Inside every folder: `ruff check .` and `python -m pytest` pass, and `backend` imports
   from that folder, not from another one.
7. Every README command was run inside its folder with no keys or `TE_*` settings in the
   environment; the outputs in the READMEs are pasted from those runs.
8. From v4 on, the `te` entry point in `pyproject.toml` was called inside each folder.
   For M1 and the reviews after it, peeling the first stage once more was checked to give
   the stage before it exactly (v12 to v11, v22 to v21, v23 to v22); from v41 the forward
   build was checked to give v40 back for every file that arrives in parts.
9. The READMEs and this file contain no emoji or em dashes.
"""
    (ROOT / "VERSIONS.md").write_text(text, encoding="utf-8")
    print("VERSIONS.md written,", len(rows), "stages")


if __name__ == "__main__":
    main()
