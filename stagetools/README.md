# stagetools

Scripts that build and check the stage folders v1 to vN on the `build-stages` branch.
Like VERSIONS.md and the v folders, this folder exists only on that branch; it is never
copied into a stage and never committed to `main`. Run everything from the repository
root, with a Python that has the project's dependencies, pytest and ruff (see SETUP.md).

| File | What it does |
|---|---|
| `peel.py` | Builds new stage folders from the finished code at the root, peeling one layer per stage with exact edits, and proves the first new stage sits exactly on the last committed one (how v1 to v30 were built) |
| `forward.py` | Builds stage folders forward from a committed stage, adding finished files whole; a file that arrives in parts is the finished file with its later pieces undone. Checks that the base stage's copies match and that the last stage equals the root (how v41 to v92 were built) |
| `specs/forward_v41.py` | v41 to v92: each file's first stage, and the pieces of `backend/cli.py` and of a few test files that arrive later. `specs/pushed_main.md` lists what v31 to v40 hold |
| `specs/peel_*.py` | The layers of each milestone: what each stage added. `peel_m0.py` and `peel_m1.py` are historical records of how v1 to v21 were built; do not rerun them, they assume the root was at that milestone |
| `specs/stages_*.py` | README and VERSIONS.md text for each stage: layer, what it adds, the commands its README runs, commit summary and description (`stages_c.py` for v31 to v92; `stages_c_m2.py`, `_m3.py` and `_m4.py` split it by milestone for VERSIONS.md) |
| `render_readmes.py` | Runs each stage's README commands inside that folder with no keys or `TE_*` settings, and writes the README around the measured output; the last stage's README is also written to the root |
| `versions_md.py` | Writes VERSIONS.md from the stage metadata and `measured.json` |
| `check_stages.py` | Checks the rules: at most 5 files per stage up to v30, 200 to 400 changed lines of code per stage from v31 (tests, docs, license texts, binaries and the README left out), every folder contains the previous one, the last folder matches the root, README file mentions exist, no caches or secrets, and (with `--run`) ruff and pytest pass inside every folder |
| `measured.json` | The measured README command outputs for every stage, used by VERSIONS.md |

## Adding the stages for a new milestone

1. Finish the milestone at the root, with tests passing.
2. Write a forward spec like `specs/forward_v41.py`: `BASE` is the newest committed stage,
   `FILES` maps each finished file to the stage it arrives in, and `PIECES` holds the
   parts of a file that arrive later (a text piece reads as its earlier text until its
   stage; a test piece leaves a test function out until its stage). Keep every stage at
   200 to 400 changed lines of code (added plus removed; tests, docs, license texts,
   binaries and the README ride along), put each file after everything it imports, and
   hold back a test until the stage whose module it needs. (v1 to v30 were peeled back
   from the finished tree with `peel.py` instead, at 5 files or fewer per stage.)
3. Write `specs/stages_<milestone>.py` with the text for each new stage (copy the shape
   of `stages_m1.py`; it imports the shared intro, policy text and file descriptions).
4. Build, render, write VERSIONS.md and check:

       python stagetools/forward.py stagetools/specs/forward_<first>.py
       TE_RENDER_STAGES=<first>-<last> python stagetools/render_readmes.py . \
         stagetools/specs/stages_<milestone>.py <label>
       python stagetools/versions_md.py . \
         stagetools/specs/stages_m0.py:M0 stagetools/specs/stages_m1.py:M1 \
         stagetools/specs/stages_v22.py:M1 stagetools/specs/stages_v23.py:"M1 review" \
         stagetools/specs/stages_c_m2.py:M2 stagetools/specs/stages_c_m3.py:M3 \
         stagetools/specs/stages_c_m4.py:M4
       python stagetools/check_stages.py . --run

   `forward.py` rewrites only the new folders. If it reports that the base stage differs
   from the pieces' text, the spec or the root is wrong; fix them, never the committed
   folders (unless carrying a real bug fix into every stage that has the code, which the
   author asked for and must be told about). `TE_RENDER_STAGES` limits the render to the
   new stages, so the READMEs already on `main` (with their measured timings) stay as
   they are; renders of different ranges can run side by side.
5. Remove `__pycache__` folders, check `git status`, then commit with the author's
   identity and push (SETUP.md section 5).

The check script and the renderer use the Python they are run with and look for `ruff`
on the PATH. The renderer runs commands such as `python -m pytest` inside each folder;
make sure `python` on the PATH is that same environment (activate the venv first).
