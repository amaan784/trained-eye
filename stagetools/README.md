# stagetools

Scripts that build and check the stage folders v1 to vN on the `build-stages` branch.
Like VERSIONS.md and the v folders, this folder exists only on that branch; it is never
copied into a stage and never committed to `main`. Run everything from the repository
root, with a Python that has the project's dependencies, pytest and ruff (see SETUP.md).

| File | What it does |
|---|---|
| `peel.py` | Builds new stage folders from the finished code at the root, peeling one layer per stage with exact edits, and proves the first new stage sits exactly on the last committed one |
| `specs/peel_*.py` | The layers of each milestone: what each stage added. `peel_m0.py` and `peel_m1.py` are historical records of how v1 to v21 were built; do not rerun them, they assume the root was at that milestone |
| `specs/stages_*.py` | README and VERSIONS.md text for each stage: layer, what it adds, the commands its README runs, commit summary and description |
| `render_readmes.py` | Runs each stage's README commands inside that folder with no keys or `TE_*` settings, and writes the README around the measured output; the last stage's README is also written to the root |
| `versions_md.py` | Writes VERSIONS.md from the stage metadata and `measured.json` |
| `check_stages.py` | Checks the rules: at most 5 files per stage up to v49, 150 to 300 changed lines per stage from v50 (README left out), every folder contains the previous one, the last folder matches the root, README file mentions exist, no caches or secrets, and (with `--run`) ruff and pytest pass inside every folder |
| `measured.json` | The measured README command outputs for every stage, used by VERSIONS.md |

## Adding the stages for a new milestone

1. Finish the milestone at the root, with tests passing.
2. Write `specs/peel_<milestone>.py`: `BASE` is the newest existing stage, `FIRST` and
   `LAST` the new stage numbers, and `PEEL[k]` undoes what stage k adds (files to remove,
   files to restore from `v<BASE>`, and exact edits). Keep every stage from v50 at 150 to
   300 changed lines (added plus removed, README left out; v1 to v49 were cut at 5 files
   or fewer, README included), and order the layers foundations first, docs and UI last.
3. Write `specs/stages_<milestone>.py` with the text for each new stage (copy the shape
   of `stages_m1.py`; it imports the shared intro, policy text and file descriptions).
4. Build, render, write VERSIONS.md and check:

       python stagetools/peel.py . stagetools/specs/peel_<milestone>.py
       python stagetools/render_readmes.py . stagetools/specs/stages_<milestone>.py <label>
       python stagetools/versions_md.py . \
         stagetools/specs/stages_m0.py:M0 stagetools/specs/stages_m1.py:M1 \
         stagetools/specs/stages_v22.py:M1 stagetools/specs/stages_<milestone>.py:<label>
       python stagetools/check_stages.py . --run

   `peel.py` rewrites only the new folders. If it reports that the first new stage does
   not peel back to `v<BASE>`, the peel edits or the root are wrong; fix them, never the
   committed folders (unless carrying a real bug fix into every stage that has the code,
   which the author asked for and must be told about).
5. Remove `__pycache__` folders, check `git status`, then commit with the author's
   identity and push (SETUP.md section 5).

The check script and the renderer use the Python they are run with and look for `ruff`
on the PATH. The renderer runs commands such as `python -m pytest` inside each folder;
make sure `python` on the PATH is that same environment (activate the venv first).
