# Version stages

The finished project split into stages v1 to v49, in the order a developer would build
it, so it can be reviewed and committed one layer at a time. These folders and this file
live only on the `build-stages` branch; they are not part of the project itself.

## Ground rules

- Each stage is one commit and adds or changes at most 5 files compared with the stage
  before it (v1 has 5 files in total). The README counts.
- Stages are cumulative: vN holds everything in v(N-1) plus what stage N adds.
- No stage deletes or renames a file the previous stage had, so each folder can be copied
  over the repository and committed.
- Every folder runs on its own: it has its own README, `pyproject.toml` (requirements and
  tool settings), tests and CI workflow, and nothing in it points into another folder.
- The last folder, v49, is the finished project: byte for byte the same as the
  repository root, apart from local-only things (this file, the v folders,
  `stagetools/`, caches, `.env`, `data/`).
- Folders hold code and docs only: no caches, generated data or secrets.

## Stages

| Stage | Layer | What it adds | Files added or changed | How to try it | Tests |
|---|---|---|---|---|---|
| v1 | M0: Settings | Project skeleton and the settings module. | `.github/workflows/ci.yml`, `backend/config.py`, `pyproject.toml`, `tests/test_config.py`, `README.md` | `python -m backend.config` | 13 passed |
| v2 | M0: Response cache | The response cache: identical Meta API requests are answered from disk for free. | `.gitignore`, `backend/meta/cache.py`, `tests/test_cache.py`, `README.md` (changed) | `python -m pytest`, then the `python -c` line in its README | 21 passed |
| v3 | M0: Spend ledger | The spend ledger, with per-skill and global caps checked before every paid call. | `backend/meta/ledger.py`, `tests/test_ledger.py`, `README.md` (changed) | `python -m pytest`, then the `python -c` line in its README | 25 passed |
| v4 | M0: te command: spend | The te command line with a spend report, and test isolation from your shell settings. | `backend/cli.py`, `pyproject.toml` (changed), `tests/conftest.py`, `tests/test_ledger.py` (changed), `README.md` (changed) | `python -m backend.cli spend` | 26 passed |
| v5 | M0: Meta API client | The Meta API client: structured output, retries with backoff and conservative spend accounting. | `backend/meta/client.py`, `pyproject.toml` (changed), `tests/test_client.py`, `README.md` (changed) | `python -m pytest` | 50 passed |
| v6 | M0: Media endpoints | The media calls: Voice Transcribe, SAM 3.1 and Muse Image. | `backend/meta/client.py` (changed), `tests/test_client.py` (changed), `README.md` (changed) | `python -m pytest` | 60 passed |
| v7 | M0: Key check commands | Two commands to check a real key (one free, one under $0.01) and the key template. | `.env.example`, `backend/cli.py` (changed), `tests/test_cli.py`, `README.md` (changed) | `python -m backend.cli models` | 64 passed |
| v8 | M0: WebSocket frame server | The WebSocket backend that receives camera frames from the phone. | `backend/cli.py` (changed), `backend/server.py`, `pyproject.toml` (changed), `tests/test_server.py`, `README.md` (changed) | `python -m backend.cli ws-selftest` | 83 passed |
| v9 | M0: W&B smoke run on Modal | A Modal app that logs one Weights & Biases run. | `backend/training/wandb_smoke.py`, `pyproject.toml` (changed), `tests/test_wandb_smoke.py`, `README.md` (changed) | `modal run backend/training/wandb_smoke.py` (needs Modal) | 85 passed |
| v10 | M0: Glasses check | The guide for checking the glasses with Meta's sample app. | `.gitattributes`, `LICENSE`, `docs/glasses-check.md`, `README.md` (changed) | `python -m pytest` | 85 passed |
| v11 | M0: Project notes | The project notes: setup and conventions, and the progress log. | `CLAUDE.md`, `PROGRESS.md`, `README.md` (changed) | `python -m backend.cli spend` | 85 passed |
| v12 | M1: Skill spec and policy check | The skill spec, the local policy check with its refusals, and te policy. | `backend/cli.py` (changed), `backend/skills/spec.py`, `pyproject.toml` (changed), `tests/test_skills.py`, `README.md` (changed) | `python -m backend.cli policy "Tell me when Sam gets here"` | 151 passed |
| v13 | M1: Skill compiler | The skill compiler, which turns a request and 3 keyframes into a spec with Muse Wonder. | `backend/meta/client.py` (changed), `backend/skills/compiler.py`, `tests/test_compiler.py`, `README.md` (changed) | `python -m pytest` | 161 passed |
| v14 | M1: Keyframe picker | The keyframe picker for the 10 second teaching clip. | `backend/dataengine/keyframes.py`, `pyproject.toml` (changed), `tests/test_keyframes.py`, `README.md` (changed) | `ffmpeg -v error -y -f lavfi -i testsrc=duration=10:size=1280x720:rate=30 -pix_fmt yuv420p /tmp/te-demo.mp4` | 167 passed |
| v15 | M1: Dataset writer | The dataset writer in Meta's Axolotl recipe format. | `backend/dataengine/dataset.py`, `tests/test_dataset.py`, `README.md` (changed) | `python -m pytest` | 172 passed |
| v16 | M1: Region of interest (SAM 3.1) | The SAM 3.1 region-of-interest step. | `backend/dataengine/roi.py`, `tests/test_roi.py`, `README.md` (changed) | `python -m pytest` | 173 passed |
| v17 | M1: Synthetic set (Muse Image) | The synthetic set: Muse Image edits of the keyframes with a prompt manifest. | `backend/dataengine/synth.py`, `tests/test_synth.py`, `README.md` (changed) | `python -m pytest`, then the `python -c` line in its README | 175 passed |
| v18 | M1: Labels (Muse Spark) | Muse Spark labels for every image, with filtering and agreement stats. | `backend/dataengine/labels.py`, `tests/test_labels.py`, `README.md` (changed) | `python -m pytest` | 176 passed |
| v19 | M1: Teach flow | The teach flow end to end, with the fallback for training on real data only. | `backend/config.py` (changed), `backend/teach.py`, `tests/test_teach.py`, `README.md` (changed) | `python -m pytest` | 182 passed |
| v20 | M1: te teach | Te teach, which prints a cost estimate and asks before spending. | `backend/cli.py` (changed), `tests/test_cli.py` (changed), `README.md` (changed) | `ffmpeg -v error -y -f lavfi -i testsrc=duration=10:size=1280x720:rate=30 -pix_fmt yuv420p /tmp/te-demo.mp4` | 185 passed |
| v21 | M1: Recording protocol and notes | The recording protocol for real clips and the updated project notes. | `CLAUDE.md` (changed), `PROGRESS.md` (changed), `docs/recording-protocol.md`, `README.md` (changed) | `python -m backend.cli spend` | 185 passed |
| v22 | M1: Handover and setup guide | The handover and the setup guide for the next session. | `CLAUDE.md` (changed), `HANDOVER.md`, `PROGRESS.md` (changed), `SETUP.md`, `README.md` (changed) | `python -m backend.cli spend` | 185 passed |
| v23 | M1 review: Fix: settings | The settings fixes: a fallback switch that fails closed, quoted .env values, number checks, and data that belongs to the project folder. | `backend/config.py` (changed), `backend/meta/ledger.py` (changed), `tests/conftest.py` (changed), `tests/test_config.py` (changed), `README.md` (changed) | `python -m backend.config` | 206 passed |
| v24 | M1 review: Fix: API client | The client fixes: the spend caps are checked again before a retry, every served reply is recorded, and cached replies keep their cost. | `backend/meta/client.py` (changed), `backend/meta/ledger.py` (changed), `tests/test_client.py` (changed), `README.md` (changed) | `python -m backend.cli spend` | 212 passed |
| v25 | M1 review: Fix: policy check and compiler | The policy and compiler fixes: names in any script, fewer false refusals, and the user's correction of a wrong read-back. | `backend/skills/compiler.py` (changed), `backend/skills/spec.py` (changed), `tests/test_compiler.py` (changed), `tests/test_skills.py` (changed), `README.md` (changed) | `python -m backend.cli policy "Tell me when José gets home"` | 266 passed |
| v26 | M1 review: Fix: keyframes | The keyframe fixes: photos come out upright and ffmpeg leaves the terminal alone. | `backend/dataengine/keyframes.py` (changed), `tests/test_keyframes.py` (changed), `README.md` (changed) | `python -m pytest` | 268 passed |
| v27 | M1 review: Fix: data engine failures | The data engine fixes: one rejected image or label no longer stops the run. | `backend/dataengine/labels.py` (changed), `backend/dataengine/roi.py` (changed), `backend/dataengine/synth.py` (changed), `tests/test_labels.py` (changed), `README.md` (changed) | `python -m pytest` | 274 passed |
| v28 | M1 review: Fix: teach flow | The teach flow fixes: checked labels, keyframe names tied to their content, corrections and progress lines. | `backend/teach.py` (changed), `tests/test_teach.py` (changed), `README.md` (changed) | `python -m pytest` | 283 passed |
| v29 | M1 review: Fix: te teach | The te teach fixes: labels checked before the estimate, a cap warning, --correct, and clear stops. | `.env.example` (changed), `backend/cli.py` (changed), `tests/test_cli.py` (changed), `README.md` (changed) | `ffmpeg -v error -y -f lavfi -i testsrc=duration=10:size=1280x720:rate=30 -pix_fmt yuv420p /tmp/te-demo.mp4` | 297 passed |
| v30 | M1 review: Fix: docs and notes | The docs for these fixes: the glasses reachability check and the visual design rules. | `CLAUDE.md` (changed), `HANDOVER.md` (changed), `PROGRESS.md` (changed), `docs/glasses-check.md` (changed), `README.md` (changed) | `python -m backend.cli spend` | 297 passed |
| v31 | M1 review: Fix: API client, second round | The client fixes: Ctrl+C and failed downloads are charged, empty transcripts are not cached, stale cache entries are fetched again, and every failed call says what it cost. | `backend/meta/client.py` (changed), `tests/test_client.py` (changed), `README.md` (changed) | `python -m backend.cli spend` | 309 passed |
| v32 | M1 review: Fix: settings and spend report | The settings and spend report fixes: ~ in TE_DATA_DIR, and a spend table that lines up for long skill names. | `backend/config.py` (changed), `backend/meta/ledger.py` (changed), `tests/test_config.py` (changed), `tests/test_ledger.py` (changed), `README.md` (changed) | `python -m backend.config` | 310 passed |
| v33 | M1 review: Fix: policy check and compiler, second round | The policy and compiler fixes: named people and coaching that the first fixes let through, corrections that rule something out, and the read-back wording. | `backend/skills/compiler.py` (changed), `backend/skills/spec.py` (changed), `tests/test_compiler.py` (changed), `tests/test_skills.py` (changed), `README.md` (changed) | `python -m backend.cli policy "Tell me when Jake gets ready"` | 361 passed |
| v34 | M1 review: Fix: keyframes and generated images | The keyframe and image fixes: hidden macOS files, WebP and HEIC folders, EXIF turns, and a manifest that is written even when generation stops. | `backend/dataengine/keyframes.py` (changed), `backend/dataengine/synth.py` (changed), `tests/test_keyframes.py` (changed), `tests/test_synth.py` (changed), `README.md` (changed) | `python -m pytest` | 372 passed |
| v35 | M1 review: Fix: labels and region of interest | The label and SAM fixes: a label that needed the larger budget is free on a rerun, and SAM skips a rejected frame like the other steps. | `backend/dataengine/labels.py` (changed), `backend/dataengine/roi.py` (changed), `tests/test_labels.py` (changed), `tests/test_roi.py` (changed), `README.md` (changed) | `python -m pytest` | 377 passed |
| v36 | M1 review: Fix: teach flow and te teach | The teach flow and te teach fixes: inputs checked before the estimate, a worst-case figure, and datasets that must hold both answers. | `backend/cli.py` (changed), `backend/teach.py` (changed), `tests/test_cli.py` (changed), `tests/test_teach.py` (changed), `README.md` (changed) | `ffmpeg -v error -y -f lavfi -i testsrc=duration=10:size=1280x720:rate=30 -pix_fmt yuv420p /tmp/te-demo.mp4` | 400 passed |
| v37 | M1 review: Fix: CI and setup | CI and setup: ffmpeg in CI, Python 3.12 and ffmpeg in the Mac setup, and the W&B team note. | `.env.example` (changed), `.github/workflows/ci.yml` (changed), `CLAUDE.md` (changed), `README.md` (changed) | `python -m pytest` | 400 passed |
| v38 | M1 review: Fix: docs and notes, second round | The docs for these fixes: the estimator figures for a real clip, the recording protocol and the notes. | `HANDOVER.md` (changed), `PROGRESS.md` (changed), `docs/recording-protocol.md` (changed), `README.md` (changed) | `python -m backend.cli spend` | 400 passed |
| v39 | M1 review: Fix: client and settings, third round | The client and settings fixes from a review of v31 to v38: costs that are not numbers, and an unknown home folder. | `backend/config.py` (changed), `backend/meta/client.py` (changed), `tests/test_client.py` (changed), `tests/test_config.py` (changed), `README.md` (changed) | `python -m backend.cli spend` | 404 passed |
| v40 | M1 review: Fix: policy check and compiler, third round | The policy and compiler fixes from that review: two-name arrivals, titles, the glasses' light, and fewer false refusals. | `backend/skills/compiler.py` (changed), `backend/skills/spec.py` (changed), `tests/test_compiler.py` (changed), `tests/test_skills.py` (changed), `README.md` (changed) | `python -m backend.cli policy "Tell me when Sam and Jake get home"` | 442 passed |
| v41 | M2: Watch trigger and calibration | The watch trigger, 3 of the last 4 frames over the threshold, then the cooldown, and the threshold calibration. | `backend/watch/trigger.py`, `tests/test_trigger.py`, `README.md` (changed) | `python -m pytest` | 452 passed |
| v42 | M2: Student scorer | The student scorer: the trained model behind an OpenAI-compatible endpoint, read at its answer token. | `backend/inference/student.py`, `tests/test_student.py`, `README.md` (changed) | `python -m pytest` | 461 passed |
| v43 | M2: Watcher | The watcher: frames in, alerts out, with the Muse Spark check and a frame log. | `backend/watch/watcher.py`, `tests/test_watcher.py`, `README.md` (changed) | `python -m pytest` | 468 passed |
| v44 | M2: Watch mode on the server | Watch mode on the WebSocket: the phone names the skills, and the server sends scores and alerts. | `backend/config.py` (changed), `backend/server.py` (changed), `tests/test_config.py` (changed), `tests/test_server.py` (changed), `README.md` (changed) | `python -m backend.config` | 474 passed |
| v45 | M2: Training bundle and config | The training data: taught skills bundled for Modal, and the Axolotl config from Meta's recipe. | `backend/training/axolotl.py`, `backend/training/prepare.py`, `tests/test_training_prep.py`, `README.md` (changed) | `python -m pytest` | 481 passed |
| v46 | M2: Modal training app | The Modal training app: one H100, the recipe's pins, the volume and both Modal Secrets. | `backend/training/modal_train.py`, `tests/test_modal_train.py`, `README.md` (changed) | `python -m pytest` | 485 passed |
| v47 | M2: te train, te watch, te calibrate | The commands: te train on Modal, te watch to replay a clip, te calibrate for the threshold. | `CLAUDE.md` (changed), `backend/cli.py` (changed), `backend/config.py` (changed), `tests/test_cli.py` (changed), `README.md` (changed) | `python -m backend.cli train --help` | 496 passed |
| v48 | M2: Docs and notes | The notes for these stages: the review round, what M2 has so far, and the steps that need the Mac or Modal. | `.env.example` (changed), `HANDOVER.md` (changed), `PROGRESS.md` (changed), `docs/recording-protocol.md` (changed), `README.md` (changed) | `python -m backend.cli spend` | 496 passed |
| v49 | M1 review: Fix: te teach messages | The last small fixes from the review of v31 to v38, in te teach. | `backend/cli.py` (changed), `backend/teach.py` (changed), `tests/test_cli.py` (changed), `tests/test_teach.py` (changed), `README.md` (changed) | `python -m pytest` | 498 passed |

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

v31 to v38 hold the second half of that review (F30 to F62) and the faults a second
check found in the v23 to v30 fixes themselves, in the same order: the API client,
settings and the spend report, the policy check and compiler, keyframes and generated
images, labels and SAM, the teach flow with te teach, CI and setup, then the docs.
v39 and v40 hold what a review of v31 to v38 found: the client and settings, then the
policy check and compiler.

M2 (v41 to v48) holds the parts of the milestone that need neither the Mac nor Modal.
Watch mode is built from the decision outwards: when a skill fires, how a frame is
scored, the watcher that joins them, then the server that serves it to the phone. The
training side follows the order a run takes: the bundle and the config, the Modal app
that trains, then the commands that drive both, and the notes last. v49 comes after
them with the last small fixes from the review of v31 to v38, in te teach.

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
added, so carry on from the first stage you have not committed yet (v39 if v1 to v38
are already on `main`). `git status` before each commit should show only the files
listed for that stage. LICENSE and .gitattributes already exist in the repository
from its first commit, and the stage copies are identical, so v10 changes only its own
files. After the last stage, `main` matches the root of the `build-stages` branch, and
you can push it.

**v1.** `Add project skeleton and settings module`

Adds pyproject.toml with the test and lint setup, a GitHub Actions workflow that runs ruff and pytest, and backend/config.py: model IDs for every job, Meta's published prices, the $10 per-skill and $120 global spend caps, and keys read from the environment or .env without ever being printed. python -m backend.config shows the resolved settings.

**v2.** `Add content-addressed cache for API responses`

Adds backend/meta/cache.py. Responses are stored under data/cache, keyed by a SHA-256 of model, endpoint and normalized request with media replaced by its own hash, so reruns of the pipeline cost nothing. Binary outputs are written exactly as returned. Extends .gitignore so local data, outputs and copies of .env are never committed.

**v3.** `Add spend ledger with per-skill and global caps`

Adds backend/meta/ledger.py: cost helpers, an append-only ledger in data/ledger.jsonl, cap checks that raise BudgetExceeded before a call could push spend past $10 for one skill or $120 overall, and a report with totals by model and by skill.

**v4.** `Add te command with spend report`

Adds backend/cli.py and the te console script in pyproject.toml, with te spend printing the ledger totals. Adds tests/conftest.py, which keeps TE_* overrides and real keys in the developer's shell out of the tests.

**v5.** `Add Meta API client with cache, ledger and retries`

Adds backend/meta/client.py with structured() and list_models(), built on httpx. Each call goes through the cache, the spend caps and a retry loop for 429, 5xx and timeouts; 4xx errors fail at once. Refusals, empty or non-JSON outputs are billed but never cached. Tests use httpx.MockTransport, so nothing reaches the network.

**v6.** `Add transcription, SAM and Muse Image calls`

Extends backend/meta/client.py with transcribe(), sam_image(), image_edit() and image_generate(). SAM's server-sent events are parsed with any line ending, unfinished streams raise instead of being cached, and generated images are stored byte for byte so their provenance metadata stays intact.

**v7.** `Add te models and te api-smoke key checks`

Adds te models and te api-smoke to backend/cli.py and a template .env.example. Errors from settings, the API or the spend caps print as 'te: ...' and exit with code 2. Without a key both commands stop before any request is made.

**v8.** `Add WebSocket backend that receives camera frames`

Adds backend/server.py (Starlette, /health and /ws) and the te serve and te ws-selftest commands. Frames are checked as real JPEGs within size limits, acknowledged, and dropped; tests confirm nothing is written to disk or logged beyond metadata.

**v9.** `Add Modal script that logs a W&B smoke run`

Adds backend/training/wandb_smoke.py and its static tests, plus a train extra in pyproject.toml. Running it needs a Modal token and a Modal Secret called wandb holding WANDB_API_KEY.

**v10.** `Add glasses check guide`

Adds docs/glasses-check.md, written from the DAT iOS SDK 1.0.1 docs and samples: Developer Mode in the Meta AI app, building CameraAccess, what success looks like, common errors, and the Mock Device Kit. LICENSE and .gitattributes are the files from the initial commit, unchanged.

**v11.** `Add project notes and M0 progress`

Adds CLAUDE.md and PROGRESS.md. PROGRESS.md records what M0 delivered, the design decisions, 21 places where Meta's cookbook or the DAT SDK differ from the spec, the checks still waiting on the Mac and the glasses, and the spend so far ($0.00).

**v12.** `Add skill spec, read-back and local policy check`

Adds backend/skills/spec.py with the version 1 SkillSpec, the read-back sentence, and a rule-based policy check that refuses person, health, covert and sports requests without calling any model. Adds te policy and pydantic as a dependency.

**v13.** `Add skill compiler using Muse Wonder`

Adds backend/skills/compiler.py: request plus 3 keyframes in, validated SkillSpec or refusal out. Adds a validate hook to MetaClient.structured so unusable answers are never cached. A local refusal costs nothing; tests run against a fake Muse Wonder.

**v14.** `Add keyframe picker for teaching clips`

Adds backend/dataengine/keyframes.py: decode a clip or read a frame folder, drop blurry frames, drop near-duplicates, keep 20 to 30 keyframes. Adds numpy as a dependency.

**v15.** `Add Axolotl dataset writer`

Adds backend/dataengine/dataset.py: padded PNGs, the JSONL in the cookbook's multimodal format, and a manifest with counts per skill, source and answer plus SHA-256 digests of the JSONL and the example ids.

**v16.** `Add SAM region-of-interest step`

Adds backend/dataengine/roi.py: SAM 3.1 per keyframe through the client, raw output saved to roi.json, a parser hook for the boxes, and the largest-confident-box rule.

**v17.** `Add Muse Image synthetic set with prompt manifest`

Adds backend/dataengine/synth.py: a deterministic plan of positives and hard negatives (150 each by default), anchored on the keyframes, with nuisance variation; generation through the cache; images stored byte for byte; manifest.jsonl with every prompt, source keyframe hash and image hash.

**v18.** `Add Muse Spark labels and filtering`

Adds backend/dataengine/labels.py: one structured Spark call per image, the keep-only-if-Spark-agrees filter, and agreement statistics overall and for positives and hard negatives.

**v19.** `Add end-to-end teach flow`

Adds backend/teach.py, which chains transcription, keyframes, compiler, read-back, SAM, Muse Image, Spark labels and the dataset, with a cost estimate helper. Adds train_on_model_outputs to the settings. An offline end-to-end test runs the whole flow twice against a fake API and checks that the second run makes no calls.

**v20.** `Add te teach command with cost estimate`

Adds te teach to backend/cli.py: per-step cost estimate, two confirmations (spend, then the read-back), the teach run and the spend report. Tests check that the estimate and a refused request make no API call.

**v21.** `Add recording protocol and M1 progress`

Adds docs/recording-protocol.md and updates PROGRESS.md and CLAUDE.md: M1 is done offline, the first live washer run and the real clips are the next steps, and spend is still $0.00.

**v22.** `Add handover and setup guide`

Adds HANDOVER.md (state, open items with commands, rules, facts from the docs, and the plan for M2 to M5) and SETUP.md (accounts, Mac, cloud session, git), and points CLAUDE.md and PROGRESS.md at them. No code changes.

**v23.** `Fix settings: fail-closed switch, quotes, project paths`

Makes TE_TRAIN_ON_MODEL_OUTPUTS fail closed and shows it in the settings, strips quotes from .env values that have a trailing comment, rejects nan and negative numbers, and anchors .env, data/, the ledger and the cache to the project folder. Tests cover each case and keep the real .env and ledger out of the test run.

**v24.** `Fix client: cap check on retries, record odd replies`

Re-checks the caps before a retry that may have been billed, records any served reply in the ledger even when it cannot be read, and stores each reply's cost in the cache (CallInfo.billed_usd). Tests cover a retry run stopped at the cap and five malformed replies.

**v25.** `Fix policy check and compiler wording rules`

Widens the named-person rule (accents, titles, curly apostrophes), stops false refusals of common object requests, narrows the re-check of the model's wording to the target and trigger, accepts placeholder refusal reasons, validates field types, and adds a correction argument to compile_skill. Tests add 30 policy cases and the compiler cases.

**v26.** `Fix keyframes: EXIF orientation, ffmpeg stdin`

Bakes EXIF orientation into folder frames when they are loaded and runs ffmpeg with -nostdin and stdin closed. Tests check a turned photo comes out upright and that decoding a video leaves stdin unread.

**v27.** `Fix data engine: skip single failures, keep costs`

Skips and records a single rejected image edit or unusable label instead of aborting, with a cap on failures; retries a label cut off for length with 4096 tokens; writes the real cost of cached images to the manifest; adds progress callbacks. Tests cover the label cases.

**v28.** `Fix teach flow: labels, keyframe names, progress`

Normalises and checks the user's labels before any spend, names keyframes by position and content hash, removes stale ones, passes corrections to the compiler, handles per-item failures and prints progress. Tests cover provenance byte for byte, costs across reruns, failures, labels and corrections.

**v29.** `Fix te teach prompts, warnings and corrections`

Checks --labels before the estimate, shows the training mode, warns about the caps, makes --audio and --request exclusive, adds --correct, treats EOF as no, explains how to continue after Ctrl+C or a cap stop, and validates the api-smoke reply. Tests cover both confirmations and every new message.

**v30.** `Update glasses check, handover and notes after review`

Rewrites step 7 of docs/glasses-check.md to test reachability during streaming, adds the visual design rules to HANDOVER.md, and records the review and fixes in PROGRESS.md and CLAUDE.md.

**v31.** `Fix client: charge interrupts, refetch stale cache`

Charges a call interrupted with Ctrl+C, stops caching empty transcripts, fetches again a cached reply that no longer passes its check, turns a failed image download into a skippable output error, and charges an image reply without images for what was asked. Failed calls carry their charge, cut-off replies their finish reason, and structured() gains a cache-only lookup. Tests cover each case.

**v32.** `Fix settings home folder and spend table width`

Expands ~ in TE_DATA_DIR, since .env files are not read by a shell, and sizes the spend report's name column to the longest skill name. Tests cover both.

**v33.** `Fix policy gaps, corrections and read-back wording`

Splits the named-person rule so possessives always count and 'gets ready' is no device state, widens the sports, covert, health and person lists, adds a capture-light rule, leaves negated phrases out of a correction's check, limits requests to 500 characters, checks all of the model's wording against the topic rules, and tidies the read-back. Tests cover every new refusal and every request that must still pass.

**v34.** `Fix frame folders and keep the image manifest current`

Skips hidden files in frame folders, reads WebP, names HEIC and unreadable frames in the error, turns EXIF-tagged frames without rewriting EXIF, writes manifest.jsonl in a finally block, adds TooManyFailures, and plans duplicate near-misses once. Tests cover every EXIF orientation and each folder case.

**v35.** `Fix label reruns and SAM failure handling`

Looks up the larger-budget label in the cache before sending the small one, retries labels cut off mid-answer, adds failed attempts to each label's cost, and makes roi.locate skip rejected frames with the same failure cap as synth and labels. Tests cover the rerun, the cut-off reply and both SAM cases.

**v36.** `Fix teach input checks, class balance and te serve`

Moves every input check of te teach before the estimate, adds the worst case to the estimate, stops one-answer datasets, requires both answers among fallback labels, removes a stale dataset, guards the keyframes folder, refuses contradicting labels, saves labels on a stop, adds the failures hint, and makes te serve print the phone address with the server's logs. Tests cover each stop and message.

**v37.** `Install ffmpeg in CI and fix the Mac setup`

Adds an ffmpeg install step to the CI workflow, switches the Mac setup in CLAUDE.md to python3.12 with ffmpeg, and replaces the unused WANDB_ENTITY line in .env.example with a note on passing the team to Modal.

**v38.** `Update notes, handover and recording protocol`

Updates the cost figures, records the second review round, its open questions and the M2 and M4 notes in HANDOVER.md and PROGRESS.md, and extends the teaching-clip section of docs/recording-protocol.md.

**v39.** `Fix cost checks and the TE_DATA_DIR home error`

Falls back to the estimate when a computed cost is not finite or is negative, so a malformed usage or duration can no longer switch the spend caps off, and turns an unexpandable ~user in TE_DATA_DIR into a ConfigError. Tests cover NaN, -9e9 and 1e999 durations and the unknown user.

**v40.** `Fix named-person gaps and false refusals in policy`

Reworks the named-arrival patterns (two names, titles, contracted 's, readings with units), narrows the capture-light, covert, sports and count rules to what they mean, adds object phrases, bounds the negation in corrections, checks the model's trigger with its 'when', and fixes the read-back's capitals. Tests cover every case the review reported.

**v41.** `Add the watch trigger and threshold calibration`

Adds the 3-of-4 trigger with cooldown, gap reset and snooze, the alert simulation and evaluation over labelled clips, the calibration that picks the lowest threshold with at most one false alarm per hour, and the reader for watch frame logs, with tests.

**v42.** `Add the student scorer for an OpenAI-style endpoint`

Adds OpenAIStudent, which sends a frame as the dataset builds it and reads the yes/no probability from the answer token's logprobs, with tests for the request, the reasoning before the answer and endpoint failures.

**v43.** `Add the watcher with the Spark check and frame log`

Adds Watcher, the alert message, the frame log and load_skill, with tests for a confirmed alert, a cloud no, a failed check, skills without the check, student errors and two skills at once.

**v44.** `Add watch mode to the WebSocket server`

Lets a session ask to watch taught skills, scores each frame in a worker thread, sends scores in the acks and alert messages, and reports a watch request that cannot be met without ending the session. Adds the student endpoint settings. Tests cover the protocol, the errors and the default watcher.

**v45.** `Add the training bundle and the Axolotl config`

Adds the bundle writer with variants, limits, validation split and volume paths, and the config builder that copies Meta's QLoRA recipe and changes only paths, names and gradient accumulation. Tests compare the config with the recipe and cover every bundle option.

**v46.** `Add the Modal training app with the recipe pins`

Adds the Modal app, image, volume and training function, the bundle upload and the local entrypoint, with static tests for the pins, the GPU, the secrets and the bundle check.

**v47.** `Add te train, te watch and te calibrate`

Adds the three commands, the H100 price setting and their CLAUDE.md lines. Tests cover the smoke, a priced run, the over-$15 question, input errors, a failed run, a replay and a calibration that watch mode then uses.

**v48.** `Update the notes for the review and M2 so far`

Records the review fixes and the offline half of M2 in HANDOVER.md and PROGRESS.md, with the next steps on Modal and the Mac and corrected cost figures, adds the student settings to .env.example, and extends the teaching-clip section of docs/recording-protocol.md.

**v49.** `Fix te teach input checks and stop messages`

Checks the normalised request text, adds a non-negative count type for --positives and --negatives, prints the failure reason before the hint, and corrects two messages. Tests cover each.

## Notes

- **Reconstructed stages.** Each milestone was written as one finished tree, and its
  stages were then made from it by removing layers with exact edits (whole files, or
  exact blocks such as the media part of the client), working backwards. No code outside
  a layer was rewritten. M1's stages were peeled from the finished M1 tree, and peeling
  v12 once more gives v11 byte for byte (README aside), so M1 sits exactly on M0; the
  same holds for v22 on v21, v23 on v22, v31 on v30 and v39 on v38. Each folder was then tested as its own
  project. The READMEs were written per stage, from commands run inside each folder.
- **Files from the first commit.** `.gitignore` enters at v2 as the original Python
  template plus Trained Eye's own entries. `LICENSE` (MIT, the original) and
  `.gitattributes` enter unchanged at v10, so the last folder matches the repository.
- **Branch-only tools.** `stagetools/` holds the scripts that built and checked these
  folders (see `stagetools/README.md`). Like this file, it lives only on this branch.
- **Large files.** None. The largest file is `HANDOVER.md` at
  about 29 KB; the whole of v49 is about 429 KB.
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
    Spark see these prompts. SAM's output is stored but not parsed yet.
  - ffmpeg on the Mac: video decoding was tested with ffmpeg on Linux.

## How the folders were checked

A script (kept outside the repository) checked every folder after it was built:

1. v1 has at most 5 files, and each later stage adds or changes at most 5 files compared
   with the one before (README included).
2. Every folder contains every file of the folder before it.
3. v49 matches the repository root byte for byte (local-only things excluded).
4. Every file a README names, in backticks or as a bare file name, exists in that folder.
5. No folder holds caches, `.env`, `data/`, build output or secrets.
6. Inside every folder: `ruff check .` and `python -m pytest` pass, and `backend` imports
   from that folder, not from another one.
7. Every README command was run inside its folder with no keys or `TE_*` settings in the
   environment; the outputs in the READMEs are pasted from those runs.
8. From v4 on, the `te` entry point in `pyproject.toml` was called inside each folder.
   For each later milestone, peeling its first stage once more was checked to give the
   stage before it exactly (v12 to v11, v22 to v21, v23 to v22, v31 to v30, v39 to v38).
9. The READMEs and this file contain no emoji or em dashes.
