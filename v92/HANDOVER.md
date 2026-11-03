# Handover

Read this first when picking the project up in a new session, local or cloud. It says
where everything is, what is done, what is not, the rules the author set, and what to
build next. Last updated 2026-10-29, at the end of milestone M1 (code).

## 1. Read in this order

1. **HANDOVER.md** (this file): state, rules, next steps.
2. **PROGRESS.md**: milestone checklist, decisions, the docs-vs-spec table, open issues,
   spend. It is the source of truth between sessions; update it at the end of every
   milestone.
3. **SETUP.md**: accounts and keys, the Mac, a cloud session, git and the stage branch.
4. **SPEC.md**: the full build spec. It is not in the repository; ask the author to
   upload it. Section 6 below summarises what is left of it.
5. On the `build-stages` branch only: **VERSIONS.md** (the stage table and how to commit
   stages) and **stagetools/README.md** (how stages are built and checked).

## 2. Where things are

- Repository: github.com/amaan784/trained-eye. All work is on the branch
  `build-stages` (the author chose the name). Its root is the finished code. The folders
  v1 to v105 are the same code in cumulative stages, for the author to commit to `main`
  one at a time. `main` holds only what the author has committed from the stages so far;
  check with `git log origin/main`.
- Hackathon: Meta Global AI Developer Hackathon, solo. Submission is due Wed Nov 4, 9 PM
  ET; the plan is to submit by 6 PM ET. The repo, video, weights and playground must be
  public by then.

Code map (all under `backend/`, namespace packages without `__init__.py`):

| File | What it does |
|---|---|
| `config.py` | Settings: model ID per job, prices, spend caps ($10 per skill, $120 global), data paths, keys from the environment or `.env` (never printed), the pinned Glimmer revision, `TE_*` overrides, the `-contributor` ban |
| `meta/cache.py` | Content-addressed cache in `data/cache/`: SHA-256 of model, endpoint and normalized request, media replaced by its own hash |
| `meta/ledger.py` | Spend ledger `data/ledger.jsonl`: each call's estimate reserved before it runs (thread lock and `data/ledger.jsonl.lock`, so concurrent te processes cannot pass a cap), actual cost after, report, image tokens measured from past calls |
| `meta/client.py` | `MetaClient`: structured output (Chat Completions, strict json_schema, optional validate hook), Voice Transcribe, SAM 3.1 (Responses API, streamed), Muse Image edit and generate, model list; retries 429/5xx/timeouts, never 4xx; conservative spend accounting; key redaction |
| `cli.py` | `te`: spend, models, api-smoke, serve, ws-selftest, policy, teach, the shared helpers (`ask`, `make_client`, `modal_ready`), and the registration of the commands in the other `cli_*.py` files |
| `server.py` | Starlette backend: `/health`, WebSocket `/ws` protocol 1 (hello, JPEG frames, acks; watch mode with scores and alerts); frames never stored |
| `training/wandb_smoke.py` | Modal app that logs one W&B run; key only from the Modal Secret `wandb` |
| `training/axolotl.py` | The Axolotl config: Meta's QLoRA recipe with this run's paths and W&B names |
| `training/prepare.py` | Bundles taught skills (real, synthetic or mixed) with a validation split for the Modal volume |
| `training/modal_train.py` | Modal app that trains on one H100 with the recipe's pins; smoke mode, detached runs, the processor preflight (`training/preflight.py`) |
| `training/runs.py`, `training/gpu_ledger.py` | Training run status for GET /skills; the Modal spend ledger `data/modal_ledger.jsonl` checked against the $150 credit |
| `inference/student.py` | Scores a frame on an OpenAI-compatible endpoint from the answer token's logprobs |
| `watch/trigger.py` | 3-of-4 smoothing with a cooldown; threshold calibration on labelled clips |
| `watch/watcher.py` | Watch mode: scores, the Muse Spark check, alerts, the frame log |
| `teach_api.py`, `teach_upload.py`, `teach_session.py` | Teaching from the phone: the upload, the checks before any spend, the session with its two questions |
| `training/export.py`, `training/export_check.py` | Modal app that merges the adapter and checks the merged model's answers |
| `inference/modal_serve.py` | Modal app serving the merged model with vLLM, behind STUDENT_API_KEY; also base Glimmer (`--base`) and several adapters on one base (`--runs`) |
| `labeltool/` | The labelling tool: true time ranges in a clip, yes/no on generated images |
| `eval/` | Metrics, SVG charts, results.md (`te eval`), Spark as a scorer, Spark's error rate, the experiment plan with its costs (`plan.py`) and `eval/plan.json` (`plan_file.py`), the regression check, `te replay`, Experiment 1 failure galleries, the README results block, the W&B report |
| `regenerate.py`, `release/` | Prompt manifests and regeneration, face blurring (with manual boxes and blur records), the model card (`model_card.py`) and the release folder, upload and collection (`package.py`: base license, blurred keyframes), the test set as a dataset, weights from the Modal volume |
| `playground/` | The web playground: clips with the live timeline, how each skill was built, teach it yourself (capped), results; a public mode and its Modal app (`modal_app.py`, `publish.py`) |
| `cli_train.py`, `cli_watch.py`, `cli_serving.py`, `cli_eval.py` | te train; te watch and te calibrate; te export and te serve-student; te eval and te label |
| `cli_experiments.py`, `cli_replay.py`, `cli_release.py`, `cli_playground.py` | te experiments, unfiltered, clips, spark-errors and regress; te replay; the M4 release commands; te playground and te playground-clip |
| `skills/spec.py` | `SkillSpec` v1, read-back sentence, local `policy_check` (people, health, covert watching, the capture light, sensitive places, sports, politics and war), `check_name` |
| `skills/compiler.py` | Muse Wonder compiler: request + 3 keyframes to a spec or a refusal; re-checks the model's wording |
| `dataengine/keyframes.py` | 20 to 24 sharp, distinct keyframes from a video (ffmpeg, bit-exact) or a frame folder |
| `dataengine/roi.py`, `dataengine/sam_text.py` | SAM 3.1 per keyframe, boxes read with our own reader of Meta's SAM 3 text format (masks not decoded), raw output stored; crops to the best box |
| `dataengine/realclips.py` | `te clips`: labelled session 1 clips as training frames, session 2 as validation only |
| `dataengine/synth.py` | Deterministic Muse Image edit plan (positives, hard negatives, 11 nuisances), images stored untouched, `manifest.jsonl` |
| `dataengine/labels.py` | Muse Spark yes/no/unsure labels, keep-if-agrees filter, agreement stats |
| `dataengine/dataset.py` | Axolotl JSONL in Meta's recipe format, 512x512 padded PNGs, manifest with provenance index |
| `teach.py` | The teach flow end to end, cached; fallback switch `TE_TRAIN_ON_MODEL_OUTPUTS=0` |

Other files: `tests/` (707 tests, all offline; one skips without OpenCV),
`docs/glasses-check.md`, `docs/recording-protocol.md`, `.env.example`,
`.github/workflows/ci.yml` (ruff and pytest on every push), `CLAUDE.md` (short
conventions file the author asked for).

Local data, all under `data/` and gitignored: `cache/`, `ledger.jsonl`, and
`skills/<skill>/` with `request.json`, `keyframes/`, `spec.json`, `roi.json`,
`synthetic/` (images and `manifest.jsonl`), `labels.jsonl`, `label_stats.json`,
`dataset/`. Real clips go in `data/clips/<skill>/` and the teaching look in
`data/teach/<skill>/` (see `docs/recording-protocol.md`).

## 3. State

### Done, and tested offline

- **M0:** repo, notes, `.env.example`, the Meta API client with cache, ledger and
  retries, `te` commands, the WebSocket backend, the W&B smoke script, the glasses guide.
- **M1:** the teach flow for one skill: spoken request, keyframes, skill spec with policy
  refusals and read-back, SAM region of interest, Muse Image variations with manifest,
  Spark labels with filtering, Axolotl dataset; `te teach`, `te policy`; the recording
  protocol.
- Each milestone had an independent code review (several reviewers, and a skeptic
  re-checking every finding). The real bugs it found were fixed before the stages were
  cut (see the Decisions in PROGRESS.md).
- **Review after M1 (Oct 31):** bugs, version compatibility, functionality against
  SPEC.md and the UI, every finding checked by one reviewer who reproduced it and one who
  tried to refute it. The 27 confirmed in F1 to F29 are fixed in stages v23 to v30, and
  the confirmed ones in F30 to F62, with the faults that review found in those fixes, in
  stages v31 to v38, and what a review of those found in v39 and v40 (PROGRESS.md,
  "Review after M1"). Two open questions from it are in item 9 below.
- **Teaching changes for the author:** keyframes are now named
  `kf_<position>_<hash>.jpg`, so a `--labels` file uses those names (from
  `data/skills/<skill>/keyframes/`); a wrong read-back is fixed with
  `--correct "what is wrong"` on the same command, which reuses the clip and transcript.
  `te teach` checks the clip, audio, request and labels before it shows the estimate,
  stops if the dataset would hold only one answer, and says when a rerun would pay again
  for failed calls. `te serve` prints the address to type into the phone.
- **M2 to M4, the code** (Nov 2; tested offline with fakes, never run against Meta,
  Modal or a phone): teaching from the phone (`/teach/start`, `/teach/status`,
  `/teach/answer`, behind `SERVER_TOKEN`; nothing is paid before the phone says yes),
  `GET /skills`, export and
  Modal serving (`te export`, `te serve-student`), the labelling tool (`te label`), the
  evaluation harness (`te eval`, `make eval`, SVG charts), the experiments (`te
  experiments`, `te unfiltered`, `te spark-errors`, `te regress`, `te watch --scorer
  spark`), the release tools (`te manifests`, `te regenerate`, `te blur`, `te release`),
  the web playground (`te playground`), the iOS app (`ios/`, not compiled) and
  CONTRIBUTIONS.md.
- **M2, the parts that run without the Mac or Modal** (tested offline with fakes): the
  watch trigger (3 of the last 4 frames over the threshold, then the cooldown) and the
  threshold calibration, the student scorer for an OpenAI-compatible endpoint (vLLM on
  Modal or llama.cpp's llama-server), the watcher (Muse Spark check, frame log), watch
  mode on the WebSocket (`"watch"` in the hello, scores in each ack, alert messages), the
  training bundle and the Axolotl config from Meta's recipe, the Modal training app, and
  `te train` and `te watch`.
- Spend so far: $0.00 Meta API, $0.00 Modal. Nothing has called a paid service yet.

### Not done yet, in priority order

Each of these needs the author's keys, Mac, glasses or footage. Record every outcome in
PROGRESS.md.

1. **Glasses check** (`docs/glasses-check.md`). Decides glasses camera or the iPhone
   camera fallback. Ask the author for: Xcode version, `sysctl hw.memsize`, which steps
   worked with exact error text, what the capture LED did, any update prompts.
2. **First live Meta calls.** `te models` (free): if the model IDs differ from
   `DEFAULT_MODELS` in `backend/config.py`, fix them there. `te api-smoke` (at most about
   $0.0024): confirms Chat Completions with a strict json_schema works. If it returns a
   400 about the request shape, adapt only `MetaClient.structured`.
3. **W&B smoke from Modal:** `modal run backend/training/wandb_smoke.py`. Creates the
   `trained-eye` project. It stays private; making it public counts as publishing.
4. **First washer teach run, small:**
   `te teach --skill washer_done --clip data/teach/washer_done/teach.mp4 --audio data/teach/washer_done/request.m4a --positives 6 --negatives 6`
   (for 20 to 24 keyframes, what a 10 s clip gives, the estimator prints about $0.48 to
   $0.52, and at most $1.11 to $1.22 if every label has to be asked again with the larger
   token budget; `te teach` prints the figure for the real clip first). Then look at
   `data/skills/washer_done/`. The
   request shapes for Voice Transcribe (multipart file + model to
   `/audio/transcriptions`), SAM (`/responses`, streamed) and Muse Image (`/images/edits`
   multipart, `/images/generations` JSON) come from SPEC.md and are unconfirmed; fix
   them in `client.py` if the API disagrees. Keep tests offline: turn real responses
   into fixtures in the tests, never call the API from a test.
5. **SAM parser.** After step 4, the raw SAM text is in `roi.json` and in the cache
   (`data/cache/*/*.json` with endpoint `/responses`). Write `parse_sam_output` in
   `backend/dataengine/roi.py` against it (boxes are x1, y1, x2, y2 in source pixels;
   masks are base85 per SPEC.md) and add a test with that text as a fixture. The
   `meta-sam-parser` package could not be installed or read; installing it is a
   download, so ask first.
6. **Full washer run:** the same command with the default 150 and 150. The estimator
   prints about $5.25 to $5.28 for 20 to 24 keyframes, and at most $11.52 to $11.63 if
   every label is asked again with the larger budget. That worst case is under the $15
   ask threshold but over the $10 skill cap: the cap would stop the run partway, and a
   rerun with a higher `TE_SKILL_CAP_USD` continues from the cache. Image tokens are
   guessed at 1,500 per image, so treat these as estimates, not bounds.
7. **Real clips** for washer_done, boil_over and bus_m4 (`docs/recording-protocol.md`):
   about 5 minutes of training and validation per skill from sessions 1 and 2, about 5
   minutes of test per skill from a different session, with near-misses.
8. **Questions for the organizers:** does self-hosting Glimmer on Modal count as partner
   inference (the cookbook lists only Together AI as a Glimmer host)? The human answer on
   training Glimmer on Muse outputs is still pending; if it is no, set
   `TE_TRAIN_ON_MODEL_OUTPUTS=0`. With it at 0 the images and answers are real and the
   author's own, but the condition text in every example (spec.positive and negatives)
   is still written by Muse Wonder: ask whether that is allowed too.
9. **Questions for the author, from the review:**
   - Starlette 1.2 and later want the `httpx2` package for `TestClient` (used by
     `te ws-selftest` and the server tests) and warn without it; pyproject.toml does not
     name it. Either add `httpx2` to the dependencies (a new download, so ask) or cap
     `starlette<1.2` (not tested here: only 1.7.0 is installed).
   - The approved-downloads list in section 4 names openai, httpx, fastapi, websockets,
     pytest, wandb, modal and opencv-python-headless, but `pip install -e ".[dev]"` also
     installs numpy, pillow, pydantic, starlette, uvicorn with its extras, and ruff.
     Ask whether those count as approved.
10. **M2 on Modal and the Mac**, after the washer dataset exists (item 6):
   - Check the H100 price in `MODAL_GPU_USD_PER_HOUR` (backend/config.py) against
     modal.com/pricing; it could not be checked from the build sandbox.
   - `te train --skills washer_done --run exp1-mixed --preflight` first (a CPU Modal job;
     asks): it renders a few examples with the pinned processor and says whether to set
     `TE_SYSTEM_IN_USER=1` before the datasets are written. `te clips --skill washer_done`
     adds the labelled session 1 clips as training frames and session 2 as validation.
   - `te train --skills washer_done --run exp1-mixed --smoke` asks, then runs the recipe's
     two-step smoke. The first run builds the image (flash-attn compiles; the CUDA image
     tag `nvidia/cuda:13.0.0-devel-ubuntu24.04` is unverified) and downloads the 30B base
     to the volume. It saves the timing that prices the real run.
   - `te train --skills washer_done --run exp1-mixed` prints the estimate from that
     timing and asks (always, if it is over $15). `--detach` keeps it running if the Mac
     sleeps (`te train --collect <run>` fetches the result); `--wandb-entity` names the
     team; `--wandb-artifacts` is off by default and asks every time. Modal spend is kept
     in `data/modal_ledger.jsonl` (`te spend`), and te train and te export ask near the
     $150 credit.
   - `te export --run exp1-mixed` merges the adapter on Modal (asks; at most 3 hours,
     not timed yet) and checks the merged model gives the same yes/no answers as base
     plus adapter on 50 examples. The cookbook evaluated QLoRA adapters on an NF4 base
     only, so `--reference nf4` exists too.
   - Serving: the cookbook documents no conversion of a fine-tuned Glimmer for the Mac,
     so the student is served from Modal. Make up a long random `STUDENT_API_KEY`, put it
     in `.env` and in a Modal Secret named "student", then `te serve-student --run
     exp1-mixed` (asks; billed while a container is up, `--stop` stops it). Check in the
     first container log that the key is not printed (vLLM may log its arguments; if it
     is, pass it as `VLLM_API_KEY` instead), and that a request returns `logprobs` with
     `top_logprobs`. Then set `TE_STUDENT_URL` (the printed URL plus `/v1`).
     `te serve-student --base` serves base Glimmer (Experiment 4, te regress) on its own
     app; `--runs a,b,c` serves several adapters on one base, each run a model name
     (untried: check that vLLM's LoRA serving works for Glimmer before relying on it).
   - `te watch --skills washer_done --clip <recorded clip>` replays a clip through watch
     mode and writes a frame log; `te serve` then serves it to the phone. Thresholds stay
     at 0.5 until `te calibrate` has labelled validation clips (`te label --clip`, M3);
     `--run` keeps a threshold per training run. Teaching from the phone needs
     `SERVER_TOKEN`; GET /skills tells the app which skills are training, trained or
     served, and cooldowns survive reconnects (`data/watch/cooldowns.json`).
   - The iOS app: `ios/README.md` (create the Xcode project, add DAT 1.0.1, the
     Info.plist keys). It has never been compiled; its TODOs are listed there.
11. **M3 on the Mac:** teach boil_over and bus_m4 as in item 4; record the clips in
   `docs/recording-protocol.md`; label the test clips with `te label --clip <clip>` and
   200 generated images with `te label --skill <skill> --sample 200`; then
   `te experiments --skills washer_done,boil_over,bus_m4` prints every training run of
   Experiments 1 to 3 with its price (steps stay fixed per run, see backend/eval/plan.py)
   and `--write-plan` writes `eval/plan.json` for `make eval` (the $150 check counts the
   final run, an export and serving per model and the Modal spend so far;
   `--endpoint-hours` sets the serving guess). `te replay --plan eval/plan.json --run <run>`
   calibrates a run on session 2 and replays its test clips into the plan's logs, once per
   served model. Experiment 2 needs `te unfiltered --skill <skill>` and
   `te spark-errors --skill <skill>`; Experiment 4 replays with `--scorer spark` (paid per
   frame; asks) and with the base endpoint; the regression check is `te regress`.
   `make eval` writes `eval/results.md` (with Spark's error rate, the regression check and
   links to the local Experiment 1 failure galleries in `eval/galleries/`, which may show
   faces) and the SVG charts; `te eval --readme README.md` writes the results block.
12. **M4, after asking:** in this order: `te manifests` (commit `manifests/`);
   `te fetch-weights --run <final>`; `te blur --src data/skills/<s>/keyframes --dst
   data/skills/<s>/keyframes_blurred` for each skill (needs `pip install
   opencv-python-headless`, approved; `--boxes <json>` adds faces the detector misses, and
   a video gets a `.blur.json` record); `te release --run <final> --out <folder>
   --keyframes <s>=<blurred folder>` (apache-2.0, the base model's license, its text as
   `LICENSE`, Meta's Usage Policy linked); `te blur` each session 3 clip, then
   `te release-test-set --clips <s>=<blurred folder> --out <folder> --license <id>`; the
   uploads (`--upload`, each asks; `huggingface_hub` is not on the approved list yet);
   `te release-collection`; `te wandb-report` (needs `wandb-workspaces`, not approved
   yet), then publish the report in the W&B UI after asking. The playground: `te
   playground --public` locally shows what the public copy holds (no generated images,
   only blurred keyframes and clips); `te playground-clip` makes a blurred clip folder in
   `data/playground/clips/`; `te playground --deploy` (asks) copies the public files to
   the private Modal Volume `trained-eye-playground` and deploys `playground/modal_app.py`
   (create the Modal Secret `meta` with `MODEL_API_KEY` first; `--total-cap` is the
   all-time cap, default $10; `modal app stop trained-eye-playground` takes it down).

## 4. The author's rules (all still in force)

### Working

- One milestone at a time. A milestone ends with tests passing, its demo command
  working, PROGRESS.md updated (done, decisions, open issues, API and GPU spend), and its
  stages built, checked and pushed to `build-stages`.
- If SPEC.md and the docs disagree, follow the docs and record it in PROGRESS.md.
- Tests never call paid APIs. Use `httpx.MockTransport` fakes or cached fixtures.
- Keys live in `.env`, environment variables or Modal Secrets (`wandb`, `huggingface`).
  Never commit, print or log them.
- Ask the author before: any single job estimated over $15 (API or GPU); deleting data;
  publishing anything (Hugging Face uploads, making the W&B project or report public,
  the playground, footage); anything physical (recording, wearing the glasses, creating
  accounts); downloading anything not already approved. Approved so far: the PyPI
  packages openai, httpx, fastapi, websockets, pytest, wandb, modal and
  opencv-python-headless; the Meta docs pages listed in SPEC.md; read-only clones of
  github.com/meta-models/meta-oss-cookbook and github.com/facebook/meta-wearables-dat-ios.
  Anything else (meta-sam-parser, mlx-vlm, llama.cpp, model weights, Docker images) needs
  a yes first.
- Never invent results. Every number in the README, playground, model card and video
  comes from the eval scripts.
- If time runs short: keep 2 skills instead of 3, cut the iOS Clips screen and playground
  polish, never cut Experiments 1 and 4.
- No AI or tool credit anywhere: not in commits, files, READMEs or the model card.

### Policy (Meta's acceptable use policies)

- No skill may identify or track a specific person. Skills cover objects, states, text,
  displays, signs and vehicles only. The compiler refuses the rest, and refusals are
  explained in one sentence; show one refusal in the demo ("Tell me when Sam gets here").
- No skills about bodies, health, medication or other sensitive traits; no covert
  monitoring; no sensitive locations; no sports analysis or coaching. Also avoid the
  hackathon's anti-projects: medical advice, nutrition coaching, sports coaching, a
  "simple image analyzer", politics or war.
- Never hide or suppress the glasses' capture LED; watch mode is always visible in the app.
- Frames are not stored unless the user taps "save clip".
- Do not strip watermarks, labels or provenance metadata from Muse Image outputs. Never
  commit generated images; commit the manifest and the regeneration path instead.
- Blur faces in anything published (README, playground, video, test set).

### Version stages

The author reviews and commits the project one layer at a time. New work becomes new
stages after the newest folder (v105 now). The rules, as the author gave them:

- Stages v1 to vN live in the root of `build-stages`, in the order a developer would
  build them: foundations first, docs and UI late.
- Each stage is one commit. From v50 on (the author's newer rule, Nov 2) a stage changes
  150 to 300 lines compared with the stage before, counting added plus removed lines and
  leaving out its README; `stagetools/check_stages.py` checks it. v1 to v49 were cut
  under the earlier rule (at most 5 files per stage, README included) and stay as they
  are. If a layer is bigger, split it into stages that each still run; if a small fix is
  under 150 lines, put it in a stage with related work. If a change cannot fit without
  breaking a stage, tell the author first.
- Cumulative: vN has everything from v(N-1) plus what stage N adds. No stage deletes or
  renames a file the previous one had.
- Each folder runs on its own: its own README, `pyproject.toml`, tests and CI file;
  nothing points into another folder; its tests and README commands pass from inside it.
- The last folder is byte-identical to the root except for local-only things: `.claude/`,
  `.env`, `data/`, outputs, caches, `VERSIONS.md`, `stagetools/` and the v folders.
- Build earlier stages by peeling layers off the finished code with exact edits, so the
  diff between two stages is exactly that layer. Do not rewrite code outside the layer.
- Folders hold code and docs only: no caches, generated data or secrets.
- Each stage README: what it adds, which files, the commands to run, what they printed
  (measured, never invented), and what the next stage adds.
- `VERSIONS.md`: ground rules; the stage table; why this order; how to review
  (`git diff --no-index vA vB`); how to commit (copy each folder in order, then commit)
  with a short imperative summary and one-paragraph description per stage; notes; how
  the folders were checked.
- Before saying a milestone is done: run every stage's tests and README commands inside
  it; confirm the 5-file limit, that each folder contains the previous one, that the last
  folder matches the root byte for byte, and that every file a README names exists.
- If a real bug turns up, fix it at the root, carry the fix into every stage that has
  that code, and tell the author.
- Plain wording, no emoji or em dashes, in READMEs, VERSIONS.md and the other docs.
- `stagetools/README.md` explains the scripts that do all of this.

### Git

- Commit only to `build-stages`, with the author as author and committer, unsigned, with
  no attribution trailers (SETUP.md section 5 has the exact command). Push with
  `git push -u origin build-stages`. Never push to `main`; the author does that. Do not
  open pull requests unless asked.

## 5. Facts from the docs that the next milestones depend on

The full table is in PROGRESS.md. Sources: the Meta OSS cookbook at commit b30ca2c
(github.com/meta-models/meta-oss-cookbook) and the DAT iOS SDK 1.0.1
(github.com/facebook/meta-wearables-dat-ios). dev.meta.ai could not be read, so the
hosted Meta Model API pages have still not been checked.

**Training recipe** (`muse-glimmer/post-training/sft/axolotl/configs/muse-glimmer-30b-qlora.yaml`):
Axolotl 0.19.0 (Python 3.12, torch 2.12.1, torchvision 0.27.1, CUDA 13.0, flash-attn
built for H100); base `meta-models/Muse-Glimmer-30B` at revision
`a4e59da52a7bc87ae7251dd5545c0dd437c44b68`, `processor_type: AutoProcessor`; QLoRA
(`load_in_4bit`), `lora_r` 32, `lora_alpha` 16, `lora_dropout` 0.05; LoRA targets only
the language model's attention q/k/v/o and MLP gate/up/down projections, so the vision
tower stays frozen; all four `lora_*_kernel` options false (gated attention); Cut Cross
Entropy plugin; `skip_prepare_dataset: true`, `remove_unused_columns: false`, sample
packing off, `train_on_inputs: false`; datasets `type: chat_template`,
`field_messages: messages`, `roles_to_train: [assistant]`, `train_on_eos: turn`;
`sequence_len` 4096; micro batch 1, gradient accumulation 4 (global batch 32 on eight
GPUs; the README says use 32 on one H100), 1 epoch, `adamw_bnb_8bit`, cosine, lr 2e-4,
warmup 0.1, bf16, tf32, gradient checkpointing, flash_attention_2, seed 42. Paths in the
config are relative; generate a per-run YAML with absolute Modal volume paths. The
dataset JSONL also holds absolute image paths from where it was written (the recipe's
format), so write it again under the volume mount before training. Meta
verified quality on 8x H100; on 1x H100 only a 2-step smoke (peak 26,995 MiB). The
cookbook has no merge or export step; the adapter was evaluated on an NF4 base.

**Dataset format** (already followed by `dataset.py`): image part first,
`{"type": "image", "path": "<absolute path to an RGB PNG>"}`, then the text part;
assistant content a plain string; 512x512 padding is the recipe's default. The recipe's
examples have no system turn: run the processor preflight on one example and, if the
system turn renders badly, write the dataset with `system_in_user=True`.

**Scoring.** Glimmer always reasons first: generations open a `to=self` channel before
the `to=user` answer, and the chat template defaults `reasoning_strength` to high. So
the spec's "probability of the first token being yes" needs this: render the prompt with
the pinned chat template, prefill the assistant turn up to the `to=user` message body,
and read the next-token logits for "yes" and "no" (check their token IDs), then
renormalize. Check first how a bare yes/no training target renders. Stop tokens are
200001 and 200008; never stop on 200007 (`<|eom|>`); read the IDs from the pinned
tokenizer.

**Serving.** No source shows Glimmer with images on MLX or mlx-vlm; the MLX builds that
exist are text-only third-party repacks needing 48 GB or more. The vision path the
cookbook verified on Apple silicon is llama.cpp (Metal, GGUF plus mmproj, release
b10355). No conversion path for a fine-tuned Glimmer is documented. On a GPU, vLLM needs
the `vllm/vllm-openai:muse-glimmer` image (stock vLLM cannot serve it), image requests
are unattested, and bf16 needs about 72 GB, so an 80 GB GPU; do not pass
`--chat-template`; parser names use underscores (`muse_glimmer`). The spec's "18-21 GB"
and "17.6 tokens/s" appear in no source; the one measurement is about 24.1 GiB (Q4_K_M,
16K context). Decide after the memory check, and test an image request first.

**Glasses (DAT 1.0.1).** SPM package github.com/facebook/meta-wearables-dat-ios at exact
version 1.0.1; products MWDATCore, MWDATCamera, and MWDATMockDevice for debug builds;
iOS 17.2 or newer, Xcode 26.4 or newer. Developer Mode in the Meta AI app is enough for
development builds (camera may still need Developer Center approval). Frame rates 2, 7,
15, 24 or 30 only; resolutions low 360x640, medium 504x896, high 720x1280. A raw stream
pauses in the background, HEVC (`hvc1`) keeps going; Meta's BirdSpotter sample streams
hvc1 at low resolution and 7 fps. No speaker API: alerts are ordinary Bluetooth audio;
speak only when the output route is Bluetooth, and stop listening while speaking.
CameraAccess records the glasses mic over Bluetooth HFP. A single touchpad tap pauses
the stream; folding the glasses stops the session; only one DAT app can use the glasses
at a time. SDK analytics and crash reporting are on by default (turn both off in
Info.plist). The mock needs `pairGlasses(model: .metaGlasses)`, power on, unfold, don,
and an HEVC feed set before streaming. DAT sample code is under Meta's Wearables
Developer Terms, not an open-source license: keep its headers, ship its LICENSE next to
copied files, and list them in CONTRIBUTIONS.md.

## 6. What comes next (from SPEC.md)

### Visual design (for ios/, the playground, the eval charts and the video)

- Look: dark and minimal, one accent colour, a clean sans font (Inter), flat shapes, no
  gradients. Every chart uses the same palette, labels values directly, and reads in
  light and dark mode.
- Four signature visuals, used in the playground and the video:
  1. The training wall: a grid of generated variations flips in, each tagged green "yes"
     or red "no" as Spark labels it; rejected ones fade out.
  2. The live timeline: the first-person video on top, the student's yes-probability as
     a line below it, the threshold dashed, and an alert marker with the spoken text
     when it fires.
  3. The cost chart: accuracy against dollars per hour, three points (fine-tuned
     Glimmer, base Glimmer, Spark), the fine-tuned point highlighted.
  4. The SAM outline on the teaching frame, showing what the glasses learned to look at.
- If time runs short, cut UI polish before cutting experiments (SPEC.md).

### M2 (Fri to Sat): training, serving, watching

- `backend/training/`: a Modal function with Axolotl in the image and data on a Modal
  volume; `te train --skills washer_done --variant mixed`. One multi-skill adapter (the
  condition is in the prompt). W&B through Axolotl (`wandb_project: trained-eye`,
  `wandb_name` like `exp1-mixed`, tags for experiment, skills, mix, image count, seed and
  git commit), WANDB_API_KEY from the Modal Secret. Log the full config, the dataset
  manifest hash and the base revision; datasets and adapters as W&B Artifacts only after
  the author approves (the project becomes public). Run a 2-step smoke first, then
  estimate the real run from seconds per step and ask if it is over $15. About 10
  experiment runs plus 1 final; stop and ask if the total would pass $150 of Modal
  credit.
- Export: `merge_and_unload()`, safetensors, a Mac build if a path exists, otherwise a
  Modal endpoint; check the merged model makes the same yes/no decisions as the adapter
  on 50 frames.
- `backend/inference/`: one student interface (Mac or Modal) scoring frames as above;
  frames at 1 fps (downscaled with `dataset.pad_image`); alert when 3 of the last 4 frames
  are over the threshold, then the skill's cooldown; threshold = the lowest meeting at
  most 1 false alarm per hour on validation footage; if `confirm_with_cloud`, send the
  last frame to Muse Spark first and log both decisions (this is the live Meta call kept
  in every demo); send `{"type": "alert", "text": ...}` to the phone over the WebSocket;
  log every scored frame (time, probability, decision) for the timeline visual.
- Built so far: everything in the first and third bullets except the threshold from
  real footage, which needs M3's labels. Since Nov 2 the export, serving on Modal and
  `ios/` are written too (section 3, item 10); none of it has run yet.
- `ios/` (SwiftUI, with the DAT facts above): Connect, Teach (hold to talk, 10 second
  capture with a countdown, read-back, confirm), Skills (list, on/off toggles, status:
  training or ready), Watch (live indicator, last alert, battery), Clips (saved by the
  user only). Configurable WebSocket URL (local Wi-Fi or a tunnel). Follow "Visual
  design" above. The backend needs an endpoint that takes the 10 second capture
  (frames and audio) and runs `teach.run_teach`; today teaching is CLI only.

### M3 (Sat to Sun): all skills, evaluation, experiments

- Teach boil_over and bus_m4. A small web labeling tool that marks the time ranges where
  the condition is true and writes JSON; test labels are human only.
- `backend/eval/` and `make eval`: frame precision, recall and F1; event recall within
  5 s; false alarms per hour; time to alert; latency per frame; dollars per hour. Writes
  `eval/results.md` and charts (one palette, values labelled, readable in dark and
  light) and logs everything to W&B. Built: the charts are SVG (matplotlib could not be
  installed), W&B logging is optional (`te eval --wandb`).
- Experiment 1: real only vs synthetic only vs mixed, with failure galleries.
  Experiment 2: hand-label 200 random training images to measure Spark's error rate;
  filtered vs unfiltered labels (`label_stats.json` already records agreement).
  Experiment 3: 25, 50, 100, 200, 300 synthetic images per skill, F1 against count with
  generation cost (decide whether steps or epochs stay fixed for the small points).
  Experiment 4: fine-tuned Glimmer vs base Glimmer zero-shot vs Muse Spark per frame:
  accuracy, latency, cost per hour, and the accuracy-vs-cost chart. Regression check: 50
  general visual questions written by the author (plus a few text-only ones), base vs
  fine-tuned at matching precision. Launch training runs early, in the background.

### M4 (Mon Nov 2): release

- The web playground (Hugging Face Space or Modal): recorded clips with the probability
  timeline, how each skill was built (keyframes, SAM outline, generated variations with
  labels), "teach it yourself" with live Meta calls (type a condition, pick a provided
  keyframe, see the compiled skill and 12 live Muse Image variations labelled by Spark;
  budget-capped per visitor and per day), and the results page (the four experiment
  charts). It uses the four signature visuals and the look in "Visual design" above. Precomputed results are fine if serving live is too slow;
  say so on the page.
- Hugging Face collection (ask before every upload): the final LoRA adapter, the Mac
  build, the experiment adapters, the human-labelled test set (faces blurred, approved
  footage only); never generated images. Apache-2.0 if the Glimmer model card allows it
  (check the card). Model card contents are listed in SPEC.md.
- Generated images are never published, so ship what regenerates them: copy each
  skill's `synthetic/manifest.jsonl` out of `data/` into a committed folder, and add a
  `te regenerate` that rebuilds the images from a manifest and the (face-blurred)
  keyframes through `synth.generate`, with the usual estimate and ledger.
- A public W&B report "Trained Eye: experiments" (ask first), the README with results,
  architecture, the Meta-model table, privacy notes, limitations, and CONTRIBUTIONS.md
  crediting the cookbook recipe and any DAT sample code.

### M5 (Tue Nov 3): demo

- Footage and the 3-minute video (shot list in SPEC.md), polish, bug fixes. Tuesday is
  the last day for questions in the Discord.

## 7. Session log

- 2026-10-29, cloud session 1: read Meta's docs from the two GitHub repos (the Meta
  docs sites and PyPI were blocked), built M0 and M1, reviewed each milestone, and cut
  stages v1 to v21. The author approved the downloads listed above,
  chose the branch name `build-stages`, asked for their own identity on every commit
  with no AI credit, chose to add new stages after each milestone, and asked for this
  handover (stage v22).
- 2026-10-31, cloud session 1 (continued): the author asked for a scan for bugs and
  version compatibility issues, a functionality check against the plan and a UI check,
  with fixes as stages v23 and later and M2 after them. There is no UI yet (the iOS app
  and the playground are M2 and M4). Fixes for F1 to F29: stages v23 to v30. The second
  half of the review, F30 to F62, and a check of the v23 to v30 fixes themselves: stages
  v31 to v38. A review of those: v39 and v40. Then the parts of M2 that need neither the
  Mac nor Modal: v41 to v48. The review's last small fixes to te teach: v49.
- 2026-11-02, cloud session 1 (continued): the author asked for the remaining
  implementations, with stages from v50 on cut at 150 to 300 changed lines. The rest of
  M2 (teaching from the phone, export and serving): v50 to v58; the iOS app, never
  compiled: v59 to v74; M3 (labelling, evaluation, the four experiments): v75 to v89; M4
  (release tools, the playground, these notes and CONTRIBUTIONS.md): v90 to v105.
  Reviews of the teach API, the iOS code and the playground ran before staging, and
  their fixes are in those stages: a token on the teach routes (SERVER_TOKEN), a clean
  stop on shutdown, size and shape limits on uploads, one watch frame in flight on the
  phone, cross-site and IPv6 checks in the playground, and a shared in-flight total for
  its ledger cap.
