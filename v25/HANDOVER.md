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
  v1 to v22 are the same code in cumulative stages, for the author to commit to `main`
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
| `meta/ledger.py` | Spend ledger `data/ledger.jsonl`: caps checked before a call, actual cost after, report |
| `meta/client.py` | `MetaClient`: structured output (Chat Completions, strict json_schema, optional validate hook), Voice Transcribe, SAM 3.1 (Responses API, streamed), Muse Image edit and generate, model list; retries 429/5xx/timeouts, never 4xx; conservative spend accounting; key redaction |
| `cli.py` | `te`: spend, models, api-smoke, serve, ws-selftest, policy, teach |
| `server.py` | Starlette backend: `/health`, WebSocket `/ws` protocol 1 (hello, JPEG frames, acks); frames never stored |
| `training/wandb_smoke.py` | Modal app that logs one W&B run; key only from the Modal Secret `wandb` |
| `skills/spec.py` | `SkillSpec` v1, read-back sentence, local `policy_check`, `check_name` |
| `skills/compiler.py` | Muse Wonder compiler: request + 3 keyframes to a spec or a refusal; re-checks the model's wording |
| `dataengine/keyframes.py` | 20 to 30 sharp, distinct keyframes from a video (ffmpeg, bit-exact) or a frame folder |
| `dataengine/roi.py` | SAM 3.1 per keyframe; raw output stored; `parse_sam_output` is still a stub |
| `dataengine/synth.py` | Deterministic Muse Image edit plan (positives, hard negatives, 11 nuisances), images stored untouched, `manifest.jsonl` |
| `dataengine/labels.py` | Muse Spark yes/no/unsure labels, keep-if-agrees filter, agreement stats |
| `dataengine/dataset.py` | Axolotl JSONL in Meta's recipe format, 512x512 padded PNGs, manifest with provenance index |
| `teach.py` | The teach flow end to end, cached; fallback switch `TE_TRAIN_ON_MODEL_OUTPUTS=0` |

Other files: `tests/` (185 tests, all offline), `docs/glasses-check.md`,
`docs/recording-protocol.md`, `.env.example`, `.github/workflows/ci.yml` (ruff and
pytest on every push), `CLAUDE.md` (short conventions file the author asked for).

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
   (the estimator says $0.42 at most). Then look at `data/skills/washer_done/`. The
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
6. **Full washer run:** the same command with the default 150 and 150 (the estimator
   says $5.26 at most, under the $10 skill cap and the $15 ask threshold).
7. **Real clips** for washer_done, boil_over and bus_m4 (`docs/recording-protocol.md`):
   about 5 minutes of training and validation per skill from sessions 1 and 2, about 5
   minutes of test per skill from a different session, with near-misses.
8. **Questions for the organizers:** does self-hosting Glimmer on Modal count as partner
   inference (the cookbook lists only Together AI as a Glimmer host)? The human answer on
   training Glimmer on Muse outputs is still pending; if it is no, set
   `TE_TRAIN_ON_MODEL_OUTPUTS=0`.

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
stages after the newest folder (v22 now). The rules, as the author gave them:

- Stages v1 to vN live in the root of `build-stages`, in the order a developer would
  build them: foundations first, docs and UI late.
- Each stage is one commit and adds or changes at most 5 files compared with the stage
  before. The README counts. If a layer needs more, split it into smaller stages that
  each still run. If a change cannot stay under 5 files without breaking a stage, tell
  the author before going over.
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
config are relative; generate a per-run YAML with absolute Modal volume paths. Meta
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
- `ios/` (SwiftUI, with the DAT facts above): Connect, Teach (hold to talk, 10 second
  capture with a countdown, read-back, confirm), Skills, Watch (live indicator, last
  alert, battery), Clips (saved by the user only). Configurable WebSocket URL (local
  Wi-Fi or a tunnel). The backend needs an endpoint that takes the 10 second capture
  (frames and audio) and runs `teach.run_teach`; today teaching is CLI only.

### M3 (Sat to Sun): all skills, evaluation, experiments

- Teach boil_over and bus_m4. A small web labeling tool that marks the time ranges where
  the condition is true and writes JSON; test labels are human only.
- `backend/eval/` and `make eval`: frame precision, recall and F1; event recall within
  5 s; false alarms per hour; time to alert; latency per frame; dollars per hour. Writes
  `eval/results.md` and PNG charts (one palette, values labelled, readable in dark and
  light) and logs everything to W&B.
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
  labels), "teach it yourself" with live Meta calls (budget-capped per visitor and per
  day), and the results page. Precomputed results are fine if serving live is too slow;
  say so on the page.
- Hugging Face collection (ask before every upload): the final LoRA adapter, the Mac
  build, the experiment adapters, the human-labelled test set (faces blurred, approved
  footage only); never generated images. Apache-2.0 if the Glimmer model card allows it
  (check the card). Model card contents are listed in SPEC.md.
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
