# Progress

The source of truth between sessions. Updated at the end of every milestone.
Last update: 2026-11-03, the code for M2 to M4 (stages v31 to v92) and the kitchen skills with
their holds (on main directly), tested offline only;
v1 to v93 are on main.

## Status

M0 and M1 code are done and tested offline. M2 to M4 are written and tested offline too
(stages v31 to v92, then the kitchen skills with a hold per skill, committed directly to
main; 1033 tests pass and one skips without OpenCV, none calls a paid API);
none of it has run against Meta, Modal or a phone yet, and the iOS app has never been
compiled. A full review after M1 (bugs, version compatibility, functionality against
SPEC.md, the UI) found 27 confirmed problems in F1 to F29, fixed in stages v23 to v30. The
rest, F30 to F62, and a second check of those fixes are fixed in the stages that hold that
code (v31 to v34, v41 and v45; the docs in v73 and v92), and so is a review of those (see
"Review after M1" below). M2 is stages v31 to v63 (v31 to v40 the foundations, the iOS
app v49 to v63), M3 v64 to v77 and M4 v78 to v92.
HANDOVER.md explains how to pick the project up; SETUP.md covers setup. What is still
open needs the Mac, the keys, the glasses or new footage:

1. the glasses check with Meta's sample app (`docs/glasses-check.md`)
2. the first live Meta calls: `te models` (free), then `te api-smoke` (under $0.01)
3. the W&B smoke run from Modal: `modal run backend/training/wandb_smoke.py`
4. the first real burner teach run: `te teach --skill burner_left_on --clip <10 s clip>
   --audio <request>`. For a 10 s clip (20 to 24 keyframes) and the default 150 positives
   and 150 negatives the estimator prints about $5.25 to $5.28, and at most $11.52 to
   $11.63 if every label has to be asked again with the larger budget (over the $10 skill
   cap, which would then stop the run partway). Run a small batch first
   (`--positives 6 --negatives 6`) and check the SAM output and the images by eye.
5. real clips for all three skills (`docs/recording-protocol.md`)

## Milestones

### M0, Thu Oct 29: foundations

- [x] Repo layout, `CLAUDE.md`, `PROGRESS.md`, `.env.example`
- [x] Meta API client with content-addressed cache, spend ledger and retries
- [x] `te spend`, `te models`, `te api-smoke`
- [ ] First live Meta call (`te models`, `te api-smoke` on the Mac)
- [x] W&B smoke script for Modal
- [ ] W&B project `trained-eye` with one test run logged from Modal
- [ ] Glasses check: camera frames, glasses mic and speakers with Meta's CameraAccess sample
- [x] Backend skeleton receiving frames over WebSocket (`te serve`, `te ws-selftest`)

### M1, Thu to Fri: teach flow for one skill (washer, since replaced by the burner)

- [x] voice to skill spec (Voice Transcribe, Wonder), with policy refusals and read-back
- [x] keyframe selection (sharpness and near-duplicate filtering)
- [x] SAM region of interest on keyframes (raw output stored; boxes are read, masks not
  decoded yet, see Open issues)
- [x] Muse Image synthetic set with prompt manifest
- [x] Muse Spark labels and filtering, with agreement stats
- [x] Axolotl dataset in the cookbook's format
- [x] `te teach` end to end, `te policy`
- [ ] First live teach run on the Mac (burner_left_on)
- [x] Ask for real clips for all 3 skills (`docs/recording-protocol.md`)
- [ ] Real clips recorded

### Review after M1, Sat Oct 31

- [x] Settings: the fallback switch fails closed (an unknown value is an error), quoted
  `.env` values with a comment lose their quotes, nan or negative numbers are refused,
  and `.env`, the ledger and the cache belong to the project, not the current folder
- [x] Client: the caps are checked again before a retry that may have been billed; any
  served reply is recorded, whatever goes wrong reading it; cached replies keep their cost
- [x] Policy: names with accents, curly apostrophes, titles; no false refusals for child
  locks, bus shelters, espresso shots, "Roomba gets stuck"; the model's wording is
  re-checked only where it decides what is watched; placeholder refusal reasons
- [x] Keyframes: EXIF orientation baked in; ffmpeg leaves the terminal alone
- [x] Data engine: one rejected image edit or unusable label no longer stops the run
  (more than 5, or 20 percent, still does); labels cut off for length are asked again
- [x] Teach: own labels normalised and checked before any spend; keyframe names carry a
  content hash; `--correct` for a wrong read-back; progress lines during paid steps
- [x] CLI: warning when the estimate is over what is left under the caps; how to
  continue after a cap stop or Ctrl+C; EOF at a prompt is a no; `--audio` and `--request`
  exclusive; `te api-smoke` never caches a failed check
- [x] Docs: the glasses check tests reachability while streaming; HANDOVER has SPEC.md's
  visual design rules
- [x] Client (v33): Ctrl+C during a call is charged; an empty transcript is billed, not
  cached; a cached reply that today's checks reject is fetched again; a failed image
  download is a skippable output error; every failed call carries its charge; a label cut
  off mid-answer is asked again; a cache-only lookup
- [x] Settings and spend report (v31): `~` in `TE_DATA_DIR`; long skill names line up
- [x] Policy and compiler (v32): plurals and more relatives; hiding the capture light,
  covert, nutrition and more sports phrasings; named people with "gets ready" or a
  possessive refused again (a v25 regression); corrections that rule something out
  ("not the bathroom light") are allowed, and a refused one says so; requests over 500
  characters stop before the call; read-back wording; duplicate near-misses; the topic
  rules over all of the model's wording
- [x] Keyframes and images (v33 and v34): macOS `._` files skipped, WebP read, clear
  errors for HEIC and unreadable frames, EXIF turns without rewriting EXIF; the image
  manifest is written even when generation stops
- [x] Labels and SAM (v33 and v34): a label that needed the larger budget is free on a
  rerun; SAM skips a rejected frame like the other steps; progress counts failed
  attempts' cost
- [x] Teach and CLI (v41 and v45): the estimate gives a worst case; a one-answer dataset
  stops the run; fallback labels need both answers; a new spec removes the old dataset;
  the clip folder is never the output folder; contradicting labels; labels saved when a
  run stops; inputs and the key checked before the spend question; the failures hint;
  closed stdin; `te serve` prints the phone address and keeps the server's log lines
- [x] CI and setup (v31 and v92): CI installs ffmpeg, so the video tests run there; Mac
  setup with python3.12 and ffmpeg; the `.env.example` W&B note
- [x] Docs (v73 and v92): estimator figures for a real clip, the recording protocol (both
  states in the teaching clip), M2 and M4 notes, open questions
- [x] A review of those fixes (v31 to v33): a reported cost of NaN or below zero is
  charged at the estimate instead (it would have switched the caps off); an unknown
  `~user` in `TE_DATA_DIR` is a settings error; the policy check refuses two names ("Sam
  and Jake arrive"), titles with brand names, "Sam's at the door" and more ways to hide
  the glasses' light, and no longer refuses brand readings, "my lift", "discreet" alerts,
  object counts or other devices' recording lights; a correction's "not" covers at most
  five words; the model's trigger is checked with its "when"
- [x] The same review's small `te teach` fixes (v41): a blank `--request` and negative
  counts are refused before the estimate, the failure reason prints before the hint, and
  a repeated label with a bad value gets the right message

### M2, Fri to Sat: training and watching

- [x] Watch trigger: 3 of the last 4 frames over the threshold, for the skill's hold
  (hold_s), then the cooldown; a stream pause starts the window and the hold again;
  `calibrate()` picks the lowest threshold with at most one false alarm per hour of
  labelled footage
- [x] Student scorer for an OpenAI-compatible endpoint, sending each frame as the
  training examples were built and reading the answer token's logprobs
- [x] Watcher: Muse Spark checks an alert first (a no snoozes the skill for a minute, a
  failed check alerts unconfirmed); every scored frame goes to a JSON-lines log
- [x] Watch mode on the WebSocket: `"watch"` in the hello, scores in each ack, alerts
- [x] Training bundle (skills combined, real, synthetic or mixed, a validation split,
  image paths on the Modal volume) and the Axolotl config, which is Meta's QLoRA recipe
  with this run's paths, W&B names and gradient accumulation 32 on one H100
- [x] Modal training app with the recipe's pins; `te train` (smoke first, then a run
  priced from the smoke, asking always over $15) and `te watch` (replay a clip)
- [ ] The two-step smoke and the first real run on Modal (asks first)
- [x] Export code: `te export` merges on Modal and compares 50 answers (not run yet)
- [x] Serving code: `te serve-student` deploys vLLM on Modal behind STUDENT_API_KEY (not run
  yet; the cookbook has no Mac path for a fine-tuned Glimmer)
- [x] Teaching from the phone: `/teach/start`, `/teach/status`, `/teach/answer` (they need
  `SERVER_TOKEN`), and `GET /skills` for the app
- [x] The iOS app's code in `ios/` (never compiled: needs Xcode on the Mac)
- [ ] The first real export and endpoint, then `TE_STUDENT_URL`; the first Xcode build
- [x] `te calibrate`: a skill's threshold from replayed validation clips and their true
  time ranges
- [ ] Thresholds from real labelled validation footage (needs footage and M3's labels)

### M3, Sat to Sun: all skills and experiments

- [x] Labelling tool: `te label --clip` (true time ranges) and `te label --skill` (200
  generated images for Experiment 2)
- [x] Evaluation harness: `te eval` and `make eval` (frame and event metrics, false alarms
  per hour, time to alert, latency, cost per hour, `eval/results.md`, SVG charts)
- [x] Experiment code: the plan with fixed steps and prices (`te experiments`), the
  unfiltered arm (`te unfiltered`), Spark's error rate (`te spark-errors`), Spark as the
  per-frame scorer (`te watch --scorer spark`), the regression check (`te regress`)
- [ ] Other two skills (needs footage), the experiment runs, real `eval/results.md`

### M4, Mon Nov 2: release

- [x] Playground code: clips with the live timeline, how each skill was built, teach it
  yourself with per-visitor and per-day caps, results (`te playground`; not deployed)
- [x] Release code: `te manifests`, `te regenerate`, `te blur`, `te release` (upload asks)
- [x] `CONTRIBUTIONS.md`
- [ ] Playground deployed; Hugging Face release (after asking); public W&B report
- [ ] README with results (needs the experiment runs)

### Review of M2 to M4, Mon Nov 2

Three reviews (the teach API, the iOS code, the playground) reproduced each defect with
fakes, and a second round with a skeptic per finding checked those fixes and the stage
cut; the fixes are in the stages that add that code (now v33 to v92).

- [x] Teach API: the routes need `SERVER_TOKEN`; a frame longer than 4:1 is refused (it
  would take gigabytes once scaled); answers have a 64 KB limit; odd `step` and endless
  `audio_s` values are a 400, and so are counts over 100000; a token with spaces or smart
  quotes is a 503 that says so; a run that fails early closes its question; on shutdown
  the next paid call is refused, a rate-limit backoff ends at once, the one in flight is
  recorded, and the Meta client checks the caps before every retry
- [x] iOS: the mock picks its feed before it is worn; one watch frame in flight at a
  time, and a backend that stops answering frames gets a fresh connection; a refused
  glasses start is not reported as watching; three teach frames a second on an even
  schedule; the permission step does not count against the first-frame wait, and a
  refused permission ends it; the LED and concurrency notes say only what was checked
- [x] Playground: never trust every proxy (`--forwarded-allow-ips "*"` is refused); an
  IPv6 visitor counts by their /64; the body size is checked as it streams; runs share
  one in-flight total for the ledger's caps; live runs take JSON from the page itself
  only, and the page cannot be framed; deeply nested JSON is a 400 on Python 3.11 too;
  a budget file that cannot be saved no longer blocks the visitor; odd `since` values
  read as 0; the wall clears between runs, counts unsure apart, and alert labels do not
  collide

### Code completion, Mon Nov 2 to Tue Nov 3

An audit compared HANDOVER, PROGRESS and the docs with the code and listed what was
missing; everything that can be written and tested here is built (offline, with fakes).
Three reviews then reproduced 12 defects in it, all fixed: playground redeploys left
removed files public, blur records counted boxes off the picture, te replay kept stale
or failed calibrations, an interrupted te train lost its GPU time, galleries outside
eval/ were not gitignored, the politics rule refused glue guns and ad campaigns but let
pistols through, GET /skills hid Ready before any deploy was recorded, Experiment 2's
arms could mix prompt layouts, and a skill-wide calibrate did not say te serve uses the
served run's threshold.

- [x] SAM 3.1 boxes read with our own reader of Meta's SAM 3 text format (masks are not
  decoded); boxes drive the outline and crops of the teaching keyframes
- [x] Inter 4.1 bundled; the model card takes the base model's license (apache-2.0) and
  ships its text
- [x] Policy: a politics and war rule; compiled cooldowns clamped to 60 to 3600 s
- [x] Spend: the ledger reserves each estimate (thread and file lock) so concurrent te
  processes cannot pass a cap; image token estimates come from past calls
- [x] Watch: cooldowns kept across reconnects; `TE_SYSTEM_IN_USER`, `TE_STUDENT_MAX_TOKENS`;
  thresholds per training run; GET /skills reports training and served
- [x] Training: `--preflight`, `--detach`/`--collect`, W&B hashes, entity and opt-in
  artifacts, base and multi-adapter serving, the Modal spend ledger and the $150 check
- [x] Data: `te clips` (session 1 trains, session 2 validates, session 3 is never used)
- [x] Evaluation: `te replay`, plan costs with exports and serving, per-run calibration,
  the pooled Experiment 4 chart, Experiment 1 failure galleries, Spark's error rate and
  the regression check in results.md, `te eval --readme`
- [x] iOS: a new session after a fold, keepalive pings, recording through an engine tap,
  medium resolution for the teach capture, teach resume, the training status
- [x] Playground: a public mode, the Modal app and `te playground --deploy`,
  `te playground-clip`; clips moved to `data/playground/clips/` (gitignored)
- [x] Release: blur records and manual boxes, blurred keyframes in the release, the test
  set as a dataset, a collection, `te fetch-weights`, `te wandb-report`
- [ ] Not built, for the author: decoding SAM masks (needs Meta's `meta-sam-parser`,
  under the SAM License), a GGUF release for the Mac (needs llama.cpp's converter),
  scoring at the answer token by prefill (needs the real model's tokenizer)

### M5, Tue Nov 3: demo

- [ ] Footage, video edit, polish

## Decisions

- **httpx instead of the OpenAI SDK.** PyPI was blocked in the build sandbox, and talking
  to the OpenAI-compatible endpoints directly gives exact control over retries, caching
  and usage logging. Only `api.meta.ai` ever gets the key.
- **Starlette instead of FastAPI for the backend.** FastAPI is built on Starlette and could
  not be installed in the sandbox; the backend needs one WebSocket and a few HTTP routes.
  Moving to FastAPI later is a small change if it is ever needed.
- **Structured output uses Chat Completions with a `json_schema` response format**, as all
  of Meta's cookbook recipes do. SAM uses the Responses API as SPEC.md describes.
- **Cache key** = SHA-256 of model, endpoint and the normalized request: keys sorted,
  media replaced by its own SHA-256, transport fields dropped, but a schema's
  `properties` order kept (strict structured output generates fields in that order).
- **Spend is counted conservatively.** Retried attempts that may have reached the server
  (read timeouts, 500/502/504) and replies that cannot be priced are charged at the
  estimate. Unusable outputs are charged and recorded as `output_error`, never cached.
- **Never use `-contributor` models**: enforced in config and again at call time, in any
  letter case.
- **Packages under `backend/` have no `__init__.py`** (namespace packages), which keeps
  each version stage small.
- **Version stages** go on the `build-stages` branch as folders v1 to vN. They go to main
  only when the author asks: one commit per stage, made from its folder, with the author
  as author and committer, the stage's text from VERSIONS.md as the message and no
  trailers. v1 to v93 are on main now.
- **Policy is checked three times.** A local rule check (`te policy`) refuses clear cases
  (named or described people, health, covert watching, sensitive places, sports coaching)
  before any call, at no cost. Muse Wonder can refuse, including through a filled-in
  refusal reason. Then the local check runs again on the spec the model wrote. Phrases are
  used instead of bare words where a word also names an object, so walk signs ("green
  man"), kitchen scales ("weight") and meat thermometers are not refused. Shown in the
  demo with "Tell me when Sam gets here".
- **The compiler schema is flat** (no nested objects or unions except one nullable
  string), the safest shape for strict structured output; code builds the SkillSpec.
- **Synthetic data are edits of the user's own keyframes** (anchored generation). The plan
  walks every (condition, keyframe, nuisance) combination once, in a fixed shuffled order,
  before repeating, so no two requests are identical (11 nuisance variations: light,
  angle, distance, blur, occlusion). Every prompt asks for no people in view. The plan is
  deterministic, so a rerun hits the cache and costs nothing.
- **Output that parses but is unusable is never cached** (for example a spec with an
  empty target): `structured()` takes a validate hook, so such answers are billed once
  and asked again on the next run.
- **Labels:** every image gets a Spark yes/no/unsure label; a generated image is kept only
  if Spark agrees with the label it was generated for, and "unsure" is dropped. Keyframes
  from the teaching clip are labelled by Spark unless the user supplies labels.
- **Dataset format follows Meta's Axolotl recipe**: image part first with an absolute PNG
  path, then the text; images padded to 512x512. The spec's system turn is kept, with a
  switch to fold it into the user text if the processor preflight shows a problem.
- **Fallback switch:** `TE_TRAIN_ON_MODEL_OUTPUTS=0` keeps generating and labelling (for
  evaluation) but trains only on real frames with the user's own labels. Without those
  labels, or with labels that are all one answer, it stops after writing the keyframes,
  before any paid call.
- **A dataset needs both answers.** If the examples would all be yes or all be no (for
  example when Spark rejects every positive edit), teach stops instead of writing a
  dataset that cannot train a yes/no check, and warns when one answer is under 10 percent.
- **Failed calls carry their charge** (`MetaAPIError.cost_usd`), so progress lines count
  them; after a stop for too many failures, the hint says a rerun pays for them again.
- **The model's wording is checked in two parts.** The target and the trigger get the
  request's full local check; everything else it wrote (description, near-misses, alert)
  gets the covert, health and sports rules, and the description also the person and
  sensitive-place rules except rooms at home.
- **Project folder:** `.env`, `data/` (ledger, cache, skills) and a relative `TE_DATA_DIR`
  are taken inside the checkout `backend/` is imported from, so `te` sees the same spend
  and caps from any folder. Tests point this at a temporary folder (tests/conftest.py).
- **Per-item failures:** an image edit rejected with 400 or 422, or a reply that cannot
  be used, skips that one item (recorded in the manifest with the error); more than
  max(5, 20 percent) failures stops the run, so a systematic problem does not keep paying.
  Auth, rate-limit, server and budget errors always stop it.
- **Keyframe names** are `kf_<position>_<first 8 hex of the SHA-256>.<ext>`, so a labels
  file made for another take of the clip cannot label the wrong frames.
- **The student is scored at its answer token.** The scorer asks the endpoint for
  logprobs and reads the last generated yes or no, renormalised over the two; a reply
  without one is an error, never a guess. It needs no chat template, so it works the same
  on vLLM and on llama-server.
- **Alerts:** a skill fires on 3 of its last 4 frames at or over its threshold, once that
  has lasted its hold, then waits for its cooldown. If Muse Spark says no, the skill
  waits a minute instead; if the check fails, the alert goes out marked unconfirmed
  (missing a burner left on is worse than one unchecked alert). Frames are never written;
  the log holds times, scores and decisions.
- **The skills and their holds (Nov 3):** the skills are burner_left_on, fridge_left_open
  and tap_left_running, with egg_whisked if it is recorded; they replace washer_done,
  boil_over and bus_m4. Each has a hold (hold_s in the spec): the condition must stay on,
  after the 3-of-4 smoothing, that long before it alerts, counted from the first frame
  over the threshold, so using the fridge or the tap briefly says nothing. The author set
  them: the fridge 30 s, the tap 20 s, the burner 10 s, the egg 0 s (`SKILL_HOLDS`); the
  compiler gives these names these holds and any other skill the model's own value, kept
  between 0 and 600 s. The evaluation counts an alert as on time when it comes within the
  skill's window after the end of the hold (`metrics.ON_TIME_S`): 5 s for the burner and
  the tap, which are seen at once and matter soonest, 10 s for the fridge, whose view
  changes as the wearer moves, and for the egg, whose foam builds up gradually; a plan's
  `on_time_s` can change them. A labelled range shorter than the hold expects no alert, and
  an alert before the hold ended is a false alarm. The burner skill is a reminder, never
  a safety device.
- **Student sampling follows Meta's serving notes:** no temperature in the request (the
  cookbook advises against greedy decoding for Glimmer) and reasoning_strength low; the
  answer token's logprobs are read either way.
- **Experiments keep the steps fixed:** every run of Experiments 1 to 3 trains the same
  number of optimizer steps (enough for 300 images per skill at the recipe's batch of
  32), so the image-count curve measures data, not steps (backend/eval/plan.py).
- **Charts are SVG**, written by hand (matplotlib could not be installed), with the light
  and dark styles inside each file.
- **Phone teach captures are kept** under `data/teach/` (gitignored) so a run can be
  repeated from the Mac; watch frames are still never stored.
- **The teach routes need a token** (`SERVER_TOKEN` in `.env`, typed once into the app):
  `te serve` listens on the whole network, and without it anyone there could agree to a
  phone run's spend. `/ws` stays open, as in M1, and so does `/skills`.
- **Session clips:** session 1 is training data, session 2 validation (never trained on;
  every run calibrates its threshold on it, so the arms stay comparable), session 3 the
  test set only. Experiment 1's real arm is the keyframes, their SAM crops and session 1.
- **SAM 3 text format, our own reader:** Meta's protocol spec (meta-models/meta-sam,
  SAM License) was read and implemented; no code or test data is copied from it.
- **Cooldowns are per skill, not per session:** te serve keeps them across reconnects,
  skill toggles and restarts, so one condition is announced once per cooldown.
- **Training config as JSON:** the Axolotl config is Meta's recipe written as JSON (YAML
  reads it), with only the paths, the W&B names and gradient accumulation changed.
- **Provenance:** generated images are stored exactly as returned; the padded dataset PNGs
  are derived copies, and the dataset manifest maps every example back to its source and
  the SHA-256 of the original image. Only real, approved footage is ever published.

## Docs vs spec

Sources read: the Meta OSS cookbook (commit b30ca2c, Oct 28) and the DAT iOS SDK repo
(1.0.1, Oct 9), cloned read-only. dev.meta.ai, wearables.developer.meta.com and
ml-explore.github.io were blocked from the build sandbox, so the hosted Meta Model API
pages, the DAT online docs and the MLX docs could not be checked. Where the reachable
sources disagree with SPEC.md, the sources win.

| # | SPEC.md says | The sources say | What we do |
|---|---|---|---|
| 1 | Base URL `https://api.meta.ai/v1`, model IDs, prices, SAM output format, image and audio endpoints | None of it is in the cookbook or the DAT repo (they cover only open-weight models) | Kept from SPEC.md, marked unverified; `te models` checks the IDs on the first live call |
| 2 | `structured(model, input, schema)`, request shape not given | Cookbook recipes use Chat Completions with `response_format` json_schema | Implemented that way |
| 3 | Score frames by the probability that the first token is yes or no | Glimmer and Spark always reason first (`to=self` channel), so the first token is not the answer; the recipe's targets are an explanation followed by the answer, not a bare yes or no | M2: the scorer generates with logprobs and reads the answer token (the last yes or no), renormalised over the two; prefilling up to the answer channel is a later speed-up; check how a bare yes/no target renders before training |
| 4 | Base model pinned "to the revision the guide uses" | Cookbook pins `a4e59da52a7bc87ae7251dd5545c0dd437c44b68` | Pinned in `backend/config.py` |
| 5 | Dataset: images inline, system prompt in every example | Recipe uses image parts with an absolute local PNG `path`; its examples have no system turn; images padded to 512x512 | M1 follows the recipe; system turn kept but can be folded into the user text |
| 6 | QLoRA verified on one H100; L40S or A100 if needed | Meta verified quality on 8x H100; 1x H100 only as a 2-step smoke (peak about 27 GB, on a 96 GB H100); L40S and A100 untested; no merge step in the cookbook | M2: own 2-step smoke on Modal first; merge is our own step |
| 7 | MLX 4-bit with mlx-vlm, about 18-21 GB, about 17.6 tokens/s on an M5 Pro | mlx-vlm is never mentioned; the only Glimmer MLX builds are text-only and need 48 GB+; the verified Mac vision path is llama.cpp (GGUF plus mmproj); no conversion path for a fine-tuned model; 18-21 GB and 17.6 tokens/s appear nowhere (one measurement: about 24.1 GiB for Q4_K_M at 16K context) | M2 risk. Decide after the memory check; never quote those numbers |
| 8 | Modal vLLM endpoint as the fallback | vLLM needs a dedicated Docker image, image input is unattested, bf16 needs about 72 GB of VRAM | Needs an 80 GB GPU; test an image request early |
| 9 | Serving on Modal counts as partner inference | The cookbook lists only Together AI as a Glimmer host | Ask the organizers |
| 10 | "About 500+ quality examples"; "track general capability" | Neither is in the cookbook; it asks for separate text and vision regression sets | Add a small text regression set next to the 50 visual questions |
| 11 | Adventurer glasses, DAT "Meta glasses" column | No "Adventurer" anywhere; "Meta Glasses" (`metaGlasses`) supported since DAT 0.8 | The glasses check decides |
| 12 | DAT 1.0.1 needs firmware V128 and Meta AI V292 | No version numbers in the repo (the table is on the blocked docs site) | Unverified; update both before the check |
| 13 | Wearables Developer Center org, project and release channel | The CameraAccess sample says a Developer Mode build needs no credentials, but two SDK docs list Camera as a capability approved in the Developer Center | Try Developer Mode alone first; if the camera is refused, register a project and request Camera |
| 14 | 1 fps from the app | DAT frame rates are 2, 7, 15, 24 or 30 | Stream at 2 or 7 fps, send 1 fps |
| 15 | 12 MP camera | Video stream is at most 720x1280; the sample default is 360x640 | Size the watcher for 360x640 to 720x1280 |
| 16 | Record the glasses mic via DAT | DAT mic audio is experimental and needs approval; the sample records the glasses mic over Bluetooth HFP | Use HFP |
| 17 | Alerts with AVSpeechSynthesizer routed to the glasses | DAT has no speaker API; audio is ordinary Bluetooth from the phone; recording over HFP makes the glasses the default input, and a spoken alert comes back through an open mic | Speak only when a Bluetooth route is present; stop listening while speaking (as Meta's BirdSpotter sample does) |
| 18 | Frames stay on your devices | DAT analytics and crash reporting are on by default | Opt out in Info.plist; say so in the privacy notes |
| 19 | Watch mode always on | A raw stream pauses in the background, HEVC (hvc1) keeps going | Use hvc1 for watch mode |
| 20 | Credit sample code in CONTRIBUTIONS.md | DAT sample code is under Meta's Wearables Developer Terms, not an open-source license | Keep Meta's headers and LICENSE next to any copied file; check the terms allow publishing |
| 21 | Never hide the capture LED | The DAT repo says nothing about the LED | Observe it during the glasses check |

## Open issues

- Waiting on the glasses check, the first live Meta calls and the W&B smoke run (Status).
- The build sandbox has no keys and blocks PyPI and the Meta docs sites, so live calls
  and the Modal run happen on the Mac. The network policy can be widened in the cloud
  environment settings if needed.
- Mac memory unknown (`sysctl hw.memsize`); it decides local vs Modal serving.
- SAM masks are not decoded: our reader takes boxes only. Outlines are boxes until the
  author approves Meta's `meta-sam-parser` (SAM License).
- SPEC.md is not in the repository, so a fresh session needs it uploaded again.
- From the review, for the author: whether to add `httpx2` (Starlette 1.2 and later warn
  without it in `TestClient`) or cap `starlette<1.2`, and whether the project's own
  dependencies count as approved downloads (HANDOVER.md, "Not done yet", item 9).
- In fallback mode the condition text in every example is still Muse Wonder's (ask the
  organizers with the training question).
- The dataset JSONL holds absolute image paths (the recipe's format); M2 writes it again
  under the Modal volume mount.
- The public playground turns the walls off and serves only blurred keyframes and clips,
  from Modal with a private Volume (`te playground --deploy`); check per-visitor budgets
  from two networks after the first deploy. Uploading to Hugging Face needs
  `huggingface_hub` and the W&B report `wandb-workspaces` (neither approved yet); face
  blurring needs `opencv-python-headless` (approved). Inter 4.1 is bundled.
- SAM output parsing follows Meta's SAM 3 protocol (meta-models/meta-sam, our own reader in
  `backend/dataengine/sam_text.py`); the first live run should confirm it reads SAM 3.1's
  real output (the raw text is stored with every frame, so it can be read again for free).
- Image token counts are unknown until the first calls: estimates assume 1,500 tokens per
  image, then use the mean measured from the ledger's compile and label calls.
- Unverified until run for real: everything on Modal (preflight, detached runs, base and
  multi-adapter serving), the W&B report's API calls, and every iOS change (not compiled;
  `ios/README.md` lists the checks on the glasses).
- The teach flow takes keyframes from a video file or a folder of frames; the iOS Teach
  screen (M2) sends the 10 s capture as frames, three a second (not compiled yet).

## Spend

| Milestone | Meta API | Modal GPU | Where the number comes from |
|---|---|---|---|
| M0 | $0.00 | $0.00 | no paid call or Modal job has run yet (`te spend` shows 0 calls) |
| M1 | $0.00 | $0.00 | built and tested offline; the first teach run happens on the Mac |
| M2 | $0.00 | $0.00 | built and tested offline by Nov 3; no Modal job has run yet |
| M3 | $0.00 | $0.00 | built and tested offline by Nov 3; the experiment runs need footage |
| M4 | $0.00 | $0.00 | built and tested offline by Nov 3; nothing deployed or published |
| Total | $0.00 of $150 credit (cap $120) | $0.00 of about $200 credit | |
