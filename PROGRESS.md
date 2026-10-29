# Progress

The source of truth between sessions. Updated at the end of every milestone.
Last update: 2026-10-29, end of M0.

## Status

M0 code is done and tested offline. Three M0 checks need the Mac, the keys or the
glasses, so they are still open:

1. the glasses check with Meta's sample app (`docs/glasses-check.md`)
2. the first live Meta calls: `te models` (free), then `te api-smoke` (under $0.01)
3. the W&B smoke run from Modal: `modal run backend/training/wandb_smoke.py`

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

### M1, Thu to Fri: teach flow for one skill (washer)

- [ ] voice to skill spec (Voice Transcribe, Wonder), with policy refusals and read-back
- [ ] SAM region of interest on keyframes
- [ ] Muse Image synthetic set with prompt manifest
- [ ] Muse Spark labels and filtering
- [ ] Axolotl dataset
- [ ] Ask for real clips for all 3 skills (train/validation and test sessions)

### M2, Fri to Sat: training and watching

- [ ] Modal QLoRA training on the washer skill, export, serving (Mac or Modal)
- [ ] Watch loop: smoothing, Spark confirmation, spoken alerts on the glasses

### M3, Sat to Sun: all skills and experiments

- [ ] Other two skills, labeling tool, eval harness
- [ ] Experiments 1 to 4, charts, `eval/results.md`

### M4, Mon Nov 2: release

- [ ] Playground deployed; Hugging Face release (after asking); public W&B report
- [ ] README with results, `CONTRIBUTIONS.md`

### M5, Tue Nov 3: demo

- [ ] Footage, video edit, polish

## Decisions

- **httpx instead of the OpenAI SDK.** PyPI was blocked in the build sandbox, and talking
  to the OpenAI-compatible endpoints directly gives exact control over retries, caching
  and usage logging. Only `api.meta.ai` ever gets the key.
- **Starlette instead of FastAPI for the backend.** FastAPI is built on Starlette and could
  not be installed in the sandbox; the backend needs one WebSocket and a health route.
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
- **Version stages** go on the `build-stages` branch as folders v1 to vN; the author
  commits them to main one at a time.

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
| 3 | Score frames by the probability that the first token is yes or no | Glimmer and Spark always reason first (`to=self` channel), so the first token is not the answer; the recipe's targets are an explanation followed by the answer, not a bare yes or no | M2: prefill up to the answer channel and read the yes/no logits there; check how a bare yes/no target renders before training |
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
- The spend cap is not safe across concurrent processes (the check and the record are not
  atomic). Fine while calls run one at a time; fix before parallel labeling.
- `meta-sam-parser` (SAM output parsing) could not be installed or read; M1 stores SAM's
  raw output and parses it once a real cached response shows the format.
- SPEC.md is not in the repository, so a fresh session needs it uploaded again.

## Spend

| Milestone | Meta API | Modal GPU | Where the number comes from |
|---|---|---|---|
| M0 | $0.00 | $0.00 | no paid call or Modal job has run yet (`te spend` shows 0 calls) |
| Total | $0.00 of $150 credit (cap $120) | $0.00 of about $200 credit | |
