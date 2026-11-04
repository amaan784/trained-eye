# Trained Eye

Teach your glasses to notice anything. Build spec: SPEC.md (kept outside the repo).
Start with HANDOVER.md (state, rules, next steps), then PROGRESS.md, the source of truth
between sessions. SETUP.md covers accounts, the Mac and cloud sessions.

## Setup (MacBook)

    brew install python@3.12 ffmpeg  # python3 may be Apple's 3.9; ffmpeg reads video clips
    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"          # add ,train for Modal and W&B
    cp .env.example .env             # then fill in the keys

## Commands

    pytest                           # tests; never call paid APIs
    ruff check .
    te spend                         # Meta API spend from data/ledger.jsonl
    te serve                         # backend; prints the ws:// address for the phone
                                     # (phone teaching needs SERVER_TOKEN in .env)
    te ws-selftest                   # synthetic frames through the WebSocket, in process
    te api-smoke                     # one tiny paid Muse Spark call (needs MODEL_API_KEY)
    te policy "<request>"            # local policy check, free
    te teach --skill burner_left_on --clip <video or frame folder> --audio <request audio>
                                     # teach flow; prints a cost estimate and asks first
                                     # --correct "<fix>" after a wrong read-back,
                                     # --labels <json> with names from keyframes/
    te train --skills burner_left_on --run exp1-mixed --smoke
                                     # the recipe's two-step smoke on Modal; asks first.
                                     # Without --smoke: the real run, priced from the smoke.
                                     # --preflight renders examples with the pinned processor
                                     # (CPU job, asks); --detach, then --collect <run>
    te clips --skill burner_left_on  # labelled session1_/session2_ clips -> dataset-clips (free)
    te watch --skills burner_left_on --clip <video or frame folder>
                                     # replay a clip through watch mode (needs TE_STUDENT_URL);
                                     # --run <run> uses that run's threshold
    te calibrate --skill burner_left_on --log <frame log> --labels <true ranges json>
                                     # threshold from labelled validation clips (--run: per run)
    te export --run exp1-mixed       # merge the adapter on Modal and check it (asks)
    te serve-student --run exp1-mixed  # the trained model on Modal (asks); --stop stops it;
                                     # --base: base Glimmer; --runs a,b: adapters on one base
    te label --clip <clip>           # mark where the condition is true (writes <clip>.labels.json)
    te label --skill burner_left_on --sample 200   # hand-label generated images (Experiment 2)
    te experiments --skills burner_left_on [--write-plan]  # every training run with its price
    te replay --run exp1-mixed       # the run's clips in eval/plan.json: threshold from its
                                     # session 2 replays, then the test clips into the logs
    te unfiltered / te spark-errors  # Experiment 2: the unfiltered arm, Spark's error rate
    te watch ... --scorer spark      # Experiment 4: Muse Spark on every frame (paid; asks)
    te regress                       # the regression check on general questions
    make eval                        # te eval: eval/results.md and the SVG charts
    te manifests / te regenerate / te blur / te release   # M4 release (upload asks)
    te fetch-weights --run <run>     # a run's adapter from the Modal volume into data/weights/
    te release-test-set / te release-collection / te wandb-report   # more releases (all ask)
    te playground                    # the web playground, locally (--public: no generated
                                     # images, blurred keyframes only; --deploy: Modal, asks)
    te playground-clip --clip <clip> --log <log> --labels <json> --name <n>  # a blurred clip
    modal run backend/training/wandb_smoke.py   # one W&B run from Modal

## Conventions

- Model IDs and prices live in backend/config.py. Never use `-contributor` models.
- Every Meta call goes through backend/meta/client.py (cache + spend ledger + retries).
- Secrets: .env locally, Modal Secrets ("wandb", "huggingface", "student", "meta") in
  Modal jobs. Never print them.
- Local data (footage, cache, ledger, generated images, weights) lives in data/ and is gitignored.
- Packages under backend/ are namespace packages (no __init__.py files).
- Ask before any job over $15, before publishing anything, and before anything physical.
- Skills: burner_left_on, fridge_left_open, tap_left_running (egg_whisked if recorded). Each
  alerts only after its hold (SKILL_HOLDS in backend/skills/spec.py: 10, 30, 20, 0 s), and
  the evaluation's on-time window per skill (ON_TIME_S in backend/eval/metrics.py) counts
  from the end of the hold.
- Policy: no skills about specific people, and none about politics or war; never hide the
  capture LED; keep Muse Image provenance; blur faces in anything published.
- Numbers in docs come only from eval scripts.

## Pointers

- docs/glasses-check.md: how to verify the glasses with Meta's DAT sample app.
- docs/recording-protocol.md: what real clips to record for the experiments.
- Teach-flow outputs go to data/skills/<skill>/ (gitignored): spec, keyframes, roi,
  synthetic images with manifest.jsonl, labels.jsonl, label_stats.json, dataset/ (with SAM
  crops), and dataset-clips/ from te clips (session 1 train, session 2 validation only).
- Meta cookbook: github.com/meta-models/meta-oss-cookbook (muse-glimmer/post-training).
- DAT iOS SDK: github.com/facebook/meta-wearables-dat-ios (samples/CameraAccess).
- ios/README.md: how to build the iPhone app in Xcode (never compiled yet).
- CONTRIBUTIONS.md: credits, licenses and the copied Meta sample files.
