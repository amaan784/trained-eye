# Version stages

The finished project split into stages v1 to v92, in the order a developer would build
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
- The last folder, v92, is the finished project: byte for byte the same as the
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
| v20 | M1: te teach | te teach, which prints a cost estimate and asks before spending. | `backend/cli.py` (changed), `tests/test_cli.py` (changed), `README.md` (changed) | `ffmpeg -v error -y -f lavfi -i testsrc=duration=10:size=1280x720:rate=30 -pix_fmt yuv420p /tmp/te-demo.mp4` | 185 passed |
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
| v31 | M2: Settings and the spend ledger | The settings and the spend ledger the later stages need: the student endpoint, the two new secrets, and spend reserved across threads and processes. | `.env.example` (changed), `.github/workflows/ci.yml` (changed), `backend/config.py` (changed), `backend/meta/ledger.py` (changed), `tests/test_config.py` (changed), `tests/test_ledger.py` (changed), `README.md` (changed) | `python -m backend.config` | 318 passed |
| v32 | M2: Policy and the skill compiler | The policy check's new rules (hiding the glasses' capture light, politics and war) and checks on the compiled skill's own wording. | `backend/skills/compiler.py` (changed), `backend/skills/spec.py` (changed), `tests/test_compiler.py` (changed), `tests/test_skills.py` (changed), `README.md` (changed) | `python -m backend.cli policy "let me know when the washer says done"` | 483 passed |
| v33 | M2: Meta calls: client, labels and image edits | The Meta client's hardening: interrupted and failed calls charged, the caps checked again before every retry, and SAM 3.1's streamed reply read by the protocol's rules. | `backend/dataengine/labels.py` (changed), `backend/dataengine/synth.py` (changed), `backend/meta/client.py` (changed), `tests/test_client.py` (changed), `tests/test_labels.py` (changed), `tests/test_synth.py` (changed), `README.md` (changed) | `python -m pytest` | 516 passed |
| v34 | M2: Keyframes, SAM boxes and the dataset | Our reader of SAM 3.1's text format, with boxes and crops for every keyframe. | `backend/dataengine/dataset.py` (changed), `backend/dataengine/keyframes.py` (changed), `backend/dataengine/roi.py` (changed), `backend/dataengine/sam_text.py`, `tests/test_keyframes.py` (changed), `tests/test_roi.py` (changed), `tests/test_sam_text.py`, `README.md` (changed) | `python -m pytest` | 565 passed |
| v35 | M2: Watch trigger and the student scorer | The watch trigger (3 of the last 4 frames over the threshold, then the cooldown) and the student scorer on an OpenAI-compatible endpoint. | `backend/inference/student.py`, `backend/watch/trigger.py`, `tests/test_config.py` (changed), `tests/test_student.py`, `tests/test_trigger.py`, `README.md` (changed) | `python -m pytest` | 588 passed |
| v36 | M2: Training run records and the Modal spend ledger | The training run records (running, done or failed, and which run is served) and the Modal spend ledger with its check against the $150 credit. | `backend/training/gpu_ledger.py`, `backend/training/runs.py`, `tests/test_gpu_ledger.py`, `tests/test_training_runs.py`, `README.md` (changed) | `python -m pytest` | 595 passed |
| v37 | M2: The watcher | Watch mode itself: the student's scores, the trigger, the Muse Spark check before an alert, cooldowns kept across sessions and the frame log. | `backend/watch/watcher.py`, `tests/test_watcher.py`, `README.md` (changed) | `python -m pytest` | 609 passed |
| v38 | M2: Training bundle and the recipe's config | The training bundle (real, synthetic or mixed, with a fixed validation split) and Meta's QLoRA recipe as the Axolotl config. | `backend/training/axolotl.py`, `backend/training/prepare.py`, `tests/test_training_prep.py`, `README.md` (changed) | `python -m pytest` | 619 passed |
| v39 | M2: The processor preflight | The processor preflight: bundle examples rendered with the pinned Glimmer processor before the paid smoke. | `backend/training/preflight.py`, `tests/test_training_preflight.py`, `README.md` (changed) | `python -m pytest` | 626 passed |
| v40 | M2: The Modal training app | The Modal training app: Meta's QLoRA recipe on one H100 with the recipe's pins, the two-step smoke, detached runs and the processor preflight on a CPU. | `backend/training/modal_train.py`, `tests/test_modal_train.py`, `tests/test_training_preflight.py` (changed), `README.md` (changed) | `python -m pytest` | 638 passed |
| v41 | M2: Teach checks, worst case and crops | te teach's checks before any money is mentioned, the worst-case price with the $15 rule, and the keyframes cropped to SAM's box. | `backend/cli.py` (changed), `backend/teach.py` (changed), `tests/test_cli.py` (changed), `tests/test_teach.py` (changed), `README.md` (changed) | `python -m backend.cli teach --help` | 668 passed |
| v42 | M2: te train on Modal | te train: the bundle, the two-step smoke on Modal, detached runs collected later, the processor preflight, and the real run's pricing from the smoke, which needs v74's experiment plan. | `backend/cli.py` (changed), `backend/cli_train.py`, `tests/test_cli_train.py`, `tests/test_modal_train.py` (changed), `README.md` (changed) | `python -m backend.cli train --help` | 679 passed |
| v43 | M2: Teach sessions from the phone | The teach session behind teaching from the phone: run_teach in a background thread, the spend question first, read-back corrections, timeouts and a clean stop. | `backend/teach_session.py`, `tests/test_teach_session.py`, `README.md` (changed) | `python -m pytest` | 689 passed |
| v44 | M2: The phone's teach API | The phone's teach API: the capture uploaded, checked and priced, then the spend and read-back questions answered over HTTP, each request with the server token. | `backend/teach_api.py`, `backend/teach_upload.py`, `tests/test_teach_api.py`, `tests/test_teach_upload.py`, `README.md` (changed) | `python -m pytest` | 715 passed |
| v45 | M2: Watch mode and teaching on the server | The server's watch mode (scores in each ack, alerts as messages, one cooldown book for every session), GET /skills, the teach routes, and a te serve that prints the address the phone uses. | `backend/cli.py` (changed), `backend/server.py` (changed), `tests/test_cli.py` (changed), `tests/test_server.py` (changed), `README.md` (changed) | `python -m backend.cli serve --help` | 730 passed |
| v46 | M2: Replaying clips and calibrating thresholds | te watch, which replays a clip through watch mode with the trained model, and te calibrate, which sets a skill's or a run's threshold from labelled frame logs. | `backend/cli.py` (changed), `backend/cli_watch.py`, `pyproject.toml` (changed), `tests/test_cli_watch.py`, `README.md` (changed) | `python -m backend.cli watch --help` | 736 passed |
| v47 | M2: Merging and checking the adapter | The export: a run's adapter merged into the bf16 base on Modal, then the merged model compared with base plus adapter on up to 50 bundle examples. | `backend/training/export.py`, `backend/training/export_check.py`, `tests/test_export.py`, `README.md` (changed) | `python -m pytest` | 743 passed |
| v48 | M2: Export and the student endpoint | te export and te serve-student: the merge run from the Mac, and the trained model, the base or the base with adapters served by vLLM on Modal. | `backend/cli.py` (changed), `backend/cli_serving.py`, `backend/inference/modal_serve.py`, `tests/test_cli_serving.py`, `tests/test_modal_serve.py`, `README.md` (changed) | `python -m backend.cli export --help` | 755 passed |
| v49 | M2: iOS: settings and the protocol | The iPhone app's first files: its settings, with the server token in the keychain, and the backend's WebSocket, /health and /skills messages as Swift types. | `ios/TrainedEye/Models/Protocol.swift`, `ios/TrainedEye/Models/Settings.swift`, `README.md` (changed) | `python -m pytest` | 755 passed |
| v50 | M2: iOS: backend and WebSocket clients | The app's two clients: plain HTTP calls to te serve with the server token, and the watch WebSocket with one frame in flight, reconnects and a keepalive. | `ios/TrainedEye/Network/BackendClient.swift`, `ios/TrainedEye/Network/WatchSocket.swift`, `README.md` (changed) | `python -m pytest` | 755 passed |
| v51 | M2: iOS: Meta's HEVC decoder and teach messages | The HEVC frame decoder and keyframe check from Meta's CameraAccess sample, copied unchanged with Meta's LICENSE, and the teach API's messages as Swift types. | `ios/TrainedEye/Meta/HEVCKeyframe.swift`, `ios/TrainedEye/Meta/LICENSE`, `ios/TrainedEye/Meta/VideoFrameDecoder.swift`, `ios/TrainedEye/Models/TeachModels.swift`, `README.md` (changed) | `python -m pytest` | 755 passed |
| v52 | M2: iOS: glasses connection and the mock device | The glasses connection (registration with the Meta AI app, link, battery, firmware) and DAT's mock glasses for Debug builds, with a drawn washer clip as the simulator's feed. | `ios/TrainedEye/Glasses/GlassesConnection.swift`, `ios/TrainedEye/Glasses/MockGlasses.swift`, `ios/TrainedEye/Resources/mock_feed.mp4`, `ios/make_mock_feed.py`, `README.md` (changed) | `python -m pytest` | 755 passed |
| v53 | M2: iOS: the glasses camera | The glasses camera: a DAT session that comes back by itself after a fold or a link drop, the camera permission asked first, and a change of stream size. | `ios/TrainedEye/Glasses/GlassesCamera.swift`, `README.md` (changed) | `python -m pytest` | 755 passed |
| v54 | M2: iOS: the glasses stream and frame throttle | The glasses stream (HEVC at 7 fps, a stopped stream added again, repeated images skipped) and the throttle that turns frames into one JPEG a second. | `ios/TrainedEye/Camera/FrameThrottle.swift`, `ios/TrainedEye/Glasses/GlassesCamera+Stream.swift`, `README.md` (changed) | `python -m pytest` | 755 passed |
| v55 | M2: iOS: phone camera and spoken alerts | The iPhone camera for when the glasses are not there, and spoken alerts that play only over Bluetooth, never from the phone's speaker. | `ios/TrainedEye/Audio/Speaker.swift`, `ios/TrainedEye/Camera/PhoneCamera.swift`, `README.md` (changed) | `python -m pytest` | 755 passed |
| v56 | M2: iOS: the spoken request recorder | The request recorder: the spoken teach request as 16 kHz WAV in memory, from the glasses' microphone when it is the input, else the iPhone's. | `ios/TrainedEye/Audio/RequestRecorder.swift`, `README.md` (changed) | `python -m pytest` | 755 passed |
| v57 | M2: iOS: the teach client and flow | The teach client and the teach flow on the phone: hold to talk, a 10 second capture, the upload, then the Mac's spend and read-back questions. | `ios/TrainedEye/Network/TeachClient.swift`, `ios/TrainedEye/Teach/TeachFlow.swift`, `README.md` (changed) | `python -m pytest` | 755 passed |
| v58 | M2: iOS: the app model | The app model that ties the parts together: which camera runs, where its frames go, what an alert does, and what went out while the screen was locked. | `ios/TrainedEye/App/AppModel.swift`, `README.md` (changed) | `python -m pytest` | 755 passed |
| v59 | M2: iOS: theme, components and the Inter fonts | The app's look: a dark, minimal theme with one accent colour in Inter 4.1, and the cards, buttons and rows every screen uses. | `ios/TrainedEye/Fonts/Inter-Bold.ttf`, `ios/TrainedEye/Fonts/Inter-Medium.ttf`, `ios/TrainedEye/Fonts/Inter-Regular.ttf`, `ios/TrainedEye/Fonts/Inter-SemiBold.ttf`, `ios/TrainedEye/Fonts/LICENSE.txt`, `ios/TrainedEye/UI/Components.swift`, `ios/TrainedEye/UI/Theme.swift`, `README.md` (changed) | `python -m pytest` | 755 passed |
| v60 | M2: iOS: app entry, tabs and project files | The app entry that configures the DAT SDK, the five tabs under a camera-on bar, and the Info.plist and entitlements, with the SDK's analytics and crash reporting off. | `ios/TrainedEye-Info.plist`, `ios/TrainedEye.entitlements`, `ios/TrainedEye/App/TrainedEyeApp.swift`, `ios/TrainedEye/UI/RootView.swift`, `README.md` (changed) | `python -m pytest` | 755 passed |
| v61 | M2: iOS: Connect and Teach screens | The Connect screen (the backend address and token, the glasses, the camera choice, the Debug mock) and the Teach screen with its capture and the Mac's questions. | `ios/TrainedEye/UI/ConnectView.swift`, `ios/TrainedEye/UI/TeachView.swift`, `README.md` (changed) | `python -m pytest` | 755 passed |
| v62 | M2: iOS: Skills and Watch screens | The Skills screen (each taught skill with its watch switch and status) and the Watch screen (alerts, scores, the connection, the glasses and the background counts). | `ios/TrainedEye/UI/SkillsView.swift`, `ios/TrainedEye/UI/WatchView.swift`, `README.md` (changed) | `python -m pytest` | 755 passed |
| v63 | M2: iOS: saved clips and the build guide | The clips the user saves, as frame folders te watch can replay, and ios/README.md: how to build the app in Xcode and what to check on the glasses. | `ios/README.md`, `ios/TrainedEye/Clips/ClipStore.swift`, `ios/TrainedEye/UI/ClipsView.swift`, `README.md` (changed) | `python -m pytest` | 755 passed |
| v64 | M3: Hand labels for generated images | The local page for hand-labelling generated training images, which shows each image with the question Muse Spark was asked and never Spark's answer. | `backend/labeltool/images.py`, `backend/labeltool/static/fonts/InterVariable.woff2`, `backend/labeltool/static/fonts/LICENSE.txt`, `backend/labeltool/static/images.html`, `backend/labeltool/static/style.css`, `tests/test_labeltool_images.py`, `README.md` (changed) | `python -m pytest` | 759 passed |
| v65 | M3: Hand labels for recorded clips | The clip labelling page: marking where a skill's condition is true in a recorded clip, saved as the time ranges te calibrate reads. | `backend/labeltool/clips.py`, `backend/labeltool/static/clips.html`, `tests/test_labeltool_clips.py`, `README.md` (changed) | `python -m pytest` | 777 passed |
| v66 | M3: Evaluation metrics and the chart style | The evaluation metrics from te watch frame logs and human labels (frame and event scores, false alarms per hour, latency, dollars per hour) and the shared SVG chart style. | `backend/eval/metrics.py`, `backend/eval/svg.py`, `tests/test_eval_metrics.py`, `tests/test_eval_svg.py`, `README.md` (changed) | `python -m pytest` | 786 passed |
| v67 | M3: Evaluation charts and the live timeline | The evaluation charts drawn as SVG by hand: grouped bars, F1 against the number of synthetic images, F1 against dollars per hour, and the live timeline of one clip. | `backend/eval/charts.py`, `backend/eval/timeline.py`, `tests/test_eval_charts.py`, `tests/test_eval_timeline.py`, `README.md` (changed) | `python -m pytest` | 794 passed |
| v68 | M3: Regression check and Spark as scorer | The regression check of the base model against the fine-tuned one on general visual questions, and Muse Spark as a per-frame scorer for Experiment 4. | `backend/eval/regression.py`, `backend/eval/spark_scorer.py`, `tests/test_regression.py`, `tests/test_spark_scorer.py`, `README.md` (changed) | `python -m pytest` | 822 passed |
| v69 | M3: The results write-up | The results write-up: the SVG charts and results.md, one section per experiment with its tables of accuracy, speed and cost. | `backend/eval/writeup.py`, `tests/test_eval_writeup.py`, `README.md` (changed) | `python -m pytest` | 828 passed |
| v70 | M3: Failure galleries for Experiment 1 | The failure galleries of Experiment 1: each run's worst false alarms and misses as small frames of the recorded clips, on a page that stays on this Mac. | `backend/eval/gallery.py`, `README.md` (changed) | `python -m pytest` | 828 passed |
| v71 | M3: The evaluation plan and report | The evaluation report: a JSON plan of evaluations scored into results.json, results.md, the charts and the galleries, with W&B logging only when asked. | `backend/eval/report.py`, `tests/test_eval_gallery.py`, `tests/test_eval_report.py`, `README.md` (changed) | `python -m pytest` | 837 passed |
| v72 | M3: te eval, te label and make eval | te eval and te label on the command line, make eval, and the README results block written from results.json only. | `Makefile`, `backend/cli.py` (changed), `backend/cli_eval.py`, `backend/eval/readme.py`, `tests/test_cli_eval.py`, `tests/test_eval_readme.py`, `README.md` (changed) | `python -m backend.cli eval --help` | 844 passed |
| v73 | M3: Session clips and Spark's error rate | Labelled session clips as real training and validation frames, Muse Spark's label error rate against hand labels, and Experiment 2's arm without the Spark filter. | `backend/dataengine/realclips.py`, `backend/eval/spark_errors.py`, `docs/recording-protocol.md` (changed), `tests/test_realclips.py`, `tests/test_spark_errors.py`, `tests/test_teach.py` (changed), `tests/test_training_prep.py` (changed), `README.md` (changed) | `python -m pytest` | 863 passed |
| v74 | M3: The experiment plan and its price | The experiment plan: every training run of Experiments 1 to 3 at the same fixed steps, priced from the smoke with exports, serving and the Modal credit. | `backend/eval/plan.py`, `tests/test_cli_serving.py` (changed), `tests/test_cli_train.py` (changed), `tests/test_cli_watch.py` (changed), `tests/test_ledger.py` (changed), `tests/test_plan.py`, `README.md` (changed) | `python -m pytest` | 877 passed |
| v75 | M3: The experiment commands and eval/plan.json | te experiments, te unfiltered, te clips, te spark-errors and te regress, and eval/plan.json written from the experiment plan with its test and validation clips. | `backend/cli.py` (changed), `backend/cli_experiments.py`, `backend/eval/plan_file.py`, `tests/test_cli_experiments.py`, `tests/test_plan_file.py`, `tests/test_realclips.py` (changed), `README.md` (changed) | `python -m backend.cli experiments --help` | 886 passed |
| v76 | M3: Replay engine and the W&B report | The replay engine that fills the plan's frame logs (warm-up, validation clips, then test clips) and the W&B report built from results.md. | `backend/eval/replay.py`, `backend/eval/wandb_report.py`, `tests/test_wandb_report.py`, `README.md` (changed) | `python -m pytest` | 892 passed |
| v77 | M3: te replay: filling the plan's logs | te replay: a run's pending clips of eval/plan.json replayed into the plan's logs, its threshold calibrated on its own validation clips before the test clips. | `backend/cli.py` (changed), `backend/cli_replay.py`, `tests/test_replay.py`, `README.md` (changed) | `python -m backend.cli replay --help` | 901 passed |
| v78 | M4: Face blurring | Face blurring for anything published: OpenCV's Haar cascades plus manual boxes, every output rebuilt without its metadata, and a blur record for each video. | `backend/release/faces.py`, `tests/test_faces.py`, `README.md` (changed) | `python -m pytest` | 923 passed |
| v79 | M4: The Hugging Face model card | The Hugging Face model card, written only from the recipe, the bundle manifest and the eval results, under the base model's apache-2.0 license. | `backend/release/model_card.py`, `tests/test_model_card.py`, `README.md` (changed) | `python -m pytest` | 925 passed |
| v80 | M4: The release folder and its upload | The release folder (the card, the weights, the prompt manifests and the base model's LICENSE), the guarded Hugging Face upload and the collection of release repos. | `backend/release/BASE_LICENSE.txt`, `backend/release/package.py`, `tests/test_release_package.py`, `README.md` (changed) | `python -m pytest` | 930 passed |
| v81 | M4: Regeneration and fetching the weights | Regeneration from the prompt manifests (copied out of data/ without images, then the edits run again) and the download of a run's weights from the Modal volume. | `backend/regenerate.py`, `backend/release/weights.py`, `tests/test_regenerate.py`, `tests/test_release_package.py` (changed), `tests/test_release_weights.py`, `README.md` (changed) | `python -m pytest` | 939 passed |
| v82 | M4: The test clips as a dataset | The human-labelled session 3 test clips as a Hugging Face dataset folder: only blurred videos with a matching blur record, under the license the author chooses. | `backend/release/test_set.py`, `tests/test_test_set.py`, `README.md` (changed) | `python -m pytest` | 942 passed |
| v83 | M4: The release commands | The M4 release commands: te manifests, te regenerate, te blur, te fetch-weights, te release, te release-test-set, te release-collection and te wandb-report, each asking before it spends or publishes unless given --yes. | `backend/cli.py` (changed), `backend/cli_release.py`, `tests/test_cli_release.py`, `README.md` (changed) | `python -m backend.cli blur --help` | 950 passed |
| v84 | M4: Playground: clips, results and media | The playground's read-only data: recorded clips with their frame logs, the evaluation's charts, and the media files a page may link to, with a public mode. | `.gitignore` (changed), `backend/playground/data.py`, `tests/test_playground_data.py`, `README.md` (changed) | `python -m pytest` | 957 passed |
| v85 | M4: Playground: how each skill was built | How each taught skill was built, for the playground: the read-back, the keyframes, SAM's box on a teaching frame, the labelled training wall and the keyframes a visitor may pick. | `backend/playground/skills.py`, `tests/test_playground_skills.py`, `README.md` (changed) | `python -m pytest` | 967 passed |
| v86 | M4: Playground: spend caps and metered runs | The playground's spend caps per visitor and per day, and the metered ledger that holds each live call to its run's reservation. | `backend/playground/budget.py`, `backend/playground/runs.py`, `tests/test_playground_budget.py`, `tests/test_playground_runs.py`, `README.md` (changed) | `python -m pytest` | 982 passed |
| v87 | M4: Playground: teach it yourself | Teach it yourself: a visitor's condition checked by the local policy for free, then compiled, with 6 edits of a provided keyframe that show it and 6 near-misses, labelled live within the caps. | `backend/playground/api.py`, `backend/playground/publish.py`, `tests/test_playground_teach.py`, `README.md` (changed) | `python -m pytest` | 988 passed |
| v88 | M4: Playground: the page and its style | The playground page, with its clips, skills, teach it yourself, results and privacy sections, the stylesheet with dark and light themes, and the bundled Inter font. | `backend/playground/static/fonts/InterVariable.woff2`, `backend/playground/static/fonts/LICENSE.txt`, `backend/playground/static/index.html`, `backend/playground/static/style.css`, `README.md` (changed) | `python -m pytest` | 988 passed |
| v89 | M4: Playground: the timeline and the wall | The playground's two drawings: the live timeline of a clip's scores against the threshold with its spoken alerts, and the training wall of labelled variations. | `backend/playground/static/timeline.js`, `backend/playground/static/wall.js`, `README.md` (changed) | `python -m pytest` | 988 passed |
| v90 | M4: Playground: the page script | The page script that ties the playground together: the clip and skill pickers, SAM's outline, a live teach run followed as it goes, the cost and the caps, and the results. | `backend/playground/static/app.js`, `README.md` (changed) | `python -m pytest` | 988 passed |
| v91 | M4: The playground app | The playground app: one page and a JSON API over the local results, a visitor cookie for the budget, live runs refused from other sites, and a public copy for the open web. | `backend/playground/app.py`, `tests/test_playground_app.py`, `README.md` (changed) | `python -m pytest` | 1000 passed |
| v92 | M4: Playground commands, Modal app and notes | te playground and te playground-clip, the public playground on Modal, and the notes in their final form, with the credits, licenses and copied Meta sample files. | `CLAUDE.md` (changed), `CONTRIBUTIONS.md`, `HANDOVER.md` (changed), `PROGRESS.md` (changed), `backend/cli.py` (changed), `backend/cli_playground.py`, `backend/playground/modal_app.py`, `tests/test_cli_playground.py`, `tests/test_playground_modal.py`, `README.md` (changed) | `python -m backend.cli playground --help` | 1012 passed |

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

**v31.** `Add M2 settings and reserve spend across processes`

Adds the settings the later stages read: STUDENT_API_KEY and SERVER_TOKEN as secrets that are never printed, the student endpoint (TE_STUDENT_URL, TE_STUDENT_MODEL, TE_STUDENT_MAX_TOKENS), TE_SYSTEM_IN_USER, ~ in TE_DATA_DIR, the Modal GPU price, the $15 rule and the $150 Modal credit. The spend ledger reserves each estimate until the call is recorded, under a lock and an flock, so calls in flight in any thread or process count against the caps together, and it measures what an image costs from the reported usage. CI installs ffmpeg. Tests cover each.

**v32.** `Extend the local policy and check compiled wording`

Refuses skills that hide the glasses' capture light and skills about politics or war (phrases, with look-alikes such as a glue gun or toy soldiers taken out first), and narrows the people, covert and sports rules. The compiler runs the policy on the model's description, negatives, trigger and alert too, leaves a correction's "not the ..." part out of its check, clamps the cooldown to a minute to an hour and caps a request at 500 characters. Adds read-back helpers. Tests cover each rule.

**v33.** `Harden Meta calls, SAM streams and image edits`

Charges interrupted and failed-after-send calls, re-checks the caps before every retry, refetches stale cache entries, skips failed image downloads as output errors, and records each call's images, text tokens, cost and finish reason. SAM 3.1 replies are read by the protocol's lane rules with confidences asked for. Labels gain a cache-only lookup and the synthetic set stops on too many failures. Tests cover each case.

**v34.** `Parse SAM 3.1 boxes and crop keyframes`

Adds backend/dataengine/sam_text.py, our reader of SAM 3.1's output text written from the grammar in Meta's SAM 3 protocol, and uses it in the region of interest: boxes scaled to the keyframe, problems kept per frame, and a crop around each box. Frame folders follow EXIF orientation and refuse HEIC, and dataset entries record their source. Tests cover the grammar, the crops and the keyframes.

**v35.** `Add the watch trigger and the student scorer`

Adds backend/watch/trigger.py (3 of the last 4 frames over the threshold, then the cooldown, and the threshold calibration on labelled clips) and backend/inference/student.py (a frame scored on an OpenAI-compatible endpoint, read at the yes or no answer token, with no guess on a cut-off reply). Tests cover both, with no network.

**v36.** `Add training run records and the Modal spend ledger`

Adds backend/training/runs.py (each run's status, a vanished te train read as failed unless detached, and the served run, for GET /skills) and backend/training/gpu_ledger.py (Modal GPU time per training, export and serving job, priced from config, and the check against the $150 credit). Neither calls Modal. Tests cover both.

**v37.** `Add the watcher: scores, Spark check, alerts`

Adds backend/watch/watcher.py: each frame scored by the student, the trigger, the Muse Spark check before an alert (a no snoozes the skill, a failed check sends the alert marked unconfirmed), thresholds per run or skill with their source logged, cooldowns kept across sessions and restarts, and a JSON line per scored frame. Tests cover each.

**v38.** `Add the training bundle and the recipe's config`

Adds backend/training/prepare.py (one deterministic bundle of taught skills for Modal, in Experiment 1's real, synthetic and mixed variants, with a fixed validation split and volume paths) and backend/training/axolotl.py (Meta's QLoRA recipe as the run's Axolotl config, with only paths, W&B names and gradient accumulation changed). Tests cover both.

**v39.** `Add the processor preflight`

Adds backend/training/preflight.py: bundle examples rendered in both layouts with the pinned Glimmer processor (on Modal, CPU only), and a local verdict on sequence length, the system prompt, the yes and no tokens, the stop tokens and TE_SYSTEM_IN_USER. Tests cover it with a stand-in processor.

**v40.** `Add the Modal training app with the recipe's pins`

Adds backend/training/modal_train.py: QLoRA training on one H100 with Meta's Axolotl recipe and its pinned versions, the trained-eye volume and the wandb and huggingface Secrets, the two-step smoke, result.json for detached runs, W&B tags and optional artifacts, and the processor preflight as a CPU function. Tests read the file without running Modal, and the preflight's Modal test comes back.

**v41.** `Check te teach inputs first and add SAM crops`

Checks every te teach input before the estimate, with the fallback's own labels needing both answers, and prices images at the measured tokens with a worst case that asks past $15 even with --yes. Adds a crop to SAM's box for each keyframe that has one, labelled like the keyframe, and stops on a one-answer dataset. te spend prints the Modal total. Tests cover the input checks, the questions, the crops and the dataset rules.

**v42.** `Add te train: the smoke, detach, collect and preflight`

Adds backend/cli_train.py, registered by cli.py: te train bundles taught skills, runs the two-step smoke on Modal and saves its timing, and asks past the Modal credit or before W&B artifacts even with --yes. Runs are tagged, recorded as running, done or failed, and their GPU time goes into the Modal ledger; --detach and --collect let a run outlive the terminal, and --preflight runs the processor check. The real run's pricing from the smoke, with its $15 question, reads backend/eval/plan.py and so works from v74. Tests cover the smoke, input errors, run states, tags, artifacts, detach and collect, and the preflight, with Modal faked; the priced run's tests come in v74.

**v43.** `Add the teach session behind phone teaching`

Adds backend/teach_session.py: a teach run in a background thread with the spend question asked first and the read-back answered later, a correction compiling the spec again from the cached transcript, an unanswered question counted as no after 600 seconds, and stop() refusing the next paid call and waiting for the one in flight. Tests cover the timeouts, the answers, a refused request, failures and their wording, and the stop.

**v44.** `Add the teach API and the phone's upload checks`

Adds backend/teach_api.py (POST /teach/start, GET /teach/status and POST /teach/answer, each needing SERVER_TOKEN, one session at a time, with size limits) and backend/teach_upload.py (the multipart upload parsed without python-multipart, checked as te teach checks its inputs, saved under data/teach/ and priced, with runs that need your own labels or could cost over $15 sent back to the Mac). Tests cover a full run, a no to the spend, a correction, the token, bad inputs and odd answers, through Starlette's TestClient and a fake Meta API.

**v45.** `Wire watch mode, GET /skills and teaching into the server`

Wires watch mode into the WebSocket: a hello listing skills gets a watcher, each ack carries the scores and alerts go to the phone, with one cooldown book for every session. Adds GET /skills for the app's Skills screen, mounts the teach routes and stops a teach run cleanly on shutdown. te serve prints the local network address and the health check, says whether teaching from the phone is on, and keeps the server's log lines. Tests cover watch mode, cooldowns across sessions, GET /skills, the teach routes, shutdown and te serve.

**v46.** `Add te watch and te calibrate`

Adds backend/cli_watch.py: te watch replays a clip through watch mode, scored by the student at its run's or skill's threshold, prints the alerts and writes a frame log, and te calibrate sets a skill's threshold, or one run's, from frame logs and labelled true ranges. pyproject.toml names backend and tests as first-party for ruff's import sorting. Tests cover the replay, the endpoint check, calibration per skill and per run, the prompt layout and the served-run note.

**v47.** `Add the adapter merge and its check on Modal`

Adds backend/training/export.py, which merges a run's adapter into the bf16 base on one H100, saves it to runs/<run>/merged with the base's small files and compares its answers with base plus adapter on up to 50 bundle examples, and backend/training/export_check.py, the plain helpers that pick the examples, read the yes or no after "to=user" and count the agreement. Tests read export.py's syntax tree without Modal and run the helpers.

**v48.** `Add te export, te serve-student and the vLLM app`

Adds backend/inference/modal_serve.py (vLLM on one H100 with the cookbook's image and arguments, serving a run's merged model, the base, or the base with runs' adapters, with the key kept off the command line) and backend/cli_serving.py (te export and te serve-student, which ask first, record GPU time in the Modal ledger and the served run for GET /skills). Tests cover both commands with Modal faked and read modal_serve.py's settings and plain functions without importing Modal.

**v49.** `Add the iOS app's settings and protocol types`

Adds Settings.swift (the backend address, the watched skills and the camera choice in UserDefaults, the server token in the keychain, and the HTTP and WebSocket addresses made from what was typed) and Protocol.swift (the WebSocket messages, the /health and /skills replies, and each skill's status line). Not compiled here, as there is no Swift toolchain; the build steps are in ios/README.md from v63.

**v50.** `Add the iOS backend and WebSocket clients`

Adds BackendClient.swift (GET /health and GET /skills with the server token, and the backend's error message on a failed status) and WatchSocket.swift (the hello, JPEG frames with one in flight, acks, alerts and errors, reconnects with a growing pause, a keepalive ping and a timeout for a stuck backend). Not compiled here, as there is no Swift toolchain; the build steps are in ios/README.md from v63.

**v51.** `Add Meta's HEVC decoder and the teach messages`

Adds VideoFrameDecoder.swift and HEVCKeyframe.swift, copied unchanged from Meta's CameraAccess sample (DAT 1.0.1) with Meta's header, and the SDK's LICENSE beside them under ios/TrainedEye/Meta/. Adds TeachModels.swift: the teach status, read leniently, and the answer. Not compiled here, as there is no Swift toolchain; the build steps are in ios/README.md from v63.

**v52.** `Add the glasses connection and the mock device`

Adds GlassesConnection.swift (registration, the chosen pair's name, link and battery, and the firmware and glasses app updates) and MockGlasses.swift (DAT's mock glasses in Debug builds, with the feed set before any stream). Adds ios/make_mock_feed.py and the synthetic washer clip it makes, Resources/mock_feed.mp4. ruff checks the script. Not compiled here, as there is no Swift toolchain; the build steps are in ios/README.md from v63.

**v53.** `Add the iOS glasses camera session`

Adds GlassesCamera.swift: the DAT session and the camera permission, a new session after a fold or a link drop (never while paused, at most 3 times without a frame), a change of stream size that keeps the session, and the status lines. Its stream half follows in v54. Not compiled here, as there is no Swift toolchain; the build steps are in ios/README.md from v63.

**v54.** `Add the glasses stream and the frame throttle`

Adds GlassesCamera+Stream.swift (the stream sizes, the hvc1 camera at 7 fps, frames decoded on the SDK's thread with repeated images skipped, and a stream that stopped on its own added again up to 3 times) and FrameThrottle.swift (JPEG frames at a set pace, one a second by default, encoding only frames that are due). Not compiled here, as there is no Swift toolchain; the build steps are in ios/README.md from v63.

**v55.** `Add the iPhone camera and spoken alerts`

Adds PhoneCamera.swift (the back camera at 640x480, turned upright, converting only frames that are due) and Speaker.swift (alerts spoken only over Bluetooth, queued, and silenced while the request is being recorded). Not compiled here, as there is no Swift toolchain; the build steps are in ios/README.md from v63.

**v56.** `Add the iOS request recorder`

Adds RequestRecorder.swift: the request as 16 kHz mono WAV in memory, the glasses' HFP microphone preferred over the iPhone's, the tap set up again after route changes, interruptions and engine changes, and which microphone recorded. Not compiled here, as there is no Swift toolchain; the build steps are in ios/README.md from v63.

**v57.** `Add the iOS teach client and teach flow`

Adds TeachClient.swift (the three teach routes, with the request and frames as multipart form data) and TeachFlow.swift (name, request, countdown and capture, upload, the spend and read-back answers, polling, and a running session picked up at launch). TeachFlow needs AppModel from v58. Not compiled here, as there is no Swift toolchain; the build steps are in ios/README.md from v63.

**v58.** `Add the iOS app model`

Adds AppModel.swift: the camera choice, frames to the socket or the teach flow, alerts spoken and shown, the last 10 frames for Save clip, the resolution and pace for watching and teaching, the background counts and the skills list. It uses ClipStore from v63. Not compiled here, as there is no Swift toolchain; the build steps are in ios/README.md from v63.

**v59.** `Add the iOS theme, components and Inter fonts`

Adds Theme.swift (the colours, spacing and text styles, Inter with the system font as the fallback, and a modifier that applies them at the root) and Components.swift (screen, card, two button styles, status dot, row, notice and frame preview), with Inter 4.1's four static faces and their license in ios/TrainedEye/Fonts/. Not compiled here, as there is no Swift toolchain; the build steps are in ios/README.md from v63.

**v60.** `Add the iOS app entry, tabs and project files`

Adds TrainedEyeApp.swift (the SDK configured at launch, the theme applied at the root, and the Meta AI app's callbacks forwarded to the SDK), RootView.swift (the five tabs, the camera-on bar and the permission question), TrainedEye-Info.plist and TrainedEye.entitlements. RootView needs the screens of v61 to v63. Not compiled here, as there is no Swift toolchain; the build steps are in ios/README.md from v63.

**v61.** `Add the iOS Connect and Teach screens`

Adds ConnectView.swift (the backend address, token and health check, the glasses' registration and updates, the camera picker and the Debug mock card) and TeachView.swift (the name, request and capture cards, then the spend question, the read-back, the progress and the result). Not compiled here, as there is no Swift toolchain; the build steps are in ios/README.md from v63.

**v62.** `Add the iOS Skills and Watch screens`

Adds SkillsView.swift (the skills with their switches, status and quiet time) and WatchView.swift (start and stop, the last alert and its cloud check, scores against thresholds, the connection, the background counts, the glasses card and Save clip). Not compiled here, as there is no Swift toolchain; the build steps are in ios/README.md from v63.

**v63.** `Add the iOS clips and the Xcode build guide`

Adds ClipStore.swift (saved clips as frame folders with a clip.json under Documents/Clips, written only on Save clip) and ClipsView.swift (the list, the frames and delete). Adds ios/README.md: the Xcode project, the DAT package, the build settings, the Info.plist, capabilities, the mock device, the files table and the checks still to do on the glasses. Nothing here is compiled, as there is no Swift toolchain; the README says how to build the app in Xcode.

**v64.** `Add the hand-labelling page for generated images`

Adds backend/labeltool/images.py: a seeded sample of a skill's generated images (never real keyframes), a local Starlette page that shows each one with Spark's question but not its answer, and every yes, no or unsure saved at once so the page can resume. Adds the shared style.css and the bundled Inter font with its license. Tests cover the sampling, what the page shows, saving and resuming, and bad image lists.

**v65.** `Add the clip labelling page`

Adds backend/labeltool/clips.py and static/clips.html: a local page that plays a recorded clip, marks with [ and ] where the condition is true, and saves the sorted, merged ranges in the format te calibrate reads. Ranges outside the clip are refused, an existing file resumes, and labels for another clip are refused. Tests cover the page, range requests for the video, saving, bad ranges and resuming.

**v66.** `Add the evaluation metrics and the shared SVG style`

Adds backend/eval/metrics.py: frame precision, recall, F1 and accuracy, event recall within 5 s and at all, false alarms per hour, time to alert, latency and dollars per hour, from frame logs and labelled true ranges. Adds backend/eval/svg.py, the palette, style, escaping, axes and marks every chart uses, light and dark. Tests cover the metrics on one clip and several, a log written by the watcher, the costs, bad inputs, and the SVG pieces.

**v67.** `Add the evaluation charts and the live timeline`

Adds backend/eval/charts.py (grouped bars, a log-scale line of F1 against synthetic images, and a scatter of F1 against dollars per hour) and backend/eval/timeline.py (one skill's probability, threshold, labelled ranges and spoken alerts over a clip). Both write plain SVG with the shared style. Tests cover the light and dark style, labels on every value, missing values, escaping, gaps for student errors and ranges outside the clip.

**v68.** `Add the regression check and the Spark scorer`

Adds backend/eval/regression.py: hand-written general questions asked of the base and the fine-tuned model through OpenAI-compatible endpoints, graded by exact, contains or choice rules, with failed requests counted as wrong and listed. Adds backend/eval/spark_scorer.py: Muse Spark's yes, no or unsure as a probability for every frame, cached and billed to the skill, with a latency log of the frames actually sent. Tests cover grading, the comparison, failed requests, bad question rows, and the scorer's billing, cache, latency and errors.

**v69.** `Add the results write-up: charts and results.md`

Adds backend/eval/writeup.py, which writes the SVG charts and renders results.md from the scored evaluations: a section per experiment with its charts, an accuracy table and a speed and cost table, pending runs listed, Spark's error rate under Experiment 2, the pooled judges under Experiment 4, and the regression check and galleries when present. Tests cover the tables, pending experiments, the charts, the threshold column and the extra sections.

**v70.** `Add the failure galleries for Experiment 1`

Adds backend/eval/gallery.py: the worst false alarms and misses of each scored Experiment 1 evaluation, spread at least 5 s apart, read again from the recorded clips as thumbnails, on one local HTML page per run with a .gitignore beside it. Its own tests arrive in v71 with the evaluation runner, since they run the galleries through run_eval.

**v71.** `Add the evaluation plan and report`

Adds backend/eval/report.py: the plan's format, checked by pydantic, with evaluations pending until their logs and labels exist, and run_eval, which scores the rest into results.json, results.md, the charts and the galleries, pools Experiment 4's judges, takes in Spark's error rate and the regression check, and logs to W&B only when asked. The gallery tests come with it. Tests cover the output, bad plans, W&B logging and its team, the pooled judges, and the galleries with and without the clip.

**v72.** `Add te eval, te label, make eval and the README block`

Adds backend/cli_eval.py with te eval (the plan scored into results.md, the charts and the galleries, with optional W&B) and te label (the clip or image labelling tool on localhost), registered by backend/cli.py, and a Makefile whose make eval runs te eval on eval/plan.json. Adds backend/eval/readme.py, which writes te eval --readme's results block from results.json only and names its SHA-256. Tests cover both commands, the block's contents, the markers and the refusals.

**v73.** `Add session clips, Spark's error rate, unfiltered arm`

Adds backend/dataengine/realclips.py (labelled session 1 and 2 clips sampled at 1 fps into dataset-clips/, frames near a change left out, session 2 for validation only) and backend/eval/spark_errors.py (Spark's error rate against hand labels with a Wilson interval, and the unfiltered arm's dataset). Adds notes on recording the teach look to docs/recording-protocol.md. Tests cover the frame answers, the clip search, the splits, unreadable videos, the error rates and the unfiltered arm, and the held-back bundle and crop tests come back.

**v74.** `Add the experiment plan with fixed steps and prices`

Adds backend/eval/plan.py: every training run of Experiments 1 to 3 with its te train arguments and the same fixed optimizer steps, runs with the same bundle marked, and the price of the whole plan from the smoke's timing with the final run, exports, serving and the spend so far, checked against the $150 Modal credit. The held-back tests of te train's pricing, te watch --scorer spark and the credit check come back with it. Tests cover the runs, their counts, the cost, the credit check and the progress read from finished runs.

**v75.** `Add the experiment commands and eval/plan.json`

Adds backend/cli_experiments.py with te experiments, te unfiltered, te clips, te spark-errors and te regress, registered by backend/cli.py, and backend/eval/plan_file.py, which writes eval/plan.json from the experiment plan: one evaluation per run, judge and skill with its clips, logs, labels and prices, and the validation clips of each run the student replays, for calibration. The held-back te clips test comes back. Tests cover each command, the plan skeleton and the shared replays.

**v76.** `Add the replay engine and the W&B report`

Adds backend/eval/replay.py: a run's validation and test clips taken from the plan, the endpoint warm-up, and one clip replayed through the watcher into a log that appears only when frames scored. Adds backend/eval/wandb_report.py, which plans the W&B report from results.md and the runs' tags and saves it as a draft only after a yes. Tests cover the report's sections and panels, the eval run check and saving; replay.py's tests come with te replay.

**v77.** `Add te replay for the plan's pending clips`

Adds backend/cli_replay.py with te replay, registered by backend/cli.py: a run's pending clips from eval/plan.json, the endpoint warm-up, the validation clips replayed and calibrated first, the test clips scored at that threshold, and Spark's frames priced before any is sent. Tests cover the warm-up, a log written only when frames scored, jobs and scorers from the plan, calibration before the test clips, refusals, the served-run warning, the Spark price and recalibration.

**v78.** `Add face blurring for published images and video`

Adds backend/release/faces.py: boxes blurred with Pillow so the detail is gone, faces found with OpenCV's Haar cascades (imported only when needed), and manual boxes for faces the detector misses, refused when they miss the picture. Every output is rebuilt from the pixels, so EXIF and other metadata are left out. Folders get a sources.json of the originals' hashes, and videos go through ffmpeg without audio and with a blur record. Tests cover the blur, the metadata, folders, videos and manual boxes; the detection test skips without OpenCV.

**v79.** `Add the Hugging Face model card`

Adds backend/release/model_card.py: the card's front matter and sections, with the recipe's values, the bundle's counts and the eval results copied from their files, the adapter or merged wording, the paragraph on generated images and the base model's apache-2.0 license with Meta's Usage Policy. Tests check that the card says only what its files say, and its wording for fixed steps and a bundle of real frames only.

**v80.** `Add the release folder, upload and collection`

Adds backend/release/package.py and the base model's license text: prepare_release assembles the card, the weights, the prompt manifests, LICENSE and release.json, with face-blurred keyframes only when blur_folder made them from frames the manifests need. upload() refuses without the author's yes, with generated or unlisted images, without weights or LICENSE, or when the repo's visibility is not the one asked for, and make_collection() checks the yes and every repo's visibility; neither shows the token in an error. Tests cover the folder, the refusals and the collection against a fake Hugging Face Hub; the keyframe test that needs te regenerate comes in v81.

**v81.** `Add regeneration from manifests and weight fetching`

Adds backend/regenerate.py (the prompt manifests copied out of data/ without images, and their edits run again through the cached, capped client, with the estimate first and a count of matching hashes) and backend/release/weights.py (a run's top-level weight files from the Modal volume, size-checked, into a complete folder). The release test of the blurred keyframes comes back. Tests cover both against a fake Meta API and a fake volume.

**v82.** `Add the labelled test clips as a dataset folder`

Adds backend/release/test_set.py: blurred session 3 clips with a matching blur record and the author's labels, checked against the video's length, written as a dataset folder with metadata.jsonl, a dataset card and release.json, under a license the author chooses. check_folder stops an upload with any image, an unlisted or unblurred video, or metadata that does not match. Tests cover what goes in, the license and label checks, and the private dataset upload, with ffmpeg and the Hub faked.

**v83.** `Add the M4 release commands`

Adds backend/cli_release.py with te manifests, te regenerate, te blur, te fetch-weights, te release, te release-test-set, te release-collection and te wandb-report, and registers them in backend/cli.py. Every command that spends or publishes asks first unless given --yes, and te regenerate asks even with --yes over $15. Tests run every command with the paid and remote calls faked, checking its output and errors and the questions of te regenerate, te release-test-set, te release-collection and te wandb-report.

**v84.** `Add the playground's clip, result and media reader`

Adds backend/playground/data.py: clips and their frame logs as timelines, the evaluation's charts and results.md, and media_path, which reaches only the files the pages link to and, in the public copy, only blurred keyframes and videos with a matching blur record. .gitignore leaves out eval/clips/ and eval/galleries/. Tests cover the clip list, the timeline, the results and what each copy serves.

**v85.** `Add the playground's skill pages and training wall`

Adds backend/playground/skills.py: each taught skill's read-back, keyframes, SAM box on a teaching frame (read again from the stored text when needed) and training wall, the precomputed teach-it-yourself examples, and the keyframes a visitor may pick. The public copy shows no generated image and only face-blurred keyframes. Tests cover each part and the public copy.

**v86.** `Add the playground's spend caps and metered runs`

Adds backend/playground/budget.py (caps per visitor and per day, a visitor known by an id and a salted hash of their address, estimates reserved and then settled with the recorded spend) and backend/playground/runs.py (a run's estimate, a ledger wrapper that holds each call to the run's reservation, and the Job the page polls). Tests cover the caps, the IPv6 /64, the saved totals, calls in flight and the estimate.

**v87.** `Add teach it yourself and the public file list`

Adds backend/playground/api.py: a live run from a visitor's condition and a provided keyframe, with the free policy check first, the estimate reserved, one run per visitor and 4 in all, the compile, the edits and the labels in a polled thread, and a stop before the next call once a cap is reached. Adds backend/playground/publish.py, the files the public copy may hold. Tests run teach it yourself against the fake Meta API: 12 labelled variations with no image written outside the call cache, a free refusal, bad input and the caps.

**v88.** `Add the playground page, stylesheet and font`

Adds backend/playground/static/index.html (the clips, skills, teach it yourself, results and privacy sections), style.css (dark and light themes, the timeline and the wall) and the Inter variable font with its license. Nothing here is tested on its own: the app's tests in stage v91 load the page and these files and check that nothing comes from another site.

**v89.** `Add the playground's timeline and training wall`

Adds backend/playground/static/timeline.js (scores, threshold, true ranges and spoken alerts over a clip, with a playhead that follows the video, hover, and a click to seek) and wall.js (labelled variations flipping in, rejected ones faded, and a summary sentence). There are no tests here: both run in the browser, and the app's tests in stage v91 check that the page loads them.

**v90.** `Add the playground's page script`

Adds backend/playground/static/app.js: the clip and skill pickers, the timeline and the wall wired to the API, SAM's box drawn on a teaching frame, teach it yourself with its status, cost and polling, the precomputed examples when live teaching is off, and the results. There are no tests here: it runs in the browser, and the app's tests in stage v91 serve it.

**v91.** `Add the playground app and its JSON API`

Adds backend/playground/app.py: the page, the static files and a JSON API over the local results, a visitor cookie for the budget, live runs refused from other sites and limited to a small JSON body, security headers on the page and the charts, the public copy, and behind_proxy for a hosted copy. Tests cover the page and its files, the data APIs, a live run, both caps, bad bodies, requests from other sites, the public copy and the proxy's address.

**v92.** `Add the playground commands, Modal app and final notes`

Adds backend/cli_playground.py (te playground locally or as the public copy, te playground --deploy to a private Modal Volume after a question, and te playground-clip for blurred test clips) and backend/playground/modal_app.py (the public copy on Modal in one container, its state committed to the Volume), and registers both commands in backend/cli.py. Brings CLAUDE.md, HANDOVER.md and PROGRESS.md to their final form and adds CONTRIBUTIONS.md with the credits, licenses and copied Meta sample files. Tests cover the commands, the deploy's question and file list, the clip checks, and the Modal app's settings read from its syntax tree.

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
- **Large files.** None. The largest file is `ios/TrainedEye/Fonts/Inter-Bold.ttf` at
  about 410 KB; the whole of v92 is about 3945 KB.
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
3. v92 matches the repository root byte for byte (local-only things excluded).
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
