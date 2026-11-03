# Consolidated stage plan for v31 onward (proposal)

Status: a proposal for the author. Nothing in the stage folders or VERSIONS.md has changed. Stages v1 to v30 stay exactly as committed to `main`.

## Summary

- The 75 planned stages v31 to v105, plus the code completed since (Nov 2 and 3, about 10,900 more lines), become **63 stages, v31 to v93**. Under the old 150 to 300 line rule the same work would have been about 110 stages.
- Every stage changes **200 to 400 lines of code** compared with the stage before it (smallest 205, largest 383, average 281). Counted as the stage checker counts (added plus removed lines), but code only: each stage's tests, docs, license texts and binary files (fonts, the mock video) ride along with it and are not counted. Counting tests too would mean about 27,200 lines, so at least 69 stages and about 75 in practice; this plan assumes code only (see the questions at the end).
- The floor for this much code (17,765 lines) at 400 lines a stage is 45 stages; the extra stages keep each one cohesive (one feature with its tests) and in dependency order.
- Each stage still runs on its own: its tests pass inside its folder, and it imports only what earlier stages have. Files over 400 lines (`cli.py`, `cli_experiments.py`, `cli_release.py`, `eval/plan.py`, `release/model_card.py`, `GlassesCamera.swift`) arrive in parts, each part with the feature that needs it.
- Later fixes are folded into the stage that adds the code, so a file is written once in its final form wherever it fits (for example `meta/client.py`, changed by old v31, v39, v50 and this week, arrives once in v31).
- The docs (HANDOVER, PROGRESS, CLAUDE.md, CONTRIBUTIONS.md) arrive in their final form in the last stage; `ios/README.md` and `docs/recording-protocol.md` with the code they describe.

## How the old stages map to the new ones

Every old stage from v31 to v105 is part of at least one new stage (the Old stages column); "new" marks the code completed since the last stage cut.

### Part A: Review fixes and later changes to files from v1 to v30

| New | Scope | Code lines | Files (code; tests listed apart) | Tests | Old stages | Depends on | Delivers |
|---|---|---|---|---|---|---|---|
| v31 (A1) | Meta calls: client, labels, image edits | 330 | meta/client.py, dataengine/labels.py, dataengine/synth.py | test_client, test_labels, test_synth | v31, v34, v35, v39, v50 + new | v30 | Retries that never re-bill blindly, the caps re-checked before every retry, SAM streams read by the protocol's lane rules, label and edit failure handling |
| v32 (A2) | Settings and the spend ledger | 274 | config.py, .env.example, meta/ledger.py, .github/workflows/ci.yml | test_config, test_ledger | v32, v37, v39, v44, v47, v48, v54 + new | v30 | Every setting and secret the later stages read, and a ledger that reserves each estimate so no two processes pass a cap |
| v33 (A3) | Policy and compiler | 324 | skills/spec.py, skills/compiler.py | test_skills, test_compiler | v33, v40 + new | A2 | The local policy check (people, health, covert, the capture light, places, sports, politics and war) and the compiler's checks of the model's wording |
| v34 (A4) | Keyframes, SAM boxes and the dataset | 334 | dataengine/keyframes.py, dataengine/sam_text.py, dataengine/roi.py, dataengine/dataset.py | test_keyframes, test_sam_text, test_roi, test_dataset | v34, v35 + new | A1 | Our reader of SAM 3.1's text format, boxes and crops per keyframe, keyframe and dataset fixes |
| v35 (A5) | The teach flow and te teach | 363 | teach.py, cli.py (te teach) | test_teach, test_cli (teach) | v36, v47, v49, v58, v88 + new | A1 to A4 | The teach flow with crops and measured image tokens; te teach's questions, the $15 rule and messages |

### Part B: M2: watch mode, training, teaching from the phone, serving

| New | Scope | Code lines | Files (code; tests listed apart) | Tests | Old stages | Depends on | Delivers |
|---|---|---|---|---|---|---|---|
| v36 (B1) | Watch trigger and the student scorer | 277 | watch/trigger.py, inference/student.py | test_trigger, test_student | v41, v42, v54 + new | A2 | 3 of 4 smoothing, cooldown, calibration; scoring a frame on an OpenAI-compatible endpoint |
| v37 (B2) | Training run records | 335 | training/runs.py, training/gpu_ledger.py | test_training_runs, test_gpu_ledger |  + new | A2 | Run status for GET /skills and the Modal spend ledger with the $150 check |
| v38 (B3) | The watcher | 303 | watch/watcher.py | test_watcher | v43, v88 + new | B1, B2 | Watch mode: scores, the Spark check, alerts, cooldowns kept across sessions, thresholds per run, the frame log |
| v39 (B4) | Training bundle and recipe config | 332 | training/prepare.py, training/axolotl.py | test_training_prep | v45, v54, v89 + new | A4 | Bundles of taught skills (and session clips), Meta's QLoRA recipe as the Axolotl config |
| v40 (B5) | Modal training app | 240 | training/modal_train.py | test_modal_train | v46, v88 + new | B4 | Training on one H100 with the recipe's pins, smoke mode, detached runs |
| v41 (B6) | Processor preflight | 226 | training/preflight.py | test_training_preflight |  + new | B5 | Rendering examples with the pinned processor before the paid smoke |
| v42 (B7) | te train | 313 | cli.py (te train) | test_cli (train) | v36, v47, v49, v58, v88 + new | B2, B4 to B6 | Smoke first, priced runs, fixed steps, detach and collect, the preflight, W&B tags, entity and artifacts |
| v43 (B8) | Phone teach session | 289 | teach_session.py, cli.py (te serve) | test_teach_session | v36, v50, v52 + new | A5 | The spend question before any paid call, the read-back, timeouts, a clean stop; te serve's teaching line |
| v44 (B9) | Phone teach upload and routes | 365 | teach_upload.py, teach_api.py | test_teach_upload, test_teach_api | v51, v52, v53, v55 | B8 | The multipart upload and its checks, /teach/start, /status, /answer behind SERVER_TOKEN |
| v45 (B10) | The server, te watch and te calibrate | 365 | server.py, cli.py (te watch, te calibrate) | test_server, test_cli (watch) | v36, v44, v47, v49, v54, v58, v88 + new | B3, B9 | Watch mode over the WebSocket, GET /skills with training and served, the teach routes mounted; te watch and te calibrate per run |
| v46 (B11) | Export on Modal | 280 | training/export.py, training/export_check.py | test_export | v56, v57 + new | B5 | Merging the adapter and checking the merged model's answers |
| v47 (B12) | Serving and te export, te serve-student | 360 | inference/modal_serve.py, cli.py (te export, te serve-student) | test_modal_serve, test_cli (export) | v36, v47, v49, v57, v58, v88 + new | B11 | vLLM on Modal behind STUDENT_API_KEY, base Glimmer, adapters on one base; the two commands |

### Part C: M2: the iOS app (never compiled; no tests)

| New | Scope | Code lines | Files (code; tests listed apart) | Tests | Old stages | Depends on | Delivers |
|---|---|---|---|---|---|---|---|
| v48 (C1) | iOS: settings and protocol | 383 | Models/Settings.swift, Models/Protocol.swift | none (not compiled) | v59 + new | v30 | The backend address and the server token (keychain); the WebSocket and /skills types |
| v49 (C2) | iOS: backend and WebSocket clients | 333 | Network/BackendClient.swift, Network/WatchSocket.swift | none | v60 + new | C1 | GET /skills with the token; the watch socket with one frame in flight, keepalive, reconnects |
| v50 (C3) | iOS: Meta's decoder and the teach types | 314 | Meta/VideoFrameDecoder.swift, Meta/HEVCKeyframe.swift, Meta/LICENSE, Models/TeachModels.swift | none | v61, v65 | C1 | Meta's two copied HEVC files with their LICENSE; the teach API's types |
| v51 (C4) | iOS: glasses connection and the mock | 309 | Glasses/GlassesConnection.swift, Glasses/MockGlasses.swift, Resources/mock_feed.mp4, ios/make_mock_feed.py | none | v62 + new | C1 | Registration, device state, the mock device with its synthetic feed |
| v52 (C5) | iOS: glasses camera, session | 260 | Glasses/GlassesCamera.swift (session) | none | v63 + new | C4 | The DAT session, start and stop, the camera permission |
| v53 (C6) | iOS: glasses camera, stream | 281 | Glasses/GlassesCamera.swift (stream), Camera/FrameThrottle.swift | none | v63, v64 + new | C5 | The stream at low and medium resolution, revival after a fold, repeated frames; the frame throttle |
| v54 (C7) | iOS: phone camera and speech | 205 | Camera/PhoneCamera.swift, Audio/Speaker.swift | none | v64 | C6 | The iPhone camera fallback; alerts spoken only on Bluetooth |
| v55 (C8) | iOS: request recorder | 321 | Audio/RequestRecorder.swift | none | v65 + new | C1 | Recording the spoken request through an engine tap, glasses microphone first |
| v56 (C9) | iOS: teach client and flow | 355 | Network/TeachClient.swift, Teach/TeachFlow.swift | none | v65, v66 + new | C2, C3, C8 | The teach API calls; hold to talk, the capture, the spend question, the read-back, resume |
| v57 (C10) | iOS: app model | 359 | App/AppModel.swift | none | v67 + new | C2 to C9 | Connection, camera source, watch socket, skills, alerts, background counts |
| v58 (C11) | iOS: theme, components and Inter | 206 | UI/Theme.swift, UI/Components.swift, Fonts/Inter-Regular.ttf, Fonts/Inter-Medium.ttf, Fonts/Inter-SemiBold.ttf, Fonts/Inter-Bold.ttf, Fonts/LICENSE.txt | none | v68 + new | C1 | The look, shared views and the camera-on bar, Inter's four faces |
| v59 (C12) | iOS: app entry, project files and tabs | 253 | App/TrainedEyeApp.swift, ios/TrainedEye-Info.plist, ios/TrainedEye.entitlements, UI/RootView.swift | none | v69, v70 + new | C10, C11 | The SDK set-up, Info.plist, entitlements and the tab layout |
| v60 (C13) | iOS: Connect and Teach screens | 305 | UI/ConnectView.swift, UI/TeachView.swift | none | v70, v71 + new | C12 | The address and token screen; the Teach screen |
| v61 (C14) | iOS: Skills and Watch screens | 218 | UI/SkillsView.swift, UI/WatchView.swift | none | v72 + new | C12 | Skills with Training, Trained or Ready; the Watch screen |
| v62 (C15) | iOS: clips and the build guide | 205 | Clips/ClipStore.swift, UI/ClipsView.swift, ios/README.md | none | v73, v74 + new | C12 | Saving the last 10 seconds by hand; ios/README.md |

### Part D: M3: labelling, evaluation, the four experiments

| New | Scope | Code lines | Files (code; tests listed apart) | Tests | Old stages | Depends on | Delivers |
|---|---|---|---|---|---|---|---|
| v63 (D1) | Labelling tool: images | 224 | labeltool/images.py, labeltool/static/images.html, labeltool/static/style.css, labeltool/static/fonts/InterVariable.woff2, labeltool/static/fonts/LICENSE.txt | test_labeltool_images | v75, v77 + new | A4 | Hand labels for generated images (Experiment 2), the shared page style with Inter |
| v64 (D2) | Labelling tool: clips and te label | 274 | labeltool/clips.py, labeltool/static/clips.html, cli.py (te label) | test_labeltool_clips, test_cli (label) | v36, v47, v49, v58, v76, v77, v88 + new | D1 | True time ranges in a clip; te label for both tools |
| v65 (D3) | Metrics and SVG pieces | 333 | eval/metrics.py, eval/svg.py | test_eval_metrics, test_eval_svg | v78, v79, v80 | B1 | Frame and event metrics, costs per hour; the chart building blocks |
| v66 (D4) | Charts and the timeline | 205 | eval/charts.py, eval/timeline.py | test_eval_charts, test_eval_timeline | v80, v81 | D3 | The experiment charts and the live timeline chart |
| v67 (D5) | Spark per frame and the regression check | 308 | eval/spark_scorer.py, eval/regression.py | test_spark_scorer, test_regression | v84, v87 + new | B1 | Muse Spark as a scorer (Experiment 4); the regression check on general questions |
| v68 (D6) | Results write-up | 284 | eval/writeup.py | test_eval_writeup | v82 + new | D3 to D5 | results.md and the chart files, every number from the metrics |
| v69 (D7) | Failure galleries | 246 | eval/gallery.py | test_eval_gallery |  + new | D4 | Experiment 1's worst false alarms and misses, local only |
| v70 (D8) | Evaluation runner | 351 | eval/report.py | test_eval_report | v83 + new | D6, D7 | The plan format, scored and pending evaluations, results.json, W&B on request |
| v71 (D9) | te eval and the README results block | 219 | eval/readme.py, Makefile, cli.py (te eval) | test_eval_readme, test_cli (eval) | v36, v47, v49, v58, v88 + new | D8 | te eval, make eval, and te eval --readme |
| v72 (D10) | Experiment 2 | 263 | eval/spark_errors.py, cli_experiments.py (unfiltered, spark-errors) | test_spark_errors, test_cli_experiments | v85, v89 + new | D1, D8 | Spark's error rate, the unfiltered arm, te unfiltered and te spark-errors |
| v73 (D11) | Session clips as training data | 244 | dataengine/realclips.py, docs/recording-protocol.md, cli_experiments.py (te clips) | test_realclips | v38, v48 + new | D2, D10 | te clips: session 1 frames to train on, session 2 to validate; the recording protocol |
| v74 (D12) | Experiment plan: runs and costs | 270 | eval/plan.py (runs, costs) | test_plan | v86 + new | B2, B4, D11 | Every training run with fixed steps; the cost with exports, serving and spend so far |
| v75 (D13) | Experiment plan: the plan file and te experiments | 255 | eval/plan.py (plan file), cli_experiments.py (te experiments) | test_plan, test_cli_experiments | v86, v89 + new | D12 | eval/plan.json with calibration and timelines; te experiments |
| v76 (D14) | Replay engine and te regress | 220 | eval/replay.py, cli_experiments.py (te regress) | test_replay, test_regression | v87, v89 + new | B3, D8, D13 | Replaying a plan's clips into its logs; te regress |
| v77 (D15) | te replay | 247 | cli_experiments.py (te replay), cli.py (registration) | test_replay, test_cli_experiments | v89 + new | D14 | Calibrating a run on session 2 and replaying its test clips |

### Part E: M4: release tools and the web playground, then the notes

| New | Scope | Code lines | Files (code; tests listed apart) | Tests | Old stages | Depends on | Delivers |
|---|---|---|---|---|---|---|---|
| v78 (E1) | Face blurring | 293 | release/faces.py | test_faces | v90 + new | v30 | Blurring with manual boxes, blur records for videos |
| v79 (E2) | Model card | 258 | release/model_card.py (card) | test_model_card | v92, v93 + new | B4 | The card from the bundle and the results, the base model's license |
| v80 (E3) | Release folder and upload | 262 | release/model_card.py (release, upload), release/BASE_LICENSE.txt | test_model_card | v92, v93 + new | E1, E2 | The release folder with blurred keyframes and LICENSE, the guarded upload, collections |
| v81 (E4) | Regeneration and fetching weights | 252 | regenerate.py, release/weights.py | test_regenerate, test_release_weights | v91 + new | E1, E3, B2 | Prompt manifests and regeneration; a run's weights from the Modal volume |
| v82 (E5) | The test set | 282 | release/test_set.py | test_test_set |  + new | E1, E3, D3 | Session 3 clips, blurred, as a dataset with its card |
| v83 (E6) | Playground: local data | 245 | playground/data.py | test_playground_data | v94 + new | D3 | Clips, timelines and results from data/ and eval/ |
| v84 (E7) | Playground: skills | 211 | playground/skills.py | test_playground_skills | v95 + new | A4, E6 | How each skill was built: read-back, keyframes, wall, outline |
| v85 (E8) | Playground: caps and runs | 242 | playground/budget.py, playground/runs.py | test_playground_budget, test_playground_runs | v96, v97, v98 + new | A2 | Per-visitor and daily caps; run accounting |
| v86 (E9) | Playground: live teaching and the public files | 247 | playground/api.py, playground/publish.py | test_playground_teach | v97, v98 + new | E7, E8 | Teach it yourself; what the public copy may hold |
| v87 (E10) | Playground: page and style | 280 | playground/static/index.html, playground/static/style.css, playground/static/fonts/InterVariable.woff2, playground/static/fonts/LICENSE.txt | none (browser files) | v99 + new | E9 | The page, the style sheet, Inter |
| v88 (E11) | Playground: timeline and wall | 227 | playground/static/timeline.js, playground/static/wall.js | none | v100 | E10 | The live timeline and the training wall in the browser |
| v89 (E12) | Playground: page logic | 297 | playground/static/app.js | none | v101 + new | E11 | app.js wiring the tabs to the API |
| v90 (E13) | Playground: app | 231 | playground/app.py | test_playground_app | v102, v104 + new | E6 to E12 | The app, its public mode, the content security policy |
| v91 (E14) | Deploy targets: Modal app and the W&B report | 267 | playground/modal_app.py, eval/wandb_report.py | test_playground_modal, test_wandb_report |  + new | D8, E13 | The playground on Modal; the W&B report |
| v92 (E15) | Release commands | 313 | cli_release.py (release) | test_cli_release | v103, v104 + new | E1 to E5, E14 | te manifests, regenerate, blur, fetch-weights, release, release-test-set, release-collection, wandb-report |
| v93 (E16) | Playground commands, the notes and the credits | 298 | cli_release.py (playground), cli.py (registration), .gitignore, HANDOVER.md, PROGRESS.md, CLAUDE.md, CONTRIBUTIONS.md | test_cli_release | v37, v38, v47, v48, v103, v104, v105 + new | E13, E15 | te playground (public, deploy), te playground-clip; HANDOVER, PROGRESS, CLAUDE.md, CONTRIBUTIONS.md, .gitignore |

Code lines for a file that arrives in parts are estimates from where the file splits by function; the real counts come from the stage checker when the stages are cut, and a part may move a few lines to keep every stage within 200 to 400.

## How it would be built

1. A new peel spec, `stagetools/specs/peel_v31.py`, peels the finished code at the root back to v30, stage by stage, with exact edits for the files that arrive in parts. Peeling v31 once more must give v30 byte for byte, which proves the new stages sit exactly on `main`.
2. The stage checker counts code lines only from v31 (tests, docs, license texts and binaries ride along), and keeps the 5-file rule for v1 to v30.
3. Every stage's README is rendered from its own measured `pytest` and `ruff` run; VERSIONS.md is rewritten with the new table, each stage's commit summary and description, and the order explained.
4. The old folders v31 to v105 are replaced by the new v31 to v93 on `build-stages` (v1 to v30 untouched).
5. You commit them to `main` one at a time as before (`rsync -a v31/ ./`, `git add -A`, `git commit` with the summary and description VERSIONS.md gives), one commit per stage.

## Questions for the author

1. **Counting:** code lines only, with tests and docs riding along (this plan, 63 stages), or tests counted too (about 75 stages: the code plus about 9,500 lines of tests)?
2. **Docs:** all doc updates in the last stage (this plan), or the notes updated along the way (adds no stage; each part's doc update rides with its last stage)?
3. **The iOS stages (v48 to v62)** cannot compile one by one (the app model and the teach flow need each other); the app builds as a whole from v62. Keep them as separate stages, or fold them into fewer, larger ones (they would still be 200 to 400 lines each, so the count stays about the same)?
