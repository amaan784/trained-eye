# Consolidated stages v31 to v92 (as built)

The 75 planned stages v31 to v105, plus the code completed since (Nov 2 and 3), became
**62 stages, v31 to v92**. Stages v1 to v30 are unchanged, as committed to `main`.

## Rules

- Every stage changes **200 to 400 lines of code** compared with the stage before it
  (smallest 205, largest 399, average 290; 17,958 in all), counted as the stage checker
  counts them: added plus removed lines, code only. Each stage's tests, docs, license
  texts and binary files (fonts, the mock video) ride along and are not counted.
- Each stage runs on its own: its tests pass inside its folder (the counts only grow,
  from 318 at v31 to 1012 at v92), ruff passes, and it imports only what earlier stages
  have.
- A file arrives once, in its final form, with the later fixes to it already in. The six
  files that were over 400 lines (`cli.py`, `cli_experiments.py`, `cli_release.py`,
  `eval/plan.py`, `release/model_card.py`, `GlassesCamera.swift`) were first split into
  feature modules at the root, with no change in behaviour, so they too arrive whole.
- What still arrives in parts: `backend/cli.py` gets each command module's
  registration and docstring lines with the stage that adds that module, and its
  `te serve` changes with the server (v45); and a few tests arrive with the stage whose
  module they need (each stage's README says which). `stagetools/specs/forward_v41.py`
  lists every such piece.
- The docs (HANDOVER, PROGRESS, CLAUDE.md, CONTRIBUTIONS.md) arrive in their final form
  in the last stage; `ios/README.md` and `docs/recording-protocol.md` with the code they
  describe.

## How they were built

v31 to v40 were built forward from v30 by hand and committed to `main` first
(`stagetools/specs/pushed_main.md` lists what each holds). v41 to v92 were built forward
from v40 by `stagetools/forward.py` with `stagetools/specs/forward_v41.py`, which also
checks that its pieces give v40 back and that v92 equals the repository root. The
README and commit texts (`stagetools/specs/stages_c.py`) were written from each
stage's diff and then checked against the code.

Changes from the proposal: one stage fewer (63 became 62) after the split of the large
files; the policy check comes before the Meta client, because image edits use one of
its helpers; and the teach flow and te teach come after the training records (v41)
rather than before them, so that v31 to v40 could go to `main` while `cli.py` was being
split.

## The stages

| Stage | Milestone | Layer | Code lines | Code files | Tests | Old stages |
|---|---|---|---|---|---|---|
| v31 | M2 | Settings and the spend ledger | 274 | backend/config.py, .env.example, backend/meta/ledger.py, .github/workflows/ci.yml | test_config, test_ledger | v32, v37, v39, v44, v47, v48, v54 |
| v32 | M2 | Policy and the skill compiler | 324 | backend/skills/spec.py, backend/skills/compiler.py | test_skills, test_compiler | v33, v40 |
| v33 | M2 | Meta calls: client, labels and image edits | 330 | backend/meta/client.py, backend/dataengine/labels.py, backend/dataengine/synth.py | test_client, test_labels, test_synth | v31, v34, v35, v39, v50 |
| v34 | M2 | Keyframes, SAM boxes and the dataset | 334 | backend/dataengine/keyframes.py, backend/dataengine/sam_text.py, backend/dataengine/roi.py, backend/dataengine/dataset.py | test_keyframes, test_sam_text, test_roi, test_dataset | v34, v35 |
| v35 | M2 | Watch trigger and the student scorer | 277 | backend/watch/trigger.py, backend/inference/student.py | test_trigger, test_student | v41, v42, v54 |
| v36 | M2 | Training run records and the Modal spend ledger | 335 | backend/training/runs.py, backend/training/gpu_ledger.py | test_training_runs, test_gpu_ledger | new |
| v37 | M2 | The watcher | 303 | backend/watch/watcher.py | test_watcher | v43, v88 |
| v38 | M2 | Training bundle and the recipe's config | 332 | backend/training/prepare.py, backend/training/axolotl.py | test_training_prep | v45, v54, v89 |
| v39 | M2 | The processor preflight | 226 | backend/training/preflight.py | test_training_preflight | new |
| v40 | M2 | The Modal training app | 240 | backend/training/modal_train.py | test_modal_train | v46, v88 |
| v41 | M2 | Teach checks, worst case and crops | 357 | backend/cli.py, backend/teach.py | test_cli, test_teach | v36, v47, v49, v58, v88, v89, v103 |
| v42 | M2 | te train on Modal | 370 | backend/cli.py, backend/cli_train.py | test_cli_train, test_modal_train | v36, v46, v47, v49, v58, v88, v89, v103 |
| v43 | M2 | Teach sessions from the phone | 239 | backend/teach_session.py | test_teach_session | v50, v52 |
| v44 | M2 | The phone's teach API | 365 | backend/teach_api.py, backend/teach_upload.py | test_teach_api, test_teach_upload | v51, v52, v53, v55 |
| v45 | M2 | Watch mode and teaching on the server | 230 | backend/cli.py, backend/server.py | test_cli, test_server | v36, v44, v47, v49, v54, v58, v88, v89, v103 |
| v46 | M2 | Replaying clips and calibrating thresholds | 236 | backend/cli.py, backend/cli_watch.py, pyproject.toml | test_cli_watch | v36, v47, v49, v58, v88, v89, v103 |
| v47 | M2 | Merging and checking the adapter | 280 | backend/training/export.py, backend/training/export_check.py | test_export | v56, v57 |
| v48 | M2 | Export and the student endpoint | 375 | backend/cli.py, backend/cli_serving.py, backend/inference/modal_serve.py | test_cli_serving, test_modal_serve | v36, v47, v49, v57, v58, v88, v89, v103 |
| v49 | M2 | iOS: settings and the protocol | 383 | ios/TrainedEye/Models/Protocol.swift, ios/TrainedEye/Models/Settings.swift | none | v59 |
| v50 | M2 | iOS: backend and WebSocket clients | 333 | ios/TrainedEye/Network/BackendClient.swift, ios/TrainedEye/Network/WatchSocket.swift | none | v60 |
| v51 | M2 | iOS: Meta's HEVC decoder and teach messages | 314 | ios/TrainedEye/Meta/HEVCKeyframe.swift, ios/TrainedEye/Meta/VideoFrameDecoder.swift, ios/TrainedEye/Models/TeachModels.swift | none | v61, v65 |
| v52 | M2 | iOS: glasses connection and the mock device | 309 | ios/TrainedEye/Glasses/GlassesConnection.swift, ios/TrainedEye/Glasses/MockGlasses.swift, ios/make_mock_feed.py | none | v62 |
| v53 | M2 | iOS: the glasses camera | 298 | ios/TrainedEye/Glasses/GlassesCamera.swift | none | v63 |
| v54 | M2 | iOS: the glasses stream and frame throttle | 260 | ios/TrainedEye/Camera/FrameThrottle.swift, ios/TrainedEye/Glasses/GlassesCamera+Stream.swift | none | v63, v64 |
| v55 | M2 | iOS: phone camera and spoken alerts | 205 | ios/TrainedEye/Audio/Speaker.swift, ios/TrainedEye/Camera/PhoneCamera.swift | none | v64 |
| v56 | M2 | iOS: the spoken request recorder | 321 | ios/TrainedEye/Audio/RequestRecorder.swift | none | v65 |
| v57 | M2 | iOS: the teach client and flow | 355 | ios/TrainedEye/Network/TeachClient.swift, ios/TrainedEye/Teach/TeachFlow.swift | none | v65, v66 |
| v58 | M2 | iOS: the app model | 359 | ios/TrainedEye/App/AppModel.swift | none | v67 |
| v59 | M2 | iOS: theme, components and the Inter fonts | 206 | ios/TrainedEye/UI/Components.swift, ios/TrainedEye/UI/Theme.swift | none | v68 |
| v60 | M2 | iOS: app entry, tabs and project files | 253 | ios/TrainedEye-Info.plist, ios/TrainedEye.entitlements, ios/TrainedEye/App/TrainedEyeApp.swift, ios/TrainedEye/UI/RootView.swift | none | v69, v70 |
| v61 | M2 | iOS: Connect and Teach screens | 305 | ios/TrainedEye/UI/ConnectView.swift, ios/TrainedEye/UI/TeachView.swift | none | v70, v71 |
| v62 | M2 | iOS: Skills and Watch screens | 218 | ios/TrainedEye/UI/SkillsView.swift, ios/TrainedEye/UI/WatchView.swift | none | v72 |
| v63 | M2 | iOS: saved clips and the build guide | 205 | ios/TrainedEye/Clips/ClipStore.swift, ios/TrainedEye/UI/ClipsView.swift | none | v73, v74 |
| v64 | M3 | Hand labels for generated images | 224 | backend/labeltool/images.py, backend/labeltool/static/images.html, backend/labeltool/static/style.css | test_labeltool_images | v75, v77 |
| v65 | M3 | Hand labels for recorded clips | 224 | backend/labeltool/clips.py, backend/labeltool/static/clips.html | test_labeltool_clips | v76, v77 |
| v66 | M3 | Evaluation metrics and the chart style | 333 | backend/eval/metrics.py, backend/eval/svg.py | test_eval_metrics, test_eval_svg | v78, v79, v80 |
| v67 | M3 | Evaluation charts and the live timeline | 205 | backend/eval/charts.py, backend/eval/timeline.py | test_eval_charts, test_eval_timeline | v80, v81 |
| v68 | M3 | Regression check and Spark as scorer | 308 | backend/eval/regression.py, backend/eval/spark_scorer.py | test_regression, test_spark_scorer | v84, v87 |
| v69 | M3 | The results write-up | 284 | backend/eval/writeup.py | test_eval_writeup | v82 |
| v70 | M3 | Failure galleries for Experiment 1 | 246 | backend/eval/gallery.py | none | new |
| v71 | M3 | The evaluation plan and report | 351 | backend/eval/report.py | test_eval_gallery, test_eval_report | v83 |
| v72 | M3 | te eval, te label and make eval | 271 | Makefile, backend/cli.py, backend/cli_eval.py, backend/eval/readme.py | test_cli_eval, test_eval_readme | v36, v47, v49, v58, v88, v89, v103 |
| v73 | M3 | Session clips and Spark's error rate | 352 | backend/dataengine/realclips.py, backend/eval/spark_errors.py | test_realclips, test_spark_errors, test_teach, test_training_prep | v36, v38, v45, v48, v49, v54, v85, v89 |
| v74 | M3 | The experiment plan and its price | 305 | backend/eval/plan.py | test_cli_serving, test_cli_train, test_cli_watch, test_ledger, test_plan | v32, v36, v47, v49, v58, v86, v88 |
| v75 | M3 | The experiment commands and eval/plan.json | 359 | backend/cli.py, backend/cli_experiments.py, backend/eval/plan_file.py | test_cli_experiments, test_plan_file, test_realclips | v36, v47, v49, v58, v86, v88, v89, v103 |
| v76 | M3 | Replay engine and the W&B report | 323 | backend/eval/replay.py, backend/eval/wandb_report.py | test_wandb_report | new |
| v77 | M3 | te replay: filling the plan's logs | 314 | backend/cli.py, backend/cli_replay.py | test_replay | v36, v47, v49, v58, v88, v89, v103 |
| v78 | M4 | Face blurring | 293 | backend/release/faces.py | test_faces | v90 |
| v79 | M4 | The Hugging Face model card | 241 | backend/release/model_card.py | test_model_card | v92, v93 |
| v80 | M4 | The release folder and its upload | 303 | backend/release/package.py | test_release_package | v92 |
| v81 | M4 | Regeneration and fetching the weights | 252 | backend/regenerate.py, backend/release/weights.py | test_regenerate, test_release_package, test_release_weights | v91 |
| v82 | M4 | The test clips as a dataset | 283 | backend/release/test_set.py | test_test_set | new |
| v83 | M4 | The release commands | 399 | backend/cli.py, backend/cli_release.py | test_cli_release | v36, v47, v49, v58, v88, v89, v103, v104 |
| v84 | M4 | Playground: clips, results and media | 250 | .gitignore, backend/playground/data.py | test_playground_data | v94 |
| v85 | M4 | Playground: how each skill was built | 211 | backend/playground/skills.py | test_playground_skills | v95 |
| v86 | M4 | Playground: spend caps and metered runs | 242 | backend/playground/budget.py, backend/playground/runs.py | test_playground_budget, test_playground_runs | v96, v97, v98 |
| v87 | M4 | Playground: teach it yourself | 247 | backend/playground/api.py, backend/playground/publish.py | test_playground_teach | v97, v98 |
| v88 | M4 | Playground: the page and its style | 280 | backend/playground/static/index.html, backend/playground/static/style.css | none | v99 |
| v89 | M4 | Playground: the timeline and the wall | 227 | backend/playground/static/timeline.js, backend/playground/static/wall.js | none | v100 |
| v90 | M4 | Playground: the page script | 297 | backend/playground/static/app.js | none | v101 |
| v91 | M4 | The playground app | 231 | backend/playground/app.py | test_playground_app | v102, v104 |
| v92 | M4 | Playground commands, Modal app and notes | 352 | backend/cli.py, backend/cli_playground.py, backend/playground/modal_app.py | test_cli_playground, test_playground_modal | v36, v37, v38, v47, v48, v49, v58, v88, v89, v103, v105 |

## Where each old stage went

Each old stage v31 to v105 is part of at least one new stage (worked out from the files
each old stage changed; a file split at the root counts for each of its parts).

| Old stage | New stages |
|---|---|
| v31 | v33 |
| v32 | v31, v74 |
| v33 | v32 |
| v34 | v33, v34 |
| v35 | v33, v34 |
| v36 | v41, v42, v45, v46, v48, v72, v73, v74, v75, v77, v83, v92 |
| v37 | v31, v92 |
| v38 | v73, v92 |
| v39 | v31, v33 |
| v40 | v32 |
| v41 | v35 |
| v42 | v35 |
| v43 | v37 |
| v44 | v31, v45 |
| v45 | v38, v73 |
| v46 | v40, v42 |
| v47 | v31, v41, v42, v45, v46, v48, v72, v74, v75, v77, v83, v92 |
| v48 | v31, v73, v92 |
| v49 | v41, v42, v45, v46, v48, v72, v73, v74, v75, v77, v83, v92 |
| v50 | v33, v43 |
| v51 | v44 |
| v52 | v43, v44 |
| v53 | v44 |
| v54 | v31, v35, v38, v45, v73 |
| v55 | v44 |
| v56 | v47 |
| v57 | v47, v48 |
| v58 | v41, v42, v45, v46, v48, v72, v74, v75, v77, v83, v92 |
| v59 | v49 |
| v60 | v50 |
| v61 | v51 |
| v62 | v52 |
| v63 | v53, v54 |
| v64 | v54, v55 |
| v65 | v51, v56, v57 |
| v66 | v57 |
| v67 | v58 |
| v68 | v59 |
| v69 | v60 |
| v70 | v60, v61 |
| v71 | v61 |
| v72 | v62 |
| v73 | v63 |
| v74 | v63 |
| v75 | v64 |
| v76 | v65 |
| v77 | v64, v65 |
| v78 | v66 |
| v79 | v66 |
| v80 | v66, v67 |
| v81 | v67 |
| v82 | v69 |
| v83 | v71 |
| v84 | v68 |
| v85 | v73 |
| v86 | v74, v75 |
| v87 | v68 |
| v88 | v37, v40, v41, v42, v45, v46, v48, v72, v74, v75, v77, v83, v92 |
| v89 | v38, v41, v42, v45, v46, v48, v72, v73, v75, v77, v83, v92 |
| v90 | v78 |
| v91 | v81 |
| v92 | v79, v80 |
| v93 | v79 |
| v94 | v84 |
| v95 | v85 |
| v96 | v86 |
| v97 | v86, v87 |
| v98 | v86, v87 |
| v99 | v88 |
| v100 | v89 |
| v101 | v90 |
| v102 | v91 |
| v103 | v41, v42, v45, v46, v48, v72, v75, v77, v83, v92 |
| v104 | v83, v91 |
| v105 | v92 |
