# Consolidated stages already on main

These stage folders were committed to `main` one at a time. The full re-cut (the peel
from the root) must rebuild them byte for byte; check with `diff -r` against
`git show origin/main` before committing the re-cut.

Each is the stage before it plus these files exactly as at build-stages 3b71962:

- v31: backend/config.py, .env.example, backend/meta/ledger.py, .github/workflows/ci.yml,
  tests/test_config.py without test_the_student_reply_budget (needs inference/student.py),
  tests/test_ledger.py without test_the_plans_spark_price_uses_the_measured_image_tokens
  (needs eval/plan.py; it comes back with that stage).
- v32: backend/skills/spec.py, backend/skills/compiler.py, tests/test_skills.py,
  tests/test_compiler.py (before the Meta calls: synth.py imports spec.distinct).
- v33: backend/meta/client.py, backend/dataengine/labels.py, backend/dataengine/synth.py,
  tests/test_client.py, tests/test_labels.py, tests/test_synth.py.
- v34: backend/dataengine/keyframes.py, sam_text.py, roi.py, dataset.py and their tests.
- v35: backend/watch/trigger.py, backend/inference/student.py, tests/test_trigger.py,
  tests/test_student.py, and the whole tests/test_config.py.

- v36: backend/training/runs.py, backend/training/gpu_ledger.py, tests/test_training_runs.py,
  tests/test_gpu_ledger.py.
- v37: backend/watch/watcher.py, tests/test_watcher.py.
- v38: backend/training/prepare.py, backend/training/axolotl.py, tests/test_training_prep.py
  without test_real_and_mixed_take_the_clips_and_session_2_only_validates,
  test_a_limit_counts_session_1_frames_but_never_session_2 and
  test_clips_from_another_spec_or_tampered_ones_are_refused (need dataengine/realclips.py;
  they come back with that stage).
- v39: backend/training/preflight.py, tests/test_training_preflight.py without
  test_run_preflight_renders_on_modal_and_saves_the_verdict (reads modal_train.py; it comes
  back with that stage) and so without its now unused `import types`.

READMEs: v31 to v34 rendered with TE_RENDER_STAGES unset (all of 31 to 35 at the time),
v35 to v39 with TE_RENDER_STAGES=35-39. The re-cut keeps the READMEs already on main
(they hold measured timings) and renders only the stages after them.

The teach flow (plan A5) and the training app (B5) follow once cli.py's split is settled.
