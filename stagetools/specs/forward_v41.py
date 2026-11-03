"""Stages v41 onward, built forward from v40 with stagetools/forward.py.

Each stage adds whole files in their final form, with their tests. cli.py and a few test
files arrive in parts (PIECES): until a piece's stage, its final text reads as it did
before (for cli.py, as in v30, or without the registration of a module not added yet).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = 40
CLI = (ROOT / "backend/cli.py").read_text(encoding="utf-8")


def between(text: str, start: str, end: str) -> str:
    """The text from `start` up to (not including) `end`; both must occur once."""
    assert text.count(start) == 1, start
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j]


FILES = {
    # the teach flow and te teach
    "backend/teach.py": 41, "backend/cli.py": 41, "tests/test_teach.py": 41,
    "tests/test_cli.py": 41,
    # te train
    "backend/cli_train.py": 42, "tests/test_cli_train.py": 42, "tests/test_modal_train.py": 40,
    # teaching from the phone
    "backend/teach_session.py": 43, "tests/test_teach_session.py": 43,
    "backend/teach_upload.py": 44, "backend/teach_api.py": 44,
    "tests/test_teach_upload.py": 44, "tests/test_teach_api.py": 44,
    "backend/server.py": 45, "tests/test_server.py": 45,
    # te watch, export, serving
    "backend/cli_watch.py": 46, "tests/test_cli_watch.py": 46,
    "backend/training/export.py": 47, "backend/training/export_check.py": 47,
    "tests/test_export.py": 47,
    "backend/inference/modal_serve.py": 48, "backend/cli_serving.py": 48,
    "tests/test_modal_serve.py": 48, "tests/test_cli_serving.py": 48,
    # the iOS app (no tests: Swift is not compiled here)
    "ios/TrainedEye/Models/Settings.swift": 49, "ios/TrainedEye/Models/Protocol.swift": 49,
    "ios/TrainedEye/Network/BackendClient.swift": 50,
    "ios/TrainedEye/Network/WatchSocket.swift": 50,
    "ios/TrainedEye/Meta/VideoFrameDecoder.swift": 51,
    "ios/TrainedEye/Meta/HEVCKeyframe.swift": 51, "ios/TrainedEye/Meta/LICENSE": 51,
    "ios/TrainedEye/Models/TeachModels.swift": 51,
    "ios/TrainedEye/Glasses/GlassesConnection.swift": 52,
    "ios/TrainedEye/Glasses/MockGlasses.swift": 52,
    "ios/TrainedEye/Resources/mock_feed.mp4": 52, "ios/make_mock_feed.py": 52,
    "ios/TrainedEye/Glasses/GlassesCamera.swift": 53,
    "ios/TrainedEye/Glasses/GlassesCamera+Stream.swift": 54,
    "ios/TrainedEye/Camera/FrameThrottle.swift": 54,
    "ios/TrainedEye/Camera/PhoneCamera.swift": 55, "ios/TrainedEye/Audio/Speaker.swift": 55,
    "ios/TrainedEye/Audio/RequestRecorder.swift": 56,
    "ios/TrainedEye/Network/TeachClient.swift": 57, "ios/TrainedEye/Teach/TeachFlow.swift": 57,
    "ios/TrainedEye/App/AppModel.swift": 58,
    "ios/TrainedEye/UI/Theme.swift": 59, "ios/TrainedEye/UI/Components.swift": 59,
    "ios/TrainedEye/Fonts/Inter-Regular.ttf": 59, "ios/TrainedEye/Fonts/Inter-Medium.ttf": 59,
    "ios/TrainedEye/Fonts/Inter-SemiBold.ttf": 59, "ios/TrainedEye/Fonts/Inter-Bold.ttf": 59,
    "ios/TrainedEye/Fonts/LICENSE.txt": 59,
    "ios/TrainedEye/App/TrainedEyeApp.swift": 60, "ios/TrainedEye-Info.plist": 60,
    "ios/TrainedEye.entitlements": 60, "ios/TrainedEye/UI/RootView.swift": 60,
    "ios/TrainedEye/UI/ConnectView.swift": 61, "ios/TrainedEye/UI/TeachView.swift": 61,
    "ios/TrainedEye/UI/SkillsView.swift": 62, "ios/TrainedEye/UI/WatchView.swift": 62,
    "ios/TrainedEye/Clips/ClipStore.swift": 63, "ios/TrainedEye/UI/ClipsView.swift": 63,
    "ios/README.md": 63,
    # M3: labelling and evaluation
    "backend/labeltool/images.py": 64, "backend/labeltool/static/images.html": 64,
    "backend/labeltool/static/style.css": 64,
    "backend/labeltool/static/fonts/InterVariable.woff2": 64,
    "backend/labeltool/static/fonts/LICENSE.txt": 64, "tests/test_labeltool_images.py": 64,
    "backend/labeltool/clips.py": 65, "backend/labeltool/static/clips.html": 65,
    "tests/test_labeltool_clips.py": 65,
    "backend/eval/metrics.py": 66, "backend/eval/svg.py": 66,
    "tests/test_eval_metrics.py": 66, "tests/test_eval_svg.py": 66,
    "backend/eval/charts.py": 67, "backend/eval/timeline.py": 67,
    "tests/test_eval_charts.py": 67, "tests/test_eval_timeline.py": 67,
    "backend/eval/spark_scorer.py": 68, "backend/eval/regression.py": 68,
    "tests/test_spark_scorer.py": 68, "tests/test_regression.py": 68,
    "backend/eval/writeup.py": 69, "tests/test_eval_writeup.py": 69,
    "backend/eval/gallery.py": 70, "tests/test_eval_gallery.py": 70,
    "backend/eval/report.py": 71, "tests/test_eval_report.py": 71,
    "backend/eval/readme.py": 72, "backend/cli_eval.py": 72, "Makefile": 72,
    "tests/test_eval_readme.py": 72, "tests/test_cli_eval.py": 72,
    "backend/eval/spark_errors.py": 73, "backend/dataengine/realclips.py": 73,
    "docs/recording-protocol.md": 73, "tests/test_spark_errors.py": 73,
    "tests/test_realclips.py": 73, "tests/test_training_prep.py": 38,
    "backend/eval/plan.py": 74, "tests/test_plan.py": 74, "tests/test_ledger.py": 31,
    "backend/eval/plan_file.py": 75, "backend/cli_experiments.py": 75,
    "tests/test_plan_file.py": 75, "tests/test_cli_experiments.py": 75,
    "backend/eval/replay.py": 76, "backend/eval/wandb_report.py": 76,
    "tests/test_wandb_report.py": 76,
    "backend/cli_replay.py": 77, "tests/test_replay.py": 77,
    # M4: release tools and the web playground
    "backend/release/faces.py": 78, "tests/test_faces.py": 78,
    "backend/release/model_card.py": 79, "tests/test_model_card.py": 79,
    "backend/release/package.py": 80, "backend/release/BASE_LICENSE.txt": 80,
    "tests/test_release_package.py": 80,
    "backend/regenerate.py": 81, "backend/release/weights.py": 81,
    "tests/test_regenerate.py": 81, "tests/test_release_weights.py": 81,
    "backend/release/test_set.py": 82, "tests/test_test_set.py": 82,
    "backend/cli_release.py": 83, "tests/test_cli_release.py": 83,
    "backend/playground/data.py": 84, "tests/test_playground_data.py": 84,
    "backend/playground/skills.py": 85, "tests/test_playground_skills.py": 85,
    "backend/playground/budget.py": 86, "backend/playground/runs.py": 86,
    "tests/test_playground_budget.py": 86, "tests/test_playground_runs.py": 86,
    "backend/playground/api.py": 87, "backend/playground/publish.py": 87,
    "tests/test_playground_teach.py": 87,
    "backend/playground/static/index.html": 88, "backend/playground/static/style.css": 88,
    "backend/playground/static/fonts/InterVariable.woff2": 88,
    "backend/playground/static/fonts/LICENSE.txt": 88,
    "backend/playground/static/timeline.js": 89, "backend/playground/static/wall.js": 89,
    "backend/playground/static/app.js": 90,
    "backend/playground/app.py": 91, "tests/test_playground_app.py": 91,
    "backend/playground/modal_app.py": 92, "backend/cli_playground.py": 92,
    "tests/test_playground_modal.py": 92, "tests/test_cli_playground.py": 92,
    # the notes, in their final form, with the last stage
    ".gitignore": 92, "CLAUDE.md": 92, "HANDOVER.md": 92, "PROGRESS.md": 92,
    "CONTRIBUTIONS.md": 92,
}

_DOC_HEAD = between(CLI, "The `te` command line. The M0", "\n  te spend") + "\n"
_REG = {
    m: f"\n    from backend.{m} import add_parsers as {n}\n\n    {n}(sub)\n"
    for m, n in (("cli_train", "train_parsers"), ("cli_watch", "watch_parsers"),
                 ("cli_serving", "serving_parsers"), ("cli_eval", "eval_parsers"),
                 ("cli_experiments", "experiment_parsers"), ("cli_replay", "replay_parsers"),
                 ("cli_release", "release_parsers"), ("cli_playground", "playground_parsers"))
}
_V30_SERVE = ('    print(f"Trained Eye backend: ws://{args.host}:{args.port}/ws  (Ctrl+C to stop)")\n'
              '    uvicorn.run("backend.server:app", host=args.host, port=args.port, '
              'log_level="info")\n')

PIECES = {
    "backend/cli.py": [
        ("text", 42, _DOC_HEAD, "The `te` command line.\n"),
        ("text", 42, between(CLI, "  te train ", "  te watch "), ""),
        ("text", 46, between(CLI, "  te watch ", "  te export "), ""),
        ("text", 48, between(CLI, "  te export ", "  te eval "), ""),
        ("text", 72, between(CLI, "  te eval ", "  te clips "), ""),
        ("text", 75, between(CLI, "  te clips ", "  te replay "), ""),
        ("text", 77, between(CLI, "  te replay ", "  te manifests"), ""),
        ("text", 83, between(CLI, "  te manifests", "  te playground"), ""),
        ("text", 92, between(CLI, "  te playground", '"""\n\nfrom __future__'), ""),
        ("text", 45, "import copy\n", ""),
        ("text", 45, "import socket\n", ""),
        ("text", 42, between(CLI, "# The cli_* modules take", "from backend.meta.client"),
         "from backend.config import ASK_ABOVE_USD, ConfigError, load_settings\n"),
        ("text", 45, between(CLI, "def lan_address()", "def cmd_serve(args)"), ""),
        ("text", 45, between(CLI, "    host = args.host\n", "    return 0\n"), _V30_SERVE),
        ("text", 42, between(CLI, "def modal_ready()", "def positive("), ""),
        ("text", 46, between(CLI, "def positive(", "def count("), ""),
        ("text", 42, _REG["cli_train"], ""),
        ("text", 46, _REG["cli_watch"], ""),
        ("text", 48, _REG["cli_serving"], ""),
        ("text", 72, _REG["cli_eval"], ""),
        ("text", 75, _REG["cli_experiments"], ""),
        ("text", 77, _REG["cli_replay"], ""),
        ("text", 83, _REG["cli_release"], ""),
        ("text", 92, _REG["cli_playground"], ""),
    ],
    "tests/test_modal_train.py": [
        ("text", 42, "import pytest\n\nfrom backend import cli_train\n\n", "import pytest\n\n"),
        ("text", 42, '    assert cli_train.TRAIN_VOLUME == constants()["VOLUME"]\n', ""),
    ],
    "tests/test_ledger.py": [
        ("test", 74, "test_the_plans_spark_price_uses_the_measured_image_tokens"),
    ],
    "tests/test_training_prep.py": [
        ("test", 73, "test_real_and_mixed_take_the_clips_and_session_2_only_validates"),
        ("test", 73, "test_a_limit_counts_session_1_frames_but_never_session_2"),
        ("test", 73, "test_clips_from_another_spec_or_tampered_ones_are_refused"),
    ],
}
