"""Peel spec for stages v50 to v105, on v49: the rest of M2 (teaching from the phone with
its review fixes, export and serving, v50 to v58), the iOS app (v59 to v74), M3 (labelling,
evaluation and the four experiments, v75 to v89) and M4 (the release tools, the playground
and the notes, v90 to v105). Each stage changes 150 to 300 lines.

backend/cli.py changes in four stages (v58 export and serving, v88 the M3 options and
commands, v89 the experiment commands, v103 the release commands) and tests/test_cli.py in
two (v58, v88); v88, v89 and v103 use exact edits and v58 restores v49's files.
backend/training/prepare.py and its test change in v54 and again in v89, where the
missing-dataset hint starts naming te unfiltered. Every other file changes in one stage.
"""

BASE, FIRST, LAST = 49, 50, 105

IOS = "ios/TrainedEye/"

# ---- backend/cli.py: the M3 stage (v88 back to v87) ----

TRAIN_ESTIMATE_NEW = '''        timing = json.loads(timing_path.read_text()) if timing_path.is_file() else None
        from backend.eval.plan import steps_samples

        samples = (steps_samples(args.max_steps, train_n, val_n) if args.max_steps
                   else train_n + val_n)
        estimate = train_estimate(samples, timing)
'''
TRAIN_ESTIMATE_OLD = '''        timing = json.loads(timing_path.read_text()) if timing_path.is_file() else None
        estimate = train_estimate(train_n + val_n, timing)
'''

WATCH_NEEDS_NEW = '''    if args.scorer == "student" and not settings.student_url:
        print("te: watch needs the trained model's endpoint: set TE_STUDENT_URL (or use "
              "--scorer spark)", file=sys.stderr)
'''
WATCH_NEEDS_OLD = '''    if not settings.student_url:
        print("te: watch needs the trained model's endpoint: set TE_STUDENT_URL",
              file=sys.stderr)
'''

WATCH_SCORER_NEW = '''    if args.log:  # a rerun replays every frame (cached ones are free): start the log afresh
        Path(args.log).unlink(missing_ok=True)
    log = FrameLog(Path(args.log) if args.log else
                   settings.data_dir / "watch" / f"replay-{stamp}-{Path(args.clip).stem}.jsonl")
    if args.scorer == "spark":  # Experiment 4: Muse Spark judges every frame, paid per frame
        from backend.eval.plan import spark_usd_per_check
        from backend.eval.spark_scorer import SparkScorer, default_log

        settings.secret("MODEL_API_KEY")
        frames = list(frames)
        cost = len(frames) * len(skills) * spark_usd_per_check(settings)
        print(f"Muse Spark on {len(frames)} frames x {len(skills)} skills: about ${cost:.2f} "
              "before cache hits")
        over = cost > ASK_ABOVE_USD
        if (over or not args.yes) and not ask("Spend that?"):
            return 1
        student = SparkScorer(client, latency_log=default_log(settings.data_dir))
        for skill in skills:  # Spark answers yes, no or unsure (0.5): only a yes counts
            skill.trigger.threshold = 1.0
    else:
        student = OpenAIStudent(settings.student_url, settings.student_model,
                                api_key=settings.secrets.get("STUDENT_API_KEY"))
    watcher = Watcher(skills, student, confirm=confirm, log=log)
'''
WATCH_SCORER_OLD = '''    log = FrameLog(settings.data_dir / "watch" / f"replay-{stamp}-{Path(args.clip).stem}.jsonl")
    watcher = Watcher(skills, OpenAIStudent(settings.student_url, settings.student_model,
                                            api_key=settings.secrets.get("STUDENT_API_KEY")),
                      confirm=confirm, log=log)
'''

CLI_M3 = [
    ('''  te label          mark where a clip's condition is true, or hand-label generated images
  te eval           score replayed, labelled clips: results.md and the charts (make eval)
''', ""),
    (TRAIN_ESTIMATE_NEW, TRAIN_ESTIMATE_OLD),
    ('''        print(f"{args.max_steps or optimizer_steps(train_n)} optimizer steps; estimate: about "''',
     '''        print(f"{optimizer_steps(train_n)} optimizer steps; estimate: about "'''),
    ('''    if args.max_steps and not args.smoke:
        tags.append(f"steps:{args.max_steps}")
''', ""),
    ('''    elif args.max_steps:
        command += ["--max-steps", str(args.max_steps)]
''', ""),
    (WATCH_NEEDS_NEW, WATCH_NEEDS_OLD),
    (WATCH_SCORER_NEW, WATCH_SCORER_OLD),
    ('''    if args.scorer == "spark":
        print(f"Spark: {json.dumps(student.summary())}")
''', ""),
    ("CUT:def cmd_eval(args) -> int:", "def positive(text: str) -> int:"),
    ('''    train.add_argument("--max-steps", type=count, help="fixed optimizer steps for Experiments 1 "
                       "to 3 (backend/eval/plan.py)")
''', ""),
    ('''    watch.add_argument("--scorer", choices=["student", "spark"], default="student",
                       help="who scores each frame: the trained model, or Muse Spark (paid)")
    watch.add_argument("--log", help="frame log path (the evaluation plan names one per clip)")
    watch.add_argument("--yes", action="store_true", help="with --scorer spark: no question "
                       "(not over $15)")
''', ""),
    ('CUT:\n    ev = sub.add_parser("eval"', "    return parser\n"),
]

CLI_EXPERIMENTS = [
    ("  te experiments, te unfiltered, te spark-errors, te regress: M3 "
     "(backend/cli_experiments.py)\n", ""),
    ('''
    from backend.cli_experiments import add_parsers as experiment_parsers

    experiment_parsers(sub)
''', ""),
]

PREPARE_HINT = [
    ('                else f"te unfiltered --skill {name} first")',
     '                else f"Experiment 2\'s unfiltered arm for {name} first")'),
]
PREPARE_HINT_TEST = [
    ('match="te unfiltered --skill pot_boils first"', 'match="unfiltered arm for pot_boils first"'),
]

CLI_RELEASE = [
    ("  te manifests, te regenerate, te blur, te release, te playground: M4 "
     "(backend/cli_release.py)\n", ""),
    ('''
    from backend.cli_release import add_parsers as release_parsers

    release_parsers(sub)
''', ""),
]

PEEL = {
    # the rest of M2: teaching from the phone, export and serving
    50: {"remove": ["backend/teach_session.py"], "restore": ["backend/meta/client.py"]},
    51: {"remove": ["backend/teach_upload.py"]},
    52: {"remove": ["tests/test_teach_session.py", "tests/test_teach_upload.py"]},
    53: {"remove": ["backend/teach_api.py"]},
    54: {"restore": ["backend/server.py", "tests/test_server.py", "backend/config.py",
                     "tests/test_config.py", ".env.example", "backend/inference/student.py",
                     "tests/test_student.py", "backend/training/prepare.py",
                     "tests/test_training_prep.py"]},
    55: {"remove": ["tests/test_teach_api.py"]},
    56: {"remove": ["backend/training/export.py", "backend/training/export_check.py"]},
    57: {"remove": ["backend/inference/modal_serve.py", "tests/test_modal_serve.py",
                    "tests/test_export.py"]},
    58: {"restore": ["backend/cli.py", "tests/test_cli.py"]},
    # the iOS app
    59: {"remove": [IOS + "Models/Settings.swift", IOS + "Models/Protocol.swift"]},
    60: {"remove": [IOS + "Network/BackendClient.swift", IOS + "Network/WatchSocket.swift"]},
    61: {"remove": [IOS + "Meta/VideoFrameDecoder.swift", IOS + "Meta/HEVCKeyframe.swift",
                    IOS + "Meta/LICENSE"]},
    62: {"remove": [IOS + "Glasses/GlassesConnection.swift", IOS + "Glasses/MockGlasses.swift"]},
    63: {"remove": [IOS + "Glasses/GlassesCamera.swift"]},
    64: {"remove": [IOS + "Camera/FrameThrottle.swift", IOS + "Camera/PhoneCamera.swift",
                    IOS + "Audio/Speaker.swift"]},
    65: {"remove": [IOS + "Models/TeachModels.swift", IOS + "Network/TeachClient.swift",
                    IOS + "Audio/RequestRecorder.swift"]},
    66: {"remove": [IOS + "Teach/TeachFlow.swift"]},
    67: {"remove": [IOS + "App/AppModel.swift"]},
    68: {"remove": [IOS + "UI/Theme.swift", IOS + "UI/Components.swift"]},
    69: {"remove": [IOS + "App/TrainedEyeApp.swift", "ios/TrainedEye-Info.plist",
                    "ios/TrainedEye.entitlements"]},
    70: {"remove": [IOS + "UI/RootView.swift", IOS + "UI/ConnectView.swift"]},
    71: {"remove": [IOS + "UI/TeachView.swift"]},
    72: {"remove": [IOS + "UI/SkillsView.swift", IOS + "UI/WatchView.swift"]},
    73: {"remove": [IOS + "Clips/ClipStore.swift", IOS + "UI/ClipsView.swift"]},
    74: {"remove": ["ios/README.md"]},
    # M3: labelling, evaluation, the experiments
    75: {"remove": ["backend/labeltool/images.py", "backend/labeltool/static/images.html",
                    "backend/labeltool/static/style.css"]},
    76: {"remove": ["backend/labeltool/clips.py", "backend/labeltool/static/clips.html"]},
    77: {"remove": ["tests/test_labeltool_images.py", "tests/test_labeltool_clips.py"]},
    78: {"remove": ["backend/eval/metrics.py"]},
    79: {"remove": ["tests/test_eval_metrics.py", "backend/eval/svg.py"]},
    80: {"remove": ["backend/eval/charts.py", "tests/test_eval_charts.py",
                    "tests/test_eval_svg.py"]},
    81: {"remove": ["backend/eval/timeline.py", "tests/test_eval_timeline.py"]},
    82: {"remove": ["backend/eval/writeup.py", "tests/test_eval_writeup.py"]},
    83: {"remove": ["backend/eval/report.py", "tests/test_eval_report.py"]},
    84: {"remove": ["backend/eval/spark_scorer.py", "tests/test_spark_scorer.py"]},
    85: {"remove": ["backend/eval/spark_errors.py", "tests/test_spark_errors.py"]},
    86: {"remove": ["backend/eval/plan.py", "tests/test_plan.py"]},
    87: {"remove": ["backend/eval/regression.py", "tests/test_regression.py"]},
    88: {"remove": ["Makefile"],
         "restore": ["backend/watch/watcher.py", "backend/training/modal_train.py",
                     "tests/test_watcher.py"],
         "edit": {"backend/cli.py": CLI_M3,
                  "tests/test_cli.py": [
                      ("CUT_TO_END:\n\n\ndef test_label_starts_the_clip_or_image_tool_on_localhost",
                       None)]}},
    89: {"remove": ["backend/cli_experiments.py", "tests/test_cli_experiments.py"],
         "edit": {"backend/cli.py": CLI_EXPERIMENTS, "backend/training/prepare.py": PREPARE_HINT,
                  "tests/test_training_prep.py": PREPARE_HINT_TEST}},
    # M4: release tools, the playground, the notes
    90: {"remove": ["backend/release/faces.py", "tests/test_faces.py"]},
    91: {"remove": ["backend/regenerate.py", "tests/test_regenerate.py"]},
    92: {"remove": ["backend/release/model_card.py"]},
    93: {"remove": ["tests/test_model_card.py"]},
    94: {"remove": ["backend/playground/data.py", "tests/test_playground_data.py"]},
    95: {"remove": ["backend/playground/skills.py", "tests/test_playground_skills.py"]},
    96: {"remove": ["backend/playground/budget.py", "tests/test_playground_budget.py"]},
    97: {"remove": ["backend/playground/runs.py", "backend/playground/api.py"]},
    98: {"remove": ["tests/test_playground_runs.py", "tests/test_playground_teach.py"]},
    99: {"remove": ["backend/playground/static/index.html",
                    "backend/playground/static/style.css"]},
    100: {"remove": ["backend/playground/static/timeline.js",
                     "backend/playground/static/wall.js"]},
    101: {"remove": ["backend/playground/static/app.js"]},
    102: {"remove": ["backend/playground/app.py"]},
    103: {"remove": ["backend/cli_release.py"], "edit": {"backend/cli.py": CLI_RELEASE}},
    104: {"remove": ["tests/test_playground_app.py", "tests/test_cli_release.py"]},
    105: {"remove": ["CONTRIBUTIONS.md"],
          "restore": ["HANDOVER.md", "PROGRESS.md", "CLAUDE.md"]},
}
