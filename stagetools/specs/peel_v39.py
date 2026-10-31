"""Peel spec for stages v39 to v49, on v38: a review of v31 to v38 (v39, v40), the parts
of M2 that need neither the Mac nor Modal (v41 to v48), then the review's last small fixes
(v49).

backend/config.py changes in three stages (v39 the ~ error, v44 the student endpoint,
v47 the GPU price), tests/test_config.py in two (v39, v44), and backend/cli.py and
tests/test_cli.py in two (v47, v49); those steps use exact edits. Every other file
changes in one stage only.
"""

BASE, FIRST, LAST = 38, 39, 49

PRICE_NEW = '''    "muse-voice-transcribe-1.0": Price(per_audio_hour=0.18),
}

# Modal GPU list price per hour, used for training estimates. Not checked: modal.com was
# not reachable from the build sandbox. Check modal.com/pricing and fix it here.
MODAL_GPU_USD_PER_HOUR = {"H100": 3.95}
ASK_ABOVE_USD = 15.0  # the author's rule: ask before any job estimated over this
'''
PRICE_OLD = '''    "muse-voice-transcribe-1.0": Price(per_audio_hour=0.18),
}
'''

STUDENT_FIELDS_NEW = '''    train_on_model_outputs: bool = True
    # The trained student's OpenAI-compatible endpoint for watch mode (llama-server on the
    # Mac or vLLM on Modal) and the model name it serves.
    student_url: str | None = None
    student_model: str = "trained-eye"
    secrets:'''
STUDENT_FIELDS_OLD = '''    train_on_model_outputs: bool = True
    secrets:'''
STUDENT_LOAD_NEW = '''                                      Settings.train_on_model_outputs),
        student_url=(merged.get("TE_STUDENT_URL") or "").strip().rstrip("/") or None,
        student_model=(merged.get("TE_STUDENT_MODEL") or "").strip() or Settings.student_model,
        secrets='''
STUDENT_LOAD_OLD = '''                                      Settings.train_on_model_outputs),
        secrets='''
STUDENT_SHOW_NEW = '''        f"  train on model outputs: {'yes' if settings.train_on_model_outputs else 'no'}",
        (f"  student: {settings.student_url or 'not set (TE_STUDENT_URL)'} "
         f"({settings.student_model})"),
'''
STUDENT_SHOW_OLD = '''        f"  train on model outputs: {'yes' if settings.train_on_model_outputs else 'no'}",
'''
STUDENT_TEST_NEW = '''

def test_student_endpoint_settings():
    s = load_settings(env={}, env_file=None)
    assert s.student_url is None and s.student_model == "trained-eye"
    assert "student: not set (TE_STUDENT_URL)" in describe(s)
    s = load_settings(env={"TE_STUDENT_URL": " http://192.168.1.23:8080/v1/ ",
                           "TE_STUDENT_MODEL": "washer-v1"}, env_file=None)
    assert (s.student_url, s.student_model) == ("http://192.168.1.23:8080/v1", "washer-v1")
'''

# Stage v49, read from the finished files when this spec was written.
CLI_V49 = [('FAILURES_HINT = ("failed calls are not cached and are paid for again on a rerun: look at "\n                 "the reason above (and the skill folder) before running it again")', 'FAILURES_HINT = ("failed calls are not cached and are paid for again on a rerun: look at "\n                 "the reasons above (and in the skill folder) before running it again")'), ('    except TooManyFailures as err:\n        print(f"te: {err}\\nte: teach stopped; {FAILURES_HINT}", file=sys.stderr)\n        return 2', '    except TooManyFailures:\n        print(f"te: teach stopped; {FAILURES_HINT}", file=sys.stderr)\n        raise'), ('    check_name(args.skill)\n    request = " ".join((args.request or "").split())\n    if not (args.audio or request):\n        raise ValueError("pass --audio or --request")', '    check_name(args.skill)\n    if not (args.audio or args.request):\n        raise ValueError("pass --audio or --request")\n    request = " ".join((args.request or "").split())'), ('    teach.add_argument("--positives", type=count, default=150)\n    teach.add_argument("--negatives", type=count, default=150)', '    teach.add_argument("--positives", type=int, default=150)\n    teach.add_argument("--negatives", type=int, default=150)'), ('def count(text: str) -> int:\n    """An argparse type: a whole number, 0 or more."""\n    value = int(text)\n    if value < 0:\n        raise argparse.ArgumentTypeError(f"must be 0 or more, got {value}")\n    return value\n\n\n', '')]

CLI_TEST_V49 = [('\n\ndef test_teach_rejects_a_blank_request_and_negative_counts_before_any_estimate(env, monkeypatch,\n                                                                               capsys):\n    calls = use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))\n    clip = str(write_clip(env / "clip"))\n    assert cli.main(["teach", "--skill", "washer_done", "--clip", clip, "--request", "   "]) == 2\n    captured = capsys.readouterr()\n    assert "pass --audio or --request" in captured.err and "estimate" not in captured.out\n    with pytest.raises(SystemExit) as stop:\n        cli.main([*TEACH, "--clip", clip, "--positives", "-100"])\n    assert stop.value.code == 2 and "must be 0 or more" in capsys.readouterr().err\n    assert calls == []\n', '')]

PEEL = {
    49: {"restore": ["backend/teach.py", "tests/test_teach.py"],
         "edit": {"backend/cli.py": CLI_V49, "tests/test_cli.py": CLI_TEST_V49}},
    48: {"restore": ["HANDOVER.md", "PROGRESS.md", ".env.example", "docs/recording-protocol.md"]},
    47: {"restore": ["backend/cli.py", "tests/test_cli.py", "CLAUDE.md"],
         "edit": {"backend/config.py": [(PRICE_NEW, PRICE_OLD)]}},
    46: {"remove": ["backend/training/modal_train.py", "tests/test_modal_train.py"]},
    45: {"remove": ["backend/training/axolotl.py", "backend/training/prepare.py",
                    "tests/test_training_prep.py"]},
    44: {"restore": ["backend/server.py", "tests/test_server.py"],
         "edit": {"backend/config.py": [(STUDENT_FIELDS_NEW, STUDENT_FIELDS_OLD),
                                        (STUDENT_LOAD_NEW, STUDENT_LOAD_OLD),
                                        (STUDENT_SHOW_NEW, STUDENT_SHOW_OLD)],
                  "tests/test_config.py": [(STUDENT_TEST_NEW, "")]}},
    43: {"remove": ["backend/watch/watcher.py", "tests/test_watcher.py"]},
    42: {"remove": ["backend/inference/student.py", "tests/test_student.py"]},
    41: {"remove": ["backend/watch/trigger.py", "tests/test_trigger.py"]},
    40: {"restore": ["backend/skills/spec.py", "backend/skills/compiler.py",
                     "tests/test_skills.py", "tests/test_compiler.py"]},
    39: {"restore": ["backend/meta/client.py", "tests/test_client.py", "backend/config.py",
                     "tests/test_config.py"]},
}
