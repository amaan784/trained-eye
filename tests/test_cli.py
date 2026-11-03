"""CLI tests for the commands that talk to the Meta API. The client is swapped for one
backed by httpx.MockTransport, so nothing is sent or billed."""

import io
import json
import subprocess
from pathlib import Path

import httpx
import pytest

from backend import cli
from backend.meta.client import MetaClient


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODEL_API_KEY", "sk-test-cli-key")
    return tmp_path


def use_transport(monkeypatch, handler):
    calls = []

    def recording(request):
        calls.append(request)
        return handler(request)

    monkeypatch.setattr(
        cli, "make_client",
        lambda settings: MetaClient(settings, transport=httpx.MockTransport(recording)),
    )
    return calls


def test_api_smoke_then_cache_hit(env, monkeypatch, capsys):
    reply = {"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 40, "completion_tokens": 30}}
    calls = use_transport(monkeypatch, lambda r: httpx.Response(200, json=reply))
    assert cli.main(["api-smoke"]) == 0
    out = capsys.readouterr().out
    assert "one structured call to muse-spark-1.3" in out
    assert 'result: {"ok": true}' in out and "cost: $0.000178" in out
    assert cli.main(["api-smoke"]) == 0
    assert "cache hit" in capsys.readouterr().out
    assert len(calls) == 1
    assert cli.main(["spend"]) == 0
    assert "smoke" in capsys.readouterr().out


def test_models_reports_missing(env, monkeypatch, capsys):
    offered = {"data": [{"id": m} for m in
                        ("muse-spark-1.3", "muse-wonder-1.0", "sam-3.1", "muse-image-1.0")]}
    use_transport(monkeypatch, lambda r: httpx.Response(200, json=offered))
    assert cli.main(["models"]) == 1
    out = capsys.readouterr().out
    assert "configured models missing for this key: muse-voice-transcribe-1.0" in out


def test_missing_key_is_a_clean_error(env, monkeypatch, capsys):
    monkeypatch.delenv("MODEL_API_KEY")
    use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    assert cli.main(["api-smoke"]) == 2
    err = capsys.readouterr().err
    assert "MODEL_API_KEY is not set" in err


def test_api_error_is_a_clean_error(env, monkeypatch, capsys):
    use_transport(monkeypatch, lambda r: httpx.Response(401, json={"error": "invalid key"}))
    assert cli.main(["models"]) == 2
    err = capsys.readouterr().err
    assert "401" in err and "sk-test-cli-key" not in err


def write_clip(folder, n=12):
    from backend.cli import synthetic_jpeg

    folder.mkdir()
    for i in range(n):
        (folder / f"f{i:03d}.jpg").write_bytes(synthetic_jpeg(i, size=(320, 240)))
    return folder


def test_teach_estimate_only_makes_no_call(env, monkeypatch, capsys):
    calls = use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    clip = write_clip(env / "clip")
    code = cli.main(["teach", "--skill", "washer_done", "--clip", str(clip), "--request",
                     "let me know when the washer says done", "--estimate-only"])
    assert code == 0 and calls == []
    out = capsys.readouterr().out
    assert out.startswith("estimate for washer_done") and "images $3.00" in out
    assert "train on model outputs: yes" in out


def test_teach_refused_request_stops_for_free(env, monkeypatch, capsys):
    calls = use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    clip = write_clip(env / "clip")
    code = cli.main(["teach", "--skill", "sam_arrives", "--clip", str(clip), "--request",
                     "Tell me when Sam gets here", "--yes"])
    assert code == 1 and calls == []
    out = capsys.readouterr().out
    assert "I can't learn that one." in out and "stopped: refused" in out


def test_teach_rejects_a_bad_skill_name_before_anything(env, monkeypatch, capsys):
    calls = use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    clip = write_clip(env / "clip")
    code = cli.main(["teach", "--skill", "washer-done", "--clip", str(clip), "--request",
                     "let me know when the washer says done", "--yes"])
    assert code == 2 and calls == []
    assert "snake_case" in capsys.readouterr().err
    assert not (env / "data" / "skills").exists()


SPEC_REPLY = {
    "refused": False, "refusal_reason": None, "name": "washer_done",
    "sam_prompt": "washing machine display",
    "target_description": "the digital display on the front of the washer",
    "positive": "the display shows the cycle has finished (e.g. 'End' or '0:00')",
    "negatives": ["display shows time remaining", "display off"],
    "alert": "Your laundry's done.", "cooldown_s": 600, "confirm_with_cloud": True,
}
TEACH = ["teach", "--skill", "washer_done", "--request", "let me know when the washer says done"]


def answer(monkeypatch, *replies):
    """Type `replies` at the [y/N] prompts, in order; returns the prompts shown."""
    prompts, queue = [], list(replies)

    def fake_input(prompt=""):
        prompts.append(prompt)
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


def spec_only(request):
    body = json.loads(request.content or b"{}")
    if body.get("response_format", {}).get("json_schema", {}).get("name") == "skill_spec":
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps(SPEC_REPLY)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1600, "completion_tokens": 120}})
    pytest.fail(f"nothing after the spec should be sent: {request.url.path}")


def test_teach_asks_before_spending(env, monkeypatch):
    calls = use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    prompts = answer(monkeypatch, "n")
    code = cli.main([*TEACH, "--clip", str(write_clip(env / "clip"))])
    assert code == 1 and calls == []
    assert len(prompts) == 1 and prompts[0].startswith("Spend up to about $")
    assert not (env / "data" / "skills").exists()


def test_teach_read_back_no_stops_after_the_spec_and_says_how_to_correct(env, monkeypatch,
                                                                         capsys):
    calls = use_transport(monkeypatch, spec_only)
    prompts = answer(monkeypatch, "y", "n")
    code = cli.main([*TEACH, "--clip", str(write_clip(env / "clip"))])
    assert code == 1 and len(calls) == 1
    assert [p.split(" ")[0] for p in prompts] == ["Spend", "Is"]
    out = capsys.readouterr().out
    assert "I'll watch the digital display" in out and "stopped: not confirmed" in out
    assert '--correct "what is wrong"' in out


def test_teach_prompt_at_eof_is_a_clean_no(env, monkeypatch, capsys):
    calls = use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert cli.main([*TEACH, "--clip", str(write_clip(env / "clip"))]) == 1
    assert calls == [] and "Spend up to about" in capsys.readouterr().out


def test_teach_audio_and_request_are_exclusive(env, monkeypatch, capsys):
    calls = use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    with pytest.raises(SystemExit) as stop:
        cli.main([*TEACH, "--clip", "x", "--audio", "request.wav"])
    assert stop.value.code == 2 and calls == []
    assert "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize("labels", ['{"kf_00.jpg": "yes"}', '["kf_00.jpg"]', "not json"])
def test_teach_rejects_unusable_labels_before_the_estimate(env, monkeypatch, capsys, labels):
    calls = use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    (env / "labels.json").write_text(labels)
    code = cli.main([*TEACH, "--clip", str(write_clip(env / "clip")), "--labels",
                     str(env / "labels.json")])
    captured = capsys.readouterr()
    assert code == 2 and calls == [] and captured.err.startswith("te: ")
    assert "estimate" not in captured.out


def test_teach_warns_when_the_estimate_is_over_what_is_left(env, monkeypatch, capsys):
    calls = use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    clip = str(write_clip(env / "clip"))
    assert cli.main([*TEACH, "--clip", clip, "--estimate-only"]) == 0
    assert "warning" not in capsys.readouterr().err
    monkeypatch.setenv("TE_SKILL_CAP_USD", "0.20")
    assert cli.main([*TEACH, "--clip", clip, "--estimate-only"]) == 0
    err = capsys.readouterr().err
    assert "only $0.20 is left under the spend caps for washer_done" in err
    assert "TE_SKILL_CAP_USD" in err and calls == []


def test_teach_interrupt_says_a_rerun_is_free(env, monkeypatch, capsys):
    def interrupt(request):
        raise KeyboardInterrupt
    use_transport(monkeypatch, interrupt)
    code = cli.main([*TEACH, "--clip", str(write_clip(env / "clip")), "--yes"])
    assert code == 130 and "running the same command again" in capsys.readouterr().err


def test_budget_stop_says_how_to_continue(env, monkeypatch, capsys):
    calls = use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    monkeypatch.setenv("TE_SKILL_CAP_USD", "0.000001")
    assert cli.main(["api-smoke"]) == 2
    assert "raise TE_SKILL_CAP_USD" in capsys.readouterr().err and calls == []


@pytest.mark.parametrize("bad", ["true", "[]", '{"ok": false}', '{"result": "ok"}'])
def test_api_smoke_bad_reply_is_not_cached(env, monkeypatch, capsys, bad):
    replies = [bad, '{"ok": true}']

    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": replies.pop(0)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 30}})
    calls = use_transport(monkeypatch, handler)
    assert cli.main(["api-smoke"]) == 2
    assert "unexpected smoke reply" in capsys.readouterr().err
    assert cli.main(["api-smoke"]) == 0
    assert len(calls) == 2


@pytest.mark.parametrize("problem, message", [
    ("missing clip", "no clip at"),
    ("empty folder", "no usable frames"),
    ("unreadable frame", "is not a readable image"),
    ("missing audio", "no audio file at"),
    ("long request", "over the 500 allowed"),
    ("repeated label", "more than once"),
])
def test_teach_checks_its_inputs_before_any_estimate(env, monkeypatch, capsys, problem, message):
    calls = use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    clip = write_clip(env / "clip")
    args = ["teach", "--skill", "washer_done", "--clip", str(clip), "--request",
            "let me know when the washer says done"]
    if problem == "missing clip":
        args[4] = str(env / "nope.mp4")
    elif problem == "empty folder":
        (env / "empty").mkdir()
        args[4] = str(env / "empty")
    elif problem == "unreadable frame":
        (clip / "f999.jpg").write_bytes(b"half a file")
    elif problem == "missing audio":
        args[5:7] = ["--audio", str(env / "nope.wav")]
    elif problem == "long request":
        args[6] = "tell me when the pot boils " * 20
    else:
        (env / "labels.json").write_text('{"kf_00_1a2b3c4d.jpg": "yes", '
                                         '"kf_00_1a2b3c4d.jpg": "no"}')
        args += ["--labels", str(env / "labels.json")]
    code = cli.main(args)
    captured = capsys.readouterr()
    assert code == 2 and calls == [] and message in captured.err
    assert "estimate" not in captured.out


def test_teach_without_a_key_stops_before_the_question(env, monkeypatch, capsys):
    calls = use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    monkeypatch.delenv("MODEL_API_KEY")
    prompts = answer(monkeypatch)
    assert cli.main([*TEACH, "--clip", str(write_clip(env / "clip"))]) == 2
    assert prompts == [] and calls == [] and "MODEL_API_KEY" in capsys.readouterr().err


def test_teach_fallback_without_labels_stops_before_the_question(env, monkeypatch, capsys):
    calls = use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    monkeypatch.setenv("TE_TRAIN_ON_MODEL_OUTPUTS", "0")
    prompts = answer(monkeypatch)
    assert cli.main([*TEACH, "--clip", str(write_clip(env / "clip"))]) == 1
    out = capsys.readouterr().out
    assert prompts == [] and calls == [] and "needs your own labels" in out
    assert list((env / "data/skills/washer_done/keyframes").glob("kf_*.jpg"))  # to label


def test_teach_shows_the_worst_case_and_asks_with_it(env, monkeypatch, capsys):
    use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    prompts = answer(monkeypatch, "n")
    assert cli.main([*TEACH, "--clip", str(write_clip(env / "clip"))]) == 1
    out = capsys.readouterr().out
    assert "at most $" in out and "at_most" not in out
    assert "(at most $" in prompts[0]


def test_teach_after_too_many_failures_says_a_rerun_pays_again(env, monkeypatch, capsys):
    def handler(request):
        if request.url.path.endswith("/chat/completions"):
            return spec_only(request)
        return httpx.Response(400, json={"error": "rejected"})  # every SAM call and edit

    use_transport(monkeypatch, handler)
    assert cli.main([*TEACH, "--clip", str(write_clip(env / "clip")), "--yes"]) == 2
    err = capsys.readouterr().err
    assert "paid for again on a rerun" in err and "failed so far" in err
    assert "without paying for them twice" not in err


def test_teach_prompt_with_stdin_closed_is_a_clean_no(env, monkeypatch, capsys):
    use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    monkeypatch.setattr("sys.stdin", None)
    assert cli.main([*TEACH, "--clip", str(write_clip(env / "clip"))]) == 1


def test_serve_prints_an_address_the_phone_can_use_and_keeps_server_logs(monkeypatch, capsys):
    import uvicorn

    ran = {}
    monkeypatch.setattr(cli, "lan_address", lambda: "192.168.1.23")
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: ran.update(kw))
    assert cli.main(["serve"]) == 0
    assert "ws://192.168.1.23:8765/ws" in capsys.readouterr().out
    assert ran["host"] == "0.0.0.0"
    assert ran["log_config"]["loggers"]["trained_eye"]["level"] == "INFO"
    monkeypatch.setattr(cli, "lan_address", lambda: None)
    cli.main(["serve", "--port", "9000"])
    assert "ws://<this Mac's Wi-Fi IP>:9000/ws" in capsys.readouterr().out


def test_serve_says_whether_the_phone_can_teach(env, monkeypatch, capsys):
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)
    monkeypatch.setenv("SERVER_TOKEN", "")
    cli.main(["serve"])
    assert "teaching from the phone: off (set SERVER_TOKEN" in capsys.readouterr().out
    monkeypatch.setenv("SERVER_TOKEN", "t0ken-never-printed")
    cli.main(["serve"])
    out = capsys.readouterr().out
    assert "teaching from the phone: on" in out and "t0ken-never-printed" not in out


def test_server_log_lines_reach_the_console_handler(capsys):
    import logging.config

    logging.config.dictConfig(cli.server_log_config())
    logging.getLogger("trained_eye.server").info("session opened")
    assert "session opened" in capsys.readouterr().err


def taught(env, skill="washer_done", n=40):
    from backend.cli import synthetic_jpeg
    from backend.dataengine.dataset import Example, write_dataset
    from backend.skills.spec import SkillSpec, Target

    root = env / "data/skills" / skill
    examples = [Example(f"{skill}-{'real' if i < 10 else 'synt'}-{i:03d}",
                        synthetic_jpeg(i, size=(64, 48)), "yes" if i % 2 else "no",
                        "real" if i < 10 else "synthetic", skill, "the display shows End",
                        ["display off"]) for i in range(n)]
    write_dataset(examples, root / "dataset", f"{skill}-teach")
    (root / "spec.json").write_text(SkillSpec(
        name=skill, request="tell me when the washer says End",
        target=Target(sam_prompt="washer display", description="the display"),
        positive="the display shows End", negatives=["display off"],
        alert="Laundry's done.", confirm_with_cloud=False).model_dump_json())
    return root


def fake_modal(monkeypatch, result=None, code=0):
    runs = []

    def run(command, cwd=None, check=False, **kw):
        if command[0] != "modal":
            return subprocess.CompletedProcess(command, 0, stdout="abc1234\n", stderr="")
        runs.append(command)
        path = command[command.index("--result") + 1]
        if result is not None:
            Path(path).write_text(json.dumps(result))
        return subprocess.CompletedProcess(command, code)

    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/" + name)
    return runs


TRAIN = ["train", "--skills", "washer_done", "--run", "exp1-mixed"]


def test_train_needs_a_smoke_timing_before_a_real_run(env, monkeypatch, capsys):
    taught(env)
    runs = fake_modal(monkeypatch)
    assert cli.main(TRAIN) == 2
    assert "run the same command with --smoke first" in capsys.readouterr().err and runs == []


def test_train_smoke_asks_runs_modal_and_saves_the_timing(env, monkeypatch, capsys):
    taught(env)
    runs = fake_modal(monkeypatch, {"seconds": 900.0, "train_runtime": 60.0,
                                    "train_samples_per_second": 0.5, "ok": True})
    prompts = answer(monkeypatch, "y")
    assert cli.main([*TRAIN, "--smoke"]) == 0
    assert prompts == ["Run the smoke on Modal? [y/N] "]
    [command] = runs
    assert command[:3] == ["modal", "run", "backend/training/modal_train.py"]
    assert command[-1] == "--smoke" and "--bundle" in command
    tags = command[command.index("--tags") + 1].split(",")
    assert {"exp:exp1", "skills:washer_done", "mix:mixed", "seed:42",
            "commit:abc1234"} <= set(tags) and any(t.startswith("examples:") for t in tags)
    timing = json.loads((env / "data/train/smoke-timing.json").read_text())
    assert timing["train_samples_per_second"] == 0.5
    manifest = json.loads((env / "data/train/exp1-mixed/manifest.json").read_text())
    assert manifest["image_root"] == "/vol/bundles/exp1-mixed"
    out = capsys.readouterr().out
    assert "bundle exp1-mixed:" in out and "smoke timing saved" in out


def test_train_prices_the_real_run_from_the_smoke_and_asks(env, monkeypatch, capsys):
    taught(env)
    (env / "data/train").mkdir(parents=True)
    (env / "data/train/smoke-timing.json").write_text(json.dumps(
        {"seconds": 900.0, "train_runtime": 60.0, "train_samples_per_second": 0.5}))
    runs = fake_modal(monkeypatch, {"seconds": 990.0, "ok": True})
    assert cli.main([*TRAIN, "--estimate-only"]) == 0
    out = capsys.readouterr().out
    # setup 840 s + 40 examples / 0.5 per s = 920 s, about 15 min
    assert "about 15 min of H100, about $1.01" in out and runs == []
    prompts = answer(monkeypatch, "n")
    assert cli.main(TRAIN) == 1 and runs == [] and "about $1.01" in prompts[0]
    assert cli.main([*TRAIN, "--yes"]) == 0 and len(runs) == 1


def test_train_over_fifteen_dollars_asks_even_with_yes(env, monkeypatch, capsys):
    taught(env)
    (env / "data/train").mkdir(parents=True)
    (env / "data/train/smoke-timing.json").write_text(json.dumps(
        {"seconds": 60.0, "train_runtime": 0.0, "train_samples_per_second": 0.001}))
    runs = fake_modal(monkeypatch, {"seconds": 1.0, "ok": True})
    prompts = answer(monkeypatch, "n")
    assert cli.main([*TRAIN, "--yes"]) == 1 and runs == [] and len(prompts) == 1
    assert "over $15" in capsys.readouterr().out


@pytest.mark.parametrize("args, message", [
    (["--skills", "washer_done", "--run", "Exp 1"], "run name"),
    (["--skills", "nope", "--run", "exp1"], "no dataset yet"),
    (["--skills", ",", "--run", "exp1"], "at least one"),
])
def test_train_input_errors(env, monkeypatch, capsys, args, message):
    taught(env)
    fake_modal(monkeypatch)
    assert cli.main(["train", *args]) == 2 and message in capsys.readouterr().err


def test_a_failed_training_run_says_where_the_log_is(env, monkeypatch, capsys):
    taught(env)
    fake_modal(monkeypatch, None, code=1)
    assert cli.main([*TRAIN, "--smoke", "--yes"]) == 1
    assert "runs/exp1-mixed/train.log" in capsys.readouterr().err


def test_watch_replays_a_clip_through_the_student(env, monkeypatch, capsys):
    from backend.inference import student

    taught(env)
    monkeypatch.setenv("TE_STUDENT_URL", "http://127.0.0.1:8080/v1")
    seen = []

    class Fake:
        def __init__(self, url, model, api_key=None, **options):
            seen.append((url, model, api_key))

        def score(self, image, spec):
            return student.Score(0.9, "yes", 0.1)

    monkeypatch.setattr(student, "OpenAIStudent", Fake)
    clip = write_clip(env / "clip", n=5)
    assert cli.main(["watch", "--skills", "washer_done", "--clip", str(clip)]) == 0
    out = capsys.readouterr().out
    assert seen == [("http://127.0.0.1:8080/v1", "trained-eye", None)]
    assert '2.0 s  washer_done: "Laundry\'s done."' in out and "cloud not checked" in out
    assert "5 frames scored at 1 fps" in out and "no calibrated threshold" in out
    [log] = (env / "data/watch").glob("replay-*-clip.jsonl")
    assert len(log.read_text().splitlines()) == 5


def test_watch_needs_the_student_endpoint(env, capsys):
    assert cli.main(["watch", "--skills", "washer_done", "--clip", "x"]) == 2
    assert "TE_STUDENT_URL" in capsys.readouterr().err


def test_calibrate_writes_the_threshold_watch_mode_then_uses(env, capsys):
    from backend.watch.watcher import load_skill

    root = taught(env)
    log, labels = env / "replay.jsonl", env / "labels.json"
    probs = [0.6 if i % 1800 < 3 else 0.2 for i in range(3600)] + [0.9] * 30
    log.write_text("".join(json.dumps({"t": float(i), "skill": "washer_done", "p": p}) + "\n"
                           for i, p in enumerate(probs)))
    labels.write_text(json.dumps({"true": [[3600, 3630]]}))
    assert cli.main(["calibrate", "--skill", "washer_done", "--log", str(log),
                     "--labels", str(labels)]) == 0
    out = capsys.readouterr().out
    assert "0 false alarms, 1 of 1 events caught" in out
    saved = json.loads((root / "threshold.json").read_text())
    assert 0.6 < saved["threshold"] <= 0.9 and saved["clips"][0]["frames"] == 3630
    assert load_skill(root).trigger.threshold == saved["threshold"]
    assert cli.main(["calibrate", "--skill", "washer_done", "--log", str(log), "--log", str(log),
                     "--labels", str(labels)]) == 2
    assert "one --labels file for each --log" in capsys.readouterr().err


def test_teach_rejects_a_blank_request_and_negative_counts_before_any_estimate(env, monkeypatch,
                                                                               capsys):
    calls = use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    clip = str(write_clip(env / "clip"))
    assert cli.main(["teach", "--skill", "washer_done", "--clip", clip, "--request", "   "]) == 2
    captured = capsys.readouterr()
    assert "pass --audio or --request" in captured.err and "estimate" not in captured.out
    with pytest.raises(SystemExit) as stop:
        cli.main([*TEACH, "--clip", clip, "--positives", "-100"])
    assert stop.value.code == 2 and "must be 0 or more" in capsys.readouterr().err
    assert calls == []


def test_export_needs_a_bundle_asks_and_reports_the_check(env, monkeypatch, capsys):
    import ast

    taught(env)
    assert cli.main(["export", "--run", "exp1-mixed"]) == 2  # no bundle yet
    (env / "data/train/exp1-mixed").mkdir(parents=True)
    (env / "data/train/exp1-mixed/manifest.json").write_text("{}")
    runs = fake_modal(monkeypatch, {"merged_dir": "/vol/runs/exp1-mixed/merged", "examples": 50,
                                    "agreement": 1.0, "all_agree": True, "bin_files": 0,
                                    "tensor_names_match": True, "disagreements": []})
    prompts = answer(monkeypatch, "n")
    assert cli.main(["export", "--run", "exp1-mixed"]) == 1 and runs == [] and prompts
    assert cli.main(["export", "--run", "exp1-mixed", "--yes"]) == 0
    [command] = runs
    assert command[:3] == ["modal", "run", "backend/training/export.py"]
    assert "same answer on 100% of 50 examples" in capsys.readouterr().out
    tree = ast.parse((cli.PROJECT_ROOT / "backend/training/export.py").read_text())
    timeout = next(eval(compile(ast.Expression(n.value), "export.py", "eval"))  # 3 * 3600
                   for n in tree.body if isinstance(n, ast.Assign)
                   and getattr(n.targets[0], "id", "") == "TIMEOUT_S")
    assert timeout == cli.EXPORT_MAX_HOURS * 3600  # the worst case printed is the real limit


def test_serve_student_deploys_with_the_run_and_stops(env, monkeypatch, capsys):
    calls = []

    def run(command, cwd=None, check=False, env=None, **kw):
        calls.append((command, (env or {}).get("TE_SERVE_RUN")))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/" + name)
    assert cli.main(["serve-student"]) == 2
    assert cli.main(["serve-student", "--run", "exp1-mixed", "--yes"]) == 0
    assert cli.main(["serve-student", "--stop"]) == 0
    assert calls == [(["modal", "deploy", "backend/inference/modal_serve.py"], "exp1-mixed"),
                     (["modal", "app", "stop", "trained-eye-student"], None)]
    assert "TE_STUDENT_URL" in capsys.readouterr().out


def test_teach_over_fifteen_dollars_asks_even_with_yes(env, monkeypatch, capsys):
    calls = use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    prompts = answer(monkeypatch, "n")
    code = cli.main([*TEACH, "--clip", str(write_clip(env / "clip")), "--positives", "600",
                     "--negatives", "600", "--yes"])
    assert code == 1 and calls == [] and len(prompts) == 1
    assert "so it needs your yes even with --yes" in capsys.readouterr().out


def test_label_starts_the_clip_or_image_tool_on_localhost(env, monkeypatch, capsys):
    import uvicorn

    ran = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: ran.append(kw))
    clip = env / "session3.mp4"
    clip.write_bytes(b"\x00" * 64)
    assert cli.main(["label", "--clip", str(clip), "--no-browser"]) == 0
    assert ran == [{"host": "127.0.0.1", "port": 8766, "log_level": "warning"}]
    assert "session3.labels.json" in capsys.readouterr().out
    assert cli.main(["label", "--skill", "nope", "--no-browser"]) == 2
    assert cli.main(["label", "--skill", "Bad-Name", "--no-browser"]) == 2


def test_eval_reports_a_missing_plan_and_scores_an_empty_one(env, capsys):
    assert cli.main(["eval", "--plan", str(env / "nope.json"), "--out", str(env / "eval")]) == 2
    (env / "plan.json").write_text('{"evaluations": []}')
    assert cli.main(["eval", "--plan", str(env / "plan.json"), "--out", str(env / "eval")]) == 0
    assert "0 of 0 evaluations scored" in capsys.readouterr().out
    assert (env / "eval/results.md").is_file()


def test_watch_with_spark_asks_first_and_only_a_yes_fires(env, monkeypatch, capsys):
    from backend.eval import spark_scorer
    from backend.inference import student

    taught(env)
    replies = []

    class Fake:
        def __init__(self, client, latency_log=None):
            pass

        def score(self, image, spec):
            return student.Score(replies.pop(0), "yes", 0.1)

        def summary(self):
            return {"frames": 5}

    monkeypatch.setattr(spark_scorer, "SparkScorer", Fake)
    clip = str(write_clip(env / "clip", n=5))
    args = ["watch", "--skills", "washer_done", "--clip", clip, "--scorer", "spark"]
    prompts = answer(monkeypatch, "n")
    assert cli.main(args) == 1 and len(prompts) == 1
    assert "Muse Spark on 5 frames x 1 skills" in capsys.readouterr().out
    log = env / "eval/frames.jsonl"
    log.parent.mkdir()
    log.write_text("a line from an earlier run\n")
    replies[:] = [0.5] * 5  # unsure is not a yes
    assert cli.main([*args, "--yes", "--log", str(log)]) == 0
    assert "washer_done:" not in capsys.readouterr().out
    assert "earlier run" not in log.read_text() and len(log.read_text().splitlines()) == 5
    replies[:] = [1.0] * 5
    assert cli.main([*args, "--yes", "--log", str(log)]) == 0
    assert 'washer_done: "Laundry\'s done."' in capsys.readouterr().out


def test_train_with_fixed_steps_prices_them_and_passes_them_on(env, monkeypatch, capsys):
    from backend.eval.plan import steps_samples

    taught(env)
    (env / "data/train").mkdir(parents=True)
    (env / "data/train/smoke-timing.json").write_text(json.dumps(
        {"seconds": 900.0, "train_runtime": 60.0, "train_samples_per_second": 0.5}))
    runs = fake_modal(monkeypatch, {"seconds": 990.0, "ok": True})
    assert cli.main([*TRAIN, "--max-steps", "10", "--yes"]) == 0
    out = capsys.readouterr().out
    manifest = json.loads((env / "data/train/exp1-mixed/manifest.json").read_text())
    samples = steps_samples(10, manifest["train_examples"], manifest.get("val_examples", 0))
    minutes = round((840 + samples / 0.5) / 60)  # the smoke's setup, then samples at 0.5/s
    assert f"10 optimizer steps; estimate: about {minutes} min of H100" in out
    [command] = runs
    assert command[-2:] == ["--max-steps", "10"]
    assert "steps:10" in command[command.index("--tags") + 1].split(",")


def status_of(env, run="exp1-mixed"):
    return json.loads((env / f"data/train/{run}.status.json").read_text())


def test_train_marks_the_run_running_then_done_or_failed(env, monkeypatch):
    taught(env)
    assert cli.main([*TRAIN, "--smoke", "--estimate-only"]) == 0
    assert not (env / "data/train/exp1-mixed.status.json").exists()  # nothing ran
    during = []
    fake_modal(monkeypatch, {"seconds": 900.0, "ok": True})
    modal = cli.subprocess.run

    def watching(command, **kw):
        if command[0] == "modal":
            during.append(status_of(env))
        return modal(command, **kw)

    monkeypatch.setattr(cli.subprocess, "run", watching)
    assert cli.main([*TRAIN, "--smoke", "--yes"]) == 0
    [running] = during
    assert (running["state"], running["skills"], running["finished"], running["smoke"]) == (
        "running", ["washer_done"], None, True)
    done = status_of(env)
    assert done["state"] == "done" and done["started"] == running["started"]
    assert done["finished"] >= done["started"]

    fake_modal(monkeypatch, None, code=1)
    assert cli.main([*TRAIN, "--smoke", "--yes"]) == 1
    assert status_of(env)["state"] == "failed"

    def interrupted(command, **kw):
        if command[0] == "modal":
            raise KeyboardInterrupt
        return subprocess.CompletedProcess(command, 0, stdout="abc1234\n", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        cli.main([*TRAIN, "--smoke", "--yes"])
    assert status_of(env)["state"] == "failed" and status_of(env)["finished"]


def test_serve_student_records_the_served_run_and_stop_clears_it(env, monkeypatch, capsys):
    from backend.training import runs

    codes = []
    monkeypatch.setattr(cli.subprocess, "run",
                        lambda command, **kw: subprocess.CompletedProcess(command, codes.pop(0)))
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/" + name)
    runs.write_status(env / "data", "exp1-mixed", "done", ["washer_done"],
                      started="2026-11-03T10:00:00Z", finished="2026-11-03T11:00:00Z")
    codes[:] = [1]
    assert cli.main(["serve-student", "--run", "exp1-mixed", "--yes"]) == 1
    assert not (env / "data/train/serving.json").exists()  # the deploy failed
    codes[:] = [0]
    assert cli.main(["serve-student", "--run", "exp1-mixed", "--yes"]) == 0
    serving = json.loads((env / "data/train/serving.json").read_text())
    assert serving["run"] == "exp1-mixed" and serving["skills"] == ["washer_done"]
    assert serving["deployed"].endswith("Z")
    assert "run exp1-mixed serves washer_done" in capsys.readouterr().out
    codes[:] = [1]
    assert cli.main(["serve-student", "--stop"]) == 1  # not stopped: still served
    assert (env / "data/train/serving.json").exists()
    codes[:] = [0]
    assert cli.main(["serve-student", "--stop"]) == 0
    assert not (env / "data/train/serving.json").exists()


def scoring(monkeypatch, p=0.9):
    """A fake OpenAIStudent: every frame scores p; returns the options it was built with."""
    from backend.inference import student

    built = []

    class Fake:
        def __init__(self, url, model, api_key=None, **options):
            built.append(options)

        def score(self, image, spec):
            return student.Score(p, "yes", 0.1)

    monkeypatch.setattr(student, "OpenAIStudent", Fake)
    return built


def replay_rows(log):
    return [json.loads(x) for x in Path(log).read_text().splitlines()]


def test_calibrate_per_run_then_watch_uses_that_run_s_threshold(env, monkeypatch, capsys):
    root = taught(env)
    monkeypatch.setenv("TE_STUDENT_URL", "http://127.0.0.1:8080/v1")
    log, labels = env / "replay.jsonl", env / "labels.json"
    probs = [0.6 if i % 1800 < 3 else 0.2 for i in range(3600)] + [0.9] * 30
    log.write_text("".join(json.dumps({"t": float(i), "skill": "washer_done", "p": p,
                                       "run": "exp1-mixed"}) + "\n"
                           for i, p in enumerate(probs)))
    labels.write_text(json.dumps({"true": [[3600, 3630]]}))
    calibrate = ["calibrate", "--skill", "washer_done", "--log", str(log), "--labels",
                 str(labels)]
    assert cli.main([*calibrate, "--run", "exp1-mixed"]) == 0
    out = capsys.readouterr().out
    saved_path = env / "data/watch/exp1-mixed/washer_done/threshold.json"
    saved = json.loads(saved_path.read_text())
    assert "washer_done for run exp1-mixed: threshold" in out and "saved: data/watch" in out
    assert saved["run"] == "exp1-mixed" and 0.6 < saved["threshold"] <= 0.9
    assert not (root / "threshold.json").exists()  # the skill-wide file is left alone
    assert cli.main([*calibrate, "--run", "exp1-real"]) == 2
    assert "was replayed for run exp1-mixed, not exp1-real" in capsys.readouterr().err
    assert cli.main([*calibrate, "--run", "../up"]) == 2
    assert "run name" in capsys.readouterr().err

    scoring(monkeypatch)
    clip = str(write_clip(env / "clip", n=4))
    replay = env / "out.jsonl"
    watch = ["watch", "--skills", "washer_done", "--clip", clip, "--log", str(replay)]
    assert cli.main([*watch, "--run", "exp1-mixed"]) == 0
    assert "no calibrated threshold" not in capsys.readouterr().out
    assert {(r["run"], r["threshold"], r["threshold_source"]) for r in replay_rows(replay)} == {
        ("exp1-mixed", saved["threshold"], "run")}
    assert cli.main([*watch, "--run", "exp1-mixed", "--threshold", "0.3"]) == 0
    assert {(r["threshold"], r["threshold_source"]) for r in replay_rows(replay)} == {
        (0.3, "flag")}
    assert cli.main([*watch, "--run", "exp1-real"]) == 0  # not calibrated for that run
    assert "no calibrated threshold yet; using 0.5" in capsys.readouterr().out
    assert {r["threshold_source"] for r in replay_rows(replay)} == {"default"}

    assert cli.main(calibrate) == 0  # no --run: the skill-wide file, with a note
    assert "te calibrate --run keeps that run's threshold apart" in capsys.readouterr().out
    assert cli.main([*watch, "--run", "exp1-real"]) == 0
    assert "no threshold for run exp1-real yet" in capsys.readouterr().out
    assert {r["threshold_source"] for r in replay_rows(replay)} == {"skill"}
    with pytest.raises(SystemExit):
        cli.main([*watch, "--threshold", "1.5"])


def test_watch_asks_the_student_as_it_was_trained_with_its_reply_budget(env, monkeypatch,
                                                                         capsys):
    root = taught(env)
    monkeypatch.setenv("TE_STUDENT_URL", "http://127.0.0.1:8080/v1")
    built = scoring(monkeypatch)
    watch = ["watch", "--skills", "washer_done", "--clip", str(write_clip(env / "clip", n=3))]
    assert cli.main(watch) == 0
    monkeypatch.setenv("TE_STUDENT_MAX_TOKENS", "900")
    monkeypatch.setenv("TE_SYSTEM_IN_USER", "1")  # for new datasets: this one is not folded
    assert cli.main(watch) == 0
    assert cli.main([*watch, "--max-tokens", "2048"]) == 0
    assert built == [{"max_tokens": 512, "system_in_user": False},
                     {"max_tokens": 900, "system_in_user": False},
                     {"max_tokens": 2048, "system_in_user": False}]
    manifest_path = root / "dataset/washer_done-teach.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps({**manifest, "system_in_user": True}))
    built.clear()
    capsys.readouterr()
    assert cli.main(watch) == 0
    assert built == [{"max_tokens": 900, "system_in_user": True}]
    assert "system prompt in the user text" in capsys.readouterr().out
    bundle = env / "data/train/exp1-mixed"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps(
        {"skills": {"washer_done": "x"}, "system_in_user": False}))
    assert cli.main([*watch, "--run", "exp1-mixed"]) == 0  # what the run trained on wins
    assert built[-1]["system_in_user"] is False
    with pytest.raises(SystemExit):
        cli.main([*watch, "--max-tokens", "0"])


# -- te train: W&B tags and team, artifacts, detach and collect, preflight; Modal spend --

SMOKE_RESULT = {"seconds": 900.0, "train_runtime": 60.0, "train_samples_per_second": 0.5,
                "ok": True}


def test_train_tags_the_data_and_base_passes_the_wandb_team_and_records_the_spend(
        env, monkeypatch):
    from backend.config import STUDENT_BASE_REVISION
    from backend.training import gpu_ledger
    from backend.training.axolotl import data_hash

    taught(env)
    runs = fake_modal(monkeypatch, SMOKE_RESULT)
    assert cli.main([*TRAIN, "--smoke", "--yes", "--limit", "20", "--wandb-entity", "team"]) == 0
    [command] = runs
    tags = command[command.index("--tags") + 1].split(",")
    manifest = json.loads((env / "data/train/exp1-mixed/manifest.json").read_text())
    assert {f"data:{data_hash(manifest)[:12]}", f"base:{STUDENT_BASE_REVISION[:12]}",
            "dataset:dataset", "images:20"} <= set(tags)
    assert command[command.index("--wandb-entity") + 1] == "team" and command[-1] == "--smoke"
    assert not {"--wandb-artifacts", "--detach", "--detached"} & set(command)
    [row] = gpu_ledger.read(env / "data")
    assert (row["kind"], row["name"], row["seconds"], row["usd"]) == (
        "train", "exp1-mixed", 900.0, 0.9875)


def test_wandb_artifacts_ask_even_with_yes_and_say_what_becomes_visible(env, monkeypatch,
                                                                        capsys):
    from backend.training.axolotl import data_hash

    taught(env)
    runs = fake_modal(monkeypatch, SMOKE_RESULT)
    prompts = answer(monkeypatch, "n")
    assert cli.main([*TRAIN, "--smoke", "--yes", "--wandb-artifacts"]) == 1
    assert runs == [] and prompts == ["Upload them to W&B? (asked even with --yes) [y/N] "]
    out, err = capsys.readouterr()
    manifest = json.loads((env / "data/train/exp1-mixed/manifest.json").read_text())
    assert f"dataset artifact exp1-mixed-{data_hash(manifest)[:12]} (40 images" in out
    assert "30 are Muse Image outputs" in out and "Faces in these images are not blurred" in out
    assert "nothing uploaded or trained" in err
    prompts = answer(monkeypatch, "y")
    assert cli.main([*TRAIN, "--smoke", "--yes", "--wandb-artifacts"]) == 0
    assert len(prompts) == 1  # --yes skipped the cost question, never this one
    [command] = runs
    assert "--wandb-artifacts" in command
    copied = env / "data/train/exp1-mixed/provenance/washer_done-teach.manifest.json"
    assert json.loads(copied.read_text())["examples_index"]  # the provenance goes along


def test_train_detach_outlives_the_client_and_collect_fetches_the_result(env, monkeypatch,
                                                                       capsys):
    from backend.training import gpu_ledger
    from backend.training import runs as run_files

    taught(env)
    calls = []

    def lost(command, cwd=None, check=False, **kw):
        if command[0] != "modal":
            return subprocess.CompletedProcess(command, 0, stdout="abc1234\n", stderr="")
        calls.append(command)
        return subprocess.CompletedProcess(command, 1)  # the connection dropped

    monkeypatch.setattr(cli.subprocess, "run", lost)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/" + name)
    assert cli.main([*TRAIN, "--smoke", "--yes", "--detach"]) == 1
    [command] = calls
    assert command[:4] == ["modal", "run", "--detach", "backend/training/modal_train.py"]
    assert "--detached" in command
    status = status_of(env)
    assert (status["state"], status["detached"], status["pid"]) == ("running", True, None)
    assert run_files.read_status(env / "data", "exp1-mixed")["state"] == "running"
    assert "te train --collect exp1-mixed" in capsys.readouterr().err

    remote = {"run": "exp1-mixed", "smoke": True, "started": status["started"],
              "finished": status["started"], **SMOKE_RESULT}
    gets = []

    def volume_get(command, cwd=None, check=False, **kw):
        gets.append(command)
        Path(command[-1], "result.json").write_text(json.dumps(remote))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", volume_get)
    assert cli.main(["train", "--collect", "exp1-mixed"]) == 0
    assert gets[0][:5] == ["modal", "volume", "get", "trained-eye", "/runs/exp1-mixed/result.json"]
    assert json.loads((env / "data/train/exp1-mixed.result.json").read_text()) == remote
    timing = json.loads((env / "data/train/smoke-timing.json").read_text())
    assert timing["train_samples_per_second"] == 0.5
    done = status_of(env)
    assert (done["state"], done["finished"], done["skills"]) == (
        "done", remote["finished"], ["washer_done"])
    out = capsys.readouterr().out
    assert "collected:" in out and "smoke timing saved" in out and "done: 900 s" in out
    assert not list((env / "data/train").glob(".collect-*"))
    assert cli.main(["train", "--collect", "exp1-mixed"]) == 0  # again: counted once
    assert [r["kind"] for r in gpu_ledger.read(env / "data")] == ["train"]


def test_collect_says_when_there_is_no_result_or_only_an_earlier_one(env, monkeypatch, capsys):
    from backend.training import runs as run_files

    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(cli.subprocess, "run",
                        lambda command, **kw: subprocess.CompletedProcess(command, 1))
    assert cli.main(["train", "--collect", "exp1-mixed"]) == 1
    assert "no result for exp1-mixed on the Modal volume yet" in capsys.readouterr().err
    assert cli.main(["train", "--collect", "Bad Run"]) == 2
    assert cli.main(["train"]) == 2 and "--skills and --run" in capsys.readouterr().err
    run_files.write_status(env / "data", "exp1-mixed", "running", ["washer_done"],
                           started="2026-11-03T15:00:00Z", detached=True)

    def stale(command, **kw):
        Path(command[-1], "result.json").write_text(json.dumps(
            {"run": "exp1-mixed", "ok": True, "seconds": 5.0, "finished": "2026-11-03T14:00:00Z"}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", stale)
    assert cli.main(["train", "--collect", "exp1-mixed"]) == 1
    assert "from an earlier run of exp1-mixed" in capsys.readouterr().err
    assert not (env / "data/train/exp1-mixed.result.json").exists()
    assert status_of(env)["state"] == "running"


def test_train_preflight_asks_runs_the_cpu_check_and_prints_the_verdict(env, monkeypatch,
                                                                       capsys):
    from backend.dataengine.dataset import SYSTEM_PROMPT
    from backend.training import gpu_ledger, preflight

    taught(env)
    render = {"answer": "yes", "tokens": 300, "text": f"<|start|>system<|message|>"
              f"{SYSTEM_PROMPT}<|eot|>...", "target_text": "yes<|eot|>",
              "target_ids": [5, 200008], "target_tokens": ["yes", "<|eot|>"]}
    rendered = {"processor": "MuseProcessor", "generation_eos": [200001, 200008],
                "renders": [{"id": "a", "layout": layout, **render}
                            for layout in preflight.LAYOUTS],
                "ids": {"yes": [5], "no": [6], "special": dict(preflight.SPECIAL),
                        "eos_token_id": 200001}}
    report = preflight.summarize(rendered, sequence_len=4096, system_prompt=SYSTEM_PROMPT,
                                 bundle_system_in_user=False)
    runs = fake_modal(monkeypatch, report)
    prompts = answer(monkeypatch, "n")
    assert cli.main([*TRAIN, "--preflight"]) == 1 and runs == [] and len(prompts) == 1
    assert cli.main([*TRAIN, "--preflight", "--estimate-only"]) == 0 and runs == []
    answer(monkeypatch, "y")
    assert cli.main([*TRAIN, "--preflight"]) == 0
    [command] = runs
    assert command[:3] == ["modal", "run", "backend/training/modal_train.py"]
    assert "--preflight" in command and "--smoke" not in command
    out = capsys.readouterr().out
    assert "verdict: TE_SYSTEM_IN_USER=0 (the default) works" in out
    assert "preflight: passed" in out and "exp1-mixed.preflight.json" in out
    assert not (env / "data/train/exp1-mixed.status.json").exists()  # nothing trained
    assert gpu_ledger.read(env / "data") == []  # a CPU job: not in the GPU ledger


def test_train_and_export_ask_when_the_modal_credit_would_run_out(env, monkeypatch, capsys):
    from backend.training import gpu_ledger

    taught(env)
    (env / "data/train").mkdir(parents=True)
    (env / "data/train/smoke-timing.json").write_text(json.dumps(SMOKE_RESULT))
    gpu_ledger.record(env / "data", "train", "earlier", 149.5 / 3.95 * 3600)
    runs = fake_modal(monkeypatch, {"seconds": 990.0, "ok": True})
    prompts = answer(monkeypatch, "n")
    assert cli.main([*TRAIN, "--yes"]) == 1 and runs == [] and len(prompts) == 1
    assert "plus this $1.01 would pass the $150 of Modal credit" in capsys.readouterr().out
    prompts = answer(monkeypatch, "n")
    (env / "data/train/exp1-mixed/manifest.json").write_text("{}")
    assert cli.main(["export", "--run", "exp1-mixed", "--yes"]) == 1 and runs == []
    assert "plus this $11.85 would pass" in capsys.readouterr().out and len(prompts) == 1


def test_export_records_the_timed_remote_call_also_when_it_fails(env, monkeypatch, capsys):
    from backend.training import gpu_ledger

    (env / "data/train/exp1-mixed").mkdir(parents=True)
    (env / "data/train/exp1-mixed/manifest.json").write_text("{}")
    fake_modal(monkeypatch, {"merged_dir": "/vol/runs/exp1-mixed/merged", "examples": 50,
                             "agreement": 1.0, "all_agree": True, "bin_files": 0,
                             "tensor_names_match": True, "disagreements": [], "seconds": 1800.0})
    assert cli.main(["export", "--run", "exp1-mixed", "--yes"]) == 0
    [row] = gpu_ledger.read(env / "data")
    assert (row["kind"], row["name"], row["seconds"], row["usd"]) == (
        "export", "exp1-mixed", 1800.0, 1.975)
    fake_modal(monkeypatch, {"run": "exp1-mixed", "ok": False, "seconds": 60.0}, code=1)
    assert cli.main(["export", "--run", "exp1-mixed", "--yes"]) == 1
    out, err = capsys.readouterr()
    assert "the last export took 30 min, about $1.98" in out and "the export failed" in err
    assert [r["seconds"] for r in gpu_ledger.read(env / "data")] == [1800.0, 60.0]


def test_spend_prints_the_modal_total(env, capsys):
    from backend.training import gpu_ledger

    gpu_ledger.record(env / "data", "train", "exp1-mixed", 3600)
    assert cli.main(["spend"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Meta API spend")
    assert ("Modal GPU spend (data/modal_ledger.jsonl, at config prices): $3.95 of $150 credit"
            in out)


def test_serve_student_base_and_adapters_get_their_own_app_and_the_time_is_recorded(
        env, monkeypatch, capsys):
    from backend.training import gpu_ledger

    calls = []

    def run(command, cwd=None, check=False, env=None, **kw):
        calls.append((command, {k: (env or {}).get(k) for k in cli.SERVE_ENVS}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/" + name)
    assert cli.main(["serve-student", "--base", "--yes"]) == 0
    assert calls[-1] == (["modal", "deploy", "backend/inference/modal_serve.py"],
                         {"TE_SERVE_RUN": "", "TE_SERVE_BASE": "1", "TE_SERVE_LORAS": ""})
    out = capsys.readouterr().out
    assert "deploys trained-eye-base serving the base model meta-models/Muse-Glimmer-30B at " \
           "revision a4e59da52a7b as muse-glimmer" in out
    assert "TE_STUDENT_MODEL=muse-glimmer" in out and "--base-model muse-glimmer" in out
    assert "te serve-student --stop --base stops it now" in out
    assert not (env / "data/train/serving.json").exists()  # the base serves no taught skill
    monkeypatch.setenv("TE_SERVE_BASE", "1")  # a stray shell setting never reaches a deploy
    assert cli.main(["serve-student", "--run", "exp1-mixed", "--yes"]) == 0
    assert calls[-1][1] == {"TE_SERVE_RUN": "exp1-mixed", "TE_SERVE_BASE": "",
                            "TE_SERVE_LORAS": ""}
    (env / "data/train/exp1-mixed.result.json").write_text('{"ok": true, "smoke": false}')
    assert cli.main(["serve-student", "--runs", "exp1-real,exp1-mixed", "--yes"]) == 0
    assert calls[-1][1]["TE_SERVE_LORAS"] == "exp1-real,exp1-mixed"
    out = capsys.readouterr().out
    assert "LoRA serving is not documented for Muse Glimmer" in out
    assert "no finished full run of exp1-real" in out and "of exp1-mixed" not in out
    assert "TE_STUDENT_MODEL=exp1-real with te watch --run exp1-real" in out
    for bad in (["--runs", "exp1,exp1"], ["--runs", "Bad Run"], ["--runs", ","],
                ["--run", "exp1-mixed", "--base"]):
        assert cli.main(["serve-student", *bad, "--yes"]) == 2, bad
    assert cli.main(["serve-student", "--stop", "--base"]) == 0
    assert calls[-1][0] == ["modal", "app", "stop", "trained-eye-base"]
    assert "trained-eye-base was deployed for" in capsys.readouterr().out
    rows = gpu_ledger.read(env / "data")
    assert [(r["kind"], r["name"]) for r in rows] == [
        ("deploy", "trained-eye-base"), ("deploy", "trained-eye-student"),
        ("deploy", "trained-eye-base"), ("serve", "trained-eye-base")]
    assert rows[-1]["upper_bound"] and set(gpu_ledger.open_deploys(env / "data")) == {
        "trained-eye-student"}


def test_eval_wandb_entity_needs_wandb(env, capsys):
    plan = env / "plan.json"
    plan.write_text('{"evaluations": []}')
    args = ["eval", "--plan", str(plan), "--out", str(env / "out"), "--wandb-entity", "team"]
    assert cli.main(args) == 2 and "--wandb-entity needs --wandb" in capsys.readouterr().err
