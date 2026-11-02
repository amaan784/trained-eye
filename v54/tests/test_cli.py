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
        def __init__(self, url, model):
            seen.append((url, model))

        def score(self, image, spec):
            return student.Score(0.9, "yes", 0.1)

    monkeypatch.setattr(student, "OpenAIStudent", Fake)
    clip = write_clip(env / "clip", n=5)
    assert cli.main(["watch", "--skills", "washer_done", "--clip", str(clip)]) == 0
    out = capsys.readouterr().out
    assert seen == [("http://127.0.0.1:8080/v1", "trained-eye")]
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
