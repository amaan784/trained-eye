"""CLI tests for te watch and te calibrate (backend/cli_watch.py): the trained model or
Muse Spark as the scorer, and the skill-wide and per-run thresholds. Fakes only: nothing is
sent or billed."""

import json
from pathlib import Path

import pytest

from backend import cli
from tests.test_cli import answer, taught, use_transport, write_clip
from tests.test_teach import FakeMeta


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODEL_API_KEY", "sk-test-cli-key")
    return tmp_path


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


def test_a_skill_wide_calibrate_says_when_te_serve_uses_the_served_run_s(env, capsys):
    from backend.training import runs

    taught(env)
    log, labels = env / "replay.jsonl", env / "labels.json"
    probs = [0.6 if i % 1800 < 3 else 0.2 for i in range(3600)] + [0.9] * 30
    log.write_text("".join(json.dumps({"t": float(i), "skill": "washer_done", "p": p}) + "\n"
                           for i, p in enumerate(probs)))
    labels.write_text(json.dumps({"true": [[3600, 3630]]}))
    calibrate = ["calibrate", "--skill", "washer_done", "--log", str(log), "--labels",
                 str(labels)]
    assert cli.main(calibrate) == 0
    assert "te serve scores" not in capsys.readouterr().out  # nothing served: no note
    runs.write_status(env / "data", "exp1-mixed", "done", ["washer_done"],
                      started="2026-11-03T10:00:00Z", finished="2026-11-03T11:00:00Z")
    runs.write_serving(env / "data", "exp1-mixed")
    run_file = env / "data/watch/exp1-mixed/washer_done/threshold.json"
    run_file.parent.mkdir(parents=True)
    run_file.write_text('{"threshold": 0.8}')
    assert cli.main(calibrate) == 0
    assert "run exp1-mixed's own threshold" in capsys.readouterr().out
    assert cli.main([*calibrate, "--run", "exp1-mixed"]) == 0
    assert "te serve scores" not in capsys.readouterr().out


def test_watch_with_spark_as_the_scorer_prices_and_asks_first(env, monkeypatch, capsys):
    taught(env)
    fake = FakeMeta(answer=lambda label, real: "yes")
    use_transport(monkeypatch, fake)
    clip = write_clip(env / "clip", n=4)
    prompts = answer(monkeypatch, "n")
    assert cli.main(["watch", "--skills", "washer_done", "--clip", str(clip),
                     "--scorer", "spark"]) == 1
    assert prompts and fake.calls == {}  # nothing paid before the yes
    log = env / "replay.jsonl"
    assert cli.main(["watch", "--skills", "washer_done", "--clip", str(clip), "--scorer",
                     "spark", "--yes", "--no-cloud", "--log", str(log)]) == 0
    out = capsys.readouterr().out
    assert "Muse Spark on 4 frames x 1 skills" in out and '"frames": 4' in out
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["p"] for r in rows] == [1.0] * 4
