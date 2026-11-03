"""te replay (backend/eval/replay.py, cli_experiments.cmd_replay): a run's pending clips of
eval/plan.json replayed into the plan's logs, its threshold calibrated on its own
validation clips first. Fakes only: no endpoint, nothing paid."""

import io
import json

import pytest
from PIL import Image

from backend import cli, config
from backend.eval import replay
from backend.eval.plan import plan_skeleton, write_plan
from backend.eval.report import Plan, run_eval
from backend.inference import student
from backend.inference.student import StudentError
from backend.watch.watcher import load_skill
from tests.test_cli import answer, taught, use_transport
from tests.test_teach import FakeMeta

TIMEOUT = StudentError("student endpoint failed: ReadTimeout: timed out")


class Clock:
    def __init__(self):
        self.now, self.slept = 0.0, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def scorer(*outcomes):
    """score() raising or returning each outcome in turn, then answering."""
    queue = list(outcomes)

    def score():
        if queue and isinstance(queue[0], Exception):
            raise queue.pop(0)
        return queue.pop(0) if queue else "answer"
    return score


def test_warm_up_retries_until_the_endpoint_answers_or_the_deadline():
    clock, said = Clock(), []
    retry = StudentError("student endpoint failed: HTTPStatusError: Server error '503 "
                         "Service Unavailable' for url 'https://x/v1/chat/completions'")
    busy = StudentError("student endpoint failed: HTTPStatusError: Client error '429 Too "
                        "Many Requests' for url 'https://x'")
    assert replay.warm_up(scorer(TIMEOUT, retry, busy), 900, clock=clock, sleep=clock.sleep,
                          say=said.append)
    assert clock.slept == [15.0] * 3 and said[-1] == "the endpoint answered after 45 s"
    clock, said = Clock(), []
    assert not replay.warm_up(scorer(*[TIMEOUT] * 10), 40, clock=clock, sleep=clock.sleep,
                              say=said.append)
    assert clock.slept == [15.0, 15.0] and "did not answer in 30 s (3 tries)" in said[-1]
    refused = StudentError("student endpoint failed: HTTPStatusError: Client error '401 "
                           "Unauthorized' for url 'https://x'")
    clock, said = Clock(), []
    assert not replay.warm_up(scorer(refused), 900, clock=clock, sleep=clock.sleep,
                              say=said.append)
    assert clock.slept == [] and "refused the request" in said[0]
    unreadable = StudentError("the reply has no logprobs (does the server support ...)")
    assert replay.warm_up(scorer(unreadable), 900, clock=clock, sleep=clock.sleep,
                          say=said.append)  # the model ran: it is up
    assert replay.answered(unreadable) is True and replay.answered(TIMEOUT) is None


def frame_folder(folder, n):
    """n frames, frame i coloured (8 i, 0, 0), so a fake student can tell which it is."""
    folder.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (64, 48), (8 * i, 0, 0)).save(folder / f"f{i:03d}.jpg", quality=95)
    return folder


def index_of(image: bytes) -> int:
    with Image.open(io.BytesIO(image)) as img:
        return round(img.convert("RGB").getpixel((32, 24))[0] / 8)


def fake_student(monkeypatch, true=lambda i: False, fail=lambda i: False):
    """OpenAIStudent's stand-in: p 0.9 on frames where `true(index)` holds, else 0.2."""
    seen = []

    class Fake:
        def __init__(self, url, model, api_key=None, **options):
            seen.append(("built", model))

        def score(self, image, spec):
            i = index_of(image)
            seen.append(i)
            if fail(i):
                raise TIMEOUT
            return student.Score(0.9 if true(i) else 0.2, "yes", 0.1)

    monkeypatch.setattr(student, "OpenAIStudent", Fake)
    return seen


def test_replay_clip_writes_the_log_only_when_frames_scored(tmp_path, monkeypatch):
    from backend.dataengine.keyframes import load_frames
    from tests.test_realclips import SPEC, skill_folder

    skill = load_skill(skill_folder(tmp_path / "skills", SPEC))
    frames = load_frames(frame_folder(tmp_path / "clip", 4), fps=1.0)
    job = replay.Job("washer_done", "test", tmp_path / "w/a.jsonl", tmp_path / "a.json",
                     tmp_path / "clip", "clip")
    fake_student(monkeypatch, fail=lambda i: i == 2)
    scoring = student.OpenAIStudent("http://x/v1", "m")
    assert replay.replay_clip(job, skill, scoring, frames, run="exp1-mixed") == (3, 1)
    rows = [json.loads(line) for line in job.log.read_text().splitlines()]
    assert [r["p"] for r in rows] == [0.2, 0.2, None, 0.2] and {r["run"] for r in rows} == {
        "exp1-mixed"}
    assert not job.log.with_name("a.jsonl.partial").exists()
    job.log.unlink()
    fake_student(monkeypatch, fail=lambda i: True)
    assert replay.replay_clip(job, skill, student.OpenAIStudent("http://x/v1", "m"), frames,
                              run="exp1-mixed") == (0, 4)
    assert not job.log.exists() and not list(job.log.parent.iterdir())  # still pending


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODEL_API_KEY", "sk-test-cli-key")
    return tmp_path


def recorded(env, labels=True):
    """washer_done taught, a 30 s validation clip (true 10 to 20 s) and a 30 s test clip
    (true 15 to 25 s), and te experiments' plan for them."""
    taught(env)
    clips = env / "data/clips/washer_done"
    frame_folder(clips / "session2_a", 30)
    frame_folder(clips / "session3_test_a", 30)
    (clips / "session3_test_a.labels.json").write_text(json.dumps({"true": [[15, 25]]}))
    if labels:
        (clips / "session2_a.labels.json").write_text(json.dumps({"true": [[10, 20]]}))
    settings = config.load_settings(env={"TE_DATA_DIR": str(env / "data")}, env_file=None)
    return write_plan(env / "eval/plan.json", plan_skeleton(["washer_done"], settings),
                      force=True)


def test_jobs_and_scorers_come_from_the_plan(env):
    plan_path = recorded(env)
    plan = Plan.model_validate_json(plan_path.read_text())
    jobs = replay.plan_jobs(plan, plan_path.parent, "exp1-mixed")
    assert [(j.kind, j.log.name) for j in jobs] == [("validation", "session2_a.jsonl"),
                                                    ("test", "session3_test_a.jsonl")]
    assert jobs[1].evaluations == ["exp1-mixed-washer_done", "exp2-filtered-washer_done",
                                   "exp4-finetuned-washer_done"]  # one replay, three reads
    assert jobs[1].clip.resolve() == (env / "data/clips/washer_done/session3_test_a").resolve()
    assert replay.plan_scorer(plan, "exp4-spark") == "spark"
    assert replay.plan_scorer(plan, "exp4-base") == "student"
    with pytest.raises(ValueError, match="with the spark scorer, not student"):
        replay.plan_scorer(plan, "exp4-spark", "student")
    with pytest.raises(ValueError, match="read the replays of exp1-mixed .*--run exp1-mixed"):
        replay.plan_scorer(plan, "exp2-filtered")
    with pytest.raises(ValueError, match="no evaluation for run final; its runs: exp1-mixed"):
        replay.plan_scorer(plan, "final")


def test_te_replay_calibrates_the_run_then_replays_its_test_clips(env, monkeypatch, capsys):
    plan = recorded(env, labels=False)
    replay_cmd = ["replay", "--plan", str(plan), "--run", "exp1-mixed"]
    assert cli.main(replay_cmd) == 2  # no endpoint
    assert "TE_STUDENT_URL" in capsys.readouterr().err
    monkeypatch.setenv("TE_STUDENT_URL", "http://127.0.0.1:8080/v1")
    seen = fake_student(monkeypatch, true=lambda i: 10 <= i <= 25)
    assert cli.main(replay_cmd) == 0  # the validation labels are missing: tests held back
    out = capsys.readouterr().out
    watch = env / "data/watch/exp1-mixed/washer_done"
    assert (watch / "session2_a.jsonl").is_file() and not (watch / "threshold.json").exists()
    assert not (watch / "session3_test_a.jsonl").exists()
    assert "held back: exp1-mixed has no threshold for washer_done yet" in out
    assert "no labels yet: te label --clip" in out
    assert seen[:3] == [("built", "trained-eye"), 0, 0]  # the warm-up frame, then the clip

    labels = env / "data/clips/washer_done/session2_a.labels.json"
    labels.write_text(json.dumps({"true": [[10, 20]]}))
    seen.clear()
    assert cli.main(replay_cmd) == 0
    out = capsys.readouterr().out
    threshold = json.loads((watch / "threshold.json").read_text())
    assert threshold["run"] == "exp1-mixed" and threshold["threshold"] == 0.9
    assert "washer_done for run exp1-mixed: threshold 0.900" in out
    rows = [json.loads(line) for line in (watch / "session3_test_a.jsonl").read_text()
            .splitlines()]
    assert {(r["run"], r["threshold"], r["threshold_source"]) for r in rows} == {
        ("exp1-mixed", 0.9, "run")}
    assert "1 clip replayed now; 2 of 2 are replayed and labelled" in out
    assert "still pending" not in out

    seen.clear()
    assert cli.main(replay_cmd) == 0  # nothing left: no endpoint call at all
    assert seen == [] and "0 clips replayed now; 2 of 2" in capsys.readouterr().out
    results = run_eval(plan, env / "eval")
    by_name = {e["name"]: e for e in results["evaluations"]}
    for name in ("exp1-mixed", "exp2-filtered", "exp4-finetuned"):
        e = by_name[f"{name}-washer_done"]
        assert e["status"] == "scored" and e["threshold"] == {"values": [0.9],
                                                              "sources": ["run"]}
    assert by_name["exp1-real-washer_done"]["status"] == "pending"

    (watch / "threshold.json").write_text(json.dumps({**threshold, "threshold": 0.5}))
    assert cli.main(replay_cmd) == 0  # calibrated again by hand: the test log is stale
    assert ("scored at 0.900, but exp1-mixed's threshold for washer_done is 0.500 now: "
            "delete this log") in capsys.readouterr().out


def test_te_replay_refuses_what_the_plan_does_not_say(env, monkeypatch, capsys):
    plan = recorded(env)
    monkeypatch.setenv("TE_STUDENT_URL", "http://127.0.0.1:8080/v1")
    for run, scorer_name, message in (
            ("exp2-filtered", None, "read the replays of exp1-mixed"),
            ("exp4-spark", "student", "with the spark scorer, not student"),
            ("Bad", None, "run name")):
        extra = ["--scorer", scorer_name] if scorer_name else []
        assert cli.main(["replay", "--plan", str(plan), "--run", run, *extra]) == 2
        assert message in capsys.readouterr().err
    assert cli.main(["replay", "--plan", str(env / "missing.json"), "--run", "exp1-mixed"]) == 2
    with pytest.raises(SystemExit):
        cli.main(["replay", "--plan", str(plan), "--run", "exp1-mixed", "--fps", "0"])
    fake_student(monkeypatch, fail=lambda i: True)  # the endpoint never answers
    code = cli.main(["replay", "--plan", str(plan), "--run", "exp1-real", "--warm-up-s", "0"])
    out = capsys.readouterr().out
    assert code == 1 and "replay failed: every frame was a student error" in out
    assert not list((env / "data/watch").rglob("*.jsonl"))


def test_te_replay_warns_when_another_run_is_served(env, monkeypatch, capsys):
    from backend.cli_experiments import serving_note

    (env / "data/train").mkdir(parents=True)
    (env / "data/train/serving.json").write_text(json.dumps({"run": "exp1-real",
                                                             "skills": []}))
    data = env / "data"
    assert "last deployed run exp1-real, not exp1-mixed" in serving_note(data, "exp1-mixed",
                                                                          "trained-eye")
    assert serving_note(data, "exp1-real", "trained-eye") is None
    assert serving_note(data, "exp1-mixed", "exp1-mixed") is None  # te serve-student --runs
    assert serving_note(data, "exp4-base", "muse-glimmer") is None


def test_te_replay_with_spark_prices_and_asks_first(env, monkeypatch, capsys):
    plan = recorded(env)
    fake = FakeMeta(answer=lambda label, real: "yes")
    use_transport(monkeypatch, fake)
    prompts = answer(monkeypatch, "n")
    replay_cmd = ["replay", "--plan", str(plan), "--run", "exp4-spark"]
    assert cli.main(replay_cmd) == 1
    assert prompts and fake.calls == {}  # nothing paid before the yes
    assert "Muse Spark on 30 frames of 1 clip:" in capsys.readouterr().out
    assert cli.main([*replay_cmd, "--yes", "--no-cloud"]) == 0
    log = env / "data/watch/exp4-spark/washer_done/session3_test_a.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert {(r["p"], r["threshold"], r["threshold_source"]) for r in rows} == {
        (1.0, 1.0, "spark")}
