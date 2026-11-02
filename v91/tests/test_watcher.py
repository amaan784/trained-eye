import json

import pytest

from backend.dataengine.labels import Label
from backend.inference.student import Score, StudentError
from backend.skills.spec import SkillSpec, Target
from backend.watch.trigger import Trigger
from backend.watch.watcher import (
    DEFAULT_THRESHOLD,
    FrameLog,
    WatchedSkill,
    Watcher,
    alert_message,
    load_skill,
)


def spec(name="washer_done", confirm=True, cooldown=600):
    return SkillSpec(name=name, request="tell me when the washer says End",
                     target=Target(sam_prompt="washer display", description="the display"),
                     positive="the display shows End", negatives=["display off"],
                     alert="Laundry's done.", cooldown_s=cooldown, confirm_with_cloud=confirm)


class Scores:
    """A fake student: p per frame from a list, per skill."""

    def __init__(self, **probs):
        self.probs = {k: list(v) for k, v in probs.items()}

    def score(self, image, spec):
        p = self.probs[spec.name].pop(0)
        if p is None:
            raise StudentError("endpoint down")
        return Score(p, "yes" if p >= 0.5 else "no", 0.2)


def watched(s, threshold=0.5):
    return WatchedSkill(s, Trigger(threshold, s.cooldown_s), calibrated=True)


def run(watcher, n):
    return [watcher.on_frame(b"jpeg", t=float(i)) for i in range(n)]


def test_an_alert_after_three_of_four_frames_confirmed_by_spark(tmp_path):
    asked = []

    def spark(image, s):
        asked.append(s.name)
        return Label("yes", "End shows", 0.003, False)

    log = FrameLog(tmp_path / "watch/s1.jsonl")
    watcher = Watcher([watched(spec())], Scores(washer_done=[0.1, 0.9, 0.8, 0.9, 0.95]),
                      confirm=spark, log=log)
    results = run(watcher, 5)
    alerts = [a for _, fired in results for a in fired]
    assert [(a.skill, a.text, a.t, a.confirmed) for a in alerts] == [
        ("washer_done", "Laundry's done.", 3.0, True)]
    assert asked == ["washer_done"]  # Spark only when the trigger fires
    rows = [json.loads(x) for x in log.path.read_text().splitlines()]
    assert [r["p"] for r in rows] == [0.1, 0.9, 0.8, 0.9, 0.95]
    assert [r.get("alert") for r in rows] == [None, None, None, True, None]
    assert rows[3]["cloud"] == "yes" and "image" not in json.dumps(rows)
    assert alert_message(alerts[0]) == {"type": "alert", "skill": "washer_done",
                                        "text": "Laundry's done.", "t": 3.0, "p": 0.9,
                                        "confirmed": True}


def test_a_cloud_no_snoozes_for_a_minute_instead_of_alerting():
    calls = []

    def spark(image, s):
        calls.append(1)
        return Label("no" if len(calls) == 1 else "yes", "", 0.003, False)

    watcher = Watcher([watched(spec())], Scores(washer_done=[0.9] * 80), confirm=spark)
    alerts = [(a.t, a.confirmed) for _, fired in run(watcher, 80) for a in fired]
    assert alerts == [(62.0, True)]  # first fire at t=2 rejected, next after 60 s


def test_a_failed_cloud_check_still_alerts_unconfirmed():
    def spark(image, s):
        raise ConnectionError("no network")

    watcher = Watcher([watched(spec())], Scores(washer_done=[0.9] * 3), confirm=spark)
    [(_, [alert])] = run(watcher, 3)[2:]
    assert alert.confirmed is None
    error = Watcher([watched(spec())], Scores(washer_done=[0.9] * 3),
                    confirm=lambda i, s: Label("unsure", "no label: 400", 0.0, False, error=True))
    assert run(error, 3)[2][1][0].confirmed is None


def test_skills_without_the_cloud_check_never_call_spark():
    watcher = Watcher([watched(spec(confirm=False))], Scores(washer_done=[0.9] * 3),
                      confirm=lambda i, s: pytest.fail("Spark must not be asked"))
    assert run(watcher, 3)[2][1][0].confirmed is None


def test_student_errors_are_logged_and_do_not_count(tmp_path):
    log = FrameLog(tmp_path / "w.jsonl")
    watcher = Watcher([watched(spec(confirm=False))],
                      Scores(washer_done=[0.9, None, 0.9, None, 0.9]), log=log)
    results = run(watcher, 5)
    assert [scores["washer_done"] for scores, _ in results] == [0.9, None, 0.9, None, 0.9]
    assert [a.t for _, fired in results for a in fired] == [4.0]
    rows = [json.loads(x) for x in log.path.read_text().splitlines()]
    assert rows[1]["error"] == "endpoint down" and rows[1]["p"] is None


def test_frames_that_were_not_timed_log_no_latency(tmp_path):
    from types import SimpleNamespace

    calls = []

    class PartlyTimed:  # e.g. Spark as the scorer, whose cached frames are not timed
        def score(self, image, spec):
            calls.append(1)
            return SimpleNamespace(p=0.1, answer="no", latency_s=0.25,
                                   latency_measured=len(calls) != 2)

    log = FrameLog(tmp_path / "w.jsonl")
    run(Watcher([watched(spec(confirm=False))], PartlyTimed(), log=log), 3)
    rows = [json.loads(x) for x in log.path.read_text().splitlines()]
    assert [r["latency_s"] for r in rows] == [0.25, None, 0.25]


def test_each_skill_has_its_own_trigger():
    watcher = Watcher([watched(spec("washer_done", confirm=False)),
                       watched(spec("pot_boils", confirm=False), threshold=0.95)],
                      Scores(washer_done=[0.9] * 3, pot_boils=[0.9] * 3))
    assert [a.skill for a in run(watcher, 3)[2][1]] == ["washer_done"]


def test_load_skill_reads_the_spec_and_the_calibrated_threshold(tmp_path):
    (tmp_path / "spec.json").write_text(spec().model_dump_json())
    skill = load_skill(tmp_path)
    assert skill.trigger.threshold == DEFAULT_THRESHOLD and not skill.calibrated
    assert skill.trigger.cooldown_s == 600
    (tmp_path / "threshold.json").write_text('{"threshold": 0.83, "false_alarms": 0}')
    assert load_skill(tmp_path).trigger.threshold == 0.83
    refused = SkillSpec(name="sam_arrives", request="tell me when Sam gets here", refused=True,
                        refusal_reason="I can't watch for or recognize specific people.")
    (tmp_path / "spec.json").write_text(refused.model_dump_json())
    with pytest.raises(ValueError, match="refused"):
        load_skill(tmp_path)
