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


def spec(name="burner_left_on", confirm=True, cooldown=600):
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
    watcher = Watcher([watched(spec())], Scores(burner_left_on=[0.1, 0.9, 0.8, 0.9, 0.95]),
                      confirm=spark, log=log)
    results = run(watcher, 5)
    alerts = [a for _, fired in results for a in fired]
    assert [(a.skill, a.text, a.t, a.confirmed) for a in alerts] == [
        ("burner_left_on", "Laundry's done.", 3.0, True)]
    assert asked == ["burner_left_on"]  # Spark only when the trigger fires
    rows = [json.loads(x) for x in log.path.read_text().splitlines()]
    assert [r["p"] for r in rows] == [0.1, 0.9, 0.8, 0.9, 0.95]
    assert [r.get("alert") for r in rows] == [None, None, None, True, None]
    assert rows[3]["cloud"] == "yes" and "image" not in json.dumps(rows)
    assert alert_message(alerts[0]) == {"type": "alert", "skill": "burner_left_on",
                                        "text": "Laundry's done.", "t": 3.0, "p": 0.9,
                                        "confirmed": True}


def test_a_cloud_no_snoozes_for_a_minute_instead_of_alerting():
    calls = []

    def spark(image, s):
        calls.append(1)
        return Label("no" if len(calls) == 1 else "yes", "", 0.003, False)

    watcher = Watcher([watched(spec())], Scores(burner_left_on=[0.9] * 80), confirm=spark)
    alerts = [(a.t, a.confirmed) for _, fired in run(watcher, 80) for a in fired]
    assert alerts == [(62.0, True)]  # first fire at t=2 rejected, next after 60 s


def test_a_failed_cloud_check_still_alerts_unconfirmed():
    def spark(image, s):
        raise ConnectionError("no network")

    watcher = Watcher([watched(spec())], Scores(burner_left_on=[0.9] * 3), confirm=spark)
    [(_, [alert])] = run(watcher, 3)[2:]
    assert alert.confirmed is None
    error = Watcher([watched(spec())], Scores(burner_left_on=[0.9] * 3),
                    confirm=lambda i, s: Label("unsure", "no label: 400", 0.0, False, error=True))
    assert run(error, 3)[2][1][0].confirmed is None


def test_skills_without_the_cloud_check_never_call_spark():
    watcher = Watcher([watched(spec(confirm=False))], Scores(burner_left_on=[0.9] * 3),
                      confirm=lambda i, s: pytest.fail("Spark must not be asked"))
    assert run(watcher, 3)[2][1][0].confirmed is None


def test_student_errors_are_logged_and_do_not_count(tmp_path):
    log = FrameLog(tmp_path / "w.jsonl")
    watcher = Watcher([watched(spec(confirm=False))],
                      Scores(burner_left_on=[0.9, None, 0.9, None, 0.9]), log=log)
    results = run(watcher, 5)
    assert [scores["burner_left_on"] for scores, _ in results] == [0.9, None, 0.9, None, 0.9]
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
    watcher = Watcher([watched(spec("burner_left_on", confirm=False)),
                       watched(spec("pot_boils", confirm=False), threshold=0.95)],
                      Scores(burner_left_on=[0.9] * 3, pot_boils=[0.9] * 3))
    assert [a.skill for a in run(watcher, 3)[2][1]] == ["burner_left_on"]


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


def test_thresholds_come_from_the_flag_then_the_run_then_the_skill(tmp_path):
    from backend.watch.watcher import run_threshold_file

    skill_dir = tmp_path / "skills/burner_left_on"
    skill_dir.mkdir(parents=True)
    (skill_dir / "spec.json").write_text(spec().model_dump_json())
    path = run_threshold_file(tmp_path, "exp1-mixed", "burner_left_on")
    assert path == tmp_path / "watch/exp1-mixed/burner_left_on/threshold.json"

    def picked(**kw):
        skill = load_skill(skill_dir, **kw)
        return skill.trigger.threshold, skill.threshold_source, skill.calibrated

    assert picked(run="exp1-mixed") == (DEFAULT_THRESHOLD, "default", False)
    (skill_dir / "threshold.json").write_text('{"threshold": 0.7}')
    assert picked(run="exp1-mixed") == (0.7, "skill", True)
    path.parent.mkdir(parents=True)
    path.write_text('{"threshold": 0.83, "run": "exp1-mixed"}')
    assert picked(run="exp1-mixed") == (0.83, "run", True)
    assert picked(run="exp1-real") == (0.7, "skill", True)  # another run: not this file
    assert picked() == (0.7, "skill", True)
    assert picked(run="exp1-mixed", threshold=0.4) == (0.4, "flag", False)
    with pytest.raises(ValueError, match="run name"):
        load_skill(skill_dir, run="../x")
    for broken in ('{"threshold": NaN}', '{"p": 1}', "not json"):
        path.write_text(broken)
        with pytest.raises(ValueError, match="te calibrate again"):
            load_skill(skill_dir, run="exp1-mixed")


def test_the_frame_log_says_which_threshold_and_run_it_used(tmp_path):
    log = FrameLog(tmp_path / "w.jsonl", {"run": "exp1-mixed"})
    skill = WatchedSkill(spec(confirm=False), Trigger(0.6, 600.0), True, "run")
    run(Watcher([skill], Scores(burner_left_on=[None, 0.9]), log=log), 2)
    rows = [json.loads(x) for x in log.path.read_text().splitlines()]
    assert [(r["run"], r["threshold"], r["threshold_source"]) for r in rows] == [
        ("exp1-mixed", 0.6, "run")] * 2


def write_manifest(skill_dir, **fields):
    folder = skill_dir / "dataset"
    folder.mkdir(parents=True)
    (folder / f"{skill_dir.name}-teach.manifest.json").write_text(json.dumps(fields))
    return skill_dir


def test_the_prompt_layout_follows_the_run_then_the_datasets(tmp_path):
    from backend.watch.watcher import system_in_user

    folded = write_manifest(tmp_path / "pot_boils", system_in_user=True)
    plain = write_manifest(tmp_path / "burner_left_on", system_in_user=False)
    untaught = tmp_path / "hot_pan"
    assert system_in_user([folded, untaught]) is True
    assert system_in_user([plain]) is False
    assert system_in_user([untaught]) is False
    assert system_in_user([untaught], default=True) is True  # TE_SYSTEM_IN_USER
    assert system_in_user([write_manifest(tmp_path / "older")]) is False  # no field: the old way
    with pytest.raises(ValueError, match="folded into the user text for pot_boils only"):
        system_in_user([folded, plain])
    # what the run trained on wins over datasets written since
    assert system_in_user([folded, plain], bundle={"system_in_user": False}) is False


def test_two_watchers_sharing_a_cooldown_book_alert_once_per_cooldown():
    from backend.watch.watcher import CooldownBook

    book = CooldownBook()
    first = Watcher([watched(spec(confirm=False))], Scores(burner_left_on=[0.9] * 3),
                    cooldowns=book)
    assert [a.t for _, fired in run(first, 3) for a in fired] == [2.0]
    # a reconnect: a new session, a new watcher, the condition still true
    second = Watcher([watched(spec(confirm=False))], Scores(burner_left_on=[0.9] * 700),
                     cooldowns=book)
    fired = [a.t for i in range(3, 700)
             for a in second.on_frame(b"jpeg", t=float(i))[1]]
    assert fired == [602.0]  # only once the first alert's cooldown is over
    alone = Watcher([watched(spec(confirm=False))], Scores(burner_left_on=[0.9] * 3))
    assert [a.t for _, f in run(alone, 3) for a in f] == [2.0]  # no book: as before


def test_a_spark_no_snooze_is_shared_and_does_not_shorten_a_newer_cooldown():
    from backend.watch.watcher import CooldownBook

    book = CooldownBook()
    no = Watcher([watched(spec())], Scores(burner_left_on=[0.9] * 3), cooldowns=book,
                 confirm=lambda i, s: Label("no", "", 0.003, False))
    assert run(no, 3)[2][1] == [] and book.quiet_until("burner_left_on") == 62.0
    other = Watcher([watched(spec(confirm=False))], Scores(burner_left_on=[0.9] * 80),
                    cooldowns=book)
    fired = [a.t for i in range(3, 80) for a in other.on_frame(b"jpeg", t=float(i))[1]]
    assert fired == [62.0]  # quiet for the snooze, then it fires
    book.set("burner_left_on", 50.0, only_if=602.0)  # an old value: left alone
    assert book.quiet_until("burner_left_on") == 662.0


def test_the_cooldown_book_survives_a_restart_in_wall_clock_time(tmp_path):
    from backend.watch.watcher import CooldownBook

    path = tmp_path / "watch/cooldowns.json"
    now = {"mono": 100.0, "wall": 1_800_000_000.0}
    book = CooldownBook(path, clock=lambda: now["mono"], wall=lambda: now["wall"])
    book.set("burner_left_on", 700.0)  # 600 s from now
    book.set("pot_boils", 50.0)  # already over: not saved
    saved = json.loads(path.read_text())
    assert saved == {"quiet_until": {"burner_left_on": 1_800_000_600.0}}
    now.update(mono=5.0, wall=1_800_000_010.0)  # te serve restarted 10 s later
    again = CooldownBook(lambda: path, clock=lambda: now["mono"], wall=lambda: now["wall"])
    assert again.quiet_until("burner_left_on") == pytest.approx(595.0)
    assert again.quiet_until("pot_boils") == float("-inf")
    path.write_text("not json")
    assert CooldownBook(path).quiet_until("burner_left_on") == float("-inf")  # lost, no crash
