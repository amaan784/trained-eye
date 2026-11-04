"""The per-skill hold: watch mode alerts only once a condition has stayed true for the
skill's hold_s, and the evaluation counts an alert as on time from the end of the hold."""

import json

import httpx
import pytest
from pydantic import ValidationError

from backend.config import load_settings
from backend.dataengine.labels import Label
from backend.eval import metrics
from backend.eval.metrics import ON_TIME_S, count_clip, hold_of, on_time_window, summarize
from backend.eval.report import Plan, run_eval
from backend.eval.writeup import render
from backend.inference.student import Score
from backend.meta.client import MetaClient
from backend.skills.compiler import COMPILE_SCHEMA, SYSTEM, compile_skill, hold_for
from backend.skills.spec import MAX_HOLD_S, SKILL_HOLDS, SkillSpec, Target
from backend.watch.trigger import Frame, Trigger, alerts, calibrate, evaluate, labelled_frames
from backend.watch.watcher import FrameLog, Watcher, load_skill


def run(trigger, probs, start=0.0, step=1.0):
    return [start + i * step for i, p in enumerate(probs) if trigger.update(p, start + i * step)]


# The trigger


def test_the_project_skills_have_the_authors_holds_and_repeats():
    from backend.skills.spec import SKILL_REPEATS

    assert SKILL_HOLDS == {"fridge_left_open": 20, "tap_left_running": 20,
                           "burner_left_on": 10, "egg_whisked": 0}
    assert SKILL_REPEATS == {"fridge_left_open": 30, "tap_left_running": 30,
                             "burner_left_on": 20}
    # each repeat is at least the hold, so the trigger repeats on the cooldown
    assert all(SKILL_REPEATS[k] >= SKILL_HOLDS[k] for k in SKILL_REPEATS)


def test_a_condition_that_stays_true_repeats_its_alert_on_the_cooldown():
    # the fridge: first after its 20 s hold, then every 30 s while the door stays open
    assert run(Trigger(0.5, cooldown_s=30, hold_s=20), [0.9] * 90) == [20.0, 50.0, 80.0]
    # the burner: after 10 s, then every 20 s until it is off
    assert run(Trigger(0.5, cooldown_s=20, hold_s=10), [0.9] * 55) == [10.0, 30.0, 50.0]


def test_a_hold_waits_until_the_condition_has_lasted_it():
    # on from t = 2 (3 of 4), counted from the first frame over, t = 0
    assert run(Trigger(0.5, 600, hold_s=10), [0.9] * 20) == [10.0]
    assert run(Trigger(0.5, 600, hold_s=0), [0.9] * 20) == [2.0]  # as before the hold
    assert run(Trigger(0.5, 600), [0.9] * 20) == [2.0]  # the default is no hold


def test_a_condition_shorter_than_the_hold_never_fires():
    # the fridge opened for 25 s, twice, with its 30 s hold: nothing to say
    probs = ([0.9] * 25 + [0.1] * 10) * 2
    assert run(Trigger(0.5, 600, hold_s=30), probs) == []
    assert run(Trigger(0.5, 600, hold_s=0), probs) == [2.0]


def test_the_3_of_4_smoothing_still_decides_when_the_condition_is_on():
    # one frame under in four keeps it on, so the hold carries on through it
    assert run(Trigger(0.5, 600, hold_s=10), [0.9] * 5 + [0.1] + [0.9] * 10) == [10.0]
    # two under in four turn it off: the hold starts again from the next frame over
    probs = [0.9] * 8 + [0.1] * 2 + [0.9] * 15
    assert run(Trigger(0.5, 600, hold_s=10), probs) == [20.0]
    # single bright frames still never fire, however long the hold
    assert run(Trigger(0.5, 600, hold_s=5), [0.9, 0.1, 0.1] * 20) == []


def test_a_pause_in_the_stream_starts_the_hold_again():
    trigger = Trigger(0.5, 600, hold_s=10)
    assert run(trigger, [0.9] * 9) == []  # 8 s on, not yet 10
    assert run(trigger, [0.9] * 15, start=30) == [40.0]  # 21 s later: from t = 30 again


def test_the_cooldown_still_follows_an_alert_after_a_hold():
    # fires once the hold is over, then once per cooldown while the tap keeps running
    assert run(Trigger(0.5, cooldown_s=30, hold_s=10), [0.9] * 75) == [10.0, 40.0, 70.0]


def test_a_held_alert_after_a_spark_no_waits_for_the_hold_again(tmp_path):
    asked = []

    def spark(image, s):
        asked.append(1)
        return Label("no" if len(asked) == 1 else "yes", "", 0.003, False)

    folder = tmp_path / "skills" / "fridge_left_open"
    folder.mkdir(parents=True)
    (folder / "spec.json").write_text(spec("fridge_left_open", hold=90).model_dump_json())
    watcher = Watcher([load_skill(folder)], Always(), confirm=spark)
    fired = [a.t for i in range(200) for a in watcher.on_frame(b"jpeg", t=float(i))[1]]
    # Spark says no at 90 s: a minute's snooze, and the hold starts again from 91 s, so the
    # next check is at 181 s, not when the snooze ends at 150 s
    assert fired == [181.0] and len(asked) == 2


def test_a_run_exactly_as_long_as_the_hold_fires_only_on_a_frame_still_over():
    # over for 10 frames (0 to 9 s), then off: at 10 s three of the last four are still
    # over, but that frame is under, so nothing fires for a burner lit 10 s at most
    assert run(Trigger(0.5, 600, hold_s=10), [0.9] * 10 + [0.1] * 5) == []
    assert run(Trigger(0.5, 600, hold_s=10), [0.9] * 11 + [0.1] * 5) == [10.0]
    # and a cooldown that ends on an under frame fires on the next frame over
    trigger = Trigger(0.5, cooldown_s=5)
    assert run(trigger, [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9]) == [2.0, 8.0]


def test_a_burner_lit_just_under_its_hold_is_never_announced(tmp_path):
    folder = tmp_path / "skills" / "burner_left_on"
    folder.mkdir(parents=True)
    (folder / "spec.json").write_text(spec(hold=10).model_dump_json())

    class Lit:  # a perfect student: the burner is lit from 10.4 s to 20.3 s
        def score(self, image, s):
            return Score(0.98 if 10.4 <= self.t <= 20.3 else 0.02, "", 0.2)

    student = Lit()
    watcher = Watcher([load_skill(folder)], student)
    fired = []
    for i in range(40):
        student.t = float(i)
        fired += watcher.on_frame(b"jpeg", t=float(i))[1]
    assert fired == []


def test_a_negative_hold_is_refused():
    with pytest.raises(ValueError):
        Trigger(0.5, 60, hold_s=-1)


# The trigger's own evaluation and calibration


def clip(truths, probs):
    return [Frame(float(i), p, t) for i, (t, p) in enumerate(zip(truths, probs, strict=True))]


def held_clip():
    """Open from 10 to 39 s (30 frames), then briefly from 50 to 57 s (8 frames)."""
    truths = [False] * 10 + [True] * 30 + [False] * 10 + [True] * 8 + [False] * 12
    return clip(truths, [0.9 if t else 0.1 for t in truths])


def test_evaluate_counts_events_from_the_end_of_the_hold():
    held = evaluate([held_clip()], 0.5, 600, hold_s=20)
    # the long one is due at 30 s and caught then; the short one is under the hold
    assert (held["events"], held["caught"], held["false_alarms"]) == (1, 1, 0)
    assert held["median_delay_s"] == 0.0
    plain = evaluate([held_clip()], 0.5, 600)
    assert (plain["events"], plain["caught"], plain["median_delay_s"]) == (2, 1, 2.0)


def test_an_alert_before_the_hold_ended_is_a_false_alarm():
    truths = [False] * 10 + [True] * 30 + [False] * 10
    probs = [0.9 if 5 <= i < 40 else 0.1 for i in range(50)]  # the student is 5 s early
    result = evaluate([clip(truths, probs)], 0.5, 600, hold_s=20)
    assert [f.t for f in alerts(clip(truths, probs), 0.5, 600, 20)] == [25.0]
    assert (result["false_alarms"], result["caught"], result["events"]) == (1, 0, 1)


def test_calibrate_uses_the_hold_so_brief_bumps_need_no_higher_threshold():
    probs = [0.7 if 5 <= i < 10 or 18 <= i < 23 else 0.1 for i in range(30)]  # bumps
    probs += [0.9] * 30 + [0.1] * 10  # then the condition, for 30 s
    truths = [False] * 30 + [True] * 30 + [False] * 10
    frames = clip(truths, probs)
    assert calibrate([frames], cooldown_s=600)["threshold"] == 0.9
    held = calibrate([frames], cooldown_s=600, hold_s=10)
    assert held["threshold"] == 0.7 and held["caught"] == 1 and held["false_alarms"] == 0


# The skill spec and the compiler


def spec(name="burner_left_on", hold=None, cooldown=600):
    return SkillSpec(name=name, request="tell me if I leave the burner on",
                     target=Target(sam_prompt="stove burner", description="the front burner"),
                     positive="the burner is lit with no pan on it",
                     negatives=["the burner is off"], alert="The burner is still on.",
                     cooldown_s=cooldown, **({} if hold is None else {"hold_s": hold}))


def test_the_spec_keeps_a_hold_between_0_and_the_maximum():
    assert spec().hold_s == 0  # a hand-written spec says its own hold
    assert spec(hold=30).hold_s == 30
    for bad in (-1, MAX_HOLD_S + 1):
        with pytest.raises(ValidationError):
            spec(hold=bad)


def test_hold_for_gives_the_project_skills_their_hold_and_clamps_the_rest():
    assert [hold_for(n, 999) for n in SKILL_HOLDS] == list(SKILL_HOLDS.values())
    assert (hold_for("hot_pan", 45), hold_for("hot_pan", -5), hold_for("hot_pan", 10**9)) == (
        45, 0, MAX_HOLD_S)


REPLY = {
    "refused": False, "refusal_reason": None, "name": "fridge_open",
    "sam_prompt": "fridge door", "target_description": "the fridge door",
    "positive": "the fridge door is open", "negatives": ["the fridge door is closed"],
    "alert": "The fridge is open.", "cooldown_s": 300, "hold_s": 5,
    "confirm_with_cloud": True,
}


def compiled(reply, name, data):
    """Compile with a fake Muse Wonder giving `reply`; each call has its own cache."""
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps(reply)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5000, "completion_tokens": 600}})

    settings = load_settings(env={"MODEL_API_KEY": "sk-test",
                                  "TE_DATA_DIR": str(data / f"data-{name}")}, env_file=None)
    client = MetaClient(settings, transport=httpx.MockTransport(handler), sleep=lambda s: None)
    return compile_skill(client, "tell me when the fridge is left open", [b"\xff\xd8x"],
                         name=name), requests


def test_the_compiler_asks_for_a_hold_and_the_project_skills_keep_theirs(tmp_path,
                                                                         monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert "hold_s" in COMPILE_SCHEMA["required"] and "hold_s:" in SYSTEM
    fridge, requests = compiled(REPLY, "fridge_left_open", tmp_path)
    assert fridge.hold_s == 20  # the author's hold, not the model's 5
    assert fridge.cooldown_s == 30  # and the author's repeat
    assert requests[0]["response_format"]["json_schema"]["schema"] == COMPILE_SCHEMA
    other, _ = compiled({**REPLY, "hold_s": 12}, "freezer_door", tmp_path)
    assert other.hold_s == 12
    without, _ = compiled({k: v for k, v in REPLY.items() if k != "hold_s"}, "freezer_lid",
                          tmp_path)
    assert without.hold_s == 0


# Watch mode


class Always:
    def score(self, image, spec):
        return Score(0.9, "yes", 0.2)


def test_a_taught_skill_watches_with_its_hold_and_logs_it(tmp_path):
    folder = tmp_path / "skills" / "tap_left_running"
    folder.mkdir(parents=True)
    tap = spec("tap_left_running", hold=20).model_copy(update={"alert": "The tap is running."})
    (folder / "spec.json").write_text(tap.model_dump_json())
    skill = load_skill(folder)
    assert skill.trigger.hold_s == 20.0
    log = FrameLog(tmp_path / "watch" / "tap.jsonl")
    watcher = Watcher([skill], Always(), log=log)
    fired = [a.t for i in range(25) for a in watcher.on_frame(b"jpeg", t=float(i))[1]]
    assert fired == [20.0]
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert {r["hold_s"] for r in rows} == {20.0}
    assert hold_of(rows, "tap_left_running") == 20.0


# The evaluation


def row(t, p, skill="burner_left_on", hold=10, alert=False):
    r = {"t": float(t), "skill": skill, "threshold": 0.5, "p": p, "over": p >= 0.5,
         "latency_s": 0.2, "fires": alert}
    if hold is not None:
        r["hold_s"] = hold
    if alert:
        r.update(alert=True, cloud="yes")
    return r


def log_rows(alert_times, skill="burner_left_on", hold=10, n=80):
    return [row(t, 0.9 if 10 <= t <= 40 else 0.1, skill, hold, t in alert_times)
            for t in range(n)]


def test_an_alert_is_on_time_from_the_end_of_the_hold():
    # on from 10 s, hold 10: due at 20 s; the burner's window is 5 s
    c = count_clip(log_rows({23}), "burner_left_on", [[10, 40]])
    assert (c.events, c.caught, c.caught_in_window, c.delays) == (1, 1, 1, [3.0])
    late = count_clip(log_rows({27}), "burner_left_on", [[10, 40]])
    assert (late.caught, late.caught_in_window, late.delays) == (1, 0, [7.0])


def test_early_alerts_are_false_alarms_and_short_events_expect_none():
    c = count_clip(log_rows({15, 60}), "burner_left_on", [[10, 40], [50, 55]])
    # 15 s is inside the range but before its hold ended; 60 s is outside every range
    assert (c.alerts, c.false_alarms, c.early_alerts) == (2, 2, 1)
    assert (c.events, c.short_events, c.caught) == (1, 1, 0)
    m = summarize([c])
    assert (m["short_events"], m["early_alerts"], m["event_recall_any"]) == (1, 1, 0.0)


def test_each_skill_has_its_own_on_time_window():
    assert ON_TIME_S == {"burner_left_on": 5.0, "tap_left_running": 5.0,
                         "fridge_left_open": 10.0, "egg_whisked": 10.0}
    assert on_time_window("hot_pan") == metrics.EVENT_WINDOW_S == 5.0
    assert on_time_window("burner_left_on", {"burner_left_on": 8}) == 8.0
    # an alert 8 s after the hold ended: on time for the fridge, late for the burner
    for skill, on_time in (("fridge_left_open", 1), ("burner_left_on", 0)):
        c = count_clip(log_rows({28}, skill=skill), skill, [[10, 40]])
        assert c.caught_in_window == on_time


def test_the_hold_comes_from_the_log_and_one_evaluation_has_one():
    assert hold_of(log_rows(set(), hold=None), "burner_left_on") == 0.0  # an older log
    mixed = log_rows(set())[:40] + log_rows(set(), hold=30)[40:]
    with pytest.raises(ValueError, match="more than one hold_s"):
        count_clip(mixed, "burner_left_on", [[10, 40]])
    # without a logged hold the alert at 12 s is on time; given the hold it is early
    plain = count_clip(log_rows({12}, hold=None), "burner_left_on", [[10, 40]])
    assert (plain.caught_in_window, plain.false_alarms) == (1, 0)
    held = count_clip(log_rows({12}, hold=None), "burner_left_on", [[10, 40]], hold_s=10)
    assert (held.caught_in_window, held.false_alarms) == (0, 1)


def test_a_plan_can_change_a_skills_window_but_not_to_nothing():
    fridge = [{"name": "exp1-fridge", "experiment": 1, "skill": "fridge_left_open",
               "clips": [{"log": "a.jsonl", "labels": "a.json"}]}]
    plan = Plan.model_validate({"on_time_s": {"fridge_left_open": 12}, "evaluations": fridge})
    assert (plan.window("fridge_left_open"), plan.window("burner_left_on")) == (12.0, 5.0)
    for bad in ({"fridge_left_open": 0}, {"Fridge Door": 5}):
        with pytest.raises(ValidationError):
            Plan.model_validate({"on_time_s": bad, "evaluations": fridge})
    with pytest.raises(ValidationError, match="no evaluation scores: fridge_left_opne"):
        Plan.model_validate({"on_time_s": {"fridge_left_opne": 30}, "evaluations": fridge})
    with pytest.raises(ValidationError):  # the old single window is gone
        Plan.model_validate({"window_s": 5, "evaluations": []})


def test_results_and_the_write_up_give_each_skills_hold_and_window(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    entries = []
    for skill, hold, alert_at in (("burner_left_on", 10, 23), ("fridge_left_open", 30, 48)):
        rows = log_rows({alert_at}, skill=skill, hold=hold)
        (clips / f"{skill}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (clips / f"{skill}.json").write_text(json.dumps({"true": [[10, 40]]}))
        entries.append({"name": f"exp1-{skill.split('_')[0]}", "experiment": 1,
                        "skill": skill, "clips": [{"log": f"../clips/{skill}.jsonl",
                                                   "labels": f"../clips/{skill}.json"}]})
    plan = tmp_path / "eval" / "plan.json"
    plan.parent.mkdir()
    plan.write_text(json.dumps({"evaluations": entries}))
    results = run_eval(plan, tmp_path / "out")
    assert results["on_time"] == {"burner_left_on": {"window_s": 5.0, "hold_s": [10.0]},
                                  "fridge_left_open": {"window_s": 10.0, "hold_s": [30.0]}}
    burner, fridge = results["evaluations"]
    assert burner["on_time"] == {"hold_s": [10.0], "window_s": 5.0}
    assert burner["metrics"]["caught_in_window"] == 1  # 3 s after its 20 s mark
    # the fridge's 30 s hold ends at 40 s, the range's last second; the alert at 48 s is
    # after the range, so it is a false alarm and the event is missed
    assert (fridge["metrics"]["events"], fridge["metrics"]["false_alarms"]) == (1, 1)
    text = render(results)
    assert "burner_left_on, a 10 s hold and a 5 s window" in text
    assert "fridge_left_open, a 30 s hold and a 10 s window" in text
    assert "Events caught on time" in text and "within 5 s" not in text


def frames_of(rows, ranges):
    """The rows as te calibrate reads them, with the same labels."""
    return labelled_frames(rows, rows[0]["skill"], ranges)


def test_labels_between_frames_are_timed_by_their_frames_as_te_calibrate_times_them():
    # [32.12, 42.12] holds the frames 33 to 42: 9 s of frames, under the 10 s hold, so no
    # alert can be due (the trigger could only fire at 43 s, after the burner went off)
    rows = [row(t, 0.9 if 33 <= t <= 42 else 0.1) for t in range(60)]
    c = count_clip(rows, "burner_left_on", [[32.12, 42.12]])
    assert (c.events, c.short_events, c.false_alarms) == (0, 1, 0)
    assert evaluate([frames_of(rows, [[32.12, 42.12]])], 0.5, 600, 10)["events"] == 0
    # one more frame and it is an event, due at 43 s, in both
    c = count_clip(rows, "burner_left_on", [[32.12, 43.12]])
    assert (c.events, c.short_events) == (1, 0)
    assert evaluate([frames_of(rows, [[32.12, 43.12]])], 0.5, 600, 10)["events"] == 1


def test_a_one_frame_gap_in_the_labels_is_one_event_as_the_smoothing_sees_it():
    ranges = [[10, 20.5], [21.5, 70]]  # frame 21 falls in the gap
    truth = [any(a <= t <= b for a, b in ranges) for t in range(90)]
    trigger = Trigger(0.5, 600, hold_s=30)
    fired = {t for t in range(90) if trigger.update(0.9 if truth[t] else 0.1, float(t))}
    assert fired == {40}  # a perfect student: on from 10 s, through the dip at 21 s
    rows = [row(t, 0.9 if truth[t] else 0.1, "fridge_left_open", 30, t in fired)
            for t in range(90)]
    c = count_clip(rows, "fridge_left_open", ranges)
    assert (c.events, c.short_events, c.caught, c.caught_in_window, c.false_alarms) == (
        1, 0, 1, 1, 0)
    result = evaluate([frames_of(rows, ranges)], 0.5, 600, 30)
    assert (result["events"], result["caught"], result["false_alarms"]) == (1, 1, 0)


def test_a_threshold_records_its_hold_and_te_replay_recalibrates_when_it_changes(tmp_path):
    from backend.cli_replay import _calibration_changed

    log, labels = tmp_path / "v.jsonl", tmp_path / "v.json"
    log.write_text("")
    labels.write_text("")
    target = tmp_path / "threshold.json"
    saved = {"threshold": 0.7, "hold_s": 10, "clips": [{"log": "v.jsonl", "labels": "v.json"}]}
    target.write_text(json.dumps(saved))
    assert not _calibration_changed(target, [log], [labels], 10.0)
    assert _calibration_changed(target, [log], [labels], 30.0)  # the spec's hold changed
    # the plan now calibrates for frames, the saved threshold was for alerts
    assert _calibration_changed(target, [log], [labels], 10.0, "frames")
    target.write_text(json.dumps({k: v for k, v in saved.items() if k != "hold_s"}))
    assert _calibration_changed(target, [log], [labels], 10.0)  # calibrated before holds


def test_the_app_is_told_each_skills_hold(tmp_path):
    from backend.server import taught_skills

    for name, hold in (("fridge_left_open", 30), ("burner_left_on", 10)):
        folder = tmp_path / "skills" / name
        folder.mkdir(parents=True)
        (folder / "spec.json").write_text(spec(name, hold=hold).model_dump_json())
    assert {s["name"]: s["hold_s"] for s in taught_skills(tmp_path)} == {
        "burner_left_on": 10, "fridge_left_open": 30}
