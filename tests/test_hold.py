"""The per-skill hold: watch mode alerts only once a condition has stayed true for the
skill's hold_s, and the evaluation counts an alert as on time from the end of the hold."""

import json

import httpx
import pytest
from pydantic import ValidationError

from backend.config import load_settings
from backend.eval import metrics
from backend.eval.metrics import ON_TIME_S, count_clip, hold_of, on_time_window, summarize
from backend.eval.report import Plan, run_eval
from backend.eval.writeup import render
from backend.inference.student import Score
from backend.meta.client import MetaClient
from backend.skills.compiler import COMPILE_SCHEMA, SYSTEM, compile_skill, hold_for
from backend.skills.spec import MAX_HOLD_S, SKILL_HOLDS, SkillSpec, Target
from backend.watch.trigger import Frame, Trigger, alerts, calibrate, evaluate
from backend.watch.watcher import FrameLog, Watcher, load_skill


def run(trigger, probs, start=0.0, step=1.0):
    return [start + i * step for i, p in enumerate(probs) if trigger.update(p, start + i * step)]


# The trigger


def test_the_project_skills_have_the_authors_holds():
    assert SKILL_HOLDS == {"fridge_left_open": 30, "tap_left_running": 20,
                           "burner_left_on": 10, "egg_whisked": 0}


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


def test_a_cloud_no_snoozes_a_held_alert_like_any_other():
    trigger = Trigger(0.5, cooldown_s=600, hold_s=10)
    assert run(trigger, [0.9] * 11) == [10.0]
    trigger.quiet_until = 10  # as if Spark said no to that alert
    trigger.snooze(10, 60)
    assert run(trigger, [0.9] * 70, start=11) == [70.0]  # after the minute, still held


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


def compiled(reply, name):
    """Compile with a fake Muse Wonder giving `reply`; each call has its own cache."""
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps(reply)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5000, "completion_tokens": 600}})

    settings = load_settings(env={"MODEL_API_KEY": "sk-test", "TE_DATA_DIR": f"data-{name}"},
                             env_file=None)
    client = MetaClient(settings, transport=httpx.MockTransport(handler), sleep=lambda s: None)
    return compile_skill(client, "tell me when the fridge is left open", [b"\xff\xd8x"],
                         name=name), requests


def test_the_compiler_asks_for_a_hold_and_the_project_skills_keep_theirs(tmp_path,
                                                                         monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert "hold_s" in COMPILE_SCHEMA["required"] and "hold_s:" in SYSTEM
    fridge, requests = compiled(REPLY, "fridge_left_open")
    assert fridge.hold_s == 30  # the author's hold, not the model's 5
    assert requests[0]["response_format"]["json_schema"]["schema"] == COMPILE_SCHEMA
    other, _ = compiled({**REPLY, "hold_s": 12}, "freezer_door")
    assert other.hold_s == 12
    without, _ = compiled({k: v for k, v in REPLY.items() if k != "hold_s"}, "freezer_lid")
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
    plan = Plan.model_validate({"on_time_s": {"fridge_left_open": 12}, "evaluations": []})
    assert (plan.window("fridge_left_open"), plan.window("burner_left_on")) == (12.0, 5.0)
    for bad in ({"fridge_left_open": 0}, {"Fridge Door": 5}):
        with pytest.raises(ValidationError):
            Plan.model_validate({"on_time_s": bad, "evaluations": []})
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
