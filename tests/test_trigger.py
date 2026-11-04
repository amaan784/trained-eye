import pytest

from backend.watch.trigger import Frame, Trigger, alerts, calibrate, evaluate, events


def run(trigger, probs, start=0.0, step=1.0):
    return [i for i, p in enumerate(probs) if trigger.update(p, start + i * step)]


def test_three_of_four_frames_fire_once_then_the_cooldown_holds():
    trigger = Trigger(threshold=0.5, cooldown_s=60)
    fired = run(trigger, [0.9, 0.1, 0.8, 0.7] + [0.9] * 30)
    assert fired == [3]  # the third frame over, within four; then quiet for 60 s


def test_single_bright_frames_never_fire():
    assert run(Trigger(0.5, 60), [0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1] * 5) == []


def test_it_fires_again_after_the_cooldown_if_the_condition_stays():
    fired = run(Trigger(0.5, cooldown_s=10), [0.9] * 30)
    assert fired == [2, 12, 22]  # the frames during the cooldown still fill the window


def test_a_pause_in_the_stream_starts_the_window_again():
    trigger = Trigger(0.5, 60)
    assert not trigger.update(0.9, 0) and not trigger.update(0.9, 1)
    assert not trigger.update(0.9, 30)  # 29 s later: the first two no longer count
    assert not trigger.update(0.9, 31) and trigger.update(0.9, 32)


def test_snooze_after_a_cloud_no():
    trigger = Trigger(0.5, cooldown_s=600)
    assert run(trigger, [0.9] * 3) == [2]
    trigger.quiet_until = 0  # as if the cloud check rejected that alert
    trigger.snooze(2, 60)
    assert run(trigger, [0.9] * 70, start=3) == [59]  # t = 62: after the 60 s snooze


def test_bad_window_settings():
    with pytest.raises(ValueError):
        Trigger(0.5, 60, window=2, needed=3)


def clip(truths, probs, step=1.0):
    return [Frame(i * step, p, t) for i, (t, p) in enumerate(zip(truths, probs, strict=True))]


def test_events_and_evaluate_count_false_alarms_and_catches():
    truths = [False] * 10 + [True] * 10 + [False] * 10
    probs = [0.6, 0.6, 0.6] + [0.1] * 7 + [0.9] * 10 + [0.1] * 10
    frames = clip(truths, probs)
    assert events(frames) == [(10.0, 19.0)]
    result = evaluate([frames], threshold=0.5, cooldown_s=5)
    assert result["false_alarms"] == 1 and result["caught"] == 1 and result["events"] == 1
    assert result["median_delay_s"] == 2.0  # fires on the third true frame
    assert [f.t for f in alerts(frames, 0.5, 5)] == [2.0, 12.0, 17.0]
    assert result["alerts"] == 3


def test_calibrate_picks_the_lowest_threshold_with_few_false_alarms():
    # Two hours of negatives with a few bright runs at 0.6, and a true event at 0.8.
    truths = [False] * 7200 + [True] * 20
    probs = [0.6 if i % 1800 < 3 else 0.2 for i in range(7200)] + [0.8] * 20
    frames = clip(truths, probs)
    result = calibrate([frames], cooldown_s=600)
    assert 0.6 < result["threshold"] <= 0.8  # 0.6 would fire 4 false alarms in 2 hours
    assert result["false_alarms"] == 0 and result["caught"] == 1
    loose = calibrate([frames], cooldown_s=600, max_per_hour=2)
    assert loose["threshold"] <= 0.6 and loose["false_alarms"] == 4


def test_calibrate_on_footage_that_always_looks_true_never_fires():
    frames = clip([False] * 100, [0.99] * 100)
    result = calibrate([frames], cooldown_s=600)
    assert result["threshold"] == 1.0 and result["alerts"] == 0
    certain = calibrate([clip([False] * 100, [1.0] * 100)], cooldown_s=600)
    assert certain["never_fires"] and certain["alerts"] == 0
    with pytest.raises(ValueError):
        calibrate([], cooldown_s=60)


def test_labelled_frames_from_a_watch_log():
    from backend.watch.trigger import labelled_frames

    rows = [{"t": 0.0, "skill": "burner_left_on", "p": 0.1},
            {"t": 1.0, "skill": "pot_boils", "p": 0.9},
            {"t": 1.0, "skill": "burner_left_on", "p": None, "error": "down"},
            {"t": 2.0, "skill": "burner_left_on", "p": 0.8}]
    frames = labelled_frames(rows, "burner_left_on", [[1.5, 3]])
    assert [(f.t, f.p, f.truth) for f in frames] == [(0.0, 0.1, False), (2.0, 0.8, True)]
    with pytest.raises(ValueError):
        labelled_frames(rows, "burner_left_on", [[3, 1]])
