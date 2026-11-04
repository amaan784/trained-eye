import json

import pytest

from backend.eval.metrics import (
    check_ranges,
    cost_per_hour,
    count_clip,
    duration_s,
    read_labels,
    read_log,
    summarize,
)
from backend.inference.student import Score
from backend.skills.spec import SkillSpec, Target
from backend.watch.trigger import Trigger
from backend.watch.watcher import FrameLog, WatchedSkill, Watcher


def row(t, p, threshold=0.5, alert=None, cloud=None, latency=0.2, skill="burner_left_on"):
    r = {"t": float(t), "skill": skill, "threshold": threshold, "p": p}
    if p is None:
        return {**r, "error": "endpoint down"}
    r.update(over=p >= threshold, latency_s=latency, fires=alert is not None)
    if alert is not None:
        r.update(alert=alert, cloud=cloud)
    return r


def clip_rows():
    """20 frames at 1 fps; true from 8 to 15 s; a false alarm at 3 s, an alert at 10 s."""
    probs = [0.9 if t in (2, 3, 9, 10, 11, 12, 13, 16) else 0.1 for t in range(20)]
    rows = [row(t, p) for t, p in enumerate(probs)]
    rows[3] = row(3, 0.9, alert=True, cloud="error: ConnectError")
    rows[10] = row(10, 0.9, alert=True, cloud="yes", latency=1.0)
    rows[11]["latency_measured"] = False  # a frame Spark had cached: no latency
    rows[5] = row(5, None)
    rows.append(row(4, 0.9, skill="fridge_left_open"))  # another skill: ignored
    return rows


def test_frame_event_and_latency_metrics_for_one_clip():
    m = summarize([count_clip(clip_rows(), "burner_left_on", [[8, 15]])])
    # over: 2, 3, 9..13, 16 (8 frames); true: 8..15 (8 frames)
    assert (m["frames"], m["student_errors"], m["clips"]) == (19, 1, 1)
    assert m["precision"] == round(5 / 8, 4) and m["recall"] == round(5 / 8, 4)
    assert m["f1"] == 0.625 and m["accuracy"] == round(13 / 19, 4)
    assert (m["events"], m["caught_in_window"], m["caught"]) == (1, 1, 1)
    assert m["event_recall_in_window"] == 1.0 and m["time_to_alert_median_s"] == 2.0
    assert (m["alerts"], m["false_alarms"]) == (2, 1)
    assert m["hours"] == round(20 / 3600, 4)
    assert m["false_alarms_per_hour"] == 180.0  # one in 20 seconds
    assert m["latency_median_s"] == 0.2 and m["latency_p90_s"] == 0.2  # 18 frames timed
    assert m["cloud_checks"] == 1  # the failed check is not counted
    assert m["frames_per_hour"] == round(19 / (20 / 3600), 4)


def test_a_late_alert_is_caught_but_not_within_the_window():
    rows = [row(t, 0.1) for t in range(30)]
    rows[20] = row(20, 0.9, alert=True)
    # two frames apart (26 and 27), so two events; one frame apart would be bridged into one
    c = count_clip(rows, "burner_left_on", [[12, 25], [28, 29]])
    assert (c.events, c.caught, c.caught_in_window, c.false_alarms) == (2, 1, 0, 0)
    m = summarize([c])
    assert m["event_recall_any"] == 0.5 and m["event_recall_in_window"] == 0.0
    assert m["time_to_alert_mean_s"] == 8.0
    assert m["precision"] == 1.0 and m["recall"] == 0.0625  # 1 of 16 true frames over


def test_clips_add_up():
    one = count_clip(clip_rows(), "burner_left_on", [[8, 15]])
    two = count_clip([row(t, 0.1, latency=0.5) for t in range(40)], "burner_left_on", [])
    total = one + two
    assert total.clips == 2 and total.seconds == 60.0 and total.events == 1
    assert len(total.latencies) == 58
    m = summarize([one, two])
    assert m["false_alarms_per_hour"] == 60.0 and m["latency_p90_s"] == 0.5
    assert summarize([])["f1"] is None  # nothing scored: no numbers


def test_bad_inputs(tmp_path):
    with pytest.raises(ValueError, match="start <= end"):
        count_clip(clip_rows(), "burner_left_on", [[9, 8]])
    with pytest.raises(ValueError, match="no rows for tap_left_running"):
        count_clip(clip_rows(), "tap_left_running", [])
    assert duration_s([4.0]) == 0.0 and duration_s([0, 2, 4]) == 6.0
    (tmp_path / "labels.json").write_text(json.dumps({"true": [[30, 40], [1, 2]]}))
    assert read_labels(tmp_path / "labels.json") == [(1.0, 2.0), (30.0, 40.0)]
    # overlapping and touching ranges are one event, not two
    assert check_ranges([[5, 9], [1, 3], [8, 12], [12, 14]]) == [(1.0, 3.0), (5.0, 14.0)]


def test_costs_per_hour():
    m = summarize([count_clip([row(t, 0.1) for t in range(3600)], "burner_left_on", [])])
    assert m["frames_per_hour"] == 3600.0
    spark = cost_per_hour(m, scorer_usd_per_frame=0.003)
    assert spark == {"student": "Mac", "gpu_usd": 0.0, "spark_checks_usd": 0.0,
                     "per_frame_scorer_usd": 10.8, "total_usd": 10.8}
    rows = [row(t, 0.9, alert=True, cloud="yes") if t in (9, 99) else row(t, 0.1)
            for t in range(3600)]
    gpu = cost_per_hour(summarize([count_clip(rows, "burner_left_on", [])]),
                        gpu_usd_per_hour=3.95, spark_usd_per_check=0.003)
    assert (gpu["student"], gpu["spark_checks_usd"], gpu["total_usd"]) == ("GPU", 0.006, 3.956)
    assert cost_per_hour(summarize([]))["total_usd"] is None
    with pytest.raises(ValueError):
        cost_per_hour(m, spark_usd_per_check=-1)


class Scores(list):
    def score(self, image, spec):
        return Score(self.pop(0), "yes", 0.25)


def test_a_log_written_by_the_watcher(tmp_path):
    spec = SkillSpec(name="burner_left_on", request="tell me when the burner is left on",
                     target=Target(sam_prompt="stove burner", description="the display"),
                     positive="the burner is lit with no pan on it", negatives=["a pan being stirred on it"],
                     alert="The burner is still on.", cooldown_s=600, confirm_with_cloud=False)
    log = FrameLog(tmp_path / "replay.jsonl")
    skill = WatchedSkill(spec, Trigger(0.5, spec.cooldown_s), calibrated=True)
    watcher = Watcher([skill], Scores([0.1] * 5 + [0.9] * 5), log=log)
    for t in range(10):
        watcher.on_frame(b"jpeg", t=float(t))
    m = summarize([count_clip(read_log(log.path), "burner_left_on", [[5, 9]])])
    assert (m["f1"], m["caught_in_window"], m["time_to_alert_median_s"]) == (1.0, 1, 2.0)
    assert m["false_alarms"] == 0 and m["latency_median_s"] == 0.25
