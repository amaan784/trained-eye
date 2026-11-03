"""eval/plan.json: the evaluation plan te eval reads, with its replays and calibration."""

import json

import pytest

from backend.config import load_settings
from backend.eval.plan_file import plan_skeleton, write_plan
from backend.eval.report import Plan
from tests.test_training_prep import teach_dataset


def test_the_plan_skeleton_is_what_te_eval_reads(tmp_path):
    settings = load_settings(env={"TE_DATA_DIR": str(tmp_path / "data")}, env_file=None)
    Plan.model_validate(plan_skeleton(["washer_done", "boil_over"], settings))  # before data
    skill = teach_dataset(settings.data_dir / "skills", "washer_done")
    (skill / "label_stats.json").write_text(json.dumps({"total": 50, "kept": 40}))
    clips = settings.data_dir / "clips" / "washer_done"
    clips.mkdir(parents=True)
    (clips / "session3_test_a.mp4").write_bytes(b"")
    (clips / "session3_test_a.labels.json").write_text('{"true": []}')
    plan = plan_skeleton(["washer_done"], settings)
    Plan.model_validate(plan)
    entries = {e["name"].removesuffix("-washer_done"): e for e in plan["evaluations"]}
    assert len(entries) == 13 and entries["exp1-mixed"]["clips"] == [{
        "clip": "../data/clips/washer_done/session3_test_a.mp4",
        "log": "../data/watch/exp1-mixed/washer_done/session3_test_a.jsonl",
        "labels": "../data/clips/washer_done/session3_test_a.labels.json",
        "timeline": True}]  # from eval/
    check = entries["exp1-mixed"]["spark_usd_per_check"]
    assert entries["exp3-25"]["generation_usd"] == round(25 * 50 / 40 * (0.01 + check), 4)
    assert entries["exp1-real"]["images"] is None is entries["exp2-unfiltered"]["images"]
    spark = entries["exp4-spark"]
    assert (spark["gpu_usd_per_hour"], spark["spark_usd_per_check"],
            spark["scorer_usd_per_frame"]) == (0.0, 0.0, check)
    assert entries["exp4-finetuned"]["images"] == 40 and entries["exp4-finetuned"]["highlight"]
    path = write_plan(tmp_path / "eval/plan.json", plan)
    with pytest.raises(FileExistsError):
        write_plan(path, plan)


def test_same_models_share_their_replays_and_every_run_calibrates_its_own(tmp_path):
    from tests.test_realclips import SPEC, skill_folder

    settings = load_settings(env={"TE_DATA_DIR": str(tmp_path / "data")}, env_file=None)
    blank = plan_skeleton(["washer_done"], settings)  # nothing recorded: protocol names
    first = blank["evaluations"][0]["clips"][0]
    assert first["clip"] == "../data/clips/washer_done/session3_test_a.mp4"
    assert blank["calibration"][0]["clips"][0]["clip"] == ("../data/clips/washer_done/"
                                                          "session2_a.mp4")
    assert "alert_text" not in blank["evaluations"][0]  # no spec.json yet: the default
    skills = settings.data_dir / "skills"
    teach_dataset(skills, "washer_done")
    skill_folder(skills, SPEC)
    clips = settings.data_dir / "clips" / "washer_done"
    clips.mkdir(parents=True)
    for name in ("session3_test_a.mp4", "session3_test_b.mp4", "session2_x.mp4",
                 "session1_y.mp4"):
        (clips / name).write_bytes(b"")
    plan = plan_skeleton(["washer_done"], settings)
    Plan.model_validate(plan)
    entries = {e["name"].removesuffix("-washer_done"): e for e in plan["evaluations"]}
    runs = {name: e["run"] for name, e in entries.items()}
    assert runs["exp2-filtered"] == runs["exp4-finetuned"] == "exp1-mixed"
    assert runs["exp3-50"] == runs["exp3-300"] == "exp1-synthetic"  # 40 images is all
    assert runs["exp3-25"] == "exp3-25" and runs["exp4-base"] == "exp4-base"
    logs = {name: [c["log"] for c in e["clips"]] for name, e in entries.items()}
    assert logs["exp2-filtered"] == logs["exp4-finetuned"] == logs["exp1-mixed"] == [
        "../data/watch/exp1-mixed/washer_done/session3_test_a.jsonl",
        "../data/watch/exp1-mixed/washer_done/session3_test_b.jsonl"]
    timelines = {name: [bool(c.get("timeline")) for c in e["clips"]]
                 for name, e in entries.items()}
    assert timelines["exp1-mixed"] == timelines["exp4-finetuned"] == [True, False]
    assert not any(any(t) for name, t in timelines.items()
                   if name not in ("exp1-mixed", "exp4-finetuned"))
    assert {e["alert_text"] for e in entries.values()} == {"Laundry's done."}
    assert entries["exp4-spark"]["scorer"] == "spark" and entries["exp1-real"]["scorer"] == \
        "student"
    calibration = {c["run"]: c for c in plan["calibration"]}
    assert list(calibration) == ["exp1-real", "exp1-synthetic", "exp1-mixed", "exp2-unfiltered",
                                 "exp3-25", "exp4-base"]  # each replayed student; not Spark
    assert calibration["exp4-base"]["clips"] == [{
        "clip": "../data/clips/washer_done/session2_x.mp4",
        "log": "../data/watch/exp4-base/washer_done/session2_x.jsonl",
        "labels": "../data/clips/washer_done/session2_x.labels.json"}]  # session 2 only
