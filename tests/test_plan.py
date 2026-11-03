"""The experiment plan: runs, their arguments, their price and eval/plan.json."""

import json

import pytest

from backend.cli import build_parser, train_estimate
from backend.config import load_settings
from backend.eval.plan import (
    final_run,
    format_plan,
    plan_cost,
    plan_skeleton,
    progress,
    training_runs,
    write_plan,
)
from backend.eval.report import Plan
from tests.test_training_prep import taught_with_clips, teach_dataset

NAMES = ["exp1-real", "exp1-synthetic", "exp1-mixed", "exp2-filtered", "exp2-unfiltered",
         "exp3-25", "exp3-50", "exp3-100", "exp3-200", "exp3-300"]
TIMING = {"seconds": 400.0, "train_runtime": 100.0, "train_samples_per_second": 0.5}


def test_every_run_has_te_train_arguments_and_the_same_fixed_steps():
    runs = training_runs(["washer_done"])
    assert [r["name"] for r in runs] == NAMES
    assert {r["max_steps"] for r in runs} == {10}
    assert {r["max_steps"] for r in training_runs(["washer_done", "boil_over", "bus_m4"])} == {29}
    by_name = {r["name"]: r for r in runs}
    assert by_name["exp2-filtered"]["same_bundle_as"] == "exp1-mixed"
    assert by_name["exp3-25"]["args"][-4:] == ["--limit", "25", "--max-steps", "10"]
    assert "dataset-unfiltered" in by_name["exp2-unfiltered"]["args"]
    for run in runs:  # everything but --max-steps is what te train already takes
        args = build_parser().parse_args(run["args"][1:-2])
        assert (args.run, args.variant, args.limit) == (run["name"], run["variant"], run["limit"])


def test_with_the_data_runs_get_counts_and_say_what_they_need(tmp_path):
    for skill in ("washer_done", "boil_over"):
        teach_dataset(tmp_path / "skills", skill)  # 10 real and 40 generated examples each
    runs = {r["name"]: r for r in training_runs(["washer_done", "boil_over"],
                                                skills_root=tmp_path / "skills")}
    assert runs["exp1-mixed"]["examples"]["train"] + runs["exp1-mixed"]["examples"]["val"] == 100
    assert runs["exp3-25"]["examples"]["per_skill"]["boil_over"]["synthetic"] == 25
    assert runs["exp3-25"]["same_bundle_as"] is None and not runs["exp3-25"]["needs"]
    for name in ("exp3-50", "exp3-300"):  # 40 is all there is: the same as exp1-synthetic
        assert runs[name]["same_bundle_as"] == "exp1-synthetic"
        assert "has 40 generated examples, under" in runs[name]["needs"][0]
    assert runs["exp2-unfiltered"]["examples"] is None
    assert "te unfiltered" in runs["exp2-unfiltered"]["needs"][0]


def test_the_cost_uses_te_trains_estimate_and_counts_repeats(tmp_path):
    teach_dataset(tmp_path / "skills", "washer_done")
    runs = training_runs(["washer_done"], skills_root=tmp_path / "skills")
    cost = plan_cost(runs, TIMING)
    mixed = next(r for r in cost["runs"] if r["name"] == "exp1-mixed")
    assert mixed["usd"] == round(train_estimate(320 + 8 * 5, TIMING)["usd"], 2)  # 8 val passes
    assert cost["training_usd"] == pytest.approx(sum(r["usd"] for r in cost["runs"]), abs=0.02)
    assert 0 < cost["repeats_usd"] < cost["training_usd"]
    # every run with its own bundle (5 of 10: exp2-filtered is exp1-mixed, exp3-50 and up
    # exp1-synthetic) is exported at te export's limit; those and the base endpoint are served
    models = [m["name"] for m in cost["models"]]
    assert models == ["exp1-real", "exp1-synthetic", "exp1-mixed", "exp2-unfiltered",
                      "exp3-25", "exp4-base"]
    assert cost["export_usd"] == round(5 * 3 * 3.95, 2) and not cost["export_measured"]
    assert cost["serving_usd"] == round(6 * 1.0 * 3.95, 2) and cost["spent_usd"] == 0
    assert cost["total_usd"] == pytest.approx(
        cost["training_usd"] + cost["export_usd"] + cost["serving_usd"], abs=0.02)
    assert not cost["over_credit"]
    text = format_plan(runs, cost)
    assert "total about $" in text and "exports about $59.25: 5 models at 3.0 h" in text
    assert "serving about $23.70: 6 models at 1 h each" in text
    assert "Modal spend so far: $0.00" in text
    unpriced = plan_cost(runs, None)
    assert not unpriced["priced"] and "--smoke" in format_plan(runs, unpriced)
    assert unpriced["export_usd"] == cost["export_usd"]  # priced without a smoke too


def test_the_credit_check_adds_the_final_run_spend_exports_and_serving(tmp_path):
    for skill in ("washer_done", "boil_over", "bus_m4"):
        teach_dataset(tmp_path / "skills", skill)
    skills = ["washer_done", "boil_over", "bus_m4"]
    runs = training_runs(skills, skills_root=tmp_path / "skills")
    final = final_run(skills, skills_root=tmp_path / "skills")
    assert final["max_steps"] is None and "--max-steps" not in final["args"]
    assert final["args"][-2:] == ["--variant", "mixed"] and final["examples"]["train"] == 135
    done = {"trained": {"exp1-real"}, "exported": {"exp1-real"}}
    cost = plan_cost(runs, TIMING, final=final, endpoint_hours=2.0, export_hours=0.5,
                     spent_usd=20.0, done=done)
    rows = {r["name"]: r for r in cost["runs"]}
    assert rows["final"]["usd"] == round(train_estimate(150, TIMING)["usd"], 2)  # one epoch
    assert rows["exp1-real"]["done"] and "exp1-real" not in cost["over_ask"]
    models = {m["name"]: m for m in cost["models"]}
    assert models["exp1-real"]["export_usd"] == 0.0  # exported already: in the spend
    assert models["final"]["export_usd"] == round(0.5 * 3.95, 2)
    assert models["exp4-base"]["export_usd"] == 0.0 and not models["exp4-base"]["export"]
    assert cost["serving_usd"] == round(len(models) * 2.0 * 3.95, 2)
    training = sum(r["usd"] for r in cost["runs"] if not r["done"])
    assert cost["training_usd"] == pytest.approx(training, abs=0.02)
    assert cost["total_usd"] == pytest.approx(20.0 + training + cost["export_usd"]
                                              + cost["serving_usd"], abs=0.03)
    text = format_plan(runs, cost, final)
    assert "final" in text and "the recipe's epoch over every example" in text
    assert "trained already, in the spend: exp1-real" in text and "exp1-real           done" \
        in text
    assert "the longest export timed so far" in text and "$20.00 (te spend)" in text
    big = plan_cost(runs, TIMING, final=final, spent_usd=140.0)
    assert big["over_credit"] and "passes $150 of Modal credit: ask first" in format_plan(
        runs, big, final)


def test_progress_reads_finished_runs_and_exports(tmp_path):
    folder = tmp_path / "train"
    folder.mkdir()
    for name, result in (("exp1-real", {"ok": True}), ("exp1-mixed", {"ok": True, "smoke": True}),
                         ("exp3-25", {"ok": False})):
        (folder / f"{name}.result.json").write_text(json.dumps(result))
    (folder / "exp1-real.export.json").write_text(json.dumps({"merged_dir": "/vol/x"}))
    (folder / "exp3-25.export.json").write_text(json.dumps({"ok": False, "merged_dir": "x"}))
    (folder / "broken.result.json").write_text("{")
    assert progress(tmp_path) == {"trained": {"exp1-real"}, "exported": {"exp1-real"}}


def test_counts_add_the_session_clips_to_real_and_mixed(tmp_path):
    taught_with_clips(tmp_path)  # 10 real and 40 generated; 24 session 1, 9 session 2 frames
    runs = {r["name"]: r for r in training_runs(["washer_done"],
                                                skills_root=tmp_path / "skills")}
    mixed, real = runs["exp1-mixed"]["examples"], runs["exp1-real"]["examples"]
    assert (mixed["train"], mixed["val"]) == (45 + 24, 5 + 9)  # 10% of the 50 taught
    assert (real["train"], real["val"]) == (10 + 24, 9)  # under 50 taught: no teach val
    assert real["per_skill"]["washer_done"]["clips_train"] == 24
    synthetic = runs["exp1-synthetic"]["examples"]
    assert (synthetic["train"], synthetic["val"]) == (40, 0)  # never the clips


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
