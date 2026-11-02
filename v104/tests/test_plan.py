"""The experiment plan: runs, their arguments, their price and eval/plan.json."""

import json

import pytest

from backend.cli import build_parser, train_estimate
from backend.config import load_settings
from backend.eval.plan import format_plan, plan_cost, plan_skeleton, training_runs, write_plan
from backend.eval.report import Plan
from tests.test_training_prep import teach_dataset

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
    assert cost["total_usd"] == pytest.approx(sum(r["usd"] for r in cost["runs"]), abs=0.02)
    assert 0 < cost["repeats_usd"] < cost["total_usd"] and not cost["over_credit"]
    assert "total about $" in format_plan(runs, cost)
    unpriced = plan_cost(runs, None)
    assert not unpriced["priced"] and "--smoke" in format_plan(runs, unpriced)


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
        "log": "../data/watch/exp1-mixed/washer_done/session3_test_a.jsonl",
        "labels": "../data/clips/washer_done/session3_test_a.labels.json"}]  # from eval/
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
