import json
import sys
import types
import xml.etree.ElementTree as ET

import pytest

from backend.eval.report import run_eval


def write_clip(folder, name, alert_at, true=(60, 90), n=120, over=None):
    """A frame log at 1 fps: over the threshold on `over` (default: the true range), an
    alert at `alert_at` (None for none), and its labels file."""
    over = range(true[0], true[1] + 1) if over is None else over
    rows = [{"t": float(t), "skill": "burner_left_on", "threshold": 0.6,
             "p": 0.9 if t in over else 0.2, "over": t in over, "fires": t == alert_at,
             "latency_s": 0.4, **({"alert": True, "cloud": "yes"} if t == alert_at else {})}
            for t in range(n)]
    (folder / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (folder / f"{name}.json").write_text(json.dumps({"true": [list(true)]}))
    return {"log": f"../clips/{name}.jsonl", "labels": f"../clips/{name}.json"}  # from eval/


def evaluation(name, experiment, clip, **extra):
    return {"name": name, "experiment": experiment, "skill": "burner_left_on",
            "clips": [clip], **extra}


@pytest.fixture
def plan(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    mixed = {**write_clip(clips, "mixed", alert_at=62), "timeline": True}
    plan = {"evaluations": [
        evaluation("exp1-real", 1, write_clip(clips, "real", 75, over=range(70, 100)),
                   variant="real only"),
        evaluation("exp1-mixed", 1, mixed, variant="mixed", alert_text="The burner is still on."),
        evaluation("exp3-25", 3, write_clip(clips, "n25", None, over=()), images=25,
                   generation_usd=0.25),
        evaluation("exp3-300", 3, write_clip(clips, "n300", 62), images=300),
        evaluation("exp4-tuned", 4, write_clip(clips, "tuned", 62), variant="fine-tuned Glimmer",
                   highlight=True, spark_usd_per_check=0.003),
        evaluation("exp4-spark", 4, write_clip(clips, "spark", 63), variant="Muse Spark",
                   scorer_usd_per_frame=0.003),
        evaluation("exp4-base", 4, {"log": "../clips/base.jsonl", "labels": "../clips/base.json"},
                   variant="base Glimmer", gpu="H100"),
    ]}
    path = tmp_path / "eval" / "plan.json"
    path.parent.mkdir()
    path.write_text(json.dumps(plan))
    return path


def test_run_eval_writes_results_tables_and_charts(plan, tmp_path):
    out = tmp_path / "out"
    results = run_eval(plan, out)
    assert json.loads((out / "results.json").read_text()) == results
    by_name = {e["name"]: e for e in results["evaluations"]}
    mixed = by_name["exp1-mixed"]["metrics"]
    assert (mixed["f1"], mixed["caught_in_window"], mixed["time_to_alert_median_s"]) == (1.0, 1, 2.0)
    real = by_name["exp1-real"]["metrics"]
    assert (real["caught_in_window"], real["caught"], real["false_alarms"]) == (0, 1, 0)
    assert by_name["exp4-spark"]["cost_per_hour"]["total_usd"] == 10.8  # 3600 frames per hour
    assert by_name["exp4-base"]["status"] == "pending"
    assert by_name["exp4-base"]["gpu_usd_per_hour"] == 3.95  # from config
    assert results["charts"] == {"1": ["exp1_bars.svg"], "3": ["exp3_burner_left_on.svg"],
                                 "4": ["exp4_cost.svg"],
                                 "timelines": ["timeline_exp1-mixed_1.svg"]}
    for names in results["charts"].values():
        for name in names:
            ET.fromstring((out / name).read_text())
    assert "not plotted, no numbers yet: base Glimmer" in (out / "exp4_cost.svg").read_text()

    md = (out / "results.md").read_text()
    two = md.split("## Experiment 2")[1].split("## Experiment 3")[0]
    assert "No runs yet: the plan has no evaluations for this experiment." in two
    assert "![Experiment 1: real frames only, generated images only, or both](exp1_bars.svg)" in md
    assert ("| exp1-mixed | mixed | burner_left_on | 1 | 2.0 min | 120 | 1.00 | 1.00 | 1.00 | "
            "1 of 1 | 1 of 1 | 0.0 | 2.0 s (2.0 s) |") in md
    assert "| exp1-real | real only | burner_left_on |" in md and "| 0 of 1 | 1 of 1 |" in md
    assert "| 25 | $0.25 |" in md and "| 300 | n/a |" in md  # no generation cost given
    assert ("| exp4-spark | 0.40 s | 0.40 s | 0 | 1 | 30.0 | $10.80 | Mac, Spark checks not "
            "priced in the plan, $0.003 per frame scored |") in md
    assert "- exp4-base: `../clips/base.jsonl`, `../clips/base.json`" in md
    assert "![Live timeline](timeline_exp1-mixed_1.svg)" in md
    assert md.isascii()  # plain text: no em or en dashes either


def test_the_footage_a_plan_names_heads_its_results(plan, tmp_path):
    out = tmp_path / "out"
    assert run_eval(plan, out)["footage"] is None
    assert (out / "results.md").read_text().startswith("# Evaluation results\n")
    data = json.loads(plan.read_text())
    data["footage"] = "other kitchens (EPIC-KITCHENS-100, head-mounted camera)"
    plan.write_text(json.dumps(data))
    results = run_eval(plan, out)
    md = (out / "results.md").read_text()
    assert results["footage"] == data["footage"]
    assert md.startswith("# Evaluation results: other kitchens (EPIC-KITCHENS-100, "
                         "head-mounted camera)\n")
    assert "Results on other footage are reported apart." in md
    assert "calibrated for frame F1" not in md
    data["threshold_objective"] = "frames"
    plan.write_text(json.dumps(data))
    assert run_eval(plan, out)["threshold_objective"] == "frames"
    assert "Thresholds here were calibrated for frame F1" in (out / "results.md").read_text()


def test_the_speed_measurement_says_why_one_model_answers_faster(plan, tmp_path):
    out = tmp_path / "out"
    out.mkdir()

    def measured(model, seconds, tokens, reasoning):
        return {"model": model, "skill": "burner_left_on", "clip": "data/x/P03_19_00093",
                "frames": 20, "median_seconds": seconds, "median_completion_tokens": tokens,
                "median_prompt_tokens": 474, "frames_with_reasoning": reasoning,
                "rows": [{"t": 0.0, "seconds": seconds}]}

    (out / "speed.json").write_text(json.dumps({
        "finetuned": measured("trained-eye", 0.9895, 5.0, 0),
        "base": measured("muse-glimmer", 1.784, 64.0, 20)}))
    results = run_eval(plan, out)
    assert "rows" not in results["speed"]["base"]
    md = (out / "results.md").read_text()
    assert "on frames of P03_19_00093" in md
    assert "| finetuned | 5 | 0 of 20 | 0.99 s |" in md
    assert "| base | 64 | 20 of 20 | 1.78 s |" in md
    (out / "speed.json").write_text(json.dumps({"base": {"model": "m"}}))
    with pytest.raises(ValueError, match="speed"):
        run_eval(plan, out)
    (out / "speed.json").unlink()
    assert "speed" not in run_eval(plan, out)


def test_bad_plans_are_refused(tmp_path):
    def check(evaluations, match):
        path = tmp_path / "plan.json"
        path.write_text(json.dumps({"evaluations": evaluations}))
        with pytest.raises(ValueError, match=match):
            run_eval(path, tmp_path / "out")

    clip = {"log": "a.jsonl", "labels": "a.json"}
    check([evaluation("a", 1, clip), evaluation("a", 2, clip)], "must be unique: a")
    check([evaluation("a", 3, clip)], "need images")
    check([evaluation("a", 1, clip, gpu_usd_per_hr=1)], "Extra inputs")
    check([evaluation("a", 1, clip, gpu="TPU")], "gpu must be one of")
    check([evaluation("../a", 1, clip)], "pattern")
    check([{**evaluation("a", 1, clip), "skill": "Burner Left On"}], "snake_case")
    (tmp_path / "a.jsonl").write_text("{}\n")
    (tmp_path / "a.json").write_text('{"true": []}')
    check([evaluation("a", 1, clip)], "a, a.jsonl: ValueError")  # no rows for the skill


def test_wandb_is_optional_and_logged_only_when_asked(plan, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)  # import wandb now raises ImportError
    results = run_eval(plan, tmp_path / "a", log_wandb=True)
    assert results["wandb"].startswith("not logged: the wandb package is not installed")
    logged = {}

    run = types.SimpleNamespace(summary={}, url="https://wandb.ai/me/trained-eye/runs/x",
                                log=logged.update, finish=lambda: logged.update(finished=True))
    fake = types.SimpleNamespace(init=lambda **kw: logged.update(init=kw) or run,
                                 Table=lambda columns, data: ("table", columns, data),
                                 Html=lambda html: ("html", len(html)))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    assert "wandb" not in run_eval(plan, tmp_path / "b")  # not asked: nothing logged
    assert logged == {}
    results = run_eval(plan, tmp_path / "c", log_wandb=True)
    assert results["wandb"] == run.url and logged["finished"]
    assert logged["init"]["project"] == "trained-eye" and logged["init"]["job_type"] == "eval"
    assert run.summary["exp1-mixed/f1"] == 1.0 and run.summary["exp4-spark/usd_per_hour"] == 10.8
    _, columns, rows = logged["eval/results"]
    assert len(rows) == 6 and columns[:4] == ["name", "experiment", "skill", "variant"]
    assert set(logged) >= {"charts/exp1_bars", "charts/timeline_exp1-mixed_1"}
    assert f"W&B: {run.url}" in (tmp_path / "c" / "results.md").read_text()


def test_wandb_runs_go_to_the_team_given(plan, tmp_path, monkeypatch):
    inits = []
    run = types.SimpleNamespace(summary={}, url="https://wandb.ai/team/trained-eye/runs/y",
                                log=lambda data: None, finish=lambda: None)
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(
        init=lambda **kw: inits.append(kw) or run, Table=lambda columns, data: None,
        Html=lambda html: None))
    run_eval(plan, tmp_path / "a", log_wandb=True, wandb_entity="team")
    run_eval(plan, tmp_path / "b", log_wandb=True)
    assert [i["entity"] for i in inits] == ["team", None]  # None: the key's own account
    assert {i["project"] for i in inits} == {"trained-eye"}


def skill_clip(folder, name, skill, alert_at, **kw):
    """write_clip for another skill."""
    clip = write_clip(folder, name, alert_at, **kw)
    log = folder / f"{name}.jsonl"
    log.write_text(log.read_text().replace('"burner_left_on"', f'"{skill}"'))
    return clip


def test_judges_pool_over_their_skills_and_side_results_reach_results_and_wandb(
        tmp_path, monkeypatch):
    from tests.test_eval_writeup import REGRESSION, SPARK_ERRORS

    clips = tmp_path / "clips"
    clips.mkdir()

    def judge(name, variant, skill, clip, **extra):
        return {"name": name, "experiment": 4, "skill": skill, "variant": variant,
                "clips": [clip], **extra}

    plan_path = tmp_path / "eval" / "plan.json"
    plan_path.parent.mkdir()
    plan_path.write_text(json.dumps({"evaluations": [
        judge("ft-w", "finetuned", "burner_left_on", write_clip(clips, "ftw", 62), gpu="H100",
              highlight=True),
        judge("ft-b", "finetuned", "fridge_left_open", skill_clip(clips, "ftb", "fridge_left_open", None,
                                                           n=240), gpu="H100", highlight=True),
        judge("base-w", "base", "burner_left_on", write_clip(clips, "bw", 75), gpu="H100"),
        judge("base-b", "base", "fridge_left_open", {"log": "../clips/none.jsonl",
                                              "labels": "../clips/none.json"}),
        judge("spark-w", "spark", "burner_left_on", write_clip(clips, "sw", 63),
              scorer_usd_per_frame=0.003)]}))
    data = tmp_path / "data"
    for skill, files in (("burner_left_on", {"spark_errors.json": {**SPARK_ERRORS,
                                                                 "unmatched_files": ["x.png"]},
                                          "label_stats.json": {"total": 250, "kept": 200,
                                                               "agreement": 0.8,
                                                               "by_kind": {}}}),
                         ("fridge_left_open", {})):
        (data / "skills" / skill).mkdir(parents=True)
        for name, content in files.items():
            (data / "skills" / skill / name).write_text(json.dumps(content))
    regression = tmp_path / "eval" / "regression.json"
    regression.write_text(json.dumps({**REGRESSION, "rows": [{"reply": "general answers"}]}))
    logged = {}
    run = types.SimpleNamespace(summary={}, url="https://wandb.ai/me/trained-eye/runs/z",
                                log=logged.update, finish=lambda: None)
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(
        init=lambda **kw: run, Table=lambda columns, data: None, Html=lambda html: None))
    results = run_eval(plan_path, tmp_path / "out", log_wandb=True, data_dir=data,
                       regression=regression)

    judges = {j["variant"]: j for j in results["experiment4"]}
    assert list(judges) == ["finetuned", "base", "spark"]
    assert judges["finetuned"]["skills"] == ["burner_left_on", "fridge_left_open"] and judges[
        "finetuned"]["highlight"]
    assert judges["finetuned"]["metrics"]["frames"] == 120 + 240  # the counts added up
    assert judges["finetuned"]["metrics"]["clips"] == 2
    by_name = {e["name"]: e for e in results["evaluations"]}
    hours = [by_name[n]["metrics"]["hours"] for n in ("ft-w", "ft-b")]
    usd = [by_name[n]["cost_per_hour"]["total_usd"] for n in ("ft-w", "ft-b")]
    assert judges["finetuned"]["usd_per_hour"] == round(
        (usd[0] * hours[0] + usd[1] * hours[1]) / sum(hours), 4)  # weighted by footage
    assert (judges["base"]["skills"], judges["base"]["of"]) == (["burner_left_on"], 2)
    assert by_name["ft-w"]["threshold"] == {"values": [0.6], "sources": []}
    svg = (tmp_path / "out" / "exp4_cost.svg").read_text()
    assert svg.count('class="mark ring"') == 1 and svg.count('class="neu ring"') == 2
    assert "base (1 of 2 skills)" in svg

    assert results["spark_errors"]["burner_left_on"]["spark"]["errors"] == 19
    assert "unmatched_files" not in results["spark_errors"]["burner_left_on"]
    assert results["spark_errors"]["fridge_left_open"] is None is results["label_stats"]["fridge_left_open"]
    assert "by_kind" not in results["label_stats"]["burner_left_on"]
    assert results["regression"]["questions"] == 2 and "rows" not in results["regression"]
    md = (tmp_path / "out" / "results.md").read_text()
    assert "## Regression check" in md and "19 of 190 (10.0%" in md
    summary = run.summary
    assert summary["exp4-pooled/finetuned/frames"] == 360
    assert summary["spark_errors/burner_left_on/spark_rate"] == 0.1
    assert summary["spark_errors/burner_left_on/spark_ci95_low"] == 0.065
    assert summary["spark_errors/burner_left_on/kept_n"] == 150
    assert summary["label_stats/burner_left_on/agreement"] == 0.8
    assert (summary["regression/base_accuracy"], summary["regression/finetuned_accuracy"],
            summary["regression/changed"]) == (1.0, 0.5, 1)
    assert summary["ft-w/threshold"] == 0.6
    assert not any(k.startswith("spark_errors/fridge_left_open") for k in summary)

    regression.write_text(json.dumps({"precision": "bf16", "base": {}}))
    with pytest.raises(ValueError, match="is not te regress's report"):
        run_eval(plan_path, tmp_path / "out", data_dir=data, regression=regression)
    regression.write_text("{")
    with pytest.raises(ValueError, match="cannot be read"):
        run_eval(plan_path, tmp_path / "out", regression=regression)
