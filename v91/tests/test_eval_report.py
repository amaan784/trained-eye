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
    rows = [{"t": float(t), "skill": "washer_done", "threshold": 0.6,
             "p": 0.9 if t in over else 0.2, "over": t in over, "fires": t == alert_at,
             "latency_s": 0.4, **({"alert": True, "cloud": "yes"} if t == alert_at else {})}
            for t in range(n)]
    (folder / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (folder / f"{name}.json").write_text(json.dumps({"true": [list(true)]}))
    return {"log": f"../clips/{name}.jsonl", "labels": f"../clips/{name}.json"}  # from eval/


def evaluation(name, experiment, clip, **extra):
    return {"name": name, "experiment": experiment, "skill": "washer_done",
            "clips": [clip], **extra}


@pytest.fixture
def plan(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    mixed = {**write_clip(clips, "mixed", alert_at=62), "timeline": True}
    plan = {"evaluations": [
        evaluation("exp1-real", 1, write_clip(clips, "real", 75, over=range(70, 100)),
                   variant="real only"),
        evaluation("exp1-mixed", 1, mixed, variant="mixed", alert_text="Laundry's done."),
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
    assert results["charts"] == {"1": ["exp1_bars.svg"], "3": ["exp3_washer_done.svg"],
                                 "4": ["exp4_cost.svg"],
                                 "timelines": ["timeline_exp1-mixed_1.svg"]}
    for names in results["charts"].values():
        for name in names:
            ET.fromstring((out / name).read_text())
    assert "not plotted, no numbers yet: base Glimmer" in (out / "exp4_cost.svg").read_text()

    md = (out / "results.md").read_text()
    two = md.split("## Experiment 2")[1].split("## Experiment 3")[0]
    assert "No runs yet: the plan has no evaluations for this experiment." in two
    assert "![Experiment 1: real only, synthetic only, mixed](exp1_bars.svg)" in md
    assert ("| exp1-mixed | mixed | washer_done | 1 | 2.0 min | 120 | 1.00 | 1.00 | 1.00 | "
            "1 of 1 | 1 of 1 | 0.0 | 2.0 s (2.0 s) |") in md
    assert "| exp1-real | real only | washer_done |" in md and "| 0 of 1 | 1 of 1 |" in md
    assert "| 25 | $0.25 |" in md and "| 300 | n/a |" in md  # no generation cost given
    assert ("| exp4-spark | 0.40 s | 0.40 s | 0 | 1 | 30.0 | $10.80 | Mac, Spark checks not "
            "priced in the plan, $0.003 per frame scored |") in md
    assert "- exp4-base: `../clips/base.jsonl`, `../clips/base.json`" in md
    assert "![Live timeline](timeline_exp1-mixed_1.svg)" in md
    assert md.isascii()  # plain text: no em or en dashes either


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
    check([{**evaluation("a", 1, clip), "skill": "Washer Done"}], "snake_case")
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
