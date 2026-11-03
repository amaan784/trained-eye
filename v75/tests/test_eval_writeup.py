import xml.etree.ElementTree as ET

from backend.eval.metrics import ClipCounts, cost_per_hour, summarize
from backend.eval.writeup import judge_label, render, threshold_cell, write_charts

ONE_HOUR = ClipCounts(clips=1, frames=3600, tp=20, fp=5, fn=5, tn=3570, seconds=3600.0,
                      events=2, caught_in_window=1, caught=2, alerts=3, false_alarms=1,
                      cloud_checks=3, delays=[3.0, 9.0], latencies=[0.3] * 3600)


def entry(name, experiment, variant, counts=ONE_HOUR, skill="washer_done", **extra):
    prices = {"gpu_usd_per_hour": None, "spark_usd_per_check": 0.003,
              "scorer_usd_per_frame": 0.0, **{k: extra.pop(k) for k in list(extra)
                                              if k.endswith(("_hour", "_check", "_frame"))}}
    m = summarize([counts])
    return {"name": name, "experiment": experiment, "skill": skill, "variant": variant,
            "status": "scored", "metrics": m, "cost_per_hour": cost_per_hour(m, **prices),
            "images": None, "generation_usd": None, "highlight": False, "alert_text": "alert",
            "clips": [{"log": "a.jsonl", "labels": "a.json", "timeline": False}], "gpu": None,
            **prices, **extra}


def pending(name, experiment, variant):
    return {"name": name, "experiment": experiment, "skill": "washer_done",
            "variant": variant, "status": "pending", "missing": ["../data/watch/x.jsonl"],
            "highlight": False, "images": None, "generation_usd": None, "clips": []}


def results(*evaluations, charts=None):
    return {"plan": "plan.json", "window_s": 5.0, "evaluations": list(evaluations),
            "charts": charts or {}}


def section(md, title):
    return md.split(f"## {title}")[1].split("\n## ")[0]


def test_tables_say_what_was_measured_and_nothing_else():
    md = render(results(
        entry("exp1-mixed", 1, "mixed | all", gpu_usd_per_hour=3.95),
        entry("exp1-real", 1, "real", counts=ClipCounts(clips=1, frames=10, tn=10, seconds=10)),
        pending("exp4-base", 4, "base Glimmer"),
        entry("exp4-spark", 4, "Spark", scorer_usd_per_frame=0.003, spark_usd_per_check=0.0)))
    one = section(md, "Experiment 1")
    assert ("| exp1-mixed | mixed \\| all | washer_done | 1 | 60.0 min | 3600 | 0.80 | 0.80 "
            "| 0.80 | 1 of 2 | 2 of 2 | 1.0 | 6.0 s (6.0 s) |") in one
    assert "| exp1-real | real | washer_done | 1 | 0.2 min | 10 | n/a | n/a | n/a | no events" in one
    assert "| exp1-mixed | 0.30 s | 0.30 s | 0 | 3 | 3.0 | $3.96 | GPU $3.95 per hour, " \
           "$0.003 per Spark check |" in one
    assert "| exp1-real | n/a | n/a | 0 | 0 | 0.0 | $0.00 | Mac, $0.003 per Spark check |" in one
    for title in ("Experiment 2", "Experiment 3"):
        assert "No runs yet: the plan has no evaluations" in section(md, title)
    four = section(md, "Experiment 4")
    assert "| Frame accuracy |" in four and "| 6.0 s (6.0 s) | n/a | 1.00 |" in four  # 3590/3600
    assert "| $10.80 | Mac, Spark checks not priced in the plan, $0.003 per frame scored |" in four
    assert "- exp4-base: `../data/watch/x.jsonl`" in four
    assert "Live timelines" not in md and "W&B" not in md
    assert md.isascii() and "--" not in md.replace("|---", "")


def test_an_experiment_with_only_pending_runs_says_so():
    md = render(results(pending("exp3-25", 3, "25 images")))
    three = section(md, "Experiment 3")
    assert "No runs scored yet." in three and "- exp3-25:" in three and "|" not in three


def test_charts_follow_the_experiments(tmp_path):
    rows = results(
        entry("exp2-filtered", 2, "filtered"), pending("exp2-unfiltered", 2, "unfiltered"),
        entry("exp2-bus", 2, "filtered", skill="bus_m4"),
        entry("exp3-25", 3, "25", images=25, generation_usd=0.25),
        entry("exp3-50", 3, "50", images=50),
        pending("exp1-real", 1, "real"))
    charts = write_charts(rows, tmp_path, tmp_path)
    assert charts == {"2": ["exp2_bars.svg"], "3": ["exp3_washer_done.svg"]}  # none for exp 1
    bars = ET.fromstring((tmp_path / "exp2_bars.svg").read_text())
    labels = ["".join(t.itertext()) for t in bars.iter("{http://www.w3.org/2000/svg}text")]
    assert {"filtered (washer_done)", "unfiltered (washer_done)", "filtered (bus_m4)"} <= set(labels)
    assert labels.count("n/a") == 2  # the pending run: no F1, no event recall
    line = (tmp_path / "exp3_washer_done.svg").read_text()
    assert "$0.25" in line and "cost n/a" in line
    md = render({**rows, "charts": charts})
    assert "![Experiment 2: labels filtered by Muse Spark, or not](exp2_bars.svg)" in md


SPARK_ERRORS = {"hand_labelled": 200, "unmatched": 0, "human_unsure": 4, "spark_failed": 1,
                "spark": {"n": 190, "errors": 19, "rate": 0.1, "ci95": [0.065, 0.151],
                          "unsure": 5},
                "intended": {"n": 195, "errors": 39, "rate": 0.2, "ci95": [0.15, 0.262]},
                "kept": {"n": 150, "errors": 6, "rate": 0.04, "ci95": [0.018, 0.085]}}
REGRESSION = {"precision": "bf16", "questions": 2, "changed": [{"id": "q2", "base": True,
                                                                "finetuned": False}],
              **{side: {"model": model, "n": 2, "correct": right, "accuracy": right / 2,
                        "with_image": {"n": 1, "correct": 1, "accuracy": 1.0},
                        "text_only": {"n": 1, "correct": right - 1,
                                      "accuracy": float(right - 1)},
                        "failed": [], "cut_off": []}
                 for side, model, right in (("base", "muse-glimmer", 2),
                                            ("finetuned", "trained-eye", 1))}}


def test_experiment_2_gets_spark_s_error_rate_and_the_regression_check_its_section():
    md = render({**results(entry("exp2-filtered", 2, "filtered")),
                 "spark_errors": {"washer_done": SPARK_ERRORS, "boil_over": None},
                 "label_stats": {"washer_done": {"total": 250, "kept": 200, "agreement": 0.8},
                                 "boil_over": None},
                 "regression": REGRESSION})
    two = section(md, "Experiment 2")
    assert "95% Wilson intervals" in two
    assert ("| washer_done | 200 | 19 of 190 (10.0%, 95% CI 6.5% to 15.1%) | 5 | 39 of 195 "
            "(20.0%, 95% CI 15.0% to 26.2%) | 6 of 150 (4.0%, 95% CI 1.8% to 8.5%) | 200 of "
            "250 kept (80.0%) |") in two
    assert "| boil_over | not measured yet (te label --skill, then te spark-errors) |" in two
    regression = section(md, "Regression check")
    assert "| All (2) | 2/2 (100%) | 1/2 (50%) |" in regression
    assert "Changed: q2 (right to wrong)." in regression and "eval/regression.json" in regression
    empty = render({**results(), "regression": None, "spark_errors": {}, "label_stats": {}})
    assert "Not run yet: te regress writes eval/regression.json." in section(
        empty, "Regression check")
    assert "Regression check" not in render(results())  # te eval was not given one
    assert md.isascii()


def test_the_threshold_and_galleries_are_in_results_md():
    scored = entry("exp1-mixed", 1, "mixed", threshold={"values": [0.62], "sources": ["run"]})
    assert threshold_cell(scored) == "0.62 (run)"
    assert threshold_cell({"threshold": {"values": [0.5, 0.62], "sources": ["default", "run"]}}
                          ) == "0.50 to 0.62 (default, run)"
    assert threshold_cell({"threshold": {"values": [0.6], "sources": []}}) == "0.60"
    assert threshold_cell({}) == "n/a"
    md = render({**results(scored), "galleries": {"exp1-mixed": {
        "page": "galleries/exp1-mixed/index.html", "false_alarms": 2, "misses": 3}}})
    assert "| 6.0 s (6.0 s) | 0.62 (run) |" in section(md, "Experiment 1")
    galleries = section(md, "Failure galleries (local only)")
    assert "run te blur on any frame before publishing it" in galleries
    assert ("- [exp1-mixed](galleries/exp1-mixed/index.html): 2 false alarms and 3 misses "
            "shown") in galleries


def test_experiment_4_draws_one_point_per_judge(tmp_path):
    pooled = [{"variant": "finetuned", "skills": ["boil_over", "washer_done"], "of": 2,
               "highlight": True, "usd_per_hour": 3.96, "metrics": summarize([ONE_HOUR] * 2)},
              {"variant": "base", "skills": ["washer_done"], "of": 2, "highlight": False,
               "usd_per_hour": 3.95, "metrics": summarize([ONE_HOUR])},
              {"variant": "spark", "skills": [], "of": 2, "highlight": False,
               "usd_per_hour": None, "metrics": None}]
    assert [judge_label(j) for j in pooled] == ["finetuned", "base (1 of 2 skills)", "spark"]
    rows = {**results(entry("exp4-finetuned-washer_done", 4, "finetuned"),
                      entry("exp4-finetuned-boil_over", 4, "finetuned", skill="boil_over")),
            "experiment4": pooled}
    charts = write_charts(rows, tmp_path, tmp_path)
    assert charts == {"4": ["exp4_cost.svg"]}
    svg = (tmp_path / "exp4_cost.svg").read_text()
    root = ET.fromstring(svg)
    dots = [c.get("class") for c in root.iter("{http://www.w3.org/2000/svg}circle")
            if "ring" in (c.get("class") or "")]
    assert dots == ["acc ring", "neu ring"]  # fine-tuned highlighted; spark has no numbers
    assert "Accuracy as frame F1, pooled over the skills" in svg
    assert "not plotted, no numbers yet: spark" in svg
    four = section(render({**rows, "charts": charts}), "Experiment 4")
    assert "Pooled over the skills" in four
    assert "| finetuned | boil_over, washer_done | 7200 | 0.80 | 1.00 | 2 of 4 | 1.0 | $3.96 |" \
        in four
