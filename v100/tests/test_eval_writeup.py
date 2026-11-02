import xml.etree.ElementTree as ET

from backend.eval.metrics import ClipCounts, cost_per_hour, summarize
from backend.eval.writeup import render, write_charts

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
    assert "| Frame accuracy |" in four and "| 6.0 s (6.0 s) | 1.00 |" in four  # 3590 of 3600
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
