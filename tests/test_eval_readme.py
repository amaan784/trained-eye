"""te eval --readme (backend/eval/readme.py): the README's results block, from results.json
only, between its two markers."""

import json

import pytest

from backend import cli
from backend.eval.readme import END, START, block, write_readme
from backend.eval.report import run_eval
from tests.test_eval_report import evaluation, write_clip
from tests.test_eval_writeup import REGRESSION, SPARK_ERRORS


@pytest.fixture
def scored(tmp_path):
    """A plan with Experiments 1, 3 and 4 scored (4 over two judges), run through te eval
    into tmp_path/eval/, with Spark's error rate and the regression check."""
    clips = tmp_path / "clips"
    clips.mkdir()
    plan = tmp_path / "eval" / "plan.json"
    plan.parent.mkdir()
    plan.write_text(json.dumps({"evaluations": [
        evaluation("exp1-mixed", 1, write_clip(clips, "mixed", 62), variant="mixed",
                   gpu="H100"),
        evaluation("exp1-real", 1, {"log": "../clips/x.jsonl", "labels": "../clips/x.json"},
                   variant="real"),
        evaluation("exp3-25", 3, write_clip(clips, "n25", None, over=()), images=25),
        evaluation("exp4-finetuned", 4, write_clip(clips, "ft", 62), variant="finetuned",
                   gpu="H100", highlight=True),
        evaluation("exp4-spark", 4, write_clip(clips, "spark", 63), variant="spark",
                   scorer_usd_per_frame=0.003)]}))
    data = tmp_path / "data"
    (data / "skills/burner_left_on").mkdir(parents=True)
    (data / "skills/burner_left_on/spark_errors.json").write_text(json.dumps(SPARK_ERRORS))
    (tmp_path / "eval/regression.json").write_text(json.dumps(REGRESSION))
    run_eval(plan, tmp_path / "eval", data_dir=data, regression=tmp_path / "eval/regression.json")
    return tmp_path / "eval" / "results.json"


def test_the_block_copies_results_json_and_links_the_full_results(scored, tmp_path):
    readme = tmp_path / "README.md"
    text = block(scored, readme)
    assert ("Every number below comes from the evaluation scripts; the full tables are in "
            "[eval/results.md](eval/results.md)." in text)
    assert "SHA-256" not in text  # plain words in the README; the hash stays in the model card
    one = text.split("### Experiment 1")[1].split("###")[0]
    assert "| Variant | Skill | Frame F1 | Events caught on time | False alarms per hour |" \
        in one
    assert "| mixed | burner_left_on | 1.00 | 1.00 | 0.0 | $3.95 |" in one and "| real |" not in one
    assert "| 25 images" not in text and "| 25 | 0.00 |" in text  # Experiment 3's images
    assert "### Experiment 2" not in text  # nothing scored there
    four = text.split("### Experiment 4")[1]
    assert "| Judge (all skills) | Frame F1 | Frame accuracy |" in four
    assert "| finetuned | 1.00 | 1.00 | 1.00 | 0.0 | $3.95 |" in four
    assert "[Experiment 1](eval/exp1_bars.svg)" in text
    assert "[Experiment 4 (cost)](eval/exp4_cost.svg)" in text and "timeline" not in text
    assert ("Regression check on general questions at bf16: base 2 of 2 (100%), fine-tuned 1 "
            "of 2 (50%); 1 question changed.") in text
    assert ("- burner_left_on: wrong on 19 of 190 hand-labelled images (10.0%, 95% CI 6.5% to "
            "15.1%), unsure on 5") in text
    assert "galleries" not in text  # local only: never in the README
    assert text.isascii() and "--" not in text.replace("|---", "").replace("--readme", "")


def test_the_block_names_the_footage_and_a_frame_calibration(scored, tmp_path):
    results = json.loads(scored.read_text())
    assert "Test footage" not in block(scored, tmp_path / "README.md")
    results.update(footage="other kitchens (EPIC-KITCHENS-100, head-mounted camera)",
                   threshold_objective="frames")
    scored.write_text(json.dumps(results))
    text = block(scored, tmp_path / "README.md")
    assert ("Test footage: other kitchens (EPIC-KITCHENS-100, head-mounted camera). Results "
            "on other footage are reported apart.") in text
    assert "calibrated for frame F1" in text


def test_the_block_goes_between_the_markers_and_nothing_else_changes(scored, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# Trained Eye\n\nIntro.\n")
    write_readme(readme, scored)
    first = readme.read_text()
    assert first.startswith("# Trained Eye\n\nIntro.\n\n" + START + "\n## Results\n")
    assert first.endswith(END + "\n")
    edited = first.replace("Intro.", "Intro, edited.") + "\n## Licence\n\nMIT.\n"
    readme.write_text(edited.replace("## Results", "## Old results"))
    write_readme(readme, scored)
    again = readme.read_text()
    assert again == edited  # the block is written afresh, the rest is left alone
    assert again.count(START) == 1 and "## Old results" not in again


def test_it_refuses_with_nothing_scored_or_broken_markers(scored, tmp_path):
    readme = tmp_path / "README.md"
    for broken in (f"{START}\n", f"{END}\n{START}\n", f"{START}\n{END}\n{START}\n{END}\n"):
        readme.write_text(broken)
        with pytest.raises(ValueError, match="markers are broken"):
            write_readme(readme, scored)
        assert readme.read_text() == broken
    results = json.loads(scored.read_text())
    for e in results["evaluations"]:
        e["status"] = "pending"
    scored.write_text(json.dumps(results))
    readme.write_text("# Trained Eye\n")
    with pytest.raises(ValueError, match="no scored evaluation yet"):
        write_readme(readme, scored)
    assert readme.read_text() == "# Trained Eye\n"


def test_te_eval_readme_writes_the_block_or_leaves_the_file(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TE_DATA_DIR", str(tmp_path / "data"))
    clips = tmp_path / "clips"
    clips.mkdir()
    plan = tmp_path / "eval" / "plan.json"
    plan.parent.mkdir()
    plan.write_text(json.dumps({"evaluations": [
        evaluation("exp1-mixed", 1, write_clip(clips, "mixed", 62), variant="mixed")]}))
    readme = tmp_path / "README.md"
    args = ["eval", "--plan", str(plan), "--out", str(tmp_path / "eval")]
    assert cli.main([*args, "--readme", str(tmp_path / "missing.md")]) == 2
    assert "--readme: no file" in capsys.readouterr().err
    readme.write_text("# Trained Eye\n")
    assert cli.main([*args, "--readme", str(readme)]) == 0
    assert "results block written into" in capsys.readouterr().out
    text = readme.read_text()
    assert "| mixed | burner_left_on | 1.00 |" in text
    assert "Regression check: not run yet (te regress)." in text  # no eval/regression.json
    plan.write_text(json.dumps({"evaluations": [
        evaluation("exp1-mixed", 1, {"log": "../clips/x.jsonl", "labels": "../clips/x.json"})]}))
    assert cli.main([*args, "--readme", str(readme)]) == 1
    assert "no scored evaluation yet" in capsys.readouterr().err
    assert readme.read_text() == text  # left as it was


@pytest.mark.parametrize("f1, how", [(0.88, "about as accurately as"),
                                     (0.95, "more accurately than"),
                                     (0.80, "less accurately than")])
def test_the_summary_says_in_one_sentence_how_the_judges_compare(f1, how):
    from backend.eval.readme import summary

    def judge(variant, f, lat, usd):
        return {"variant": variant, "usd_per_hour": usd,
                "metrics": {"f1": f, "latency_median_s": lat}}

    results = {"footage": "other kitchens (EPIC-KITCHENS-100, head-mounted camera)",
               "experiment4": [judge("finetuned", f1, 0.983, 3.95), judge("base", 0.89, 2.071, 3.95),
                               judge("spark", 0.86, 11.156, 16.5276)]}
    text = summary(results)
    assert text == (f"on other kitchens, the fine-tuned model spotted the condition {how} base "
                    f"Glimmer and Muse Spark (frame F1 {f1:.2f}, 0.89 and 0.86), answering in "
                    "0.98 s a frame against 2.07 s and 11.16 s, at $3.95 an hour against "
                    "$16.53 for Spark.")
    assert summary({"experiment4": results["experiment4"][:2]}) is None  # Spark not scored


def test_the_block_starts_with_the_summary_when_experiment_4_is_scored(scored, tmp_path):
    text = block(scored, tmp_path / "README.md")
    assert "In short:" not in text  # the fixture scores no base judge


def test_the_summary_compares_with_each_judge_on_its_own():
    from backend.eval.readme import summary

    def judge(variant, f, lat, usd):
        return {"variant": variant, "usd_per_hour": usd,
                "metrics": {"f1": f, "latency_median_s": lat}}

    # Nov 4's phone clips: close to base, well behind Spark: never "about as accurately as" both
    text = summary({"footage": "other kitchens filmed first-person on phones (handheld)",
                    "experiment4": [judge("finetuned", 0.87, 1.64, 4.0),
                                    judge("base", 0.88, 2.53, 4.05),
                                    judge("spark", 0.94, 11.8, 16.53)]})
    assert text.startswith("on other kitchens filmed first-person on phones, the fine-tuned "
                           "model spotted the condition about as accurately as base Glimmer "
                           "and less accurately than Muse Spark (frame F1 0.87, 0.88 and 0.94)")


def test_a_second_footage_gets_its_own_block_and_heading(scored, tmp_path):
    from backend.eval.readme import END, START, markers, write_readme

    readme = tmp_path / "README.md"
    readme.write_text("# Title\n")
    write_readme(readme, scored)
    write_readme(readme, scored, section="Results on EPIC-KITCHENS-100")
    text = readme.read_text()
    start, end = markers("Results on EPIC-KITCHENS-100")
    assert text.count(START) == 1 and text.count(END) == 1  # the main block untouched
    assert text.count(start) == 1 and text.count(end) == 1
    assert "## Results on EPIC-KITCHENS-100" in text.split(start)[1]
    write_readme(readme, scored, section="Results on EPIC-KITCHENS-100")
    assert readme.read_text() == text  # rewriting changes nothing


def test_per_skill_shows_each_skill_and_a_default_threshold():
    """Experiment 4's frame F1 per skill, so a pooled number cannot hide a skill that went
    the other way; a skill run at the default threshold says so; one skill gives None."""
    from backend.eval.readme import per_skill

    def row(variant, skill, f1, source="run", value=0.8):
        return {"experiment": 4, "status": "scored", "variant": variant, "skill": skill,
                "metrics": {"f1": f1}, "threshold": {"values": [value], "sources": [source]}}

    rows = [row("finetuned", "fridge", 0.95), row("base", "fridge", 0.91),
            row("spark", "fridge", 0.92, "spark", 1.0),
            row("finetuned", "burner", 0.64, "default", 0.5),
            row("base", "burner", 0.79, "default", 0.5),
            row("spark", "burner", 0.97, "spark", 1.0)]
    line = per_skill({"evaluations": rows})
    assert line == ("By skill, the fine-tuned model was the least accurate of the three on "
                    "burner (frame F1 0.64, against 0.79 for base Glimmer and 0.97 for Muse "
                    "Spark) and the most accurate of the three on fridge (frame F1 0.95, "
                    "against 0.91 and 0.92). burner has no calibration clip, so the fine-tuned "
                    "model runs it at the default threshold, 0.50.")
    assert per_skill({"evaluations": rows[:3]}) is None
    assert per_skill({"evaluations": rows[:5]}) is None  # burner lacks Spark
