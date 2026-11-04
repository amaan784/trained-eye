"""te eval --readme (backend/eval/readme.py): the README's results block, from results.json
only, between its two markers."""

import hashlib
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


def test_the_block_copies_results_json_and_names_its_hash(scored, tmp_path):
    readme = tmp_path / "README.md"
    text = block(scored, readme)
    digest = hashlib.sha256(scored.read_bytes()).hexdigest()
    assert f"from `eval/results.json` (SHA-256 `{digest}`): 4 of 5 evaluations scored" in text
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
