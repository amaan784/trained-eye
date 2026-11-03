"""Experiment 2: Spark's error rate against hand labels, and the unfiltered dataset."""

import json

import pytest

from backend.eval.spark_errors import spark_error_rate, wilson, write_unfiltered_dataset
from backend.teach import TeachOptions, run_teach
from backend.training.prepare import bundle
from tests.test_teach import FakeMeta, client_for, clip


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def synthetic(name, label, spark, error=False):
    return {"file": f"synthetic/{name}", "source": "synthetic", "label": label, "kind": "x",
            "spark_label": spark, "label_error": error}


def test_error_rates_against_the_hand_labels(tmp_path):
    labels = write_jsonl(tmp_path / "labels.jsonl", [
        synthetic("s1.png", "yes", "yes"),
        synthetic("s2.png", "yes", "no"),
        synthetic("s3.png", "no", "no"),  # kept by the filter, but the author says yes
        synthetic("s4.png", "no", "unsure"),
        synthetic("s5.png", "yes", "unsure", error=True),
        synthetic("s6.png", "yes", "yes"),
        {"file": "keyframes/kf_00_1a2b3c4d.jpg", "source": "real", "human_label": None,
         "spark_label": "no", "label_error": False},
    ])
    human = tmp_path / "human.json"
    human.write_text(json.dumps({"labels": {
        "s1.png": "yes", "s2.png": "yes", "s3.png": "yes", "s4.png": "no", "s5.png": "no",
        "s6.png": "unsure", "kf_00_1a2b3c4d.jpg": "no", "ghost.png": "yes"}}))
    result = spark_error_rate(labels, human)
    assert (result["hand_labelled"], result["unmatched"], result["unmatched_files"]) == (
        8, 1, ["ghost.png"])
    assert result["human_unsure"] == 1 and result["spark_failed"] == 1
    assert result["confusion"] == {"yes": {"yes": 1, "no": 2, "unsure": 0},
                                   "no": {"yes": 0, "no": 1, "unsure": 1}}
    spark = result["spark"]
    assert (spark["n"], spark["errors"], spark["rate"]) == (4, 2, 0.5)
    assert (spark["unsure"], spark["unsure_rate"]) == (1, 0.2)
    assert spark["ci95"] == wilson(2, 4)
    assert (result["intended"]["n"], result["intended"]["errors"]) == (5, 2)  # s3 and s5
    assert (result["kept"]["n"], result["kept"]["errors"]) == (2, 1)  # s1 right, s3 wrong


def test_wilson_interval():
    assert wilson(2, 10) == [0.0567, 0.5098]
    assert wilson(0, 20) == [0.0, 0.1611]
    assert wilson(0, 0) is None


@pytest.mark.parametrize("content, message", [
    ({"labels": {"s1.png": "maybe"}}, "yes, no or unsure"),
    ({"s1.png": "yes"}, "expected"),
    ({"labels": {"other.png": "yes"}}, "none of the 1"),
])
def test_unusable_hand_labels_are_refused(tmp_path, content, message):
    labels = write_jsonl(tmp_path / "labels.jsonl", [synthetic("s1.png", "yes", "yes")])
    (tmp_path / "human.json").write_text(json.dumps(content))
    with pytest.raises((ValueError, TypeError), match=message):
        spark_error_rate(labels, tmp_path / "human.json")


@pytest.fixture
def taught(tmp_path):
    """A finished fake teach run: Spark disagrees with some generated positives."""
    opts = TeachOptions(skill="washer_done", clip=clip(tmp_path / "clip"),
                        request="let me know when the washer says done", positives=12,
                        negatives=12)
    report = run_teach(client_for(tmp_path, FakeMeta()), opts, tmp_path / "skills",
                       confirm=lambda s: True, say=lambda s: None)
    assert report["label_stats"]["kept"] < 24
    return tmp_path / "skills" / "washer_done"


def index(manifest):
    return {e["id"]: e for e in manifest["examples_index"]}


def test_the_unfiltered_arm_keeps_every_generated_image_with_its_intended_label(taught,
                                                                               tmp_path):
    rows = [json.loads(x) for x in (taught / "labels.jsonl").read_text().splitlines()]
    teach = index(json.loads((taught / "dataset/washer_done-teach.manifest.json").read_text()))
    manifest = write_unfiltered_dataset(taught)
    ours = index(manifest)
    assert manifest["name"] == "washer_done-teach"
    assert (taught / "dataset-unfiltered/washer_done-teach.jsonl").is_file()
    real = {k: v for k, v in teach.items() if v["source"] == "real"}
    assert {k: v for k, v in ours.items() if v["source"] == "real"} == real
    intended = sorted(r["label"] for r in rows if r["source"] == "synthetic")
    assert sorted(v["answer"] for v in ours.values() if v["source"] == "synthetic") == intended
    assert len(ours) == len(real) + 24 and set(teach) < set(ours)  # same ids as te teach
    names = {p.name for p in (taught / "dataset/images").iterdir()}
    assert names < {p.name for p in (taught / "dataset-unfiltered/images").iterdir()}
    trained = bundle([taught], tmp_path / "bundle", "/vol/bundles/exp2-unfiltered",
                     dataset_dir="dataset-unfiltered")
    assert trained["train_examples"] + trained.get("val_examples", 0) == len(ours)

    human = tmp_path / "human.json"  # the author agrees with every intended label
    human.write_text(json.dumps({"labels": {r["file"].split("/")[-1]: r["label"]
                                            for r in rows if r["source"] == "synthetic"}}))
    result = spark_error_rate(taught / "labels.jsonl", human)
    assert result["intended"]["errors"] == 0 == result["kept"]["errors"]
    assert result["spark"]["errors"] == 24 - len([v for v in teach.values()
                                                  if v["source"] == "synthetic"])


def test_the_unfiltered_arm_needs_a_finished_teach_run_with_model_outputs(taught, tmp_path):
    with pytest.raises(ValueError, match="model outputs is off"):
        write_unfiltered_dataset(taught, train_on_model_outputs=False)
    (tmp_path / "skills/boil_over").mkdir()
    with pytest.raises(ValueError, match="no finished teach run"):
        write_unfiltered_dataset(tmp_path / "skills/boil_over")


def test_the_unfiltered_arm_puts_the_system_prompt_where_te_teach_did(taught):
    manifest = taught / "dataset/washer_done-teach.manifest.json"
    assert write_unfiltered_dataset(taught, system_in_user=False)["system_in_user"] is False
    teach = json.loads(manifest.read_text())
    manifest.write_text(json.dumps({**teach, "system_in_user": True}))  # as if folded
    folded = write_unfiltered_dataset(taught)  # no setting given: follow te teach
    assert folded["system_in_user"] is True
    record = json.loads((taught / "dataset-unfiltered/washer_done-teach.jsonl").read_text()
                        .splitlines()[0])
    assert [m["role"] for m in record["messages"]] == ["user", "assistant"]
    with pytest.raises(ValueError, match="TE_SYSTEM_IN_USER=1, but it is 0 now"):
        write_unfiltered_dataset(taught, system_in_user=False)
