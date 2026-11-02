import io
import json

import pytest
from PIL import Image

from backend.dataengine.dataset import Example, write_dataset
from backend.training.axolotl import RECIPE, qlora_config
from backend.training.prepare import bundle


def png(seed):
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (seed % 256, 80, 160)).save(buf, format="PNG")
    return buf.getvalue()


def teach_dataset(root, skill, real=10, synthetic=40):
    examples = [Example(f"{skill}-{src[:4]}-{i:03d}", png(i), "yes" if i % 2 else "no", src,
                        skill, "the display shows End", ["display off"])
                for src, n in (("real", real), ("synthetic", synthetic)) for i in range(n)]
    write_dataset(examples, root / skill / "dataset", f"{skill}-teach")
    return root / skill


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_bundle_mixes_skills_and_points_images_at_the_volume(tmp_path):
    skills = [teach_dataset(tmp_path / "skills", "washer_done"),
              teach_dataset(tmp_path / "skills", "pot_boils", real=6, synthetic=6)]
    manifest = bundle(skills, tmp_path / "out", "/vol/bundles/exp1-mixed")
    train, val = read(tmp_path / "out/train.jsonl"), read(tmp_path / "out/val.jsonl")
    assert manifest["train_examples"] + manifest["val_examples"] == 62 == len(train) + len(val)
    assert 0 < len(val) < 15
    for record in train + val:
        path = record["messages"][-2]["content"][0]["path"]
        assert path.startswith("/vol/bundles/exp1-mixed/images/")
        assert (tmp_path / "out/images" / path.rsplit("/", 1)[1]).is_file()
    assert set(manifest["skills"]) == {"washer_done", "pot_boils"}
    again = bundle(skills, tmp_path / "out2", "/vol/bundles/exp1-mixed")
    assert again == manifest  # deterministic: same inputs, same split and hashes


@pytest.mark.parametrize("variant, sources", [("real", {"real"}), ("synthetic", {"synt"})])
def test_variants_keep_only_their_source(tmp_path, variant, sources):
    skill = teach_dataset(tmp_path / "skills", "washer_done")
    manifest = bundle([skill], tmp_path / "out", "/vol/b", variant=variant)
    records = read(tmp_path / "out/train.jsonl")
    if (tmp_path / "out/val.jsonl").exists():
        records += read(tmp_path / "out/val.jsonl")
    assert {r["id"].split("-")[1] for r in records} == sources
    assert all(f"/{variant}/" in k for k in manifest["counts"])


def test_limit_takes_a_fixed_subset_per_skill_and_small_sets_skip_validation(tmp_path):
    skill = teach_dataset(tmp_path / "skills", "washer_done")
    small = bundle([skill], tmp_path / "out", "/vol/b", limit=12)
    assert small["train_examples"] == 12 and "val_examples" not in small
    assert not (tmp_path / "out/val.jsonl").exists()
    ids = [r["id"] for r in read(tmp_path / "out/train.jsonl")]
    bigger = bundle([skill], tmp_path / "out", "/vol/b", limit=20)
    assert bigger["train_examples"] == 20
    assert set(ids) <= {r["id"] for r in read(tmp_path / "out/train.jsonl")}  # nested subsets


def test_bundle_refuses_stale_or_missing_datasets(tmp_path):
    with pytest.raises(ValueError, match="no dataset yet"):
        bundle([tmp_path / "skills/none"], tmp_path / "out", "/vol/b")
    skill = teach_dataset(tmp_path / "skills", "washer_done")
    jsonl = skill / "dataset/washer_done-teach.jsonl"
    jsonl.write_text(jsonl.read_text().replace('"no"', '"yes"'))
    with pytest.raises(ValueError, match="does not match its manifest"):
        bundle([skill], tmp_path / "out", "/vol/b")
    with pytest.raises(ValueError, match="absolute"):
        bundle([skill], tmp_path / "out", "vol/b")
    with pytest.raises(ValueError, match="variant"):
        bundle([skill], tmp_path / "out", "/vol/b", variant="best")


def test_the_config_is_meta_s_recipe_with_this_run_s_paths():
    config = qlora_config("exp1-mixed", "/vol/bundles/exp1-mixed/train.jsonl",
                          "/vol/runs/exp1-mixed", val_jsonl="/vol/bundles/exp1-mixed/val.jsonl")
    assert {k: config[k] for k in RECIPE} == RECIPE
    assert config["base_model"] == "meta-models/Muse-Glimmer-30B"
    assert config["model_revision"] == "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
    assert config["gradient_accumulation_steps"] == 32  # global batch 32 on one H100
    assert config["datasets"][0]["path"] == "/vol/bundles/exp1-mixed/train.jsonl"
    assert config["test_datasets"][0]["split"] == "train"
    assert config["wandb_project"] == "trained-eye" and config["wandb_name"] == "exp1-mixed"
    assert config["wandb_log_model"] is None and "max_steps" not in config
    assert json.loads(json.dumps(config)) == config  # written as JSON, which YAML reads
    smoke = qlora_config("s", "/vol/t.jsonl", "/vol/o", smoke=True)
    assert smoke["max_steps"] == 2 and smoke["gradient_accumulation_steps"] == 1
    assert "test_datasets" not in smoke
    with pytest.raises(ValueError, match="absolute"):
        qlora_config("x", "train.jsonl", "/vol/o")


def test_optimizer_steps_at_global_batch_32():
    from backend.training.axolotl import optimizer_steps

    assert [optimizer_steps(n) for n in (1, 32, 33, 300)] == [1, 1, 2, 10]


def test_bundle_can_read_the_unfiltered_dataset_folder(tmp_path):
    skill = teach_dataset(tmp_path / "skills", "washer_done")
    (skill / "dataset").rename(skill / "dataset-unfiltered")
    with pytest.raises(ValueError, match="no dataset yet.*te teach for it first"):
        bundle([skill], tmp_path / "out", "/vol/b")
    plain = teach_dataset(tmp_path / "skills", "pot_boils")
    with pytest.raises(ValueError, match="te unfiltered --skill pot_boils first"):
        bundle([plain], tmp_path / "out", "/vol/b", dataset_dir="dataset-unfiltered")
    manifest = bundle([skill], tmp_path / "out", "/vol/b", dataset_dir="dataset-unfiltered")
    assert manifest["dataset_dir"] == "dataset-unfiltered" and manifest["train_examples"]
    with pytest.raises(ValueError, match="dataset_dir"):
        bundle([skill], tmp_path / "out", "/vol/b", dataset_dir="../elsewhere")
