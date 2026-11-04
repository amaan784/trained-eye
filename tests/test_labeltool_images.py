import json

import pytest
from PIL import Image
from starlette.testclient import TestClient

from backend.labeltool.images import create_image_app, sample_images
from backend.skills.spec import SkillSpec

SPEC = SkillSpec(name="burner_left_on", request="Tell me when the burner is left on", alert="Done.",
                 target={"sam_prompt": "stove burner", "description": "the stove burner"},
                 positive="The burner is lit with no pan on it", negatives=["the burner is off"])


@pytest.fixture
def skill(tmp_path):
    root = tmp_path / "skills" / "burner_left_on"
    (root / "synthetic").mkdir(parents=True)
    (root / "spec.json").write_text(SPEC.model_dump_json())
    rows = [{"file": "keyframes/kf_00.jpg", "source": "real", "spark_label": "yes"}]
    for i in range(10):
        Image.new("RGB", (8, 8), (i * 20, 0, 0)).save(root / "synthetic" / f"v{i:03d}.png")
        rows.append({"file": f"synthetic/v{i:03d}.png", "source": "synthetic",
                     "label": "yes" if i % 2 else "no", "spark_label": "no"})
    (root / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "reversed.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[::-1]))
    return root


def test_sample_images_is_deterministic_and_synthetic_only(skill):
    picked = sample_images(skill / "labels.jsonl", 5, seed=7)
    assert picked == sample_images(skill / "labels.jsonl", 5, seed=7)
    assert picked == sample_images(skill / "reversed.jsonl", 5, seed=7)  # row order is moot
    assert picked != sample_images(skill / "labels.jsonl", 5, seed=8)
    assert len(set(picked)) == 5 and all(p.parent == skill / "synthetic" for p in picked)
    everything = sample_images(skill / "labels.jsonl", 10, seed=1)
    assert sorted(p.name for p in everything) == [f"v{i:03d}.png" for i in range(10)]
    for n in (0, 11):
        with pytest.raises(ValueError, match="10 generated images"):
            sample_images(skill / "labels.jsonl", n, seed=1)


def test_page_shows_the_question_and_images_but_not_spark(skill, tmp_path):
    images = sample_images(skill / "labels.jsonl", 3, seed=0)
    client = TestClient(create_image_app(images, tmp_path / "human.json"))
    assert client.get("/").status_code == 200 and "Inter" in client.get("/static/style.css").text
    state = client.get("/state").json()
    assert state["question"].startswith("Condition: The burner is lit with no pan on it. Not: the burner is off")
    assert state["images"] == [p.name for p in images] and state["labels"] == {}
    assert "spark" not in json.dumps(state)
    assert client.get("/image/2").content == images[2].read_bytes()
    assert client.get("/image/3").status_code == 404
    other = TestClient(create_image_app(images, tmp_path / "h.json", question="Is it red?"))
    assert other.get("/state").json()["question"] == "Is it red?"


def test_every_answer_is_saved_and_the_page_resumes(skill, tmp_path):
    images = sample_images(skill / "labels.jsonl", 3, seed=0)
    names, out = [p.name for p in images], tmp_path / "human.json"
    out.write_text(json.dumps({"labels": {"old.png": "no"}}))  # kept, but not shown
    client = TestClient(create_image_app(images, out))
    reply = client.post("/answer", json={"file": names[0], "label": "yes"})
    assert reply.json() == {"done": 1, "total": 3}
    assert json.loads(out.read_text()) == {"labels": {"old.png": "no", names[0]: "yes"}}
    client.post("/answer", json={"file": names[1], "label": "unsure"})
    client.post("/answer", json={"file": names[0], "label": "no"})  # back, and changed
    for bad in ({"file": names[2], "label": "maybe"}, {"file": "old.png", "label": "yes"},
                {"file": names[2]}, ["yes"], "{"):
        raw = bad.encode() if isinstance(bad, str) else json.dumps(bad).encode()
        assert client.post("/answer", content=raw).status_code == 400
    resumed = TestClient(create_image_app(images, out)).get("/state").json()
    assert resumed["labels"] == {names[0]: "no", names[1]: "unsure"}  # old.png stays in the file
    assert json.loads(out.read_text())["labels"] == {"old.png": "no", **resumed["labels"]}


def test_refuses_bad_image_lists(skill, tmp_path):
    image, copy = skill / "synthetic" / "v000.png", tmp_path / "copy" / "v000.png"
    copy.parent.mkdir()
    copy.write_bytes(image.read_bytes())
    for images in ([], [image, copy]):
        with pytest.raises(ValueError):
            create_image_app(images, tmp_path / "h.json")
    with pytest.raises(FileNotFoundError):
        create_image_app([image, skill / "synthetic" / "gone.png"], tmp_path / "h.json")
    no_spec = TestClient(create_image_app([copy], tmp_path / "h.json"))  # no spec.json above
    assert no_spec.get("/state").json()["question"] == ""
