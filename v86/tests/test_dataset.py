import io
import json

import pytest
from PIL import Image

from backend.dataengine.dataset import (
    IMAGE_SIZE,
    SYSTEM_PROMPT,
    Example,
    pad_image,
    user_prompt,
    write_dataset,
)
from backend.skills.spec import SkillSpec, Target

SPEC = SkillSpec(
    name="washer_done",
    request="let me know when the washer says done",
    target=Target(sam_prompt="washing machine display", description="the washer display"),
    positive="the display shows the cycle has finished (e.g. 'End' or '0:00')",
    negatives=["display shows time remaining", "display off", "a different appliance"],
    alert="Your laundry's done.",
)


def jpeg(size=(1280, 720), color=(90, 90, 90)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def test_pad_image_keeps_aspect_with_black_bars():
    img = pad_image(jpeg((1280, 720), (200, 200, 200)))
    assert img.size == (IMAGE_SIZE, IMAGE_SIZE) and img.mode == "RGB"
    assert img.getpixel((256, 5)) == (0, 0, 0)  # top bar
    assert img.getpixel((256, 256))[0] > 150  # picture in the middle


def test_prompt_template_matches_spec_md():
    assert user_prompt("the pot is about to boil over.", ["a gentle simmer", "steam only."]) == (
        "Condition: the pot is about to boil over. Not: a gentle simmer; steam only. "
        "Is the condition true in this frame?")


def test_write_dataset_axolotl_format(tmp_path):
    examples = [
        Example("washer_done-real-kf_00", jpeg(), "no", "real", "washer_done",
                SPEC.positive, SPEC.negatives),
        Example("washer_done-synt-pos-000", jpeg(color=(0, 200, 0)), "yes", "synthetic",
                "washer_done", SPEC.positive, SPEC.negatives),
    ]
    manifest = write_dataset(examples, tmp_path / "ds", "washer_done-teach")
    lines = (tmp_path / "ds" / "washer_done-teach.jsonl").read_text().splitlines()
    assert len(lines) == 2 and manifest["examples"] == 2
    record = json.loads(lines[1])
    assert record["id"] == "washer_done-synt-pos-000"
    system, user, assistant = record["messages"]
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert [p["type"] for p in user["content"]] == ["image", "text"]
    image_path = user["content"][0]["path"]
    assert image_path.startswith("/") and image_path.endswith(".png")
    assert Image.open(image_path).size == (IMAGE_SIZE, IMAGE_SIZE)
    assert assistant == {"role": "assistant", "content": "yes"}
    assert manifest["counts"] == {"washer_done/real/no": 1, "washer_done/synthetic/yes": 1}
    assert [e["source"] for e in manifest["examples_index"]] == ["real", "synthetic"]
    assert len(manifest["examples_index"][1]["original_sha256"]) == 64
    assert len(manifest["jsonl_sha256"]) == 64

    folded = write_dataset(examples, tmp_path / "ds2", "x", system_in_user=True)
    record = json.loads((tmp_path / "ds2" / "x.jsonl").read_text().splitlines()[0])
    assert record["messages"][0]["role"] == "user" and folded["system_in_user"] is True
    assert record["messages"][0]["content"][1]["text"].startswith(SYSTEM_PROMPT)


def test_write_dataset_rejects_bad_answers(tmp_path):
    bad = Example("x", jpeg(), "unsure", "real", "s", "p", ["n"])
    with pytest.raises(ValueError):
        write_dataset([bad], tmp_path, "x")
    with pytest.raises(ValueError):
        write_dataset([], tmp_path, "x")


def test_rerun_rewrites_images(tmp_path):
    first = Example("washer_done-real-kf_00", jpeg(color=(10, 10, 10)), "no", "real",
                    "washer_done", SPEC.positive, SPEC.negatives)
    write_dataset([first], tmp_path / "ds", "x")
    second = Example(first.id, jpeg(color=(250, 250, 250)), "yes", "real", "washer_done",
                     SPEC.positive, SPEC.negatives)
    write_dataset([second], tmp_path / "ds", "x")
    png = Image.open(tmp_path / "ds" / "images" / f"{first.id}.png")
    assert png.getpixel((256, 256))[0] > 200
