"""The whole teach flow against a fake Meta API (httpx.MockTransport): nothing is sent
or billed. The fake image editor paints positives green and negatives red, and the fake
labeler reads the colour back, with one deliberate mistake to exercise filtering."""

import base64
import io
import json
import wave

import httpx
import numpy as np
from PIL import Image

from backend.config import load_settings
from backend.meta.client import MetaClient
from backend.skills.compiler import compile_skill
from backend.teach import TeachOptions, estimate, run_teach

SPEC_REPLY = {
    "refused": False, "refusal_reason": None, "name": "washer_done",
    "sam_prompt": "washing machine display",
    "target_description": "the digital display on the front of the washer",
    "positive": "the display shows the cycle has finished (e.g. 'End' or '0:00')",
    "negatives": ["display shows time remaining", "display off"],
    "alert": "Your laundry's done.", "cooldown_s": 600, "confirm_with_cloud": True,
}


def png(color, serial=0) -> bytes:
    img = Image.new("RGB", (64, 64), color)
    img.putpixel((0, 0), (serial % 256, serial // 256, 0))  # every generated image differs
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def chat(content: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1600, "completion_tokens": 120}}


class FakeMeta:
    def __init__(self):
        self.calls = {}
        self.edits = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/v1")
        self.calls[path] = self.calls.get(path, 0) + 1
        if path == "/audio/transcriptions":
            return httpx.Response(200, json={"text": "let me know when the washer says done",
                                             "duration": 3.0})
        if path == "/responses":
            return httpx.Response(200, json={"output": [{"content": [
                {"type": "output_text", "text": "<sam raw output>"}]}]})
        if path == "/images/edits":
            self.edits += 1
            negative = b"clearly not the case" in request.content
            color = (220, 0, 0) if negative else (0, 220, 0)
            return httpx.Response(200, json={"data": [
                {"b64_json": base64.b64encode(png(color, self.edits)).decode()}]})
        body = json.loads(request.content)
        name = body["response_format"]["json_schema"]["name"]
        if name == "skill_spec":
            return httpx.Response(200, json=chat(SPEC_REPLY))
        image_url = body["messages"][1]["content"][0]["image_url"]["url"]
        img = Image.open(io.BytesIO(base64.b64decode(image_url.split(",", 1)[1]))).convert("RGB")
        r, g, _ = img.getpixel((32, 32))
        if g > 150 and r < 80:
            label = "yes" if self.calls[path] % 9 else "no"  # an occasional disagreement
        elif r > 150 and g < 80:
            label = "no"
        else:
            label = "no"  # real keyframes: the washer is mid-cycle
        return httpx.Response(200, json=chat({"label": label, "reason": "fake"}))


def clip(folder, n=26):
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        blocks = rng.integers(0, 255, size=(6, 8), dtype=np.uint8)
        img = Image.fromarray(blocks).resize((320, 240), Image.NEAREST).convert("RGB")
        img.save(folder / f"f{i:03d}.jpg", format="JPEG", quality=90)
    return folder


def wav(path, seconds=3.0):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * int(16000 * seconds))
    return path


def client_for(tmp_path, fake):
    settings = load_settings(env={"TE_DATA_DIR": str(tmp_path / "data"),
                                  "MODEL_API_KEY": "sk-test"}, env_file=None)
    return MetaClient(settings, transport=httpx.MockTransport(fake), sleep=lambda s: None)


def test_teach_end_to_end(tmp_path):
    fake = FakeMeta()
    client = client_for(tmp_path, fake)
    opts = TeachOptions(skill="washer_done", clip=clip(tmp_path / "clip"),
                        audio=wav(tmp_path / "request.wav"), positives=12, negatives=12)
    said = []
    report = run_teach(client, opts, tmp_path / "skills", confirm=lambda s: True, say=said.append)

    root = tmp_path / "skills" / "washer_done"
    spec = json.loads((root / "spec.json").read_text())
    assert spec["target"]["sam_prompt"] == "washing machine display"
    assert report["read_back"].startswith("I'll watch the digital display")
    assert fake.edits == 24
    manifest_rows = (root / "synthetic" / "manifest.jsonl").read_text().splitlines()
    assert len(manifest_rows) == 24
    stats = report["label_stats"]
    assert stats["total"] == 24 and 0 < stats["kept"] < 24
    ds = report["dataset"]
    assert ds["examples"] == stats["kept"] + len(list((root / "keyframes").iterdir()))
    assert (root / "dataset" / "washer_done-teach.jsonl").is_file()
    assert any(s.startswith("dataset:") for s in said)

    spent = client.ledger.spent("washer_done")
    calls_before = dict(fake.calls)
    again = run_teach(client, opts, tmp_path / "skills", confirm=lambda s: True, say=said.append)
    assert fake.calls == calls_before  # second run: every call is a cache hit
    assert client.ledger.spent("washer_done") == spent
    assert again["dataset"]["jsonl_sha256"] == ds["jsonl_sha256"]


def test_teach_stops_when_not_confirmed(tmp_path):
    fake = FakeMeta()
    client = client_for(tmp_path, fake)
    opts = TeachOptions(skill="washer_done", clip=clip(tmp_path / "clip"),
                        request="let me know when the washer says done")
    report = run_teach(client, opts, tmp_path / "skills", confirm=lambda s: False,
                       say=lambda s: None)
    assert report["stopped"] == "not confirmed"
    assert "/images/edits" not in fake.calls and "/audio/transcriptions" not in fake.calls


def test_refusal_costs_nothing(tmp_path):
    fake = FakeMeta()
    client = client_for(tmp_path, fake)
    spec = compile_skill(client, "Tell me when Sam gets here", [], name="sam_arrives")
    assert spec.refused and "specific people" in spec.refusal_reason
    assert fake.calls == {} and client.ledger.entries() == []


def test_fallback_trains_on_real_frames_with_human_labels_only(tmp_path):
    fake = FakeMeta()
    settings = load_settings(env={"TE_DATA_DIR": str(tmp_path / "data"), "MODEL_API_KEY": "k",
                                  "TE_TRAIN_ON_MODEL_OUTPUTS": "0"}, env_file=None)
    client = MetaClient(settings, transport=httpx.MockTransport(fake), sleep=lambda s: None)
    opts = TeachOptions(skill="washer_done", clip=clip(tmp_path / "clip"),
                        request="let me know when the washer says done", positives=4,
                        negatives=4, real_labels={"kf_00.jpg": "no", "kf_01.jpg": "yes"})
    report = run_teach(client, opts, tmp_path / "skills", confirm=lambda s: True,
                       say=lambda s: None)
    assert report["dataset"]["examples"] == 2
    assert set(report["dataset"]["counts"]) == {"washer_done/real/no", "washer_done/real/yes"}


def test_estimate_for_default_washer_run(tmp_path):
    client = client_for(tmp_path, FakeMeta())
    parts = estimate(client, audio_s=5, keyframes=24, positives=150, negatives=150)
    assert parts["images"] == 3.0
    assert parts["sam"] == 24 * 0.0025
    assert 3.0 < parts["total"] < client.settings.skill_cap_usd


def test_fallback_without_labels_stops_before_any_paid_call(tmp_path):
    fake = FakeMeta()
    settings = load_settings(env={"TE_DATA_DIR": str(tmp_path / "data"), "MODEL_API_KEY": "k",
                                  "TE_TRAIN_ON_MODEL_OUTPUTS": "0"}, env_file=None)
    client = MetaClient(settings, transport=httpx.MockTransport(fake), sleep=lambda s: None)
    opts = TeachOptions(skill="washer_done", clip=clip(tmp_path / "clip"),
                        audio=wav(tmp_path / "request.wav"))
    report = run_teach(client, opts, tmp_path / "skills", confirm=lambda s: True,
                       say=lambda s: None)
    assert "needs your own labels" in report["stopped"]
    assert fake.calls == {} and client.ledger.entries() == []
    assert any((tmp_path / "skills" / "washer_done" / "keyframes").iterdir())
