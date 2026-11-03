"""The whole teach flow against a fake Meta API (httpx.MockTransport): nothing is sent
or billed. The fake image editor paints positives green and negatives red, and the fake
labeler reads the colour back, with one deliberate mistake to exercise filtering."""

import base64
import hashlib
import io
import json
import wave

import httpx
import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from backend.config import load_settings
from backend.dataengine.keyframes import load_frames, select_keyframes
from backend.dataengine.synth import TooManyFailures
from backend.meta.client import MetaAPIError, MetaClient
from backend.meta.ledger import Entry, Ledger, token_cost
from backend.skills.compiler import compile_skill
from backend.teach import TeachOptions, estimate, normalise_labels, run_teach

PROVENANCE = "fake-c2pa: made with Muse Image"

SPEC_REPLY = {
    "refused": False, "refusal_reason": None, "name": "washer_done",
    "sam_prompt": "washing machine display",
    "target_description": "the digital display on the front of the washer",
    "positive": "the display shows the cycle has finished (e.g. 'End' or '0:00')",
    "negatives": ["display shows time remaining", "display off"],
    "alert": "Your laundry's done.", "cooldown_s": 600, "confirm_with_cloud": True,
}


def png(color, serial=0, provenance=False) -> bytes:
    img = Image.new("RGB", (64, 64), color)
    img.putpixel((0, 0), (serial % 256, serial // 256, 0))  # every generated image differs
    info = PngImagePlugin.PngInfo()
    if provenance:  # a metadata chunk that any decode and re-save would drop
        info.add_text("c2pa", PROVENANCE)
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def chat(content: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1600, "completion_tokens": 120}}


SAM_MASK = "<|mask;x=0;y=0;data=2,2,!!!!#Kq|>"
SAM_BOX = (100, 60, 220, 180)  # half-open, in a 320x240 keyframe


def boxed(n, size):
    """SAM text: no object on every fourth call from the first, else SAM_BOX."""
    if n % 4 == 1:
        return ""
    x1, y1, x2, y2 = SAM_BOX
    return (f"<0f>0<|box;x1={x1};y1={y1};x2={x2 - 1};y2={y2 - 1};w={size[0]};h={size[1]};"
            f"c=0.9|>{SAM_MASK}")


class FakeMeta:
    def __init__(self, reject_edit=lambda prompt: False, refuse_label=lambda n: False,
                 answer=lambda label, real: label, sam=lambda n, size: "<sam raw output>"):
        self.calls = {}
        self.edits = 0
        self.returned = []  # every image the fake editor sent back, as sent
        self.reject_edit = reject_edit
        self.refuse_label = refuse_label
        self.answer = answer  # overrides the colour-read label: (label, real frame) -> label
        self.sam = sam  # (call number, image size) -> SAM's text; the default has no box

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/v1")
        self.calls[path] = self.calls.get(path, 0) + 1
        if path == "/audio/transcriptions":
            return httpx.Response(200, json={"text": "let me know when the washer says done",
                                             "duration": 3.0})
        if path == "/responses":
            url = json.loads(request.content)["input"][0]["content"][1]["image_url"]
            size = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size
            return httpx.Response(200, json={"output": [{"content": [
                {"type": "output_text", "text": self.sam(self.calls[path], size)}]}]})
        if path == "/images/edits":
            self.edits += 1
            if self.reject_edit(request.content):
                return httpx.Response(400, json={"error": "content_policy_violation"})
            negative = b"clearly not the case" in request.content
            color = (220, 0, 0) if negative else (0, 220, 0)
            self.returned.append(png(color, self.edits, provenance=True))
            return httpx.Response(200, json={"data": [
                {"b64_json": base64.b64encode(self.returned[-1]).decode()}]})
        body = json.loads(request.content)
        name = body["response_format"]["json_schema"]["name"]
        if name == "skill_spec":
            return httpx.Response(200, json=chat(SPEC_REPLY))
        if self.refuse_label(self.calls[path]):
            return httpx.Response(200, json={
                "choices": [{"message": {"content": None, "refusal": "no"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1600, "completion_tokens": 5}})
        image_url = body["messages"][1]["content"][0]["image_url"]["url"]
        img = Image.open(io.BytesIO(base64.b64decode(image_url.split(",", 1)[1]))).convert("RGB")
        r, g, _ = img.getpixel((32, 32))
        real = False
        if g > 150 and r < 80:
            label = "yes" if self.calls[path] % 9 else "no"  # an occasional disagreement
        elif r > 150 and g < 80:
            label = "no"
        else:
            label, real = "no", True  # real keyframes: the washer is mid-cycle
        label = self.answer(label, real)
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
    fake = FakeMeta(sam=boxed)
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
    rows = [json.loads(line) for line in manifest_rows]
    saved = [(root / "synthetic" / r["file"]).read_bytes() for r in rows]
    assert saved == fake.returned  # byte for byte: provenance metadata kept
    assert all(PROVENANCE.encode() in data for data in saved)
    assert [hashlib.sha256(data).hexdigest() for data in saved] == [
        r["image_sha256"] for r in rows]
    stats = report["label_stats"]
    assert stats["total"] == 24 and 0 < stats["kept"] < 24
    ds = report["dataset"]
    n_kf = len(list((root / "keyframes").iterdir()))
    with_box = report["roi_frames_with_box"]
    assert with_box + report["roi_frames_without_box"] == n_kf
    assert 0 < with_box < n_kf and report["crop_examples"] == with_box  # one crop per box
    assert ds["examples"] == stats["kept"] + n_kf + with_box
    assert (root / "dataset" / "washer_done-teach.jsonl").is_file()
    assert any(s.startswith("dataset:") for s in said)
    assert {e["original_sha256"] for e in ds["examples_index"] if e["source"] == "synthetic"} <= {
        r["image_sha256"] for r in rows}
    entries = client.ledger.entries()
    assert {e["endpoint"] for e in entries} == {"/audio/transcriptions", "/chat/completions",
                                                "/responses", "/images/edits"}
    assert {e["skill"] for e in entries} == {"washer_done"}  # every call counts to the cap
    assert len(json.loads((root / "roi.json").read_text())) == n_kf
    assert len((root / "labels.jsonl").read_text().splitlines()) == n_kf + 24
    assert json.loads((root / "label_stats.json").read_text()) == stats
    assert (f"region of interest: {with_box} of {n_kf} keyframes with a box "
            f"({n_kf - with_box} without one get no crop)") in said
    assert "  images 24/24  $0.24 spent  (0 from cache)" in said
    assert any(s.startswith(f"  labels {n_kf + 24}/{n_kf + 24}") for s in said)

    spent = client.ledger.spent("washer_done")
    calls_before = dict(fake.calls)
    said.clear()
    again = run_teach(client, opts, tmp_path / "skills", confirm=lambda s: True, say=said.append)
    assert fake.calls == calls_before  # second run: every call is a cache hit
    assert client.ledger.spent("washer_done") == spent
    assert again["dataset"]["jsonl_sha256"] == ds["jsonl_sha256"]
    assert "  images 24/24  $0.00 spent  (24 from cache)" in said
    rerun_rows = [json.loads(x) for x in
                  (root / "synthetic" / "manifest.jsonl").read_text().splitlines()]
    assert all(r["cached"] for r in rerun_rows)  # and each still says what it cost
    edits = [e for e in entries if e["endpoint"] == "/images/edits"]
    assert sum(r["cost_usd"] for r in rerun_rows) == pytest.approx(
        sum(e["actual_usd"] for e in edits))


def test_one_bad_edit_or_label_does_not_stop_the_run(tmp_path):
    fake = FakeMeta(reject_edit=lambda body: b"dim evening light" in body,
                    refuse_label=lambda n: n == 30)
    client = client_for(tmp_path, fake)
    opts = TeachOptions(skill="washer_done", clip=clip(tmp_path / "clip"),
                        request="let me know when the washer says done", positives=12,
                        negatives=12)
    report = run_teach(client, opts, tmp_path / "skills", confirm=lambda s: True,
                       say=lambda s: None)
    root = tmp_path / "skills" / "washer_done"
    rows = [json.loads(x) for x in (root / "synthetic/manifest.jsonl").read_text().splitlines()]
    failed = [r for r in rows if r["file"] is None]
    assert len(rows) == 24 and failed and all("content_policy" in r["error"] for r in failed)
    assert report["generation_failed"] == len(failed)
    assert report["label_stats"]["label_errors"] == 1
    assert (root / "dataset" / "washer_done-teach.jsonl").is_file()


def test_systematic_edit_failures_stop_the_run(tmp_path):
    fake = FakeMeta(reject_edit=lambda body: True)
    client = client_for(tmp_path, fake)
    opts = TeachOptions(skill="washer_done", clip=clip(tmp_path / "clip"),
                        request="let me know when the washer says done", positives=20,
                        negatives=20)
    with pytest.raises(MetaAPIError, match="image edits failed"):
        run_teach(client, opts, tmp_path / "skills", confirm=lambda s: True,
                  say=lambda s: None)
    assert fake.edits == 9  # stops after max(5, 20%) failures instead of paying for all 40
    assert not (tmp_path / "skills/washer_done/labels.jsonl").exists()


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


def fallback_client(tmp_path, fake):
    settings = load_settings(env={"TE_DATA_DIR": str(tmp_path / "data"), "MODEL_API_KEY": "k",
                                  "TE_TRAIN_ON_MODEL_OUTPUTS": "0"}, env_file=None)
    return MetaClient(settings, transport=httpx.MockTransport(fake), sleep=lambda s: None)


def keyframe_names(tmp_path):
    return sorted(p.name for p in (tmp_path / "skills/washer_done/keyframes").iterdir())


def test_fallback_trains_on_real_frames_with_human_labels_only(tmp_path):
    fake = FakeMeta()
    client = fallback_client(tmp_path, fake)
    opts = TeachOptions(skill="washer_done", clip=clip(tmp_path / "clip"),
                        request="let me know when the washer says done", positives=4,
                        negatives=4)
    first = run_teach(client, opts, tmp_path / "skills", confirm=lambda s: True,
                      say=lambda s: None)
    assert "needs your own labels" in first["stopped"]
    names = keyframe_names(tmp_path)
    opts.real_labels = {names[0]: "no", names[1]: "yes"}
    report = run_teach(client, opts, tmp_path / "skills", confirm=lambda s: True,
                       say=lambda s: None)
    assert report["dataset"]["examples"] == 2
    assert set(report["dataset"]["counts"]) == {"washer_done/real/no", "washer_done/real/yes"}


def test_own_labels_are_normalised_and_win_over_spark(tmp_path):
    fake = FakeMeta()
    client = client_for(tmp_path, fake)
    opts = TeachOptions(skill="washer_done", clip=clip(tmp_path / "clip"),
                        request="let me know when the washer says done", positives=2,
                        negatives=2)
    run_teach(client, opts, tmp_path / "skills", confirm=lambda s: False, say=lambda s: None)
    names = keyframe_names(tmp_path)
    opts.real_labels = {f"keyframes/{names[0]}": "Yes", names[1]: " NO "}
    report = run_teach(client, opts, tmp_path / "skills", confirm=lambda s: True,
                       say=lambda s: None)
    rows = [json.loads(x) for x in
            (tmp_path / "skills/washer_done/labels.jsonl").read_text().splitlines()]
    human = {r["file"]: r["human_label"] for r in rows if r["source"] == "real"}
    assert human[f"keyframes/{names[0]}"] == "yes" and human[f"keyframes/{names[1]}"] == "no"
    assert report["dataset"]["counts"]["washer_done/real/yes"] == 1  # Spark said no to all


@pytest.mark.parametrize("labels", [{"kf_99_00000000.jpg": "yes"}, {"FIRST": "maybe"},
                                    {"f005.jpg": "yes"}, ["kf_00.jpg"]])
def test_unusable_own_labels_stop_before_any_paid_call(tmp_path, labels):
    fake = FakeMeta()
    client = client_for(tmp_path, fake)
    opts = TeachOptions(skill="washer_done", clip=clip(tmp_path / "clip"),
                        request="let me know when the washer says done", real_labels=labels)
    if labels == {"FIRST": "maybe"}:
        run_teach(client, TeachOptions(skill="washer_done", clip=opts.clip, request="x"),
                  tmp_path / "skills", confirm=lambda s: False, say=lambda s: None)
        opts.real_labels = {keyframe_names(tmp_path)[0]: "maybe"}
        fake.calls.clear()
    report = run_teach(client, opts, tmp_path / "skills", confirm=lambda s: True,
                       say=lambda s: None)
    assert "--labels" in report["stopped"] and fake.calls == {}


def test_labels_for_another_take_do_not_carry_over(tmp_path):
    fake = FakeMeta()
    client = fallback_client(tmp_path, fake)
    first = clip(tmp_path / "a", n=30)
    opts = TeachOptions(skill="washer_done", clip=first,
                        request="let me know when the washer says done")
    run_teach(client, opts, tmp_path / "skills", confirm=lambda s: True, say=lambda s: None)
    old_names = keyframe_names(tmp_path)
    second = tmp_path / "b"
    second.mkdir()
    for p in sorted(first.iterdir())[5:25]:  # a new, shorter take of the same scene
        (second / p.name).write_bytes(p.read_bytes())
    opts = TeachOptions(skill="washer_done", clip=second,
                        request="let me know when the washer says done",
                        real_labels=dict.fromkeys(old_names, "no"))
    report = run_teach(client, opts, tmp_path / "skills", confirm=lambda s: True,
                       say=lambda s: None)
    assert "--labels" in report["stopped"]  # old names no longer match: nothing is mislabelled
    # only the new take's keyframes are left to label
    assert len(keyframe_names(tmp_path)) == len(select_keyframes(load_frames(second)))


def test_correction_recompiles_and_is_recorded(tmp_path):
    fake = FakeMeta()
    client = client_for(tmp_path, fake)
    opts = TeachOptions(skill="washer_done", clip=clip(tmp_path / "clip"),
                        audio=wav(tmp_path / "request.wav"))
    run_teach(client, opts, tmp_path / "skills", confirm=lambda s: False, say=lambda s: None)
    assert fake.calls["/chat/completions"] == 1
    opts.correction = "the light on the lid"
    report = run_teach(client, opts, tmp_path / "skills", confirm=lambda s: False,
                       say=lambda s: None)
    assert report["stopped"] == "not confirmed"
    assert fake.calls["/chat/completions"] == 2 and fake.calls["/audio/transcriptions"] == 1
    request = json.loads((tmp_path / "skills/washer_done/request.json").read_text())
    assert request["correction"] == "the light on the lid"


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


def teach(tmp_path, client, **kw):
    said = []
    opts = TeachOptions(skill="washer_done", clip=kw.pop("clip", None) or clip(tmp_path / "clip"),
                        request="let me know when the washer says done",
                        **{"positives": 6, "negatives": 6, **kw})
    report = run_teach(client, opts, tmp_path / "skills", confirm=lambda s: True,
                       say=said.append)
    return report, said


def test_a_one_class_dataset_stops_instead_of_being_written(tmp_path):
    report, _ = teach(tmp_path, client_for(tmp_path, FakeMeta(answer=lambda label, real: "no")))
    assert "only 'no' examples" in report["stopped"] and report["label_stats"]["kept"] == 6
    assert not (tmp_path / "skills/washer_done/dataset").exists()


def test_the_dataset_line_shows_both_answers(tmp_path):
    report, said = teach(tmp_path, client_for(tmp_path, FakeMeta()))
    counts = report["dataset"]["counts"]
    yes = sum(n for k, n in counts.items() if k.endswith("/yes"))
    assert any(line.startswith(f"dataset: {report['dataset']['examples']} examples ({yes} yes, ")
               for line in said)


def test_spark_unsure_on_a_real_frame_leaves_it_out(tmp_path):
    fake = FakeMeta(answer=lambda label, real: "unsure" if real else label)
    report, _ = teach(tmp_path, client_for(tmp_path, fake))
    assert not any("/real/" in k for k in report["dataset"]["counts"])


def test_fallback_labels_must_say_both_yes_and_no(tmp_path):
    fake = FakeMeta()
    client = fallback_client(tmp_path, fake)
    first, _ = teach(tmp_path, client)
    assert "yes or no (both are needed)" in first["stopped"]
    names = keyframe_names(tmp_path)
    report, _ = teach(tmp_path, client, clip=tmp_path / "clip",
                      real_labels=dict.fromkeys(names[:5], "no"))
    assert "all 5 say 'no'" in report["stopped"] and fake.calls == {}


def test_a_new_spec_removes_the_old_dataset(tmp_path):
    fake = FakeMeta()
    client = client_for(tmp_path, fake)
    teach(tmp_path, client)
    assert (tmp_path / "skills/washer_done/dataset/washer_done-teach.jsonl").is_file()
    opts = TeachOptions(skill="washer_done", clip=tmp_path / "clip", correction="the lid light",
                        request="let me know when the washer says done")
    report = run_teach(client, opts, tmp_path / "skills", confirm=lambda s: False,
                       say=lambda s: None)
    assert report["stopped"] == "not confirmed"
    assert not (tmp_path / "skills/washer_done/dataset").exists()


def test_the_skill_folder_is_never_read_as_the_clip_and_only_keyframes_are_cleaned(tmp_path):
    client = fallback_client(tmp_path, FakeMeta())
    teach(tmp_path, client)
    kf_dir = tmp_path / "skills/washer_done/keyframes"
    with pytest.raises(ValueError, match="inside this skill's own folder"):
        teach(tmp_path, client, clip=kf_dir)
    (kf_dir / "notes.txt").write_text("frame 3 is the best")
    (kf_dir / "kf_extra").mkdir()
    (kf_dir / "kf_00_deadbeef.jpg").write_bytes(b"from an earlier take")
    teach(tmp_path, client, clip=tmp_path / "clip")
    assert (kf_dir / "notes.txt").is_file() and (kf_dir / "kf_extra").is_dir()
    assert not (kf_dir / "kf_00_deadbeef.jpg").exists()


def test_contradicting_label_entries_are_refused():
    with pytest.raises(ValueError, match="opposite of an earlier entry"):
        normalise_labels({"kf_00_1a2b3c4d.jpg": "yes", "keyframes/kf_00_1a2b3c4d.jpg": "no"},
                         ["kf_00_1a2b3c4d.jpg"])
    same = normalise_labels({"kf_00_1a2b3c4d.jpg": "yes", "keyframes/kf_00_1a2b3c4d.jpg": "Y"},
                            ["kf_00_1a2b3c4d.jpg"])
    assert same == {"kf_00_1a2b3c4d.jpg": "yes"}


def test_labels_so_far_are_saved_when_labelling_stops(tmp_path):
    fake = FakeMeta(refuse_label=lambda n: n > 1)  # every label after the compile call
    with pytest.raises(TooManyFailures, match="labels failed so far") as err:
        teach(tmp_path, client_for(tmp_path, fake))
    rows = [json.loads(x) for x in
            (tmp_path / "skills/washer_done/labels.jsonl").read_text().splitlines()]
    assert rows and all(r["label_error"] for r in rows)
    assert str(err.value).startswith(f"{len(rows)} of ")


def test_a_short_clip_warns_about_few_keyframes(tmp_path):
    _, said = teach(tmp_path, fallback_client(tmp_path, FakeMeta()), clip=clip(tmp_path / "c", 8))
    assert any(line.startswith("warning: fewer than 20 keyframes") for line in said)


def test_estimate_prices_the_label_retry_as_an_upper_bound(tmp_path):
    client = client_for(tmp_path, FakeMeta())
    parts = estimate(client, audio_s=5, keyframes=24, positives=150, negatives=150)
    assert parts["at_most"] > parts["total"] + parts["labels"]  # 4096 tokens after 1024


def test_a_repeated_label_with_an_unusable_value_says_what_is_wrong():
    with pytest.raises(ValueError) as err:
        normalise_labels({"kf_00_1a2b3c4d.jpg": "yes", "keyframes/kf_00_1a2b3c4d.jpg": "maybe"},
                         ["kf_00_1a2b3c4d.jpg"])
    assert "'maybe'" in str(err.value) and "opposite" not in str(err.value)


def measured_label_call(settings) -> Entry:
    """One recorded label call whose image took 800 input tokens (1050 less 250 of text)."""
    return Entry(settings.model("labeler"), "/chat/completions", 0.01, 0.002,
                 usage={"prompt_tokens": 1050, "completion_tokens": 9}, input_images=1,
                 input_text_tokens=250)


def test_estimate_prices_images_at_the_tokens_measured_in_the_ledger(tmp_path):
    from backend.dataengine.labels import MAX_TOKENS

    client = client_for(tmp_path, FakeMeta())
    s = client.settings
    guessed = estimate(client, audio_s=5, keyframes=24, positives=150, negatives=150)
    client.ledger.record(measured_label_call(s))
    measured = estimate(client, audio_s=5, keyframes=24, positives=150, negatives=150)
    label, wonder = s.price(s.model("labeler")), s.price(s.model("compiler"))
    assert guessed["labels"] == pytest.approx(324 * token_cost(label, 1500 + 250, MAX_TOKENS[0]))
    assert measured["labels"] == pytest.approx(324 * token_cost(label, 800 + 250, MAX_TOKENS[0]))
    assert measured["compile"] == pytest.approx(token_cost(wonder, 3 * 800 + 900, 4096))
    assert measured["total"] < guessed["total"] and measured["images"] == guessed["images"]


def test_te_teach_says_which_image_tokens_it_used(tmp_path, monkeypatch, capsys):
    from backend import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TE_DATA_DIR", str(tmp_path / "data"))
    args = ["teach", "--skill", "washer_done", "--clip", str(clip(tmp_path / "clip")),
            "--request", "let me know when the washer says done", "--estimate-only"]
    assert cli.main(args) == 0
    assert "image tokens: 1500 per image, a guess" in capsys.readouterr().out
    settings = load_settings(env={"TE_DATA_DIR": str(tmp_path / "data")}, env_file=None)
    Ledger(settings.ledger_path, 10.0, 120.0).record(measured_label_call(settings))
    assert cli.main(args) == 0
    assert ("image tokens: 800 per image, measured over 1 call in the ledger"
            in capsys.readouterr().out)


def test_te_system_in_user_folds_the_system_prompt_into_the_dataset(tmp_path):
    settings = load_settings(env={"TE_DATA_DIR": str(tmp_path / "data"), "MODEL_API_KEY": "sk",
                                  "TE_SYSTEM_IN_USER": "1"}, env_file=None)
    client = MetaClient(settings, transport=httpx.MockTransport(FakeMeta()), sleep=lambda s: None)
    report, _ = teach(tmp_path, client)
    assert report["dataset"]["system_in_user"] is True
    root = tmp_path / "skills/washer_done/dataset"
    record = json.loads((root / "washer_done-teach.jsonl").read_text().splitlines()[0])
    assert [m["role"] for m in record["messages"]] == ["user", "assistant"]
    assert json.loads((root / "washer_done-teach.manifest.json").read_text())["system_in_user"]


def test_the_fallback_adds_no_crops(tmp_path):
    client = fallback_client(tmp_path, FakeMeta(sam=boxed))
    teach(tmp_path, client)  # stops for labels, after writing the keyframes
    names = keyframe_names(tmp_path)
    report, _ = teach(tmp_path, client, clip=tmp_path / "clip",
                      real_labels={names[1]: "yes", names[2]: "no"})
    assert report["roi_frames_with_box"] > 0 and report["crop_examples"] == 0
    assert report["dataset"]["examples"] == 2  # SAM chose a crop's pixels: a model output


def test_a_box_over_the_whole_frame_adds_no_crop(tmp_path):
    def whole(n, size):
        return (f"<0f>0<|box;x1=0;y1=0;x2={size[0] - 1};y2={size[1] - 1};w={size[0]};"
                f"h={size[1]}|>{SAM_MASK}")

    report, _ = teach(tmp_path, client_for(tmp_path, FakeMeta(sam=whole)))
    assert report["roi_frames_with_box"] > 0 and report["roi_frames_without_box"] == 0
    assert report["crop_examples"] == 0  # the same picture as its keyframe
