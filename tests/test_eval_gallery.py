"""Experiment 1's failure galleries (backend/eval/gallery.py): the worst false alarms and
misses of each run, read back from the recorded clips. Frame folders only: no ffmpeg."""

import io
import json

from PIL import Image

from backend.dataengine import keyframes
from backend.eval import gallery
from backend.eval.report import run_eval


def rows(ps, skill="washer_done", threshold=0.6, step=1.0):
    return [{"t": i * step, "skill": skill, "p": p, "threshold": threshold,
             "over": p is not None and p >= threshold} for i, p in enumerate(ps)]


def test_pick_takes_the_worst_frames_spread_apart():
    ps = [0.2] * 40
    ps[3], ps[4], ps[12], ps[30] = 0.95, 0.9, 0.7, 0.65  # false alarms (true is 20 to 29)
    for t in range(20, 30):
        ps[t] = 0.8
    ps[21], ps[22], ps[28] = 0.1, 0.15, 0.3  # misses
    ps[25] = None  # a student error is neither
    alarms, misses = gallery.pick(rows(ps), "washer_done", [[20, 29]], size=3)
    assert [r["t"] for r in alarms] == [3.0, 12.0, 30.0]  # 4.0 is under 5 s from 3.0
    assert [r["t"] for r in misses] == [21.0, 28.0]  # 22.0 is too close to 21.0
    alarms, _ = gallery.pick(rows(ps), "washer_done", [[20, 29]], size=1, gap_s=0)
    assert [r["t"] for r in alarms] == [3.0]
    assert gallery.pick(rows(ps), "boil_over", [[20, 29]], size=3) == ([], [])


def test_frame_step_and_finding_the_clip(tmp_path):
    assert gallery.frame_step(rows([0.1] * 5)) == 1.0
    assert gallery.frame_step(rows([0.1] * 5, step=0.5)) == 0.5
    assert gallery.frame_step(rows([0.1])) == 1.0
    clips = tmp_path / "clips"
    (clips / "session3_test_a").mkdir(parents=True)  # a frame folder
    (clips / "session3_test_a.labels.json").write_text('{"true": []}')
    (clips / "session3_test_b.mp4").write_bytes(b"")
    (clips / "session3_test_b.labels.json").write_text('{"true": []}')
    (clips / "session3_test_b.jsonl").write_text("")
    base = tmp_path / "eval"  # the plan's folder
    base.mkdir()
    assert gallery.find_clip(None, "../clips/session3_test_a.labels.json", base) == (
        base / "../clips/session3_test_a")
    assert gallery.find_clip(None, "../clips/session3_test_b.labels.json", base).name == (
        "session3_test_b.mp4")
    assert gallery.find_clip("../clips/session3_test_b.mp4", "x.json", base).name == (
        "session3_test_b.mp4")
    assert gallery.find_clip("../clips/gone.mp4", "x.json", base) is None
    assert gallery.find_clip(None, "../nowhere/a.labels.json", base) is None


def frame_folder(folder, n):
    folder.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (640, 480), (i * 2 % 256, 90, 160)).save(folder / f"f{i:03d}.jpg")
    return folder


def write_log(path, ps):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows(ps)))


def test_te_eval_writes_a_gallery_page_per_run_reading_each_clip_once(tmp_path, monkeypatch):
    clips = tmp_path / "data/clips/washer_done"
    frame_folder(clips / "session3_test_a", 60)
    (clips / "session3_test_a.labels.json").write_text(json.dumps({"true": [[30, 45]]}))
    watch = tmp_path / "data/watch"
    for run, alarm, miss in (("exp1-mixed", 10, 40), ("exp1-real", 5, 33)):
        ps = [0.2] * 60
        for t in range(30, 46):
            ps[t] = 0.9
        ps[alarm], ps[miss] = 0.8, 0.1
        (watch / run / "washer_done").mkdir(parents=True)
        write_log(watch / run / "washer_done/session3_test_a.jsonl", ps)
    stale = tmp_path / "eval/galleries/exp1-old"
    stale.mkdir(parents=True)

    def evaluation(run):
        return {"name": f"{run}-washer_done", "experiment": 1, "skill": "washer_done",
                "variant": run.removeprefix("exp1-"), "run": run, "clips": [{
                    "clip": "../data/clips/washer_done/session3_test_a",
                    "log": f"../data/watch/{run}/washer_done/session3_test_a.jsonl",
                    "labels": "../data/clips/washer_done/session3_test_a.labels.json"}]}

    plan = tmp_path / "eval/plan.json"
    plan.write_text(json.dumps({"evaluations": [
        evaluation("exp1-mixed"), evaluation("exp1-real"),
        {**evaluation("exp1-synthetic"), "clips": [{"log": "../data/watch/none.jsonl",
                                                     "labels": "../none.json"}]}]}))
    reads = []
    real = keyframes.load_frames
    monkeypatch.setattr(keyframes, "load_frames",
                        lambda source, fps=3.0: reads.append((source, fps)) or real(source, fps))
    results = run_eval(plan, tmp_path / "eval")
    assert len(reads) == 1 and reads[0][1] == 1.0  # one clip, two runs: read once, at 1 fps
    assert results["galleries"] == {
        "exp1-mixed": {"page": "galleries/exp1-mixed/index.html", "false_alarms": 1,
                       "misses": 1},
        "exp1-real": {"page": "galleries/exp1-real/index.html", "false_alarms": 1,
                      "misses": 1}}  # exp1-synthetic is pending: no gallery
    assert not stale.exists()  # galleries/ is written afresh
    ignore = tmp_path / "eval/galleries/.gitignore"  # frames of footage, wherever --out is
    assert ignore.read_text().splitlines()[-1] == "*"
    folder = tmp_path / "eval/galleries/exp1-mixed"
    thumbs = sorted(p.name for p in folder.glob("*.jpg"))
    assert thumbs == ["exp1-mixed-washer_done-session3_test_a-fa-01.jpg",
                      "exp1-mixed-washer_done-session3_test_a-miss-01.jpg"]
    with Image.open(io.BytesIO((folder / thumbs[0]).read_bytes())) as img:
        assert max(img.size) == gallery.THUMB_PX and img.format == "JPEG"
        assert img.getpixel((5, 5))[0] in range(15, 26)  # frame 10: red 20, give or take JPEG
    page = (folder / "index.html").read_text()
    assert "0:10 (10.0 s), p 0.80, threshold 0.60" in page
    assert "0:40 (40.0 s), p 0.10, threshold 0.60" in page
    assert "run te blur on any frame before publishing it" in page
    md = (tmp_path / "eval/results.md").read_text()
    assert "- [exp1-real](galleries/exp1-real/index.html): 1 false alarms and 1 misses" in md


def test_without_the_clip_or_ffmpeg_the_page_lists_the_frames(tmp_path, monkeypatch):
    out = tmp_path / "out"
    log = tmp_path / "a.jsonl"
    write_log(log, [0.9 if t in (3, 20) else 0.2 for t in range(30)])
    (tmp_path / "a.labels.json").write_text(json.dumps({"true": [[18, 25]]}))
    entry = {"name": "exp1-mixed-washer_done", "experiment": 1, "status": "scored",
             "skill": "washer_done", "variant": "mixed", "run": "exp1-mixed",
             "clips": [{"log": "a.jsonl", "labels": "a.labels.json"}]}
    found = gallery.write_galleries({"evaluations": [entry]}, tmp_path, out)
    assert found["exp1-mixed"] == {"page": "galleries/exp1-mixed/index.html",
                                   "false_alarms": 1, "misses": 2}  # 18 s and 23 s
    page = (out / "galleries/exp1-mixed/index.html").read_text()
    assert "the clip is not on this Mac, so its frames are not shown." in page
    assert page.count("no picture") == 3 and not list((out / "galleries").rglob("*.jpg"))

    (tmp_path / "a.mp4").write_bytes(b"not a video")

    def no_ffmpeg(source, fps=3.0):
        raise RuntimeError("ffmpeg is needed to read video files (brew install ffmpeg)")

    monkeypatch.setattr(keyframes, "load_frames", no_ffmpeg)
    gallery.write_galleries({"evaluations": [entry]}, tmp_path, out)
    page = (out / "galleries/exp1-mixed/index.html").read_text()
    assert "the frames could not be read from a.mp4: RuntimeError." in page
