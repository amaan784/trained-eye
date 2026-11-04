import io
from pathlib import Path

import pytest
from PIL import Image

from backend.dataengine.roi import Detection, best_detection, locate, parse_sam_output
from backend.dataengine.synth import TooManyFailures
from backend.meta.client import MetaAPIError, OutputError
from backend.skills.spec import SkillSpec, Target

MASK = "<|mask;x=0;y=0;data=2,2,!!!!#Kq|>"


def test_roi_best_detection_and_sam_text():
    dets = [Detection((0, 0, 10, 10), 0.9, "0"), Detection((0, 0, 100, 100), 0.3, "1"),
            Detection((5, 5, 60, 60), None, "2")]
    best = best_detection(dets)  # every confident match, not the unsure one
    assert (best.box, best.score, best.object_id) == ((0, 0, 60, 60), 0.9, "0+2")
    assert best_detection(dets[2:]) is dets[2]  # one match: that one
    assert best_detection([Detection((0, 0, 1, 1), 0.1)]) is None
    text = ("<0f>0<|box;x1=10;y1=20;x2=49;y2=39;w=100;h=50;c=0.9|>" + MASK +
            ",1<|box;x1=0;y1=0;x2=9;y2=9;w=100;h=50;c=0.2|>" + MASK + "\n<0f>2 broken\n")
    detections, parser, problems = parse_sam_output(text)
    assert [(d.object_id, d.box, d.score) for d in detections] == [
        ("0", (10, 20, 50, 40), 0.9), ("1", (0, 0, 10, 10), 0.2)]
    assert parser == "sam3-text" and problems == ["line 2: text between tokens"]
    scaled, _, _ = parse_sam_output(text, (200, 100))  # SAM saw the keyframe at half size
    assert scaled[0].box == (20, 40, 100, 80)
    assert parse_sam_output("") == ([], "sam3-text", [])  # SAM found nothing


def test_the_region_holds_every_door_sam_found_in_a_real_frame():
    # SAM 3.1's live reply for the first fridge keyframe (tests/fixtures/meta): the open
    # right door (0.56) and the shut left door (0.93), which is larger. The region must
    # hold both, or the crop shows only the shut door for a frame labelled "open".
    from backend.meta.client import parse_sse, sam_stream_text

    stream = (Path(__file__).parent / "fixtures" / "meta" / "sam_stream.txt").read_text()
    text = sam_stream_text(parse_sse(stream))
    detections, _, problems = parse_sam_output(text, (1552, 2064))
    assert [(d.object_id, d.box) for d in detections] == [
        ("0", (1101, 654, 1552, 2056)), ("1", (0, 584, 484, 2064))]
    assert problems == []
    best = best_detection(detections)
    assert best.box == (0, 584, 1552, 2064) and best.object_id == "0+1"
    assert best.score == pytest.approx(0.9298245906829834)


SPEC = SkillSpec(name="burner_left_on", request="tell me when the burner is left on",
                 target=Target(sam_prompt="stove burner", description="the stove burner"),
                 positive="the burner is lit with no pan on it", negatives=["a pan being stirred on it"],
                 alert="The burner is still on.")


def jpeg(shade):
    buf = io.BytesIO()
    Image.new("RGB", (8, 6), (shade, shade, shade)).save(buf, format="JPEG")
    return buf.getvalue()


class FakeSam:
    def __init__(self, fail):
        self.fail = fail  # shade -> error to raise

    def sam_image(self, prompt, image, skill=None):
        shade = Image.open(io.BytesIO(image)).getpixel((0, 0))[0]
        for want, err in self.fail.items():
            if abs(shade - want) < 8:
                err.cost_usd = 0.002 if isinstance(err, OutputError) else 0.0
                raise err
        info = type("Info", (), {"cached": False, "cost_usd": 0.002})()
        text = "<0f>0<|box;x1=1;y1=1;x2=4;y2=3;w=8;h=6;c=0.8|>" + MASK
        return type("Sam", (), {"text": text, "info": info})()


def test_a_frame_sam_rejects_gets_no_box_and_the_run_goes_on():
    frames = [jpeg(s) for s in (20, 100, 180)]
    ticks = []
    rois = locate(FakeSam({100: MetaAPIError(422, "image rejected")}), SPEC, frames,
                  progress=lambda cached, cost: ticks.append(cost))
    assert [r.parser.startswith("error") for r in rois] == [False, True, False]
    assert ticks == [0.002, 0.0, 0.002]
    assert rois[0].best.box == (1, 1, 5, 4) and rois[0].best.score == 0.8
    assert rois[1].best is None and rois[0].problems == []


def test_sam_failing_on_most_frames_stops_the_run():
    frames = [jpeg(s) for s in range(10, 250, 30)]  # 8 frames
    fail = {s: OutputError("SAM returned no output") for s in range(10, 250, 30)}
    with pytest.raises(TooManyFailures, match="6 of 8 SAM calls failed"):
        locate(FakeSam(fail), SPEC, frames)
    with pytest.raises(MetaAPIError, match="401"):
        locate(FakeSam({10: MetaAPIError(401, "bad key")}), SPEC, frames)


@pytest.mark.parametrize("box, size, crop", [
    ((100, 60, 220, 180), (320, 240), (70, 30, 250, 210)),  # a 25% margin on every side
    ((280, 200, 320, 240), (320, 240), (256, 176, 320, 240)),  # moved inside at the corner
    ((10, 10, 12, 12), (320, 240), (0, 0, 64, 64)),  # grown to 64 a side, then moved in
    ((0, 0, 320, 240), (320, 240), (0, 0, 320, 240)),  # shrunk to the frame, aspect kept
    ((0, 100, 320, 110), (320, 240), (0, 100, 320, 110)),  # the frame wins over 64 px
    ((100.4, 60.6, 140.2, 80.9), (320, 240), (58, 39, 183, 103)),  # unrounded box
    ((-5, -5, 30, 20), (320, 240), (0, 0, 96, 64)),  # cut to the frame, then grown
])
def test_crop_box_adds_the_margin_keeps_the_aspect_and_stays_in_the_frame(box, size, crop):
    from backend.dataengine.roi import crop_box

    got = crop_box(box, size)
    assert got == crop
    left, top, right, bottom = got
    assert 0 <= left < right <= size[0] and 0 <= top < bottom <= size[1]
    x1, y1, x2, y2 = (max(0, box[0]), max(0, box[1]), box[2], box[3])
    assert left <= x1 and top <= y1 and right >= round(x2) and bottom >= round(y2)  # whole box


def test_crop_box_keeps_the_aspect_of_the_box():
    from backend.dataengine.roi import crop_box

    left, top, right, bottom = crop_box((100, 100, 140, 120), (640, 480))  # 2:1, small
    assert (right - left, bottom - top) == (128, 64)  # grown so the short side is 64
    with pytest.raises(ValueError, match="frame size"):
        crop_box((0, 0, 1, 1), (0, 10))


def test_crop_image_cuts_the_pixels_out_as_png():
    from backend.dataengine.roi import crop_image

    buf = io.BytesIO()
    img = Image.new("RGB", (100, 80), (0, 0, 0))
    img.paste((250, 10, 10), (20, 30, 60, 50))
    img.save(buf, format="PNG")
    data = crop_image(buf.getvalue(), (20, 30, 60, 50))
    with Image.open(io.BytesIO(data)) as part:
        assert part.format == "PNG" and part.size == (40, 20)
        assert part.getpixel((0, 0)) == part.getpixel((39, 19)) == (250, 10, 10)


def test_sam_several_at_a_time_keeps_the_keyframe_order():
    import random
    import time

    class Jittery(FakeSam):
        def sam_image(self, prompt, image, skill=None):
            time.sleep(random.random() / 50)  # replies come back out of order
            return super().sam_image(prompt, image, skill)

    frames = [jpeg(s) for s in range(10, 250, 30)]
    rois = locate(Jittery({}), SPEC, frames, workers=4)
    from backend.meta.cache import sha256_hex

    assert [r.keyframe_sha256 for r in rois] == [sha256_hex(f) for f in frames]
    assert all(r.best is not None for r in rois)
