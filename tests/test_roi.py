import io

import pytest
from PIL import Image

from backend.dataengine.roi import Detection, best_detection, locate, parse_sam_output
from backend.dataengine.synth import TooManyFailures
from backend.meta.client import MetaAPIError, OutputError
from backend.skills.spec import SkillSpec, Target

MASK = "<|mask;x=0;y=0;data=2,2,!!!!#Kq|>"


def test_roi_best_detection_and_sam_text():
    dets = [Detection((0, 0, 10, 10), 0.9), Detection((0, 0, 100, 100), 0.3),
            Detection((5, 5, 60, 60), None)]
    assert best_detection(dets).box == (5, 5, 60, 60)
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


SPEC = SkillSpec(name="washer_done", request="tell me when the washer says End",
                 target=Target(sam_prompt="washer display", description="the washer display"),
                 positive="the display shows End", negatives=["display off"],
                 alert="Laundry's done.")


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
