import io

import pytest
from PIL import Image

from backend.dataengine.roi import Detection, best_detection, locate, parse_sam_output
from backend.dataengine.synth import TooManyFailures
from backend.meta.client import MetaAPIError, OutputError
from backend.skills.spec import SkillSpec, Target


def test_roi_best_detection_and_unparsed_output():
    dets = [Detection((0, 0, 10, 10), 0.9), Detection((0, 0, 100, 100), 0.3),
            Detection((5, 5, 60, 60), None)]
    assert best_detection(dets).box == (5, 5, 60, 60)
    assert best_detection([Detection((0, 0, 1, 1), 0.1)]) is None
    detections, parser = parse_sam_output("<raw special tokens>")
    assert detections == [] and parser in ("none", "unwired")


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
        return type("Sam", (), {"text": "<box/>", "info": info})()


def test_a_frame_sam_rejects_gets_no_box_and_the_run_goes_on():
    frames = [jpeg(s) for s in (20, 100, 180)]
    ticks = []
    rois = locate(FakeSam({100: MetaAPIError(422, "image rejected")}), SPEC, frames,
                  progress=lambda cached, cost: ticks.append(cost))
    assert [r.parser.startswith("error") for r in rois] == [False, True, False]
    assert ticks == [0.002, 0.0, 0.002]


def test_sam_failing_on_most_frames_stops_the_run():
    frames = [jpeg(s) for s in range(10, 250, 30)]  # 8 frames
    fail = {s: OutputError("SAM returned no output") for s in range(10, 250, 30)}
    with pytest.raises(TooManyFailures, match="6 of 8 SAM calls failed"):
        locate(FakeSam(fail), SPEC, frames)
    with pytest.raises(MetaAPIError, match="401"):
        locate(FakeSam({10: MetaAPIError(401, "bad key")}), SPEC, frames)
