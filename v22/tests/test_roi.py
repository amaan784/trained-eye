from backend.dataengine.roi import Detection, best_detection, parse_sam_output


def test_roi_best_detection_and_unparsed_output():
    dets = [Detection((0, 0, 10, 10), 0.9), Detection((0, 0, 100, 100), 0.3),
            Detection((5, 5, 60, 60), None)]
    assert best_detection(dets).box == (5, 5, 60, 60)
    assert best_detection([Detection((0, 0, 1, 1), 0.1)]) is None
    detections, parser = parse_sam_output("<raw special tokens>")
    assert detections == [] and parser in ("none", "unwired")
