from backend.dataengine.labels import filter_generated


def test_filter_generated_stats():
    rows = [
        {"label": "yes", "spark_label": "yes", "kind": "positive"},
        {"label": "yes", "spark_label": "no", "kind": "positive"},
        {"label": "no", "spark_label": "no", "kind": "hard_negative"},
        {"label": "no", "spark_label": "unsure", "kind": "hard_negative"},
    ]
    kept, stats = filter_generated(rows)
    assert len(kept) == 2
    assert stats == {
        "total": 4, "kept": 2, "unsure": 1, "disagree": 1, "agreement": 0.5,
        "by_kind": {"positive": {"total": 2, "agree": 1, "unsure": 0, "agreement": 0.5},
                    "hard_negative": {"total": 2, "agree": 1, "unsure": 1, "agreement": 0.5}},
    }
