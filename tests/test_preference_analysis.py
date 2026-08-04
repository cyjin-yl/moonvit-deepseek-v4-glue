"""Pair-level analysis resamples complete minimal pairs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from analyze_paired_preference import pair_metric_rows


def test_pair_metric_rows_reports_strict_preference_and_pair_mean_margin():
    rows = [
        {"pair_id": "p1", "correct_margin": 0.8, "failure": None},
        {"pair_id": "p1", "correct_margin": 0.2, "failure": None},
        {"pair_id": "p2", "correct_margin": 0.4, "failure": None},
        {"pair_id": "p2", "correct_margin": -0.1, "failure": None},
    ]

    preference = pair_metric_rows(rows, "paired_preference")
    margin = pair_metric_rows(rows, "mean_margin")

    assert preference == [{"id": "p1", "score": 1.0}, {"id": "p2", "score": 0.0}]
    assert margin[0] == {"id": "p1", "score": 0.5}
    assert margin[1]["id"] == "p2"
    assert abs(margin[1]["score"] - 0.15) < 1e-12
