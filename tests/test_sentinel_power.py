from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from analyze_sentinel_power import (
    select_minimum_candidate,
    subsample_metric_rows,
    wilson_interval,
)


def test_subsample_metric_rows_keeps_identical_complete_pair_ids() -> None:
    current = [{"id": f"pair-{index}", "score": float(index % 2)} for index in range(10)]
    reference = [{"id": f"pair-{index}", "score": 1.0} for index in range(10)]

    current_a, reference_a = subsample_metric_rows(
        current, reference, pairs=4, seed=20260826
    )
    current_b, reference_b = subsample_metric_rows(
        current, reference, pairs=4, seed=20260826
    )

    assert current_a == current_b
    assert reference_a == reference_b
    assert {row["id"] for row in current_a} == {row["id"] for row in reference_a}
    assert len(current_a) == 4


def test_subsample_metric_rows_rejects_mismatched_or_too_small_sources() -> None:
    current = [{"id": "pair-a", "score": 0.0}]
    reference = [{"id": "pair-b", "score": 1.0}]

    with pytest.raises(ValueError, match="identical"):
        subsample_metric_rows(current, reference, pairs=1, seed=1)
    with pytest.raises(ValueError, match="available"):
        subsample_metric_rows(current, current, pairs=2, seed=1)


def test_wilson_interval_is_bounded_and_conservative() -> None:
    low, high = wilson_interval(190, 200)

    assert 0.90 < low < 0.95
    assert 0.95 < high < 1.0
    assert wilson_interval(0, 200)[0] == 0.0
    assert wilson_interval(200, 200)[1] == 1.0


def test_minimum_candidate_requires_point_and_wilson_guards() -> None:
    criteria = {
        "minimum_count_recall": 0.95,
        "minimum_count_recall_ci95_low": 0.90,
        "minimum_exact_decision_rate": 0.90,
        "minimum_exact_decision_ci95_low": 0.85,
        "maximum_familywise_false_trigger_rate": 0.05,
        "maximum_familywise_false_trigger_ci95_high": 0.10,
    }
    rows = [
        {
            "pairs_per_task": 16,
            "count_recall": 0.96,
            "count_recall_ci95_low": 0.91,
            "exact_decision_rate": 0.92,
            "exact_decision_ci95_low": 0.86,
            "familywise_false_trigger_rate": 0.04,
            "familywise_false_trigger_ci95_high": 0.11,
        },
        {
            "pairs_per_task": 25,
            "count_recall": 0.97,
            "count_recall_ci95_low": 0.92,
            "exact_decision_rate": 0.93,
            "exact_decision_ci95_low": 0.88,
            "familywise_false_trigger_rate": 0.03,
            "familywise_false_trigger_ci95_high": 0.08,
        },
        {
            "pairs_per_task": 50,
            "count_recall": 1.0,
            "count_recall_ci95_low": 0.98,
            "exact_decision_rate": 1.0,
            "exact_decision_ci95_low": 0.98,
            "familywise_false_trigger_rate": 0.0,
            "familywise_false_trigger_ci95_high": 0.02,
        },
    ]

    assert select_minimum_candidate(rows, criteria)["pairs_per_task"] == 25


def test_preregistered_sentinel_protocol_freezes_power_and_timing_rules() -> None:
    root = Path(__file__).resolve().parent.parent
    config = json.loads(
        (root / "configs" / "perception-sentinel-power-v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["pairs_per_task"] == [8, 16, 25, 50, 100]
    assert config["trials"] == 200
    assert config["bootstrap_samples"] == 2000
    assert config["expected_full_trigger_tasks"] == ["count"]
    assert config["timing_protocol"]["states"] == [
        "exchange-step50",
        "ordinary-step75",
    ]
    assert config["timing_protocol"]["teacher_conditions"] == ["vision"]
    assert config["timing_protocol"]["generation"] is False
    assert config["hard_constraints"]["change_training_examples"] is False
