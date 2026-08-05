from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from analyze_sentinel_timing import (
    achieved_overhead,
    minimum_interval_steps,
    next_power_of_two,
    validate_repeat_summaries,
)


def test_minimum_interval_solves_the_registered_overhead_formula() -> None:
    interval = minimum_interval_steps(
        evaluation_seconds=22.5,
        training_step_seconds=0.9,
        maximum_overhead=0.05,
    )

    assert interval == 475
    assert achieved_overhead(22.5, 0.9, interval) <= 0.05
    assert achieved_overhead(22.5, 0.9, interval - 1) > 0.05


def test_next_power_of_two_only_rounds_up() -> None:
    assert next_power_of_two(1) == 1
    assert next_power_of_two(475) == 512
    assert next_power_of_two(512) == 512


def test_repeat_validation_requires_identical_rows_and_preference_hashes() -> None:
    base = {
        "status": "valid",
        "metadata": {"git_sha": "frozen", "teacher_forced_seconds": 22.5},
        "states": 2,
        "teacher_conditions": ["vision"],
        "generation_skipped": True,
        "teacher_forced_records_per_cell": 300,
        "preference_rows": 600,
        "generation_rows": 0,
        "files": {"preference_records.jsonl": {"sha256": "same"}},
        "final_half_scored": False,
    }
    summaries = [base, {**base, "metadata": {**base["metadata"], "teacher_forced_seconds": 22.7}}]

    validate_repeat_summaries(
        summaries,
        expected_repeats=2,
        expected_git_sha="frozen",
        expected_records_per_state=300,
    )

    drifted = {**base, "files": {"preference_records.jsonl": {"sha256": "drift"}}}
    with pytest.raises(ValueError, match="hash"):
        validate_repeat_summaries(
            [base, drifted],
            expected_repeats=2,
            expected_git_sha="frozen",
            expected_records_per_state=300,
        )
