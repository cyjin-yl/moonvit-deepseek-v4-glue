from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from analyze_batch_stratification import classify_batch_effect
from measure_task_gradient_norms import gradient_cosine
from verify_batch_stratification import summarize_order_rows

import pytest
import torch


def test_order_summary_detects_balanced_and_global_batches() -> None:
    tasks = ["color", "shape"]
    task_by_id = {
        "c0": "color",
        "c1": "color",
        "c2": "color",
        "c3": "color",
        "s0": "shape",
        "s1": "shape",
        "s2": "shape",
        "s3": "shape",
    }
    balanced = [
        {"step": 1, "ids": ["c0", "c1", "s0", "s1"]},
        {"step": 2, "ids": ["c2", "c3", "s2", "s3"]},
    ]
    global_random = [
        {"step": 1, "ids": ["c0", "c1", "c2", "s0"]},
        {"step": 2, "ids": ["c3", "s1", "s2", "s3"]},
    ]

    balanced_summary = summarize_order_rows(
        balanced, task_by_id=task_by_id, tasks=tasks, batch_size=4
    )
    random_summary = summarize_order_rows(
        global_random, task_by_id=task_by_id, tasks=tasks, batch_size=4
    )

    assert balanced_summary["unique_ids"] == 8
    assert balanced_summary["balanced_batches"] == 2
    assert random_summary["balanced_batches"] == 0
    assert random_summary["max_task_count_in_batch"] == 3


def test_batch_effect_requires_endpoint_paired_evidence() -> None:
    supported = classify_batch_effect(
        [
            {"task": "overall", "ci95_low": 0.01, "ci95_high": 0.08},
            {"task": "count", "ci95_low": -0.01, "ci95_high": 0.12},
        ]
    )
    mixed = classify_batch_effect(
        [
            {"task": "overall", "ci95_low": -0.02, "ci95_high": 0.03},
            {"task": "count", "ci95_low": 0.01, "ci95_high": 0.12},
            {"task": "shape", "ci95_low": -0.05, "ci95_high": 0.02},
        ]
    )

    assert supported == "balanced_batch_effect_supported"
    assert mixed == "mixed_or_underpowered"


def test_batch_effect_can_support_global_random() -> None:
    result = classify_batch_effect(
        [
            {"task": "overall", "ci95_low": -0.09, "ci95_high": -0.01},
            {"task": "count", "ci95_low": -0.10, "ci95_high": 0.02},
        ]
    )

    assert result == "global_random_effect_supported"


def test_gradient_cosine_preserves_conflict_sign() -> None:
    assert gradient_cosine(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])) == pytest.approx(-1.0)
    assert gradient_cosine(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0])) == pytest.approx(0.0)
