"""Matched replay 的 pair 保留与机械触发规则。"""

import sys
from collections import Counter
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from replay_order import transform_replay_batches
from analyze_replay_sentinel import select_trigger_tasks
from train_shape_adaptation import validate_fixed_training_budget
from build_triggered_replay_config import build_triggered_config


TASKS = ("color", "coordinate", "count", "ocr", "shape", "spatial")


def _pair_records(prefix: str, pairs_per_task: int) -> list[dict]:
    return [
        {
            "id": f"{prefix}-{task}-{pair}-{variant}",
            "task": task,
            "pair_id": f"{prefix}-{task}-{pair}",
            "pair_variant": variant,
        }
        for task in TASKS
        for pair in range(pairs_per_task)
        for variant in ("a", "b")
    ]


def test_replay_replaces_complete_pairs_and_preserves_total_examples() -> None:
    history = _pair_records("history", 2)
    source = _pair_records("source", 2)
    records = history + source
    by_id = {row["id"]: index for index, row in enumerate(records)}
    history_batches = [[by_id[row["id"]] for row in history]]
    source_batches = [[by_id[row["id"]] for row in source]]

    transformed, provenance = transform_replay_batches(
        records,
        source_batches=source_batches,
        history_batches=history_batches,
        tasks=list(TASKS),
        replay_tasks=["count", "shape"],
        pairs_per_task_per_window=2,
        window_batch_count=1,
        seed=19,
    )

    assert len(transformed) == 1
    assert len(transformed[0]) == len(source_batches[0]) == 24
    counts = Counter(records[index]["task"] for index in transformed[0])
    assert counts == {
        "color": 2,
        "coordinate": 2,
        "count": 8,
        "ocr": 2,
        "shape": 8,
        "spatial": 2,
    }
    assert provenance["added_records_by_task"] == {"count": 4, "shape": 4}
    assert provenance["removed_records_by_task"] == {
        "color": 2,
        "coordinate": 2,
        "ocr": 2,
        "spatial": 2,
    }
    assert provenance["all_replacements_are_complete_pairs"] is True
    assert set(provenance["added_ids"]) <= {row["id"] for row in history}
    assert set(provenance["removed_ids"]) <= {row["id"] for row in source}


def test_replay_order_is_deterministic_for_the_same_seed() -> None:
    history = _pair_records("history", 3)
    source = _pair_records("source", 3)
    records = history + source
    split = len(history)
    kwargs = {
        "source_batches": [list(range(split, len(records)))],
        "history_batches": [list(range(split))],
        "tasks": list(TASKS),
        "replay_tasks": ["count", "shape"],
        "pairs_per_task_per_window": 2,
        "window_batch_count": 1,
        "seed": 23,
    }

    first = transform_replay_batches(records, **kwargs)
    second = transform_replay_batches(records, **kwargs)

    assert first == second


def test_replay_rejects_an_insufficient_complete_pair_pool() -> None:
    history = _pair_records("history", 1)
    source = _pair_records("source", 2)
    records = history + source
    split = len(history)

    with pytest.raises(ValueError, match="complete replay pairs"):
        transform_replay_batches(
            records,
            source_batches=[list(range(split, len(records)))],
            history_batches=[list(range(split))],
            tasks=list(TASKS),
            replay_tasks=["count", "shape"],
            pairs_per_task_per_window=2,
            window_batch_count=1,
            seed=29,
        )


def test_trigger_needs_both_absolute_drop_and_supported_paired_ci() -> None:
    contrasts = [
        {"task": "count", "mean_gap": -0.15, "ci95_low": -0.23, "ci95_high": -0.04},
        {"task": "shape", "mean_gap": -0.09, "ci95_low": -0.16, "ci95_high": -0.02},
        {"task": "color", "mean_gap": -0.20, "ci95_low": -0.31, "ci95_high": 0.01},
        {"task": "ocr", "mean_gap": 0.02, "ci95_low": -0.04, "ci95_high": 0.07},
    ]

    selected = select_trigger_tasks(contrasts, minimum_drop=0.10, max_tasks=2)

    assert selected == ["count"]


def test_trigger_ranks_supported_declines_without_manual_ties() -> None:
    contrasts = [
        {"task": "count", "mean_gap": -0.14, "ci95_low": -0.20, "ci95_high": -0.02},
        {"task": "shape", "mean_gap": -0.21, "ci95_low": -0.28, "ci95_high": -0.11},
        {"task": "ocr", "mean_gap": -0.12, "ci95_low": -0.18, "ci95_high": -0.01},
    ]

    selected = select_trigger_tasks(contrasts, minimum_drop=0.10, max_tasks=2)

    assert selected == ["shape", "count"]


def test_fixed_training_budget_accepts_the_exact_preregistered_total() -> None:
    result = validate_fixed_training_budget(
        {
            "fixed_continuation_steps": 50,
            "fixed_continuation_examples": 1200,
        },
        initial_step=50,
        final_step=100,
        batch_size=24,
    )

    assert result == {"steps": 50, "examples": 1200}


def test_fixed_training_budget_rejects_any_extra_step() -> None:
    with pytest.raises(ValueError, match="fixed training budget"):
        validate_fixed_training_budget(
            {
                "fixed_continuation_steps": 50,
                "fixed_continuation_examples": 1200,
            },
            initial_step=50,
            final_step=101,
            batch_size=24,
        )


def test_triggered_config_derivation_keeps_the_remaining_budget_fixed(tmp_path: Path) -> None:
    base = {
        "run_id": "matched-v1",
        "base_projector": "/old/step50",
        "dataset": {"tasks": list(TASKS)},
        "training": {
            "initial_step": 50,
            "steps": 100,
            "checkpoint_steps": [50, 75, 100],
            "batch_size": 24,
            "fixed_continuation_steps": 50,
            "fixed_continuation_examples": 1200,
        },
        "arms": {"ordinary_continuation": {}, "fixed_replay": {}},
        "matched_replay": {
            "trigger_rule": {"maximum_tasks": 2},
            "triggered_policy_template": {
                "pairs_per_task_per_window": 10,
                "window_steps": 25,
                "history_start_step_exclusive": 0,
                "history_end_step_inclusive": 50,
                "seed": 31,
            },
        },
    }
    decision = {
        "status": "valid",
        "reference_state": "exchange-step50",
        "current_state": "ordinary-step75",
        "trigger_tasks": ["count", "shape"],
    }
    summary = {
        "status": "valid",
        "checkpoints": {
            "step-000075": {
                "step": 75,
                "files": {
                    "projector.safetensors": {"sha256": "projector-hash"},
                    "training_state.pt": {"sha256": "optimizer-hash"},
                },
            }
        },
    }

    derived = build_triggered_config(
        base,
        decision,
        summary,
        ordinary_run=tmp_path / "ordinary",
        decision_path=tmp_path / "decision.json",
        decision_sha256="decision-hash",
    )

    assert derived["training"]["initial_step"] == 75
    assert derived["training"]["fixed_continuation_steps"] == 25
    assert derived["training"]["fixed_continuation_examples"] == 600
    assert list(derived["arms"]) == ["triggered_replay"]
    arm = derived["arms"]["triggered_replay"]
    assert arm["expected_optimizer_step"] == 75
    assert arm["replay_policy"]["tasks"] == ["count", "shape"]
    assert arm["trigger_decision"]["sha256"] == "decision-hash"
