"""Batched visual prompts must not confuse padding with image placeholders."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from eval_checkpoint_trajectory import (
    apply_screening_overrides,
    benchmark_summary,
    checkpoint_training_metrics,
    find_random_checkpoint,
    scoring_record_for_condition,
    visual_prompt_batch,
)


class FakeTokenizer:
    pad_token_id = 99

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(1, len(text.split()) + 1))


def test_visual_batch_uses_safe_padding_when_pad_id_is_placeholder():
    input_ids, attention_mask = visual_prompt_batch(
        FakeTokenizer(),
        "User: {image} {question} Assistant:",
        ["short", "a much longer question"],
        placeholder_token_id=99,
        device=torch.device("cpu"),
    )
    assert int(input_ids.eq(99).sum()) == 2
    assert int(attention_mask.sum()) < input_ids.numel()


def test_paired_image_generation_is_scored_against_the_swapped_visual_answer():
    record = {"id": "p-a", "answers": ["red"], "question": "color?"}
    pair_index = {
        "p-a": {
            "counterfactual_answer": "blue",
            "paired_image_id": "p-b",
        }
    }

    scored = scoring_record_for_condition(
        record, "paired_counterfactual_image", pair_index
    )

    assert scored["answers"] == ["blue"]
    assert record["answers"] == ["red"]


def test_generation_screening_override_is_written_into_every_denominator():
    config = {
        "checkpoints": [{"id": "c0"}, {"id": "c1"}],
        "aliases": [{"id": "current", "source": "c1"}],
        "datasets": [
            {"name": "synthetic", "expected_records": 2400},
            {"name": "benchmarks", "expected_records": 700},
        ],
        "heldout_shuffle_loss": {"expected_records": 32},
    }

    screened = apply_screening_overrides(
        config, limit=4, checkpoint_ids=["c0"]
    )

    assert [row["id"] for row in screened["checkpoints"]] == ["c0"]
    assert screened["aliases"] == []
    assert [row["expected_records"] for row in screened["datasets"]] == [4, 4]
    assert screened["heldout_shuffle_loss"]["expected_records"] == 4


def test_generation_screening_override_caps_limit_at_fixed_dataset_size():
    config = {
        "checkpoints": [{"id": "c0"}],
        "aliases": [],
        "datasets": [
            {"name": "synthetic", "expected_records": 2400},
            {"name": "benchmarks", "expected_records": 700},
        ],
        "heldout_shuffle_loss": {"expected_records": 32},
    }

    screened = apply_screening_overrides(
        config, limit=128, checkpoint_ids=None
    )

    assert [row["limit"] for row in screened["datasets"]] == [128, 128]
    assert [row["expected_records"] for row in screened["datasets"]] == [128, 128]
    assert screened["heldout_shuffle_loss"]["limit"] == 32
    assert screened["heldout_shuffle_loss"]["expected_records"] == 32


def test_checkpoint_metrics_support_jsonl_adaptation_history(tmp_path: Path):
    history = tmp_path / "train_history.jsonl"
    history.write_text(
        "".join(
            f'{{"step": {step}, "loss": {float(step)}}}\n'
            for step in range(1, 81)
        ),
        encoding="utf-8",
    )

    metrics = checkpoint_training_metrics(
        {
            "kind": "trained",
            "history_path": str(history),
            "history_max_step": 50,
        }
    )

    assert metrics["last_train_loss"] == 50.0
    assert metrics["mean_last_50_train_loss"] == 25.5
    assert len(metrics["history_sha256"]) == 64


def test_balanced_projector_eval_checkpoints_keep_required_training_provenance():
    config_path = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "perception-multitask-projector-eval-v1.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "id",
        "kind",
        "optimizer_steps",
        "examples_seen",
        "effective_epochs",
    }

    assert all(required <= set(checkpoint) for checkpoint in config["checkpoints"])


def test_saved_checkpoint_trajectory_does_not_require_random_control():
    trained = [{"id": "step-1", "kind": "trained"}]

    assert find_random_checkpoint(trained) is None
    assert find_random_checkpoint(
        [*trained, {"id": "random", "kind": "random", "random_seed": 7}]
    )["id"] == "random"


def test_grounding_summary_reports_parse_and_coordinate_collapse_counts():
    rows = [
        {
            "scores": {
                "grounding": {
                    "parse_ok": parse_ok,
                    "correct": correct,
                    "prediction_point": point,
                    "error": 0.0 if point else None,
                }
            },
            "score": float(correct),
            "metric": "accuracy",
            "normalized_prediction": prediction,
            "failure": None,
        }
        for parse_ok, correct, point, prediction in (
            (True, True, [500.0, 500.0], "500 500"),
            (True, False, [500.0, 500.0], "500 500"),
            (True, False, [100.0, 200.0], "100 200"),
            (False, False, None, "description"),
        )
    ]

    summary = benchmark_summary(rows)

    assert summary["coordinate_parse"] == {
        "numerator": 3,
        "denominator": 4,
        "value": 0.75,
    }
    assert summary["most_common_coordinate"] == [500.0, 500.0]
    assert summary["most_common_coordinate_ratio"] == {
        "numerator": 2,
        "denominator": 4,
        "value": 0.5,
    }
