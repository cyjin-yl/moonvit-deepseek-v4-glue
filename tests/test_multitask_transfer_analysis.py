from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from analyze_multitask_transfer import classify_checkpoint_transfer, pair_metric_rows


def test_pair_metric_rows_preserves_complete_pair_as_bootstrap_unit() -> None:
    preference = [
        {"pair_id": "p0", "correct_margin": margin}
        for margin in (2.0, 1.0)
    ] + [
        {"pair_id": "p1", "correct_margin": margin}
        for margin in (1.0, -1.0)
    ]
    generation = [
        {"pair_id": "p0", "correct": True, "normalized_prediction": prediction}
        for prediction in ("red", "blue")
    ] + [
        {"pair_id": "p1", "correct": correct, "normalized_prediction": "red"}
        for correct in (True, False)
    ]

    assert pair_metric_rows(preference, "paired_preference") == [
        {"id": "p0", "score": 1.0},
        {"id": "p1", "score": 0.0},
    ]
    assert pair_metric_rows(generation, "generation_paired") == [
        {"id": "p0", "score": 1.0},
        {"id": "p1", "score": 0.0},
    ]
    assert pair_metric_rows(generation, "prediction_flip") == [
        {"id": "p0", "score": 1.0},
        {"id": "p1", "score": 0.0},
    ]


def test_transfer_classification_keeps_nonmonotonic_valid_checkpoint():
    flags = {
        "step25": {
            task: {"validated_preference_transfer": task in {"a", "b", "c", "shape"}}
            for task in ("a", "b", "c", "d", "e", "shape")
        },
        "step100": {
            task: {"validated_preference_transfer": task == "shape"}
            for task in ("a", "b", "c", "d", "e", "shape")
        },
    }

    decision = classify_checkpoint_transfer(flags, shape_task="shape")

    assert decision["broad_non_shape_transfer_supported"] is True
    assert decision["broad_supporting_checkpoints"] == ["step25"]
    assert decision["shape_specific_checkpoints"] == ["step100"]
