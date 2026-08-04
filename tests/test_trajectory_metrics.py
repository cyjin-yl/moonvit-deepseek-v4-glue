"""Trajectory summaries retain exact numerators and pair denominators."""

import torch

from moonvit_glue.trajectory_metrics import (
    control_image_records,
    derangement_indices,
    resolve_control_features,
    summarize_synthetic_rows,
)


def test_synthetic_summary_reports_accuracy_pairs_and_prediction_flips():
    rows = [
        {"id": "p1-a", "pair_id": "p1", "task": "color", "correct": True, "prediction": "red"},
        {"id": "p1-b", "pair_id": "p1", "task": "color", "correct": True, "prediction": "blue"},
        {"id": "p2-a", "pair_id": "p2", "task": "color", "correct": True, "prediction": "green"},
        {"id": "p2-b", "pair_id": "p2", "task": "color", "correct": False, "prediction": "green"},
    ]

    summary = summarize_synthetic_rows(rows)
    assert summary["accuracy"] == {"numerator": 3, "denominator": 4, "value": 0.75}
    assert summary["paired_accuracy"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    assert summary["answer_flip_accuracy"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    assert summary["prediction_flip_rate"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    assert summary["failures"] == {"numerator": 0, "denominator": 4, "value": 0.0}


def test_failures_remain_in_the_denominator():
    rows = [
        {"id": "p1-a", "pair_id": "p1", "task": "ocr", "correct": True, "prediction": "A2"},
        {"id": "p1-b", "pair_id": "p1", "task": "ocr", "correct": False, "prediction": None, "failure": "OOM"},
    ]
    summary = summarize_synthetic_rows(rows)
    assert summary["accuracy"]["denominator"] == 2
    assert summary["paired_accuracy"]["denominator"] == 1
    assert summary["failures"]["numerator"] == 1


def test_seeded_derangement_has_no_fixed_points():
    first = derangement_indices(17, seed=9)
    assert first == derangement_indices(17, seed=9)
    assert sorted(first) == list(range(17))
    assert all(index != other for index, other in enumerate(first))


def test_patch_permutation_preserves_values_but_changes_spatial_order():
    features = {
        "sample": [torch.arange(24).reshape(6, 2, 2).float()],
        "other": [torch.full((6, 2, 2), 99.0)],
        "control:selection:blank": [torch.zeros((6, 2, 2))],
        "control:shape-blank": [torch.full((6, 2, 2), -1.0)],
        "control:selection:same": [torch.ones((6, 2, 2))],
    }
    cache_get = lambda sample_id: [tensor.clone() for tensor in features[sample_id]]
    control = {
        "shuffled_image_id": "other",
        "patch_permutation": {"seed": 123},
    }
    permuted = resolve_control_features(
        "patch_permutation", "sample", "selection", control, cache_get
    )[0]
    original = features["sample"][0]
    assert not torch.equal(permuted, original)
    assert torch.equal(permuted.flatten().sort().values, original.flatten().sort().values)
    assert torch.equal(
        resolve_control_features("blank", "sample", "selection", control, cache_get)[0],
        features["control:selection:blank"][0],
    )
    control["blank_image_id"] = "control:shape-blank"
    assert torch.equal(
        resolve_control_features("blank", "sample", "selection", control, cache_get)[0],
        features["control:shape-blank"][0],
    )
    assert resolve_control_features("blind", "sample", "selection", control, cache_get) is None


def test_control_image_records_deduplicate_fixed_images():
    controls = [
        {"split": "selection", "blank_image": "blank.png", "same_image": "same.png"},
        {"split": "selection", "blank_image": "blank.png", "same_image": "same.png"},
    ]
    assert control_image_records(controls) == [
        {"id": "control:selection:blank", "image": "blank.png", "question": "control image", "answers": ["n/a"], "metric": "exact_match"},
        {"id": "control:selection:same", "image": "same.png", "question": "control image", "answers": ["n/a"], "metric": "exact_match"},
    ]
