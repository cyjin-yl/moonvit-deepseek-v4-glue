"""逐层机制分析的纯函数必须先通过泄漏与干预合同。"""

import math

import pytest
import torch

from moonvit_glue.mechanism_probe import (
    apply_linear_probe,
    fit_linear_probe,
    pair_bootstrap_accuracy_delta,
    pair_bootstrap_mean,
    pair_label_permutation_test,
    aligned_effect_delta,
    patch_hidden_output,
    last_active_indices,
    masked_token_mean,
    square_grid_region_mask,
    pool_token_grid,
    select_complete_task_pairs,
)


def test_aligned_effect_delta_matches_ids_and_preserves_pairs():
    rows_a = [
        {"id": "a", "pair_id": "p1", "effect_vs_counterfactual": 0.6},
        {"id": "b", "pair_id": "p1", "effect_vs_counterfactual": 0.2},
    ]
    rows_b = [
        {"id": "b", "pair_id": "p1", "effect_vs_counterfactual": -0.1},
        {"id": "a", "pair_id": "p1", "effect_vs_counterfactual": 0.1},
    ]
    result = aligned_effect_delta(rows_a, rows_b)
    assert result == [
        {"id": "a", "pair_id": "p1", "effect_delta": 0.5},
        {"id": "b", "pair_id": "p1", "effect_delta": 0.30000000000000004},
    ]


def test_aligned_effect_delta_rejects_mismatched_samples():
    with pytest.raises(ValueError, match="sample IDs"):
        aligned_effect_delta(
            [{"id": "a", "pair_id": "p1", "effect_vs_counterfactual": 1.0}],
            [{"id": "b", "pair_id": "p1", "effect_vs_counterfactual": 0.0}],
        )


def test_select_complete_task_pairs_rejects_incomplete_pair():
    rows = [
        {"id": "a", "pair_id": "p", "pair_variant": "a", "task": "shape"},
        {"id": "x", "pair_id": "q", "pair_variant": "a", "task": "color"},
    ]
    with pytest.raises(ValueError, match="complete a/b pair"):
        select_complete_task_pairs(rows, "shape")


def test_select_complete_task_pairs_is_deterministic():
    rows = [
        {"id": "p2-b", "pair_id": "p2", "pair_variant": "b", "task": "shape"},
        {"id": "p1-a", "pair_id": "p1", "pair_variant": "a", "task": "shape"},
        {"id": "p2-a", "pair_id": "p2", "pair_variant": "a", "task": "shape"},
        {"id": "p1-b", "pair_id": "p1", "pair_variant": "b", "task": "shape"},
    ]
    selected = select_complete_task_pairs(rows, "shape")
    assert [row["id"] for row in selected] == ["p1-a", "p1-b", "p2-a", "p2-b"]


def test_pool_token_grid_preserves_fixed_spatial_contract():
    tokens = torch.arange(2 * 16 * 3, dtype=torch.float32).reshape(2, 16, 3)
    assert pool_token_grid(tokens, "global_mean").shape == (2, 3)
    assert pool_token_grid(tokens, "center_mean").shape == (2, 3)
    pooled = pool_token_grid(tokens, "spatial_2x2")
    assert pooled.shape == (2, 12)
    expected_top_left = tokens[:, [0, 1, 4, 5]].mean(dim=1)
    assert torch.equal(pooled[:, :3], expected_top_left)


def test_pool_token_grid_rejects_non_square_token_axis():
    with pytest.raises(ValueError, match="square token grid"):
        pool_token_grid(torch.zeros(2, 6, 3), "global_mean")


def test_ridge_linear_probe_separates_classes_and_records_fixed_alpha():
    features = torch.tensor(
        [[-2.0, -1.0], [-1.0, -2.0], [2.0, 1.0], [1.0, 2.0]]
    )
    labels = torch.tensor([0, 0, 1, 1])
    probe = fit_linear_probe(features, labels, class_count=2, alpha=1.0)
    prediction, scores = apply_linear_probe(probe, features)
    assert torch.equal(prediction, labels)
    assert scores.shape == (4, 2)
    assert probe.alpha == 1.0


def test_pair_bootstrap_delta_resamples_pairs_not_individual_rows():
    result = pair_bootstrap_accuracy_delta(
        correct_a=[True, False, True, False],
        correct_b=[False, False, False, False],
        pair_ids=["p1", "p1", "p2", "p2"],
        seed=7,
        samples=200,
    )
    assert result["pairs"] == 2
    assert result["records"] == 4
    assert math.isclose(result["mean_delta"], 0.5)
    assert result["ci95_low"] <= result["mean_delta"] <= result["ci95_high"]


def test_pair_bootstrap_mean_aggregates_two_directions_before_resampling():
    result = pair_bootstrap_mean(
        values=[1.0, 3.0, -1.0, 1.0],
        pair_ids=["p1", "p1", "p2", "p2"],
        seed=11,
        samples=200,
    )
    assert result["pairs"] == 2
    assert result["records"] == 4
    assert math.isclose(result["mean"], 1.0)


def test_pair_label_permutation_test_preserves_complete_pair_units():
    result = pair_label_permutation_test(
        predictions=[0, 1, 2, 3, 0, 2],
        labels=[0, 1, 2, 3, 0, 2],
        pair_ids=["p1", "p1", "p2", "p2", "p3", "p3"],
        seed=13,
        samples=200,
    )
    assert result["pairs"] == 3
    assert result["records"] == 6
    assert result["observed_accuracy"] == 1.0
    assert 0.0 < result["p_value"] <= 1.0
    assert result["null_ci95_high"] <= 1.0


def test_patch_hidden_output_supports_tensor_and_tuple_outputs():
    hidden = torch.zeros(2, 4, 3)
    donor = torch.ones_like(hidden)
    mask = torch.tensor(
        [[True, False, False, False], [False, False, True, False]]
    )
    patched = patch_hidden_output(hidden, donor, mask)
    assert torch.equal(patched[0, 0], torch.ones(3))
    assert torch.equal(patched[0, 1], torch.zeros(3))

    tuple_output = (hidden, torch.tensor(5))
    patched_tuple = patch_hidden_output(tuple_output, donor, mask)
    assert torch.equal(patched_tuple[0], patched)
    assert patched_tuple[1].item() == 5


def test_patch_hidden_output_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="donor hidden shape"):
        patch_hidden_output(
            torch.zeros(1, 2, 3), torch.zeros(1, 3, 3), torch.ones(1, 2, dtype=torch.bool)
        )


def test_last_active_and_masked_mean_handle_left_padding():
    mask = torch.tensor([[False, True, True], [True, True, False]])
    assert torch.equal(last_active_indices(mask), torch.tensor([2, 1]))
    hidden = torch.tensor(
        [[[9.0], [1.0], [3.0]], [[2.0], [4.0], [9.0]]]
    )
    assert torch.equal(masked_token_mean(hidden, mask), torch.tensor([[2.0], [3.0]]))


def test_square_grid_region_mask_partitions_center_and_outer():
    center = square_grid_region_mask(100, "center")
    outer = square_grid_region_mask(100, "outer")
    assert center.dtype == torch.bool
    assert center.sum().item() == 16
    assert torch.equal(center.logical_not(), outer)
