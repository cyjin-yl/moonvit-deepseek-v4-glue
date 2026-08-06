import math

import pytest
import torch

from moonvit_glue.representation_retention import (
    compare_geometry,
    decide_representation_action,
    pairwise_geometry,
    summarize_representation,
    summarize_token_sequences,
)


def test_representation_summary_recovers_rank_and_pairwise_geometry():
    values = torch.eye(4, dtype=torch.float64)
    summary = summarize_representation(values)

    assert summary["samples"] == 4
    assert summary["width"] == 4
    assert summary["effective_rank_participation"] == pytest.approx(3.0)
    assert summary["effective_rank_entropy"] == pytest.approx(3.0)
    assert summary["top1_variance_fraction"] == pytest.approx(1 / 3)
    assert summary["relative_between_image_spread"] == pytest.approx(
        math.sqrt(3) / 2
    )
    assert summary["pairwise_cosine_similarity"]["mean"] == pytest.approx(0.0)
    assert summary["pairwise_rms_distance"]["mean"] == pytest.approx(
        math.sqrt(0.5)
    )

    rows = pairwise_geometry(values)
    assert len(rows) == 6
    assert rows[0] == {
        "left_index": 0,
        "right_index": 1,
        "cosine_similarity": pytest.approx(0.0),
        "rms_distance": pytest.approx(math.sqrt(0.5)),
    }


def test_token_sequence_summary_separates_between_and_within_image_variance():
    sequences = [
        torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
        torch.tensor([[0.0, 2.0], [2.0, 2.0]]),
    ]
    result = summarize_token_sequences(sequences)

    assert torch.equal(
        result.pooled,
        torch.tensor([[1.0, 0.0], [1.0, 2.0]], dtype=torch.float64),
    )
    assert result.token_counts == [2, 2]
    assert result.per_image_within_rms == pytest.approx(
        [math.sqrt(0.5), math.sqrt(0.5)]
    )
    assert result.mean_within_image_rms == pytest.approx(math.sqrt(0.5))
    assert result.representation["between_image_rms"] == pytest.approx(
        math.sqrt(0.5)
    )


def test_token_sequence_summary_accepts_single_token_images():
    result = summarize_token_sequences(
        [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])]
    )

    assert result.token_counts == [1, 1]
    assert result.per_image_within_rms == [0.0, 0.0]
    assert result.mean_within_image_rms == 0.0


def test_geometry_comparison_is_scale_invariant_and_detects_reordering():
    first = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=torch.float64,
    )
    scaled = first * 7.0
    same = compare_geometry(first, scaled)
    assert same["linear_cka"] == pytest.approx(1.0)
    assert same["pairwise_distance_pearson"] == pytest.approx(1.0)

    reordered = first[[0, 2, 1, 3]]
    changed = compare_geometry(first, reordered)
    assert changed["linear_cka"] < 1.0
    assert changed["pairwise_distance_pearson"] < 1.0


def test_action_rule_requires_both_preregistered_collapse_guards():
    contract = {
        "gross_collapse_rule": {
            "requires_all": True,
            "current_over_step0_relative_spread_below": 0.25,
            "current_over_step0_effective_rank_below": 0.5,
        },
        "actions": {
            "gross_collapse": "repair",
            "diversity_retained": "margin",
        },
    }
    step0 = {
        "relative_between_image_spread": 0.8,
        "effective_rank_participation": 20.0,
    }
    collapsed = {
        "relative_between_image_spread": 0.1,
        "effective_rank_participation": 5.0,
    }
    result = decide_representation_action(step0, collapsed, contract)
    assert result == {
        "action": "repair",
        "gross_collapse": True,
        "relative_spread_ratio": pytest.approx(0.125),
        "effective_rank_ratio": pytest.approx(0.25),
        "relative_spread_guard_triggered": True,
        "effective_rank_guard_triggered": True,
    }

    one_guard = dict(collapsed, effective_rank_participation=15.0)
    result = decide_representation_action(step0, one_guard, contract)
    assert result["gross_collapse"] is False
    assert result["action"] == "margin"


@pytest.mark.parametrize(
    "values",
    [torch.ones(1, 3), torch.empty(3, 0), torch.tensor([[1.0, float('nan')]] * 2)],
)
def test_representation_summary_fails_closed(values):
    with pytest.raises(ValueError):
        summarize_representation(values)
