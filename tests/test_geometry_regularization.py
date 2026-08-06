import math

import pytest
import torch

from moonvit_glue.geometry_regularization import (
    geometry_regularization_loss,
    global_gradient_norm,
    pool_projector_batch,
)


def test_pool_projector_batch_concatenates_groups_before_mean():
    outputs = [
        [torch.tensor([[1.0, 0.0], [3.0, 0.0]]), torch.tensor([[5.0, 0.0]])],
        [torch.tensor([[0.0, 2.0], [0.0, 4.0]])],
    ]

    pooled = pool_projector_batch(outputs)

    assert torch.equal(pooled, torch.tensor([[3.0, 0.0], [0.0, 3.0]]))


def test_exact_reference_has_zero_loss_and_zero_gradient():
    reference = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    current = reference.clone().requires_grad_(True)

    result = geometry_regularization_loss(current, reference)
    result.total.backward()

    assert float(result.total) == pytest.approx(0.0, abs=1e-12)
    assert torch.count_nonzero(current.grad) == 0
    assert result.metrics["rms_ratio"] == pytest.approx(1.0)
    assert result.metrics["relative_spread_ratio"] == pytest.approx(1.0)


def test_common_direction_explosion_triggers_scale_and_spread_terms():
    reference = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    current = (reference + torch.tensor([100.0, 100.0])).requires_grad_(True)

    result = geometry_regularization_loss(current, reference)

    assert float(result.scale) > 20.0
    assert float(result.relative_spread) > 20.0
    assert float(result.centered_gram) == pytest.approx(0.0, abs=1e-10)
    assert result.metrics["rms_ratio"] > 100.0
    assert result.metrics["relative_spread_ratio"] < 0.01


def test_normalized_centered_gram_allows_rotation_and_detects_rank_collapse():
    reference = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    rotated = reference @ rotation
    same_geometry = geometry_regularization_loss(rotated, reference)
    assert float(same_geometry.total) == pytest.approx(0.0, abs=1e-12)

    collapsed = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    changed = geometry_regularization_loss(collapsed, reference)
    assert float(changed.centered_gram) > 0.1


def test_global_gradient_norm_uses_all_tensors():
    result = global_gradient_norm(
        [torch.tensor([3.0, 4.0]), None, torch.tensor([12.0])]
    )
    assert float(result) == pytest.approx(13.0)


@pytest.mark.parametrize(
    "current,reference",
    [
        (torch.ones(1, 3), torch.ones(1, 3)),
        (torch.ones(2, 3), torch.ones(2, 4)),
        (torch.tensor([[1.0, float("nan")]] * 2), torch.ones(2, 2)),
    ],
)
def test_geometry_loss_fails_closed(current, reference):
    with pytest.raises(ValueError):
        geometry_regularization_loss(current, reference)
