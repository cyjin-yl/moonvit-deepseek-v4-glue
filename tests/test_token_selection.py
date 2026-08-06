import pytest
import torch

from moonvit_glue.token_selection import select_visual_tokens


def test_prefix_and_uniform_are_deterministic_and_cover_expected_axis():
    features = torch.arange(20 * 2, dtype=torch.float32).reshape(20, 2)
    prefix = select_visual_tokens(features, 4, "prefix")
    uniform = select_visual_tokens(features, 4, "uniform")
    assert torch.equal(prefix, features[:4])
    assert torch.equal(uniform, features[[0, 6, 13, 19]])
    assert torch.equal(uniform, select_visual_tokens(features, 4, "uniform"))


def test_mean_pool_preserves_spatial_sequence_coverage():
    features = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    pooled = select_visual_tokens(features, 4, "mean_pool")
    assert torch.equal(pooled[:, 0], torch.tensor([0.5, 2.5, 4.5, 6.5]))


def test_short_sequence_is_unchanged_for_all_modes():
    features = torch.randn(3, 2)
    for mode in ("prefix", "uniform", "mean_pool"):
        result = select_visual_tokens(features, 8, mode)
        assert torch.equal(result, features)
        assert result.is_contiguous()


@pytest.mark.parametrize("kwargs", [{"max_tokens": 0}, {"max_tokens": 2, "mode": "bad"}])
def test_selection_validates_arguments(kwargs):
    with pytest.raises(ValueError):
        select_visual_tokens(torch.ones(3, 2), **kwargs)
