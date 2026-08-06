import json

import pytest
import torch

from moonvit_glue.projector import PatchMergerProjector, ProjectorConfig


def _features():
    return [torch.randn(5, 4, 3), torch.randn(2, 4, 3)]


def test_k3_v2_uses_bias_free_mlp_and_trainable_post_rmsnorm():
    projector = PatchMergerProjector(
        ProjectorConfig(
            vision_width=3,
            language_width=8,
            merge_factor=4,
            projector_variant="kimi_k3_v2",
        )
    )
    assert projector.config.projector_variant == "kimi_k3_v2"
    assert isinstance(projector.pre_norm, torch.nn.Identity)
    assert projector.linear_1.bias is None
    assert projector.linear_2.bias is None
    assert isinstance(projector.output_norm, torch.nn.RMSNorm)
    assert projector.output_norm.weight.requires_grad
    outputs = projector(_features())
    assert [tuple(item.shape) for item in outputs] == [(5, 8), (2, 8)]
    assert all(torch.isfinite(item).all() for item in outputs)


def test_k3_v2_checkpoint_round_trip_preserves_variant(tmp_path):
    torch.manual_seed(17)
    projector = PatchMergerProjector(
        ProjectorConfig(
            vision_width=3,
            language_width=8,
            merge_factor=4,
            projector_variant="kimi_k3_v2",
        )
    )
    features = _features()
    expected = projector(features)
    projector.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "projector_config.json").read_text())
    assert config["projector_variant"] == "kimi_k3_v2"
    restored = PatchMergerProjector.from_pretrained(tmp_path)
    actual = restored(features)
    assert all(torch.equal(left, right) for left, right in zip(expected, actual, strict=True))


def test_k3_v2_rejects_legacy_normalization_or_residual_options():
    with pytest.raises(ValueError, match="post RMSNorm"):
        ProjectorConfig(
            vision_width=3,
            language_width=8,
            projector_variant="kimi_k3_v2",
            output_norm="layernorm",
        )
    with pytest.raises(ValueError, match="residual_mode"):
        ProjectorConfig(
            vision_width=3,
            language_width=8,
            projector_variant="kimi_k3_v2",
            residual_mode="gated",
        )

