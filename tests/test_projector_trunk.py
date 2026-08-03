"""Trunk warm-start: pre_norm + linear_1 transfer across backbones, linear_2 stays fresh."""

import pytest
import torch

from moonvit_glue.projector import PatchMergerProjector, ProjectorConfig


def _donor(tmp_path, **overrides):
    config = ProjectorConfig(
        vision_width=overrides.get("vision_width", 4),
        language_width=overrides.get("language_width", 7),
        merge_factor=2,
        projector_width=overrides.get("projector_width", 8),
    )
    donor = PatchMergerProjector(config)
    with torch.no_grad():  # make donor weights recognizable
        donor.pre_norm.weight.fill_(0.5)
        donor.linear_1.weight.fill_(0.25)
    donor.save_pretrained(tmp_path)
    return donor


def test_load_trunk_transfers_language_agnostic_weights(tmp_path):
    donor = _donor(tmp_path, language_width=7)
    target_config = ProjectorConfig(vision_width=4, language_width=4096, merge_factor=2, projector_width=8)
    target = PatchMergerProjector(target_config)
    fresh_linear_2 = target.linear_2.weight.detach().clone()

    target.load_trunk(tmp_path)

    assert torch.equal(target.pre_norm.weight, donor.pre_norm.weight)
    assert torch.equal(target.linear_1.weight, donor.linear_1.weight)
    assert torch.equal(target.linear_1.bias, donor.linear_1.bias)
    # linear_2 output width differs across backbones: fresh init must survive
    assert torch.equal(target.linear_2.weight, fresh_linear_2)
    assert target.linear_2.out_features == 4096


def test_load_trunk_rejects_vision_width_mismatch(tmp_path):
    _donor(tmp_path, vision_width=4)
    v1_target = PatchMergerProjector(
        ProjectorConfig(vision_width=1152, language_width=4096, merge_factor=2, projector_width=8)
    )
    with pytest.raises(ValueError, match="vision_width"):
        v1_target.load_trunk(tmp_path)


def test_load_trunk_rejects_projector_width_mismatch(tmp_path):
    _donor(tmp_path, projector_width=8)
    target = PatchMergerProjector(
        ProjectorConfig(vision_width=4, language_width=4096, merge_factor=2, projector_width=16)
    )
    with pytest.raises(ValueError, match="projector_width"):
        target.load_trunk(tmp_path)
