import json

import pytest
import torch

from moonvit_glue.projector import PatchMergerProjector, ProjectorConfig


def _config(mode: str = "none"):
    return ProjectorConfig(
        vision_width=3,
        language_width=8,
        merge_factor=4,
        projector_width=8,
        residual_mode=mode,
    )


def _features():
    return [torch.randn(5, 4, 3), torch.randn(2, 4, 3)]


def _copy_base_state(source, target):
    common = {
        key: value
        for key, value in source.state_dict().items()
        if key in target.state_dict()
    }
    target.load_state_dict(common, strict=False)


@pytest.mark.parametrize("mode", ["zero_init", "gated"])
def test_residual_variants_preserve_step0_output(mode):
    torch.manual_seed(11)
    baseline = PatchMergerProjector(_config())
    torch.manual_seed(19)
    variant = PatchMergerProjector(_config(mode))
    _copy_base_state(baseline, variant)
    features = _features()
    expected = baseline(features)
    actual = variant(features)
    assert all(torch.equal(left, right) for left, right in zip(expected, actual, strict=True))
    assert variant.config.residual_mode == mode
    if mode == "zero_init":
        assert torch.count_nonzero(variant.residual.weight) == 0
    else:
        assert torch.count_nonzero(variant.residual.weight) > 0
        assert variant.residual_gate.item() == 0.0


@pytest.mark.parametrize("mode", ["zero_init", "gated"])
def test_residual_branch_receives_gradient(mode):
    torch.manual_seed(23)
    projector = PatchMergerProjector(_config(mode))
    loss = projector(_features())[0].square().mean()
    loss.backward()
    assert projector.residual.weight.grad is not None
    assert torch.isfinite(projector.residual.weight.grad).all()
    if mode == "zero_init":
        assert torch.count_nonzero(projector.residual.weight.grad) > 0
    else:
        # gate=0 时首个反向只更新 gate；branch 梯度在 gate 打开后出现。
        assert torch.count_nonzero(projector.residual.weight.grad) == 0
        assert projector.residual_gate.grad is not None
        assert torch.isfinite(projector.residual_gate.grad)
        assert projector.residual_gate.grad.abs() > 0


def test_residual_config_and_checkpoint_round_trip(tmp_path):
    projector = PatchMergerProjector(_config("gated"))
    features = _features()
    expected = projector(features)
    projector.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "projector_config.json").read_text())
    assert config["residual_mode"] == "gated"
    restored = PatchMergerProjector.from_pretrained(tmp_path)
    actual = restored(features)
    assert all(torch.equal(left, right) for left, right in zip(expected, actual, strict=True))


def test_residual_variants_add_only_expected_parameters():
    baseline = PatchMergerProjector(_config())
    zero = PatchMergerProjector(_config("zero_init"))
    gated = PatchMergerProjector(_config("gated"))
    added = 8 * 8
    assert sum(parameter.numel() for parameter in zero.parameters()) == sum(
        parameter.numel() for parameter in baseline.parameters()
    ) + added
    assert sum(parameter.numel() for parameter in gated.parameters()) == sum(
        parameter.numel() for parameter in baseline.parameters()
    ) + added + 1
    assert set(zero.state_dict()) == set(baseline.state_dict()) | {"residual.weight"}
    assert set(gated.state_dict()) == set(baseline.state_dict()) | {
        "residual.weight",
        "residual_gate",
    }


def test_residual_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="residual_mode"):
        _config("adapter")
