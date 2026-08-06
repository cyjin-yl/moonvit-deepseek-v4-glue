import json

import pytest
import torch

from moonvit_glue.projector import PatchMergerProjector, ProjectorConfig


def _features():
    return [torch.randn(5, 4, 3), torch.randn(2, 4, 3)]


def test_none_output_norm_is_backward_compatible_with_legacy_config():
    legacy = PatchMergerProjector(
        ProjectorConfig(vision_width=3, language_width=8, merge_factor=4)
    )
    explicit = PatchMergerProjector(
        ProjectorConfig(
            vision_width=3,
            language_width=8,
            merge_factor=4,
            output_norm="none",
        )
    )
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    assert legacy.state_dict().keys() == explicit.state_dict().keys()
    features = _features()
    expected = legacy(features)
    actual = explicit(features)
    assert all(torch.equal(left, right) for left, right in zip(expected, actual, strict=True))


@pytest.mark.parametrize("mode", ["layernorm", "rmsnorm"])
def test_output_norm_has_no_trainable_parameters_and_preserves_width(mode):
    baseline = PatchMergerProjector(
        ProjectorConfig(vision_width=3, language_width=8, merge_factor=4)
    )
    projector = PatchMergerProjector(
        ProjectorConfig(
            vision_width=3,
            language_width=8,
            merge_factor=4,
            output_norm=mode,
        )
    )
    assert sum(parameter.numel() for parameter in projector.parameters()) == sum(
        parameter.numel() for parameter in baseline.parameters()
    )
    assert projector.state_dict().keys() == baseline.state_dict().keys()
    outputs = projector(_features())
    assert [tuple(item.shape) for item in outputs] == [(5, 8), (2, 8)]
    assert all(torch.isfinite(item).all() for item in outputs)
    if mode == "layernorm":
        assert torch.allclose(outputs[0].mean(dim=-1), torch.zeros(5), atol=2e-5)
    else:
        rms = outputs[0].pow(2).mean(dim=-1).sqrt()
        # 冻结 eps=1e-5 后 RMS 会略低于 1，容差覆盖该确定性数值误差。
        assert torch.allclose(rms, torch.ones(5), atol=5e-4)


def test_output_norm_config_round_trip(tmp_path):
    projector = PatchMergerProjector(
        ProjectorConfig(vision_width=3, language_width=8, merge_factor=4, output_norm="rmsnorm")
    )
    features = _features()
    expected = projector(features)
    projector.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "projector_config.json").read_text())
    assert config["output_norm"] == "rmsnorm"
    restored = PatchMergerProjector.from_pretrained(tmp_path)
    actual = restored(features)
    assert all(torch.equal(left, right) for left, right in zip(expected, actual, strict=True))


def test_output_norm_rejects_unknown_mode():
    with pytest.raises(ValueError, match="output_norm"):
        ProjectorConfig(vision_width=3, language_width=8, output_norm="batchnorm")
