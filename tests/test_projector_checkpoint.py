import json

import pytest
import torch

from moonvit_glue.projector import PatchMergerProjector, ProjectorConfig


def test_projector_checkpoint_round_trip_preserves_outputs(tmp_path):
    config = ProjectorConfig(
        vision_width=3,
        language_width=5,
        merge_factor=2,
        projector_width=6,
    )
    projector = PatchMergerProjector(config)
    feature_groups = [torch.randn(4, 2, 3), torch.randn(1, 2, 3)]
    expected = projector(feature_groups)

    projector.save_pretrained(tmp_path)
    restored = PatchMergerProjector.from_pretrained(tmp_path)
    actual = restored(feature_groups)

    assert json.loads((tmp_path / "projector_config.json").read_text())["language_width"] == 5
    assert (tmp_path / "projector.safetensors").exists()
    assert len(actual) == 2
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_projector_rejects_wrong_moonvit_feature_shape():
    projector = PatchMergerProjector(
        ProjectorConfig(vision_width=3, language_width=5, merge_factor=4)
    )

    with pytest.raises(ValueError, match=r"expected \[tokens, 4, 3\]"):
        projector([torch.randn(7, 3, 3)])
