import pytest
import torch

from moonvit_glue.moonvit import MoonViTEncoder


class FakeMoonViT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, pixel_values, image_grid_hws):
        del image_grid_hws
        return [pixel_values.new_ones((3, 4, 6)) * self.weight]


def test_frozen_moonvit_returns_valid_patch_groups_without_an_autograd_graph():
    raw = FakeMoonViT()
    encoder = MoonViTEncoder(
        raw,
        vision_width=6,
        merge_factor=4,
        freeze=True,
    )

    features = encoder(torch.randn(1, 3, 4, 4), torch.tensor([[1, 2, 2]]))

    assert features[0].shape == (3, 4, 6)
    assert features[0].requires_grad is False
    assert raw.training is False
    assert raw.weight.requires_grad is False


def test_moonvit_output_contract_is_checked_before_projector_use():
    encoder = MoonViTEncoder(FakeMoonViT(), vision_width=1152, merge_factor=4)

    with pytest.raises(ValueError, match=r"expected \[tokens, 4, 1152\]"):
        encoder(torch.randn(1), torch.tensor([[1, 1, 1]]))
