import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from moonvit_glue.moonvit_v2 import (
    build_moonvit_v2,
    load_moonvit_v2_encoder,
    register_sdpa_attention,
)

TINY_CONFIG = dict(
    vt_hidden_size=64,
    vt_num_hidden_layers=2,
    vt_num_attention_heads=4,
    vt_intermediate_size=128,
    qkv_hidden_size=64,
)


def _tiny_encoder(**overrides):
    cfg = {**TINY_CONFIG, **overrides}
    torch.manual_seed(0)
    return load_moonvit_v2_encoder(attn_implementation="eager", **cfg)


def test_v2_tower_forward_matches_patch_group_contract():
    encoder = _tiny_encoder()

    features = encoder(torch.randn(16, 3, 14, 14), torch.tensor([[1, 4, 4]]))

    assert isinstance(features, list) and len(features) == 1
    assert features[0].shape == (4, 4, 64)
    assert features[0].requires_grad is False
    assert encoder.model.training is False
    assert all(not p.requires_grad for p in encoder.model.parameters())


def test_v2_real_config_shape_reports_1024_width():
    encoder = load_moonvit_v2_encoder(attn_implementation="eager")

    assert encoder.vision_width == 1024
    assert encoder.merge_factor == 4
    params = sum(p.numel() for p in encoder.model.parameters())
    assert params == pytest.approx(401.2e6, rel=0.01)


def test_v2_sdpa_attention_matches_eager():
    register_sdpa_attention()
    torch.manual_seed(0)
    eager = build_moonvit_v2(attn_implementation="eager", **TINY_CONFIG)
    sdpa = build_moonvit_v2(attn_implementation="sdpa", **TINY_CONFIG)
    sdpa.load_state_dict(eager.state_dict())
    eager.eval()
    sdpa.eval()

    pixel_values = torch.randn(16, 3, 14, 14)
    grid = torch.tensor([[1, 4, 4]])
    with torch.no_grad():
        eager_out = eager(pixel_values, grid)[0]
        sdpa_out = sdpa(pixel_values, grid)[0]

    torch.testing.assert_close(sdpa_out, eager_out, rtol=1e-5, atol=1e-5)


def test_v2_processor_adapter_supplies_glue_contract_keys():
    encoder = _tiny_encoder()
    image = Image.new("RGB", (56, 42), color=(128, 64, 200))

    batch = encoder.preprocess(image, device="cpu")

    assert set(batch) == {"pixel_values", "image_grid_hws"}
    assert batch["pixel_values"].ndim == 4  # (tokens, 3, 14, 14)
    assert batch["pixel_values"].shape[1:] == (3, 14, 14)
    assert batch["image_grid_hws"].shape == (1, 3)
    tokens = int(batch["image_grid_hws"][0].prod())
    assert batch["pixel_values"].shape[0] == tokens

    features = encoder(batch["pixel_values"], batch["image_grid_hws"])
    groups = int(tokens // 4)
    assert features[0].shape == (groups, 4, 64)


def test_v2_state_dict_loads_with_or_without_vision_tower_prefix(tmp_path):
    reference = _tiny_encoder()
    pixel_values = torch.randn(16, 3, 14, 14)
    grid = torch.tensor([[1, 4, 4]])
    with torch.no_grad():
        expected = reference(pixel_values, grid)[0]

    for name, prefix in (("bare", ""), ("prefixed", "vision_tower.")):
        state = {
            prefix + key: value.clone()
            for key, value in reference.model.state_dict().items()
        }
        path = tmp_path / f"{name}.safetensors"
        save_file(state, str(path))

        loaded = _tiny_encoder()
        from moonvit_glue.moonvit_v2 import load_vision_tower_state_dict

        loaded.model.load_state_dict(load_vision_tower_state_dict(path), strict=True)
        loaded.model.eval()
        with torch.no_grad():
            actual = loaded(pixel_values, grid)[0]
        torch.testing.assert_close(actual, expected)


def test_v2_weights_path_loads_strictly(tmp_path):
    reference = _tiny_encoder()
    path = tmp_path / "tower.safetensors"
    save_file(reference.model.state_dict(), str(path))

    loaded = load_moonvit_v2_encoder(
        path, attn_implementation="eager", **TINY_CONFIG
    )

    pixel_values = torch.randn(16, 3, 14, 14)
    grid = torch.tensor([[1, 4, 4]])
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(pixel_values, grid)[0], reference(pixel_values, grid)[0]
        )
