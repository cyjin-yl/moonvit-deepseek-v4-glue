"""MoonViT-V2 (Kimi K3 vision tower) adapter for moonvit_glue.

Provides the same contract as ``moonvit.MoonViTEncoder`` — preprocessing plus
forward returning one ``[tokens, merge, width]`` feature tensor per image —
but for the K3 MoonViT3d tower (vision width 1024, 2x2 merge, ``sd2_tpool``).

The tower is built from vendored code (``vendor.kimi_k3``), so neither the
full Kimi-K3 repository nor ``trust_remote_code`` downloads are required.
Weights come from a standalone safetensors file; keys may be bare
(``patch_embed...``) or carry the ``vision_tower.`` prefix used inside the
full K3 checkpoints — both load strictly.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .moonvit import MoonViTEncoder
from .vendor.kimi_k3.configuration_kimi_k3 import KimiK3VisionConfig
from .vendor.kimi_k3.kimi_k3_vision_processing import KimiK3VisionProcessor
from .vendor.kimi_k3.modeling_moonvit_v2 import (
    VL_VISION_ATTENTION_FUNCTIONS,
    MoonViT3dPretrainedModel,
    VisionTowerConfig,
)

_VISION_TOWER_PREFIX = "vision_tower."
_DEFAULT_PROCESSOR_CONFIG = (
    Path(__file__).parent / "vendor" / "kimi_k3" / "preprocessor_config.json"
)


def sdpa_varlen_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_cu_seqlens: Tensor | None = None,
    k_cu_seqlens: Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    deterministic: bool = False,
) -> Tensor:
    """Non-causal varlen attention via ``F.scaled_dot_product_attention``.

    Drop-in replacement for the flash-attn ``multihead_attention`` in the
    vendored K3 code, for hardware without flash-attention support (e.g.
    sm_70 V100). ``q``/``k``/``v`` are packed ``(total_tokens, heads,
    head_dim)``; each ``[start, end)`` segment of ``q_cu_seqlens`` is
    attended independently. Sequences share q/k segment boundaries here
    (self-attention within one image).
    """
    if q_cu_seqlens is None:
        q_cu_seqlens = torch.tensor(
            [0, q.shape[0]], dtype=torch.long, device=q.device
        )
    out = torch.empty_like(q)
    bounds = q_cu_seqlens.tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        # (1, heads, tokens, head_dim)
        qs = q[start:end].unsqueeze(0).transpose(1, 2)
        ks = k[start:end].unsqueeze(0).transpose(1, 2)
        vs = v[start:end].unsqueeze(0).transpose(1, 2)
        attended = F.scaled_dot_product_attention(qs, ks, vs)
        out[start:end] = attended.transpose(1, 2).squeeze(0)
    return out.flatten(start_dim=-2)


def register_sdpa_attention() -> None:
    """Make ``attn_implementation="sdpa"`` usable in the vendored tower."""
    VL_VISION_ATTENTION_FUNCTIONS.setdefault("sdpa", sdpa_varlen_attention)


def build_moonvit_v2(
    *,
    attn_implementation: str = "eager",
    **config_overrides: Any,
) -> MoonViT3dPretrainedModel:
    """Instantiate MoonViT3d from a (possibly overridden) K3 vision config.

    ``attn_implementation`` is applied per encoder block after construction:
    the vendored attention registry only knows ``flash_attention_2`` and
    ``eager``; ``"sdpa"`` is registered on demand and works everywhere.
    """
    if attn_implementation not in ("eager", "sdpa", "flash_attention_2"):
        raise ValueError(
            f"Unsupported attn_implementation {attn_implementation!r}; "
            "use 'eager', 'sdpa', or 'flash_attention_2'"
        )
    if attn_implementation == "sdpa":
        register_sdpa_attention()
    vision_config = KimiK3VisionConfig(**config_overrides)
    tower_config = VisionTowerConfig(vision_config)
    model = MoonViT3dPretrainedModel(tower_config)
    for block in model.encoder.blocks:
        block.attn_implementation = attn_implementation
    return model


def load_vision_tower_state_dict(
    weights_path: str | Path,
) -> dict[str, Tensor]:
    """Read a safetensors file, stripping any ``vision_tower.`` prefix."""
    from safetensors import safe_open

    state: dict[str, Tensor] = {}
    with safe_open(str(weights_path), framework="pt") as handle:
        for key in handle.keys():
            bare = key
            if bare.startswith(_VISION_TOWER_PREFIX):
                bare = bare[len(_VISION_TOWER_PREFIX):]
            state[bare] = handle.get_tensor(key)
    return state


class _K3VisionProcessorAdapter:
    """Adapt ``KimiK3VisionProcessor`` to the glue preprocessing contract.

    The K3 processor returns a ``BatchFeature`` with ``pixel_values`` and
    ``grid_thws``; ``MoonViTEncoder.preprocess`` expects an object exposing
    ``pixel_values`` and ``image_grid_hws`` attributes.
    """

    def __init__(self, processor: KimiK3VisionProcessor) -> None:
        self._processor = processor

    def __call__(self, images: Any, return_tensors: str = "pt") -> Any:
        if not isinstance(images, (list, tuple)):
            images = [images]
        medias = [{"type": "image", "image": image} for image in images]
        batch = self._processor.preprocess(medias, return_tensors=return_tensors)
        return SimpleNamespace(
            pixel_values=batch["pixel_values"],
            image_grid_hws=batch["grid_thws"],
        )


def load_moonvit_v2_processor(
    processor_config_path: str | Path | None = None,
) -> _K3VisionProcessorAdapter:
    """Build the K3 image processor from (vendored) preprocessor config."""
    path = Path(processor_config_path) if processor_config_path else _DEFAULT_PROCESSOR_CONFIG
    with open(path, encoding="utf-8") as handle:
        media_proc_cfg = json.load(handle)["media_proc_cfg"]
    return _K3VisionProcessorAdapter(
        KimiK3VisionProcessor(media_proc_cfg=media_proc_cfg)
    )


def load_moonvit_v2_encoder(
    weights_path: str | Path | None = None,
    *,
    attn_implementation: str = "eager",
    torch_dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    freeze: bool = True,
    processor_config_path: str | Path | None = None,
    **config_overrides: Any,
) -> MoonViTEncoder:
    """Build a glue-compatible ``MoonViTEncoder`` around MoonViT-V2.

    ``weights_path`` may be ``None`` (random init, for tests), a standalone
    vision-tower safetensors, or a full-K3 shard containing ``vision_tower.*``
    keys. Loading is strict in all cases.
    """
    model = build_moonvit_v2(
        attn_implementation=attn_implementation, **config_overrides
    )
    if weights_path is not None:
        state = load_vision_tower_state_dict(weights_path)
        model.load_state_dict(state, strict=True)
    if torch_dtype is not None or device is not None:
        model = model.to(device=device, dtype=torch_dtype)

    merge_kernel = tuple(model.config.merge_kernel_size)
    encoder = MoonViTEncoder(
        model,
        processor=load_moonvit_v2_processor(processor_config_path),
        vision_width=int(model.config.hidden_size),
        merge_factor=int(merge_kernel[0] * merge_kernel[1]),
        freeze=freeze,
    )
    return encoder
