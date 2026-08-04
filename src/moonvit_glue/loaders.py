from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .model import VisionCausalLM
from .projector import PatchMergerProjector


DEEPSEEK_FLASH_0731 = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEFAULT_IMAGE_TOKEN = "<｜image｜>"
DEEPSEEK_PAD_TOKEN = "<｜▁pad▁｜>"


@dataclass(frozen=True)
class LoadedVisionLM:
    model: VisionCausalLM
    tokenizer: Any


# Auto-detect order: DeepSeek's native token first, then the Qwen VL family
# token that also ships in Qwen text-model vocabularies.
PLACEHOLDER_CANDIDATES = (DEFAULT_IMAGE_TOKEN, "<|image_pad|>")


def resolve_placeholder_token_id(tokenizer: Any, token: str | None = None) -> int:
    """Resolve a pre-existing token and deliberately refuse vocabulary mutation.

    With ``token=None`` the known candidates are tried in order; an explicit
    token stays strict so a typo fails loudly instead of silently switching.
    """

    vocab = tokenizer.get_vocab()
    if token is None:
        for candidate in PLACEHOLDER_CANDIDATES:
            if candidate in vocab:
                return int(vocab[candidate])
        raise ValueError(
            f"No known placeholder token in the tokenizer; tried {PLACEHOLDER_CANDIDATES!r}. "
            "The token must already exist: the frozen language embedding and "
            "DeepSeek Hash-MoE routing table must not be resized"
        )
    if token not in vocab:
        raise ValueError(
            f"The placeholder token {token!r} must already exist in the tokenizer; "
            "the frozen language embedding and DeepSeek Hash-MoE routing table must not be resized"
        )
    return int(vocab[token])


def load_deepseek_flash_0731(
    projector_directory: str | Path,
    *,
    model_id: str = DEEPSEEK_FLASH_0731,
    image_token: str = DEFAULT_IMAGE_TOKEN,
    device_map: str | dict[str, Any] | None = "auto",
    dtype: torch.dtype | str = "auto",
    revision: str | None = None,
    **model_kwargs: Any,
) -> LoadedVisionLM:
    """Load the frozen 0731 text backbone and an independently saved projector.

    This function intentionally does not merge state dictionaries. Keeping the
    three weight sources separate makes revisions and hashes auditable:
    MoonViT, DeepSeek, and the trained projector.
    """

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(model_id, revision=revision)
    if getattr(config, "model_type", None) != "deepseek_v4":
        raise ValueError(f"Expected a deepseek_v4 config, got {config.model_type!r}")
    projector = PatchMergerProjector.from_pretrained(projector_directory)
    if projector.config.language_width != int(config.hidden_size):
        raise ValueError(
            "Projector/language-model mismatch: projector emits "
            f"{projector.config.language_width}, model hidden_size is {config.hidden_size}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    placeholder_token_id = resolve_placeholder_token_id(tokenizer, image_token)
    language_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        device_map=device_map,
        dtype=dtype,
        **model_kwargs,
    )
    vocab = tokenizer.get_vocab()
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = int(vocab.get(DEEPSEEK_PAD_TOKEN, 0))

    return LoadedVisionLM(
        model=VisionCausalLM(
            language_model=language_model,
            projector=projector,
            placeholder_token_id=placeholder_token_id,
            backbone_kind="deepseek_v4",
            freeze_language_model=True,
            pad_token_id=pad_token_id,
        ),
        tokenizer=tokenizer,
    )
