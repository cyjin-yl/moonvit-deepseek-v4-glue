"""Shared helpers for tools/eval_vlm.py and tools/train_overfit.py."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from moonvit_glue import MoonViTEncoder

_IMAGE_SENTINEL = "\x00image\x00"


def build_prompt_ids(tokenizer, template: str, question: str, placeholder_token_id: int, device):
    """Tokenize a prompt template, inserting the placeholder as exactly one token.

    The image token string of one tokenizer (e.g. DeepSeek's <｜image｜>) splits
    into many tokens under another tokenizer, so the placeholder must be placed
    by id, not by text.
    """

    rendered = template.replace("{image}", _IMAGE_SENTINEL).format(question=question)
    before, after = rendered.split(_IMAGE_SENTINEL)
    ids = (
        tokenizer.encode(before, add_special_tokens=False)
        + [placeholder_token_id]
        + tokenizer.encode(after, add_special_tokens=False)
    )
    return torch.tensor([ids], device=device)


def next_batch(records: list, cursor: int, batch_size: int) -> tuple[list, int]:
    """``batch_size`` distinct consecutive records starting at ``cursor``.

    Wraps around the epoch boundary in indexing only; the returned cursor stays
    monotonic so resume can reconstruct it as ``start_step * batch_size``.
    """

    batch = [records[(cursor + offset) % len(records)] for offset in range(batch_size)]
    return batch, cursor + batch_size


def encode_image(moonvit: MoonViTEncoder, image_path: Path, max_image_side: int | None = None):
    image = Image.open(image_path).convert("RGB")
    if max_image_side:
        image.thumbnail((max_image_side, max_image_side), Image.LANCZOS)
    image_inputs = moonvit.preprocess(image)
    return moonvit(**image_inputs)
