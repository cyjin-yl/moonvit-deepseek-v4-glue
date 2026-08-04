"""Shared helpers for tools/eval_vlm.py and tools/train_overfit.py."""

from __future__ import annotations

import io
import json
from pathlib import Path

import torch
from PIL import Image

from moonvit_glue import MoonViTEncoder

_IMAGE_SENTINEL = "\x00image\x00"


def validate_text_only_backbone_config(config) -> None:
    """Reject a native VLM as the language-side projector training target.

    A stock multimodal model is useful as an evaluator/data positive control,
    but training our projector against its already vision-aligned text stack
    would not establish transfer into a genuinely text-only backbone.
    """

    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None:
        architectures = getattr(config, "architectures", None) or []
        architecture = architectures[0] if architectures else type(config).__name__
        raise ValueError(
            "--text-model must be a text-only causal LM for projector training; "
            f"{architecture} exposes vision_config and is a native multimodal model. "
            "Use tools/eval_stock_vlm.py for native-VLM positive controls."
        )


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


def load_records(data_path: Path) -> list[dict]:
    """Load ``{image, question, answers}`` records from JSONL or packed parquet.

    Parquet mode reads every ``*.parquet`` under a directory (or a single
    file) in sorted order and preserves the packed row order (see
    ``tools/pack_to_parquet.py``); ``image_bytes`` stays compressed until
    ``encode_image`` decodes it per record.
    """
    data_path = Path(data_path)
    if data_path.is_dir():
        files = sorted(data_path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet shards in {data_path}")
    elif data_path.suffix == ".parquet":
        files = [data_path]
    else:
        return [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    import pyarrow.parquet as pq

    records = []
    for shard in files:
        records.extend(pq.read_table(shard).to_pylist())
    return records


def encode_image(moonvit: MoonViTEncoder, source, max_image_side: int | None = None,
                 base_dir: Path | None = None):
    """Encode one image through frozen MoonViT.

    ``source`` is either an image path, or a dataset record dict — packed
    parquet rows carry their bytes in ``image_bytes``; JSONL rows resolve
    ``image`` against ``base_dir``.
    """
    if isinstance(source, dict):
        if source.get("image_bytes"):
            image = Image.open(io.BytesIO(source["image_bytes"])).convert("RGB")
        else:
            image = Image.open(Path(base_dir) / source["image"]).convert("RGB")
    else:
        image = Image.open(source).convert("RGB")
    if max_image_side:
        image.thumbnail((max_image_side, max_image_side), Image.LANCZOS)
    image_inputs = moonvit.preprocess(image)
    return moonvit(**image_inputs)
