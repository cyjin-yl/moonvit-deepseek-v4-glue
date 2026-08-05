from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class MultimodalInputs:
    """Expanded inputs ready for a causal language model.

    ``routing_input_ids`` mirrors the expanded embedding sequence. It matters for
    token-id-routed models such as DeepSeek-V4 even when the model consumes
    ``inputs_embeds`` for the actual hidden states.
    """

    inputs_embeds: Tensor
    routing_input_ids: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    labels: Tensor | None = None


def _validate_inputs(
    input_ids: Tensor,
    text_embeddings: Tensor,
    image_embeddings: Sequence[Tensor],
    placeholder_token_id: int,
    attention_mask: Tensor,
) -> None:
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be rank 2, got shape {tuple(input_ids.shape)}")
    if text_embeddings.ndim != 3 or text_embeddings.shape[:2] != input_ids.shape:
        raise ValueError(
            "text_embeddings must have shape [batch, sequence, hidden] matching "
            f"input_ids; got {tuple(text_embeddings.shape)} and {tuple(input_ids.shape)}"
        )
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
    active = attention_mask.to(dtype=torch.bool)
    placeholder_count = int(
        (input_ids.eq(placeholder_token_id) & active).sum().item()
    )
    if placeholder_count != len(image_embeddings):
        raise ValueError(
            f"Found {placeholder_count} image placeholder token(s) but received "
            f"{len(image_embeddings)} image feature tensor(s)"
        )
    hidden_size = text_embeddings.shape[-1]
    for index, features in enumerate(image_embeddings):
        if features.ndim != 2 or features.shape[-1] != hidden_size:
            raise ValueError(
                f"image_embeddings[{index}] must have shape [tokens, {hidden_size}], "
                f"got {tuple(features.shape)}"
            )
        if features.shape[0] == 0:
            raise ValueError(f"image_embeddings[{index}] contains no tokens")


def expand_image_placeholders(
    *,
    input_ids: Tensor,
    text_embeddings: Tensor,
    image_embeddings: Sequence[Tensor],
    placeholder_token_id: int,
    attention_mask: Tensor | None = None,
    labels: Tensor | None = None,
    pad_token_id: int = 0,
    ignore_index: int = -100,
) -> MultimodalInputs:
    """Replace each placeholder with one variable-length image-token sequence.

    Image tensors are consumed in row-major placeholder order. The returned
    token IDs repeat the placeholder ID over every injected image token. This
    keeps a routing sequence available to architectures whose expert choice
    depends on token IDs.
    """

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    _validate_inputs(
        input_ids,
        text_embeddings,
        image_embeddings,
        placeholder_token_id,
        attention_mask,
    )
    if labels is not None and labels.shape != input_ids.shape:
        raise ValueError("labels must have the same shape as input_ids")

    rows: list[tuple[Tensor, Tensor, Tensor, Tensor | None]] = []
    image_index = 0
    for batch_index in range(input_ids.shape[0]):
        row_embeds: list[Tensor] = []
        row_ids: list[Tensor] = []
        row_masks: list[Tensor] = []
        row_labels: list[Tensor] = []
        for token_index in range(input_ids.shape[1]):
            token_id = int(input_ids[batch_index, token_index].item())
            is_active = bool(attention_mask[batch_index, token_index].item())
            if token_id == placeholder_token_id and is_active:
                features = image_embeddings[image_index].to(
                    device=text_embeddings.device, dtype=text_embeddings.dtype
                )
                image_index += 1
                token_count = features.shape[0]
                row_embeds.append(features)
                row_ids.append(
                    torch.full(
                        (token_count,),
                        placeholder_token_id,
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    )
                )
                row_masks.append(
                    torch.ones(
                        token_count,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                )
                if labels is not None:
                    row_labels.append(
                        torch.full(
                            (token_count,),
                            ignore_index,
                            dtype=labels.dtype,
                            device=labels.device,
                        )
                    )
            else:
                row_embeds.append(text_embeddings[batch_index, token_index : token_index + 1])
                row_ids.append(input_ids[batch_index, token_index : token_index + 1])
                row_masks.append(attention_mask[batch_index, token_index : token_index + 1])
                if labels is not None:
                    row_labels.append(labels[batch_index, token_index : token_index + 1])

        rows.append(
            (
                torch.cat(row_embeds, dim=0),
                torch.cat(row_ids, dim=0),
                torch.cat(row_masks, dim=0),
                torch.cat(row_labels, dim=0) if labels is not None else None,
            )
        )

    max_length = max(row[0].shape[0] for row in rows)
    embed_rows: list[Tensor] = []
    id_rows: list[Tensor] = []
    mask_rows: list[Tensor] = []
    label_rows: list[Tensor] = []
    for embeds, ids, mask, row_label in rows:
        padding = max_length - embeds.shape[0]
        embed_rows.append(torch.nn.functional.pad(embeds, (0, 0, 0, padding)))
        id_rows.append(torch.nn.functional.pad(ids, (0, padding), value=pad_token_id))
        mask_rows.append(torch.nn.functional.pad(mask, (0, padding), value=0))
        if row_label is not None:
            label_rows.append(
                torch.nn.functional.pad(row_label, (0, padding), value=ignore_index)
            )

    final_mask = torch.stack(mask_rows)
    position_ids = (final_mask.long().cumsum(dim=-1) - 1).clamp_min(0)
    return MultimodalInputs(
        inputs_embeds=torch.stack(embed_rows),
        routing_input_ids=torch.stack(id_rows),
        attention_mask=final_mask,
        position_ids=position_ids,
        labels=torch.stack(label_rows) if labels is not None else None,
    )
