from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Literal, Sequence

import torch
from torch import Tensor, nn

from .merge import MultimodalInputs, expand_image_placeholders
from .projector import PatchMergerProjector


BackboneKind = Literal["auto", "generic", "deepseek_v4"]


@contextmanager
def _override_embedding_lookup(
    embedding: nn.Module, replacement: Tensor
) -> Iterator[None]:
    """Temporarily replace one embedding lookup while preserving its token IDs."""

    def replace_output(_module: nn.Module, _args: tuple[Any, ...], output: Tensor) -> Tensor:
        if output.shape != replacement.shape:
            raise ValueError(
                "DeepSeek routing IDs and multimodal embeddings disagree: "
                f"lookup produced {tuple(output.shape)}, replacement is {tuple(replacement.shape)}"
            )
        return replacement.to(device=output.device, dtype=output.dtype)

    handle = embedding.register_forward_hook(replace_output)
    try:
        yield
    finally:
        handle.remove()


class VisionCausalLM(nn.Module):
    """Inject projected image tokens into an otherwise text-only causal LM.

    The wrapper owns only the projector checkpoint. The vision tower and text
    backbone remain independently loadable, which makes weight provenance and
    local deployment easier to audit.
    """

    def __init__(
        self,
        *,
        language_model: nn.Module,
        projector: PatchMergerProjector,
        placeholder_token_id: int,
        backbone_kind: BackboneKind = "auto",
        freeze_language_model: bool = True,
        ignore_index: int = -100,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.projector = projector
        self.placeholder_token_id = placeholder_token_id
        self.ignore_index = ignore_index
        self.pad_token_id = pad_token_id
        model_type = getattr(getattr(language_model, "config", None), "model_type", None)
        self.backbone_kind = (
            "deepseek_v4"
            if backbone_kind == "auto" and model_type == "deepseek_v4"
            else "generic"
            if backbone_kind == "auto"
            else backbone_kind
        )
        self.freeze_language_model = freeze_language_model
        if freeze_language_model:
            self.language_model.requires_grad_(False)
            self.language_model.eval()

    def train(self, mode: bool = True) -> "VisionCausalLM":
        super().train(mode)
        if self.freeze_language_model:
            self.language_model.eval()
        return self

    def _language_forward(self, merged: MultimodalInputs, **kwargs: Any) -> Any:
        common = {
            "attention_mask": merged.attention_mask,
            "position_ids": merged.position_ids,
            "use_cache": False,
            **kwargs,
        }
        if merged.labels is not None:
            common["labels"] = merged.labels

        if self.backbone_kind == "generic":
            return self.language_model(inputs_embeds=merged.inputs_embeds, **common)
        if self.backbone_kind == "deepseek_v4":
            embedding = self.language_model.get_input_embeddings()
            with _override_embedding_lookup(embedding, merged.inputs_embeds):
                return self.language_model(
                    input_ids=merged.routing_input_ids,
                    **common,
                )
        raise ValueError(f"Unsupported backbone_kind: {self.backbone_kind}")

    def forward(
        self,
        *,
        input_ids: Tensor,
        image_feature_groups: Sequence[Tensor] | None = None,
        image_embeddings: Sequence[Tensor] | None = None,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        **language_model_kwargs: Any,
    ) -> Any:
        if (image_feature_groups is None) == (image_embeddings is None):
            raise ValueError(
                "Provide exactly one of image_feature_groups or image_embeddings"
            )
        if image_embeddings is None:
            image_embeddings = self.projector(image_feature_groups or [])

        embedding = self.language_model.get_input_embeddings()
        text_embeddings = embedding(input_ids)
        merged = expand_image_placeholders(
            input_ids=input_ids,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            placeholder_token_id=self.placeholder_token_id,
            attention_mask=attention_mask,
            labels=labels,
            pad_token_id=self.pad_token_id,
            ignore_index=self.ignore_index,
        )
        return self._language_forward(merged, **language_model_kwargs)
