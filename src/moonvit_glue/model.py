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
    embedding: nn.Module, replacement: Tensor, *, strict: bool = True
) -> Iterator[None]:
    """Temporarily replace one embedding lookup while preserving its token IDs.

    In strict mode a shape mismatch raises, since training forwards must never
    silently skip the replacement. Generation runs non-strict: the override
    fires only while the lookup shape matches the (expanded) prefill, and
    single-token decode steps fall back to the normal embedding lookup.
    """

    def replace_output(_module: nn.Module, _args: tuple[Any, ...], output: Tensor) -> Tensor:
        if output.shape != replacement.shape:
            if strict:
                raise ValueError(
                    "DeepSeek routing IDs and multimodal embeddings disagree: "
                    f"lookup produced {tuple(output.shape)}, replacement is {tuple(replacement.shape)}"
                )
            return output
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
        receiver_adapter: nn.Module | None = None,
        placeholder_token_id: int,
        backbone_kind: BackboneKind = "auto",
        freeze_language_model: bool = True,
        ignore_index: int = -100,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.projector = projector
        self.receiver_adapter = receiver_adapter
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

    def _receiver_embeddings(self, image_embeddings: Sequence[Tensor]) -> list[Tensor]:
        """把 canonical projector 输出映射到当前代理主干的接收宽度。"""

        if self.receiver_adapter is None:
            return list(image_embeddings)
        return [self.receiver_adapter(item) for item in image_embeddings]

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
        image_embeddings = self._receiver_embeddings(image_embeddings)

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

    @torch.no_grad()
    def generate(
        self,
        *,
        input_ids: Tensor,
        image_feature_groups: Sequence[Tensor] | None = None,
        image_embeddings: Sequence[Tensor] | None = None,
        attention_mask: Tensor | None = None,
        **generate_kwargs: Any,
    ) -> Tensor:
        """Decode with image tokens injected at the placeholder positions.

        For DeepSeek-V4 the expanded routing IDs stay attached for the whole
        decode so Hash-MoE keeps its routing table; the embedding override
        fires only on the prefill (its shape matches the expanded sequence),
        while single-token decode steps use the normal embedding lookup.
        Position IDs are left to the model's own generate bookkeeping.
        """

        if (image_feature_groups is None) == (image_embeddings is None):
            raise ValueError(
                "Provide exactly one of image_feature_groups or image_embeddings"
            )
        if image_embeddings is None:
            image_embeddings = self.projector(image_feature_groups or [])
        image_embeddings = self._receiver_embeddings(image_embeddings)

        embedding = self.language_model.get_input_embeddings()
        text_embeddings = embedding(input_ids)
        merged = expand_image_placeholders(
            input_ids=input_ids,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            placeholder_token_id=self.placeholder_token_id,
            attention_mask=attention_mask,
            pad_token_id=self.pad_token_id,
            ignore_index=self.ignore_index,
        )
        if self.backbone_kind == "generic":
            # Passing routing IDs alongside inputs_embeds makes generate return
            # the full sequence (expanded prefix + continuation), matching the
            # deepseek_v4 branch instead of a continuation-only tensor.
            return self.language_model.generate(
                input_ids=merged.routing_input_ids,
                inputs_embeds=merged.inputs_embeds,
                attention_mask=merged.attention_mask,
                **generate_kwargs,
            )
        if self.backbone_kind == "deepseek_v4":
            with _override_embedding_lookup(
                embedding, merged.inputs_embeds, strict=False
            ):
                return self.language_model.generate(
                    input_ids=merged.routing_input_ids,
                    attention_mask=merged.attention_mask,
                    **generate_kwargs,
                )
        raise ValueError(f"Unsupported backbone_kind: {self.backbone_kind}")
