from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Sequence

import torch
from torch import Tensor, nn


class MoonViTEncoder(nn.Module):
    """Validated wrapper around the standalone MoonViT feature extractor."""

    def __init__(
        self,
        model: nn.Module,
        *,
        processor: Any | None = None,
        vision_width: int = 1152,
        merge_factor: int = 4,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.vision_width = vision_width
        self.merge_factor = merge_factor
        self.freeze_encoder = freeze
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True) -> "MoonViTEncoder":
        super().train(mode)
        if self.freeze_encoder:
            self.model.eval()
        return self

    def forward(
        self, pixel_values: Tensor, image_grid_hws: Tensor
    ) -> list[Tensor]:
        context = torch.no_grad() if self.freeze_encoder else nullcontext()
        with context:
            features = self.model(pixel_values, image_grid_hws)
        if not isinstance(features, (list, tuple)):
            raise ValueError(
                "MoonViT must return one [tokens, merge, width] tensor per image"
            )
        checked: list[Tensor] = []
        expected = (self.merge_factor, self.vision_width)
        for index, item in enumerate(features):
            if item.ndim != 3 or tuple(item.shape[1:]) != expected:
                raise ValueError(
                    f"MoonViT output {index} expected [tokens, {self.merge_factor}, "
                    f"{self.vision_width}], got {list(item.shape)}"
                )
            checked.append(item)
        return checked

    def preprocess(self, images: Sequence[Any] | Any, *, device: Any = None) -> dict[str, Tensor]:
        if self.processor is None:
            raise RuntimeError("No image processor was attached to this MoonViTEncoder")
        batch = self.processor(images, return_tensors="pt")
        pixel_values = batch.pixel_values
        image_grid_hws = batch.image_grid_hws
        parameter = next(self.model.parameters())
        target_device = device or parameter.device
        return {
            "pixel_values": pixel_values.to(
                device=target_device, dtype=parameter.dtype
            ),
            "image_grid_hws": image_grid_hws.to(device=target_device),
        }

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "moonshotai/MoonViT-SO-400M",
        *,
        revision: str | None = None,
        torch_dtype: torch.dtype | str = "auto",
        device_map: str | dict[str, Any] | None = None,
        freeze: bool = True,
    ) -> "MoonViTEncoder":
        from transformers import AutoImageProcessor, AutoModel

        common = {
            "revision": revision,
            "trust_remote_code": True,
        }
        try:
            model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
                **common,
            )
        except AttributeError as exc:
            # The MoonViT remote code predates Transformers v5 and never
            # defines all_tied_weights_keys, which v5's loader requires.
            # MoonViT is a plain ViT with no tied weights, so an empty
            # mapping is the semantically correct shim.
            if "all_tied_weights_keys" not in str(exc):
                raise
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            remote_cls = get_class_from_dynamic_module(
                "modeling_moonvit.MoonVitPretrainedModel", model_id, revision=revision
            )
            remote_cls.all_tied_weights_keys = {}
            model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
                **common,
            )
        processor = AutoImageProcessor.from_pretrained(model_id, **common)
        config = model.config
        merge_kernel = tuple(getattr(config, "merge_kernel_size", (2, 2)))
        return cls(
            model,
            processor=processor,
            vision_width=int(config.hidden_size),
            merge_factor=int(merge_kernel[0] * merge_kernel[1]),
            freeze=freeze,
        )
