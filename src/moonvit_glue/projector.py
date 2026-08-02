from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn


@dataclass(frozen=True)
class ProjectorConfig:
    vision_width: int
    language_width: int
    merge_factor: int = 4
    projector_width: int | None = None
    layer_norm_eps: float = 1e-5

    @property
    def flattened_vision_width(self) -> int:
        return self.vision_width * self.merge_factor

    @property
    def effective_projector_width(self) -> int:
        return self.projector_width or self.flattened_vision_width

    def __post_init__(self) -> None:
        for field_name in ("vision_width", "language_width", "merge_factor"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.projector_width is not None and self.projector_width <= 0:
            raise ValueError("projector_width must be positive when provided")


class PatchMergerProjector(nn.Module):
    """Kimi-style PatchMerger MLP with a configurable language width."""

    config_filename = "projector_config.json"
    weights_filename = "projector.safetensors"

    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.config = config
        self.pre_norm = nn.LayerNorm(config.vision_width, eps=config.layer_norm_eps)
        self.linear_1 = nn.Linear(
            config.flattened_vision_width, config.effective_projector_width
        )
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(
            config.effective_projector_width, config.language_width
        )

    def forward(self, feature_groups: Sequence[Tensor]) -> list[Tensor]:
        projected: list[Tensor] = []
        expected = (
            self.config.merge_factor,
            self.config.vision_width,
        )
        for index, item in enumerate(feature_groups):
            if item.ndim != 3 or tuple(item.shape[1:]) != expected:
                raise ValueError(
                    f"feature_groups[{index}] expected [tokens, "
                    f"{self.config.merge_factor}, {self.config.vision_width}], "
                    f"got {list(item.shape)}"
                )
            normalized = self.pre_norm(item)
            flattened = normalized.reshape(item.shape[0], -1)
            projected.append(
                self.linear_2(self.activation(self.linear_1(flattened)))
            )
        return projected

    def save_pretrained(self, directory: str | Path) -> None:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / self.config_filename).write_text(
            json.dumps(asdict(self.config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        state = {key: value.detach().contiguous().cpu() for key, value in self.state_dict().items()}
        save_file(
            state,
            str(destination / self.weights_filename),
            metadata={"format": "moonvit-patchmerger-v1"},
        )

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> "PatchMergerProjector":
        source = Path(directory)
        config = ProjectorConfig(
            **json.loads((source / cls.config_filename).read_text(encoding="utf-8"))
        )
        projector = cls(config)
        state = load_file(str(source / cls.weights_filename), device=str(device))
        projector.load_state_dict(state, strict=True)
        return projector.to(device=device, dtype=dtype)
