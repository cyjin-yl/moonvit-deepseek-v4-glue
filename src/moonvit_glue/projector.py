from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn


class _ParameterFreeRMSNorm(nn.Module):
    """不引入新参数的 RMSNorm，用于结构筛选的可迁移输出边界。"""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.width = int(width)
        self.eps = float(eps)

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.shape[-1] != self.width:
            raise ValueError(
                f"RMSNorm width differs: expected {self.width}, got {hidden.shape[-1]}"
            )
        variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        return hidden * torch.rsqrt(variance + self.eps).to(hidden.dtype)


@dataclass(frozen=True)
class ProjectorConfig:
    vision_width: int
    language_width: int
    merge_factor: int = 4
    projector_width: int | None = None
    layer_norm_eps: float = 1e-5
    output_norm: str = "none"

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
        if self.output_norm not in {"none", "layernorm", "rmsnorm"}:
            raise ValueError(
                "output_norm must be one of none, layernorm, or rmsnorm"
            )


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
        if config.output_norm == "layernorm":
            # affine=False 保持参数量和 DeepSeek 迁移边界不变。
            self.output_norm = nn.LayerNorm(
                config.language_width,
                eps=config.layer_norm_eps,
                elementwise_affine=False,
            )
        elif config.output_norm == "rmsnorm":
            self.output_norm = _ParameterFreeRMSNorm(
                config.language_width, config.layer_norm_eps
            )
        else:
            self.output_norm = nn.Identity()

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
                self.output_norm(self.linear_2(self.activation(self.linear_1(flattened))))
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

    def load_trunk(self, directory: str | Path) -> None:
        """Warm-start the language-agnostic trunk from a donor projector.

        ``pre_norm`` and ``linear_1`` only touch vision-side dimensions, so a
        projector aligned against a small text model can donate them to a
        projector targeting DeepSeek-V4; ``linear_2`` keeps its fresh init
        because its output width differs across backbones. All vision-side
        config fields must match — a V1 (1152-dim) donor cannot warm-start a
        V2 (1024-dim) projector.
        """

        source = Path(directory)
        donor = ProjectorConfig(
            **json.loads((source / self.config_filename).read_text(encoding="utf-8"))
        )
        for field_name in ("vision_width", "merge_factor", "layer_norm_eps"):
            if getattr(donor, field_name) != getattr(self.config, field_name):
                raise ValueError(
                    f"donor {field_name}={getattr(donor, field_name)} does not match "
                    f"{getattr(self.config, field_name)}"
                )
        if donor.effective_projector_width != self.config.effective_projector_width:
            raise ValueError(
                f"donor projector_width={donor.effective_projector_width} does not match "
                f"{self.config.effective_projector_width}"
            )
        state = load_file(str(source / self.weights_filename), device="cpu")
        trunk = {
            key: value
            for key, value in state.items()
            if key.startswith(("pre_norm.", "linear_1."))
        }
        self.load_state_dict(trunk, strict=False)


def seeded_projector(config: ProjectorConfig, *, seed: int) -> PatchMergerProjector:
    """Build a CPU projector from an isolated, explicit PyTorch RNG seed.

    The exact weights should still be serialized for long-lived contracts; this
    helper makes regeneration deterministic without perturbing the caller's CPU
    RNG stream.
    """

    if not 0 <= int(seed) < 2**63:
        raise ValueError("projector seed must be an integer in [0, 2**63)")
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(seed))
        return PatchMergerProjector(config)
