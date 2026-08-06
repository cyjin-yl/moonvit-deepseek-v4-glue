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
    residual_mode: str = "none"
    # ``legacy_pre_norm`` is kept as the backward-compatible implementation
    # used by historical checkpoints. ``kimi_k3_v2`` mirrors the vendored
    # Kimi-K3/MoonViT-V2 PatchMergerMLPV2 contract: bias-free MLP + post RMSNorm.
    projector_variant: str = "legacy_pre_norm"

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
        if self.residual_mode not in {"none", "zero_init", "gated"}:
            raise ValueError(
                "residual_mode must be one of none, zero_init, or gated"
            )
        if self.projector_variant not in {"legacy_pre_norm", "kimi_k3_v2"}:
            raise ValueError(
                "projector_variant must be one of legacy_pre_norm or kimi_k3_v2"
            )
        if self.projector_variant == "kimi_k3_v2":
            if self.output_norm != "none":
                raise ValueError(
                    "kimi_k3_v2 owns its post RMSNorm; keep output_norm=none"
                )
            if self.residual_mode != "none":
                raise ValueError("kimi_k3_v2 cannot combine with residual_mode")


class PatchMergerProjector(nn.Module):
    """Kimi-style PatchMerger MLP with a configurable language width.

    The default preserves the historical project checkpoints. The explicit
    ``kimi_k3_v2`` variant is source-aligned with Kimi-K3's
    ``PatchMergerMLPV2`` and is deliberately opt-in so old hashes retain their
    original meaning.
    """

    config_filename = "projector_config.json"
    weights_filename = "projector.safetensors"

    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.config = config
        if config.projector_variant == "kimi_k3_v2":
            # This is the exact Kimi-K3/MoonViT-V2 shape contract. The official
            # implementation has no vision-side pre-norm and no linear bias.
            self.pre_norm = nn.Identity()
            self.linear_1 = nn.Linear(
                config.flattened_vision_width,
                config.effective_projector_width,
                bias=False,
            )
            self.activation = nn.GELU()
            self.linear_2 = nn.Linear(
                config.effective_projector_width,
                config.language_width,
                bias=False,
            )
            self.output_norm = nn.RMSNorm(
                config.language_width, eps=config.layer_norm_eps
            )
            for module in (self.linear_1, self.linear_2):
                nn.init.trunc_normal_(
                    module.weight,
                    std=(2.0 / module.in_features) ** 0.5,
                )
        else:
            self.pre_norm = nn.LayerNorm(
                config.vision_width, eps=config.layer_norm_eps
            )
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
        if config.residual_mode != "none":
            # 残差分支保持 canonical 4096 宽度；分支初值由结构合同冻结。
            self.residual = nn.Linear(
                config.language_width, config.language_width, bias=False
            )
            if config.residual_mode == "zero_init":
                nn.init.zeros_(self.residual.weight)
            else:
                # gate=0 让初始输出精确等于旧 projector，同时保留 gate 梯度。
                self.residual_gate = nn.Parameter(torch.zeros((), dtype=torch.float32))

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
            base_output = self.output_norm(
                self.linear_2(self.activation(self.linear_1(flattened)))
            )
            if self.config.residual_mode == "none":
                projected.append(base_output)
                continue
            residual = self.residual(base_output)
            if self.config.residual_mode == "gated":
                residual = residual * self.residual_gate.to(dtype=base_output.dtype)
            projected.append(base_output + residual)
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

        ``linear_1`` (and the legacy ``pre_norm``) only touch vision-side
        dimensions, so a projector aligned against a small text model can
        donate that trunk to a projector targeting DeepSeek-V4; ``linear_2``
        and the receiver-facing norm keep their fresh init when the output
        width differs. All vision-side config fields and the projector variant
        must match — a V1 (1152-dim) donor cannot warm-start a V2 (1024-dim)
        projector.
        """

        source = Path(directory)
        donor = ProjectorConfig(
            **json.loads((source / self.config_filename).read_text(encoding="utf-8"))
        )
        for field_name in (
            "vision_width",
            "merge_factor",
            "layer_norm_eps",
            "projector_variant",
        ):
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
        prefixes = ("linear_1.",)
        if self.config.projector_variant == "legacy_pre_norm":
            prefixes = ("pre_norm.", "linear_1.")
        trunk = {key: value for key, value in state.items() if key.startswith(prefixes)}
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
