"""在 canonical 4096 projector 边界保持尺度与跨图几何。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor


def pool_projector_batch(outputs: Sequence[Sequence[Tensor]]) -> Tensor:
    """拼接每个样本的 feature groups，再对 visual tokens 做均值。"""

    if len(outputs) < 2:
        raise ValueError("geometry regularization requires at least two images")
    pooled: list[Tensor] = []
    width: int | None = None
    for sample_index, groups in enumerate(outputs):
        if not groups:
            raise ValueError(f"projector output groups are empty: {sample_index}")
        sequences: list[Tensor] = []
        for group_index, values in enumerate(groups):
            if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
                raise ValueError(
                    f"projector output must be [tokens,width]: {sample_index}/{group_index}"
                )
            if width is None:
                width = int(values.shape[1])
            elif int(values.shape[1]) != width:
                raise ValueError("projector output widths differ")
            if not bool(torch.isfinite(values).all()):
                raise ValueError("projector output contains non-finite values")
            sequences.append(values)
        pooled.append(torch.cat(sequences, dim=0).mean(dim=0))
    return torch.stack(pooled)


@dataclass(frozen=True)
class GeometryRegularizationResult:
    total: Tensor
    scale: Tensor
    relative_spread: Tensor
    centered_gram: Tensor
    metrics: dict[str, float]


def _matrix(values: Tensor, *, name: str) -> Tensor:
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError(f"{name} must have shape [batch>=2,width>0]")
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} contains non-finite values")
    return values


def geometry_regularization_loss(
    current: Tensor,
    reference: Tensor,
    *,
    scale_weight: float = 1.0,
    relative_spread_weight: float = 1.0,
    centered_gram_weight: float = 1.0,
    epsilon: float = 1e-8,
) -> GeometryRegularizationResult:
    """保持 batch 级尺度、relative spread 与 normalized centered Gram。

    normalized Gram 允许全局正交旋转，因此 projector 仍可学习语言空间方向；
    scale 与 relative-spread 项阻止巨大公共方向掩盖跨图差异。
    """

    current = _matrix(current, name="current")
    reference = _matrix(reference, name="reference").detach()
    if current.shape != reference.shape:
        raise ValueError("current and reference shapes differ")
    weights = (scale_weight, relative_spread_weight, centered_gram_weight)
    if any(not torch.isfinite(torch.tensor(value)) or value < 0 for value in weights):
        raise ValueError("geometry weights must be finite and non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    def statistics(values: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rms = torch.sqrt(torch.mean(values.square()))
        centered = values - values.mean(dim=0, keepdim=True)
        between_rms = torch.sqrt(torch.mean(centered.square()))
        relative_spread = between_rms / rms.clamp_min(epsilon)
        gram = centered @ centered.T / values.shape[1]
        normalized_gram = gram / torch.linalg.matrix_norm(gram).clamp_min(epsilon)
        return rms, between_rms, relative_spread, normalized_gram

    current_rms, current_between, current_spread, current_gram = statistics(current)
    reference_rms, reference_between, reference_spread, reference_gram = statistics(
        reference
    )
    scale = torch.log(
        current_rms.clamp_min(epsilon) / reference_rms.clamp_min(epsilon)
    ).square()
    relative_spread = torch.log(
        current_spread.clamp_min(epsilon) / reference_spread.clamp_min(epsilon)
    ).square()
    centered_gram = torch.sum((current_gram - reference_gram).square())
    total = (
        float(scale_weight) * scale
        + float(relative_spread_weight) * relative_spread
        + float(centered_gram_weight) * centered_gram
    )
    metrics = {
        "current_rms": float(current_rms.detach()),
        "reference_rms": float(reference_rms.detach()),
        "rms_ratio": float((current_rms / reference_rms.clamp_min(epsilon)).detach()),
        "current_between_rms": float(current_between.detach()),
        "reference_between_rms": float(reference_between.detach()),
        "current_relative_spread": float(current_spread.detach()),
        "reference_relative_spread": float(reference_spread.detach()),
        "relative_spread_ratio": float(
            (current_spread / reference_spread.clamp_min(epsilon)).detach()
        ),
    }
    return GeometryRegularizationResult(
        total=total,
        scale=scale,
        relative_spread=relative_spread,
        centered_gram=centered_gram,
        metrics=metrics,
    )


def global_gradient_norm(gradients: Sequence[Tensor | None]) -> Tensor:
    """以 float64 累积全参数 L2 norm，拒绝非有限梯度。"""

    squares = []
    for gradient in gradients:
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError("gradient contains non-finite values")
        norm = torch.linalg.vector_norm(gradient.detach().to(torch.float32))
        squares.append(norm.to(torch.float64).square())
    if not squares:
        return torch.tensor(0.0, dtype=torch.float64)
    return torch.sqrt(torch.stack(squares).sum())


def geometry_payload(result: GeometryRegularizationResult) -> dict[str, Any]:
    """把 loss tensor 转成可写入 JSONL 的逐步审计记录。"""

    return {
        "total": float(result.total.detach()),
        "scale": float(result.scale.detach()),
        "relative_spread": float(result.relative_spread.detach()),
        "centered_gram": float(result.centered_gram.detach()),
        **result.metrics,
    }
