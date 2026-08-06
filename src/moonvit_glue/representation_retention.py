"""冻结视觉表示的跨图多样性与几何保留诊断。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor


def _matrix(values: Tensor, *, label: str) -> Tensor:
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError(f"{label} must have shape [samples>=2, width>0]")
    result = values.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not torch.isfinite(result).all():
        raise ValueError(f"{label} contains non-finite values")
    return result


def _distribution(values: Tensor) -> dict[str, float]:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("distribution requires a non-empty vector")
    return {
        "minimum": float(values.min()),
        "p10": float(torch.quantile(values, 0.10)),
        "median": float(torch.quantile(values, 0.50)),
        "mean": float(values.mean()),
        "p90": float(torch.quantile(values, 0.90)),
        "maximum": float(values.max()),
    }


def pairwise_geometry(values: Tensor) -> list[dict[str, float | int]]:
    """返回稳定样本顺序下的全部无向 pairwise 几何。"""

    matrix = _matrix(values, label="representation")
    width = int(matrix.shape[1])
    norms = torch.linalg.vector_norm(matrix, dim=1).clamp_min(1e-30)
    rows: list[dict[str, float | int]] = []
    for left in range(matrix.shape[0]):
        for right in range(left + 1, matrix.shape[0]):
            cosine = torch.dot(matrix[left], matrix[right]) / (
                norms[left] * norms[right]
            )
            distance = torch.linalg.vector_norm(matrix[left] - matrix[right])
            rows.append(
                {
                    "left_index": left,
                    "right_index": right,
                    "cosine_similarity": float(cosine.clamp(-1.0, 1.0)),
                    "rms_distance": float(distance / math.sqrt(width)),
                }
            )
    return rows


def summarize_representation(values: Tensor) -> dict[str, Any]:
    """汇总 token-mean 表示的尺度、跨图 spread、秩与 pairwise 几何。"""

    matrix = _matrix(values, label="representation")
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    sample_rms = torch.sqrt(torch.mean(matrix.square()))
    between_rms = torch.sqrt(torch.mean(centered.square()))
    relative_spread = (
        between_rms / sample_rms if float(sample_rms) > 0.0 else torch.tensor(0.0)
    )

    # 50×50 Gram 比直接对 50×4096 矩阵做 SVD 更快，非零谱完全相同。
    gram = centered @ centered.T / matrix.shape[1]
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    total = eigenvalues.sum()
    if float(total) > 0.0:
        probabilities = eigenvalues / total
        positive = probabilities > 0
        participation = total.square() / eigenvalues.square().sum()
        entropy_rank = torch.exp(
            -(probabilities[positive] * probabilities[positive].log()).sum()
        )
        top1_fraction = eigenvalues[-1] / total
    else:
        participation = torch.tensor(0.0, dtype=torch.float64)
        entropy_rank = torch.tensor(0.0, dtype=torch.float64)
        top1_fraction = torch.tensor(0.0, dtype=torch.float64)

    pairs = pairwise_geometry(matrix)
    distances = torch.tensor(
        [row["rms_distance"] for row in pairs], dtype=torch.float64
    )
    cosines = torch.tensor(
        [row["cosine_similarity"] for row in pairs], dtype=torch.float64
    )
    return {
        "samples": int(matrix.shape[0]),
        "width": int(matrix.shape[1]),
        "sample_rms": float(sample_rms),
        "between_image_rms": float(between_rms),
        "relative_between_image_spread": float(relative_spread),
        "effective_rank_participation": float(participation),
        "effective_rank_entropy": float(entropy_rank),
        "top1_variance_fraction": float(top1_fraction),
        "pairwise_rms_distance": _distribution(distances),
        "pairwise_cosine_similarity": _distribution(cosines),
    }


@dataclass(frozen=True)
class TokenSequenceSummary:
    pooled: Tensor
    token_counts: list[int]
    per_image_within_rms: list[float]
    mean_within_image_rms: float
    representation: dict[str, Any]


def summarize_token_sequences(sequences: Sequence[Tensor]) -> TokenSequenceSummary:
    """先逐图聚合可变长 token，再分离图内与跨图方差。"""

    if len(sequences) < 2:
        raise ValueError("at least two image token sequences are required")
    width: int | None = None
    pooled: list[Tensor] = []
    token_counts: list[int] = []
    within: list[float] = []
    for index, raw in enumerate(sequences):
        item = _matrix(raw, label=f"sequences[{index}]") if raw.shape[0] >= 2 else None
        if item is None:
            if raw.ndim != 2 or raw.shape[0] != 1 or raw.shape[1] == 0:
                raise ValueError(f"sequences[{index}] must have shape [tokens>=1, width]")
            item = raw.detach().to(device="cpu", dtype=torch.float64).contiguous()
            if not torch.isfinite(item).all():
                raise ValueError(f"sequences[{index}] contains non-finite values")
        if width is None:
            width = int(item.shape[1])
        elif int(item.shape[1]) != width:
            raise ValueError("all token sequences must share one width")
        mean = item.mean(dim=0)
        pooled.append(mean)
        token_counts.append(int(item.shape[0]))
        within.append(float(torch.sqrt(torch.mean((item - mean).square()))))
    pooled_matrix = torch.stack(pooled)
    return TokenSequenceSummary(
        pooled=pooled_matrix,
        token_counts=token_counts,
        per_image_within_rms=within,
        mean_within_image_rms=float(sum(within) / len(within)),
        representation=summarize_representation(pooled_matrix),
    )


def _pearson(first: Tensor, second: Tensor) -> float:
    left = first - first.mean()
    right = second - second.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0.0:
        return 1.0 if torch.equal(first, second) else 0.0
    return float((torch.dot(left, right) / denominator).clamp(-1.0, 1.0))


def compare_geometry(first: Tensor, second: Tensor) -> dict[str, float]:
    """比较两组同 ID 表示的 centered geometry。"""

    left = _matrix(first, label="first")
    right = _matrix(second, label="second")
    if left.shape[0] != right.shape[0]:
        raise ValueError("geometry comparison requires identical sample counts")
    left_centered = left - left.mean(dim=0, keepdim=True)
    right_centered = right - right.mean(dim=0, keepdim=True)
    left_gram = left_centered @ left_centered.T
    right_gram = right_centered @ right_centered.T
    denominator = torch.linalg.matrix_norm(left_gram) * torch.linalg.matrix_norm(
        right_gram
    )
    cka = (
        float(torch.sum(left_gram * right_gram) / denominator)
        if float(denominator) > 0.0
        else 0.0
    )
    left_distances = torch.tensor(
        [row["rms_distance"] for row in pairwise_geometry(left)],
        dtype=torch.float64,
    )
    right_distances = torch.tensor(
        [row["rms_distance"] for row in pairwise_geometry(right)],
        dtype=torch.float64,
    )
    return {
        "linear_cka": cka,
        "pairwise_distance_pearson": _pearson(left_distances, right_distances),
    }


def decide_representation_action(
    step0: dict[str, Any],
    current: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """严格执行预注册的 gross-collapse 双门槛。"""

    spread_base = float(step0["relative_between_image_spread"])
    rank_base = float(step0["effective_rank_participation"])
    if spread_base <= 0.0 or rank_base <= 0.0:
        raise ValueError("step0 representation has no measurable diversity")
    spread_ratio = float(current["relative_between_image_spread"]) / spread_base
    rank_ratio = float(current["effective_rank_participation"]) / rank_base
    rule = contract["gross_collapse_rule"]
    spread_guard = spread_ratio < float(
        rule["current_over_step0_relative_spread_below"]
    )
    rank_guard = rank_ratio < float(
        rule["current_over_step0_effective_rank_below"]
    )
    guards = (spread_guard, rank_guard)
    gross = all(guards) if bool(rule["requires_all"]) else any(guards)
    action_key = "gross_collapse" if gross else "diversity_retained"
    return {
        "action": contract["actions"][action_key],
        "gross_collapse": gross,
        "relative_spread_ratio": spread_ratio,
        "effective_rank_ratio": rank_ratio,
        "relative_spread_guard_triggered": spread_guard,
        "effective_rank_guard_triggered": rank_guard,
    }
