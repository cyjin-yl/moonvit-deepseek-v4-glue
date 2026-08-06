"""Projector 表征健康监控、固定 probe 与自动止损规则。

这个模块只依赖一组视觉 token 序列和一个已经冻结的合同。它不读取
Qwen/DeepSeek 的 tokenizer、chat template 或视觉模块，因此可以直接复用在
canonical 4096 维 projector 边界上。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor


HEALTH_CONTRACT_FORMAT_VERSION = "projector-health-contract-v1"
DEFAULT_PROBE_STEPS = (0, 1, 2, 5, 10, 20, 30, 50, 75, 100)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    """以追加方式写一条可恢复的结构化记录。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def probe_due(step: int, *, max_step: int | None = None, every_after: int = 50) -> bool:
    """判断固定的高频 probe 调度是否命中。"""

    step = int(step)
    if step < 0:
        raise ValueError("probe step must be non-negative")
    if every_after <= 0:
        raise ValueError("every_after must be positive")
    due = step in DEFAULT_PROBE_STEPS or (
        step > DEFAULT_PROBE_STEPS[-1] and step % int(every_after) == 0
    )
    if max_step is not None and step > int(max_step):
        return False
    return due


def _finite_matrix(value: Tensor, *, name: str) -> Tensor:
    if value.ndim != 2 or value.shape[0] < 2 or value.shape[1] <= 0:
        raise ValueError(f"{name} must have shape [samples>=2, width>0]")
    matrix = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return matrix


def _sequence_summary(sequences: Sequence[Tensor], *, name: str) -> dict[str, Any]:
    """汇总可变长 token 序列，所有计算在 CPU float64 上完成以保证可复核。"""

    if len(sequences) < 2:
        raise ValueError(f"{name} requires at least two image sequences")
    pooled: list[Tensor] = []
    within: list[float] = []
    width: int | None = None
    token_counts: list[int] = []
    for index, raw in enumerate(sequences):
        if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] <= 0:
            raise ValueError(
                f"{name}[{index}] must have shape [tokens>=1, width>0]"
            )
        item = raw.detach().to(device="cpu", dtype=torch.float64).contiguous()
        if not bool(torch.isfinite(item).all()):
            raise ValueError(f"{name}[{index}] contains NaN or Inf")
        if width is None:
            width = int(item.shape[1])
        elif int(item.shape[1]) != width:
            raise ValueError(f"{name} sequences have different widths")
        mean = item.mean(dim=0)
        pooled.append(mean)
        within.append(float(torch.sqrt(torch.mean((item - mean).square()))))
        token_counts.append(int(item.shape[0]))

    matrix = _finite_matrix(torch.stack(pooled), name=f"{name}.pooled")
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    sample_rms = torch.sqrt(torch.mean(matrix.square()))
    between_rms = torch.sqrt(torch.mean(centered.square()))
    relative_spread = (
        between_rms / sample_rms if float(sample_rms) > 0.0 else torch.tensor(0.0)
    )
    mean_vector = matrix.mean(dim=0)
    mean_direction_fraction = (
        mean_vector.square().sum() / matrix.square().sum()
        if float(matrix.square().sum()) > 0.0
        else torch.tensor(0.0, dtype=torch.float64)
    )
    # 样本 Gram 的谱等于中心化表示的非零谱，避免对 50×4096 做昂贵 SVD。
    gram = centered @ centered.T / matrix.shape[1]
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    total = eigenvalues.sum()
    if float(total) > 0.0:
        probabilities = eigenvalues / total
        positive = probabilities > 0.0
        participation = total.square() / eigenvalues.square().sum().clamp_min(1e-30)
        entropy = torch.exp(
            -(probabilities[positive] * probabilities[positive].log()).sum()
        )
        top1 = eigenvalues[-1] / total
        top5 = eigenvalues[-min(5, eigenvalues.numel()) :].sum() / total
    else:
        participation = torch.tensor(0.0, dtype=torch.float64)
        entropy = torch.tensor(0.0, dtype=torch.float64)
        top1 = torch.tensor(0.0, dtype=torch.float64)
        top5 = torch.tensor(0.0, dtype=torch.float64)

    return {
        "samples": int(matrix.shape[0]),
        "width": int(matrix.shape[1]),
        "token_counts": token_counts,
        "output_rms": float(sample_rms),
        "between_image_rms": float(between_rms),
        "within_image_token_rms": float(sum(within) / len(within)),
        "relative_spread": float(relative_spread),
        "mean_direction_fraction": float(mean_direction_fraction),
        "effective_rank": float(participation),
        "entropy_rank": float(entropy),
        "top1_variance_fraction": float(top1),
        "top5_variance_fraction": float(top5),
        "pooled": matrix,
    }


def _pairwise_distances(matrix: Tensor) -> Tensor:
    centered = matrix[:, None, :] - matrix[None, :, :]
    distances = torch.linalg.vector_norm(centered, dim=-1)
    indices = torch.triu_indices(matrix.shape[0], matrix.shape[0], offset=1)
    return distances[indices[0], indices[1]] / math.sqrt(matrix.shape[1])


def _pearson(first: Tensor, second: Tensor) -> float:
    left = first - first.mean()
    right = second - second.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0.0:
        return 1.0 if torch.equal(first, second) else 0.0
    return float((torch.dot(left, right) / denominator).clamp(-1.0, 1.0))


def _centered_gram_similarity(first: Tensor, second: Tensor) -> float:
    left = first - first.mean(dim=0, keepdim=True)
    right = second - second.mean(dim=0, keepdim=True)
    left_gram = left @ left.T
    right_gram = right @ right.T
    denominator = torch.linalg.matrix_norm(left_gram) * torch.linalg.matrix_norm(
        right_gram
    )
    if float(denominator) == 0.0:
        return 1.0 if torch.equal(left_gram, right_gram) else 0.0
    return float((torch.sum(left_gram * right_gram) / denominator).clamp(-1.0, 1.0))


def summarize_batch_embeddings(
    projector_sequences: Sequence[Tensor],
    receiver_sequences: Sequence[Tensor],
) -> dict[str, Any]:
    """计算逐 optimizer step 可廉价记录的 batch 健康量。"""

    projector = _sequence_summary(projector_sequences, name="projector_sequences")
    receiver = _sequence_summary(receiver_sequences, name="receiver_sequences")
    if projector["samples"] != receiver["samples"]:
        raise ValueError("projector and receiver sample counts differ")
    return {
        "projector_output_rms": projector["output_rms"],
        "receiver_output_rms": receiver["output_rms"],
        "between_image_rms": projector["between_image_rms"],
        "receiver_between_image_rms": receiver["between_image_rms"],
        "within_image_token_rms": projector["within_image_token_rms"],
        "receiver_within_image_token_rms": receiver["within_image_token_rms"],
        "relative_spread": projector["relative_spread"],
        "projector_relative_spread": projector["relative_spread"],
        "receiver_relative_spread": receiver["relative_spread"],
        "mean_direction_fraction": projector["mean_direction_fraction"],
        "receiver_mean_direction_fraction": receiver["mean_direction_fraction"],
        "projector_effective_rank": projector["effective_rank"],
        "receiver_effective_rank": receiver["effective_rank"],
        "projector_top1_variance_fraction": projector["top1_variance_fraction"],
        "receiver_top1_variance_fraction": receiver["top1_variance_fraction"],
        "projector_top5_variance_fraction": projector["top5_variance_fraction"],
        "receiver_top5_variance_fraction": receiver["top5_variance_fraction"],
        "projector_pooled": projector["pooled"],
        "receiver_pooled": receiver["pooled"],
    }


def summarize_probe(
    projector_sequences: Sequence[Tensor],
    receiver_sequences: Sequence[Tensor],
    *,
    step0_projector_sequences: Sequence[Tensor],
    step0_receiver_sequences: Sequence[Tensor],
    step: int,
) -> dict[str, Any]:
    """汇总冻结 probe，并与 step0 绑定比较。"""

    current = summarize_batch_embeddings(projector_sequences, receiver_sequences)
    step0 = summarize_batch_embeddings(
        step0_projector_sequences, step0_receiver_sequences
    )
    result: dict[str, Any] = {
        "step": int(step),
        "projector": {
            key.removeprefix("projector_"): value
            for key, value in current.items()
            if key.startswith("projector_") and key not in {"projector_pooled"}
        },
        "receiver": {
            key.removeprefix("receiver_"): value
            for key, value in current.items()
            if key.startswith("receiver_") and key not in {"receiver_pooled"}
        },
    }
    for role, pooled_key, base in (
        ("projector", "projector_pooled", step0_projector_sequences),
        ("receiver", "receiver_pooled", step0_receiver_sequences),
    ):
        current_matrix = current[pooled_key]
        base_matrix = summarize_batch_embeddings(
            step0_projector_sequences,
            step0_receiver_sequences,
        )[pooled_key]
        current_summary = _sequence_summary(
            projector_sequences if role == "projector" else receiver_sequences,
            name=f"{role}_probe",
        )
        base_summary = _sequence_summary(
            base,
            name=f"{role}_step0",
        )
        role_result = result[role]
        role_result.update(
            {
                "relative_spread_ratio": float(
                    current_summary["relative_spread"]
                    / max(base_summary["relative_spread"], 1e-30)
                ),
                "effective_rank_ratio": float(
                    current_summary["effective_rank"]
                    / max(base_summary["effective_rank"], 1e-30)
                ),
                "sample_rms_ratio": float(
                    current_summary["output_rms"]
                    / max(base_summary["output_rms"], 1e-30)
                ),
                "pairwise_distance_correlation": _pearson(
                    _pairwise_distances(base_matrix),
                    _pairwise_distances(current_matrix),
                ),
                "centered_gram_similarity": _centered_gram_similarity(
                    base_matrix, current_matrix
                ),
            }
        )
    # pooled tensors只用于计算，不能进入 JSON；调用方序列化前会得到纯标量。
    return result


def jsonable_probe(payload: dict[str, Any]) -> dict[str, Any]:
    """把 probe 中的临时 Tensor/标量转换为 JSON-safe 结构。"""

    if isinstance(payload, Tensor):
        raise ValueError("probe payload still contains an internal tensor")
    if isinstance(payload, dict):
        return {key: jsonable_probe(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [jsonable_probe(value) for value in payload]
    if isinstance(payload, tuple):
        return [jsonable_probe(value) for value in payload]
    if isinstance(payload, (float, int, str, bool)) or payload is None:
        return payload
    raise TypeError(f"unsupported probe payload value: {type(payload)!r}")


def _probe_value(probe: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = probe
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def evaluate_guards(
    current: dict[str, Any],
    *,
    previous: dict[str, Any] | None,
    state: dict[str, int],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """执行冻结 guards，并返回是否应自动止损。"""

    guards = contract["guards"]
    hard = guards["hard"]
    warning: list[str] = []
    critical: list[str] = []

    def check_min(path: tuple[str, ...], threshold: float, label: str) -> bool:
        value = _probe_value(current, path)
        return value is not None and value >= float(threshold)

    projector_spread_ok = check_min(
        ("projector", "relative_spread_ratio"),
        hard["relative_spread_ratio_min"],
        "projector_relative_spread",
    )
    receiver_spread_ok = check_min(
        ("receiver", "relative_spread_ratio"),
        hard["relative_spread_ratio_min"],
        "receiver_relative_spread",
    )
    projector_rank_ok = check_min(
        ("projector", "effective_rank_ratio"),
        hard["effective_rank_ratio_min"],
        "projector_effective_rank",
    )
    receiver_rank_ok = check_min(
        ("receiver", "effective_rank_ratio"),
        hard["effective_rank_ratio_min"],
        "receiver_effective_rank",
    )
    hard_fail = not all(
        (projector_spread_ok, receiver_spread_ok, projector_rank_ok, receiver_rank_ok)
    )
    state["consecutive_hard_failures"] = (
        state.get("consecutive_hard_failures", 0) + 1 if hard_fail else 0
    )
    if hard_fail:
        warning.append("representation_hard_guard_failed")

    for role in ("projector", "receiver"):
        top1 = _probe_value(current, (role, "top1_variance_fraction"))
        rms_ratio = _probe_value(current, (role, "sample_rms_ratio"))
        if top1 is not None and top1 > float(guards["warning"]["top1_variance_fraction"]):
            warning.append(f"{role}_top1_variance_warning")
        if top1 is not None and top1 > float(guards["critical"]["top1_variance_fraction"]):
            critical.append(f"{role}_top1_variance_critical")
        if rms_ratio is not None and rms_ratio > float(guards["warning"]["output_rms_ratio"]):
            warning.append(f"{role}_output_rms_warning")
        if rms_ratio is not None and rms_ratio > float(guards["critical"]["output_rms_ratio"]):
            critical.append(f"{role}_output_rms_critical")

    preference = _probe_value(current, ("causal", "correct_preference"))
    shuffled_preference = _probe_value(current, ("causal", "shuffled_preference"))
    if preference is not None and shuffled_preference is not None:
        bad_preference = preference <= shuffled_preference
        state["consecutive_bad_preference"] = (
            state.get("consecutive_bad_preference", 0) + 1 if bad_preference else 0
        )
        if bad_preference:
            warning.append("correct_preference_not_above_shuffled")
        if state["consecutive_bad_preference"] >= int(
            guards["critical"]["causal_consecutive_probe_points"]
        ):
            critical.append("causal_preference_critical")

    logp_delta = _probe_value(current, ("causal", "vision_minus_shuffle_correct_logp"))
    if logp_delta is not None:
        bad_logp = logp_delta <= 0.0
        state["consecutive_nonpositive_logp"] = (
            state.get("consecutive_nonpositive_logp", 0) + 1 if bad_logp else 0
        )
        if state["consecutive_nonpositive_logp"] >= int(
            guards["critical"]["causal_consecutive_probe_points"]
        ):
            critical.append("vision_minus_shuffle_logp_critical")

    if previous is not None:
        for role in ("projector", "receiver"):
            old_rms = _probe_value(previous, (role, "sample_rms_ratio"))
            new_rms = _probe_value(current, (role, "sample_rms_ratio"))
            old_spread = _probe_value(previous, (role, "relative_spread_ratio"))
            new_spread = _probe_value(current, (role, "relative_spread_ratio"))
            if (
                old_rms is not None
                and new_rms is not None
                and old_spread is not None
                and new_spread is not None
                and new_rms > old_rms
                and new_spread < old_spread
            ):
                critical.append(f"{role}_rms_rising_spread_falling")

    nonfinite = bool(current.get("has_nan_or_inf", False))
    if nonfinite:
        critical.append("nan_or_inf")
    gradient_before = _probe_value(current, ("training", "gradient_norm_before_clip"))
    gradient_after = _probe_value(current, ("training", "gradient_norm_after_clip"))
    if gradient_before is not None and gradient_before > float(guards["critical"]["gradient_norm_max"]):
        critical.append("gradient_norm_exploded")
    if gradient_after is not None and gradient_after == 0.0:
        critical.append("gradient_norm_zero_after_clip")

    if state.get("consecutive_hard_failures", 0) >= int(
        guards["critical"]["consecutive_hard_failures"]
    ):
        critical.append("consecutive_representation_hard_failures")

    stop = bool(critical)
    return {
        "stop": stop,
        "warnings": sorted(set(warning)),
        "critical": sorted(set(critical)),
        "hard_guard_pass": not hard_fail,
        "state": dict(state),
    }


def validate_health_contract(contract: dict[str, Any]) -> None:
    """在训练开始前验证所有阈值，防止运行中临时改门槛。"""

    if contract.get("format_version") != HEALTH_CONTRACT_FORMAT_VERSION:
        raise ValueError("unsupported projector health contract format")
    if int(contract.get("canonical_projector_width", 0)) <= 0:
        raise ValueError("canonical projector width must be positive")
    schedule = contract.get("probe_schedule", {})
    if tuple(schedule.get("initial_steps", [])) != DEFAULT_PROBE_STEPS:
        raise ValueError("initial probe schedule differs from the frozen contract")
    if int(schedule.get("every_after_step", 0)) <= 0:
        raise ValueError("probe schedule every_after_step must be positive")
    guards = contract.get("guards", {})
    hard = guards.get("hard", {})
    warning = guards.get("warning", {})
    critical = guards.get("critical", {})
    if float(hard.get("relative_spread_ratio_min", 0.0)) != 0.25:
        raise ValueError("relative-spread hard threshold is not the frozen 0.25")
    if float(hard.get("effective_rank_ratio_min", 0.0)) != 0.50:
        raise ValueError("effective-rank hard threshold is not the frozen 0.50")
    if float(warning.get("top1_variance_fraction", 0.0)) != 0.80:
        raise ValueError("top1 warning threshold differs")
    if float(critical.get("top1_variance_fraction", 0.0)) != 0.90:
        raise ValueError("top1 critical threshold differs")
    if float(warning.get("output_rms_ratio", 0.0)) != 10.0:
        raise ValueError("RMS warning threshold differs")
    if float(critical.get("output_rms_ratio", 0.0)) != 50.0:
        raise ValueError("RMS critical threshold differs")
