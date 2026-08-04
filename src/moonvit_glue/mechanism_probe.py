"""逐层线性 probe 与激活替换共用的透明机制分析原语。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor


def select_complete_task_pairs(records: Sequence[dict], task: str) -> list[dict]:
    """按 pair/id 排序选择一个任务，并拒绝不完整或重复的 a/b 对。"""
    selected = [record for record in records if str(record.get("task")) == task]
    by_pair: dict[str, list[dict]] = {}
    for record in selected:
        by_pair.setdefault(str(record["pair_id"]), []).append(record)
    if not by_pair:
        raise ValueError(f"task {task!r} has no records")
    for pair_id, rows in by_pair.items():
        variants = {str(row.get("pair_variant")) for row in rows}
        ids = {str(row["id"]) for row in rows}
        if len(rows) != 2 or variants != {"a", "b"} or len(ids) != 2:
            raise ValueError(f"task {task!r} pair {pair_id!r} is not a complete a/b pair")
    return [
        row
        for pair_id in sorted(by_pair)
        for row in sorted(by_pair[pair_id], key=lambda value: str(value["id"]))
    ]


def _flatten_merged_tokens(tokens: Tensor) -> Tensor:
    if tokens.ndim == 4:
        return tokens.flatten(start_dim=2)
    if tokens.ndim == 3:
        return tokens
    raise ValueError(
        "tokens must have shape [batch, tokens, hidden] or "
        "[batch, tokens, merge, width]"
    )


def pool_token_grid(tokens: Tensor, mode: str) -> Tensor:
    """对固定正方形 token 网格做全局、中心或 2×2 透明池化。"""
    flat = _flatten_merged_tokens(tokens)
    token_count = int(flat.shape[1])
    side = math.isqrt(token_count)
    if side * side != token_count:
        raise ValueError(f"pooling requires a square token grid, got {token_count}")
    grid = flat.reshape(flat.shape[0], side, side, flat.shape[-1])
    if mode == "global_mean":
        return grid.mean(dim=(1, 2))
    if mode == "center_mean":
        width = max(1, round(side * 0.4))
        start = (side - width) // 2
        return grid[:, start : start + width, start : start + width].mean(dim=(1, 2))
    if mode == "spatial_2x2":
        midpoint = side // 2
        if midpoint == 0:
            raise ValueError("spatial_2x2 requires at least a 2x2 token grid")
        regions = (
            grid[:, :midpoint, :midpoint],
            grid[:, :midpoint, midpoint:],
            grid[:, midpoint:, :midpoint],
            grid[:, midpoint:, midpoint:],
        )
        return torch.cat([region.mean(dim=(1, 2)) for region in regions], dim=-1)
    raise ValueError(f"unknown pooling mode: {mode}")


@dataclass(frozen=True)
class LinearProbe:
    """固定正则的多类 ridge 线性读出，全部参数可直接序列化。"""

    mean: Tensor
    scale: Tensor
    coefficients: Tensor
    class_count: int
    alpha: float


def fit_linear_probe(
    features: Tensor,
    labels: Tensor,
    *,
    class_count: int,
    alpha: float = 1.0,
) -> LinearProbe:
    """用 class-balanced dual ridge 拟合固定 alpha 的多类线性 probe。"""
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("features/labels must have shapes [records, dims] and [records]")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("linear probe input cannot be empty")
    if class_count < 2 or alpha <= 0:
        raise ValueError("class_count must be >=2 and alpha must be positive")
    labels = labels.to(dtype=torch.long, device=features.device)
    if int(labels.min()) < 0 or int(labels.max()) >= class_count:
        raise ValueError("labels fall outside class_count")

    work = features.to(dtype=torch.float64)
    mean = work.mean(dim=0)
    scale = work.std(dim=0, correction=0).clamp_min(1e-6)
    standardized = (work - mean) / scale / math.sqrt(work.shape[1])
    design = torch.cat(
        [standardized, torch.ones(work.shape[0], 1, dtype=work.dtype, device=work.device)],
        dim=1,
    )
    counts = torch.bincount(labels, minlength=class_count).to(dtype=work.dtype)
    if bool((counts == 0).any()):
        raise ValueError("every probe class needs at least one training record")
    sample_weights = work.shape[0] / (class_count * counts[labels])
    root_weights = sample_weights.sqrt().unsqueeze(1)
    weighted_design = design * root_weights
    targets = torch.nn.functional.one_hot(labels, class_count).to(dtype=work.dtype)
    weighted_targets = targets * root_weights
    gram = weighted_design @ weighted_design.T
    gram.diagonal().add_(float(alpha))
    dual = torch.linalg.solve(gram, weighted_targets)
    coefficients = weighted_design.T @ dual
    return LinearProbe(
        mean=mean.to(dtype=torch.float32).cpu(),
        scale=scale.to(dtype=torch.float32).cpu(),
        coefficients=coefficients.to(dtype=torch.float32).cpu(),
        class_count=class_count,
        alpha=float(alpha),
    )


def apply_linear_probe(probe: LinearProbe, features: Tensor) -> tuple[Tensor, Tensor]:
    """返回类别预测和未归一化线性分数。"""
    if features.ndim != 2 or features.shape[1] != probe.mean.numel():
        raise ValueError("probe feature dimension mismatch")
    device = features.device
    work = features.to(dtype=torch.float32)
    mean = probe.mean.to(device=device)
    scale = probe.scale.to(device=device)
    standardized = (work - mean) / scale / math.sqrt(work.shape[1])
    design = torch.cat(
        [standardized, torch.ones(work.shape[0], 1, device=device)], dim=1
    )
    scores = design @ probe.coefficients.to(device=device)
    return scores.argmax(dim=1), scores


def pair_bootstrap_accuracy_delta(
    *,
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
    pair_ids: Sequence[str],
    seed: int,
    samples: int = 2000,
) -> dict[str, float | int]:
    """以 pair 为抽样单位计算两个逐记录准确率的配对差值区间。"""
    if not (len(correct_a) == len(correct_b) == len(pair_ids)) or not pair_ids:
        raise ValueError("paired bootstrap inputs must have the same non-zero length")
    grouped: dict[str, list[float]] = {}
    for a, b, pair_id in zip(correct_a, correct_b, pair_ids, strict=True):
        grouped.setdefault(str(pair_id), []).append(float(bool(a)) - float(bool(b)))
    pair_deltas = torch.tensor(
        [sum(values) / len(values) for _, values in sorted(grouped.items())],
        dtype=torch.float64,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(
        0,
        pair_deltas.numel(),
        (int(samples), pair_deltas.numel()),
        generator=generator,
    )
    draws = pair_deltas[indices].mean(dim=1)
    low, high = torch.quantile(
        draws, torch.tensor([0.025, 0.975], dtype=draws.dtype)
    ).tolist()
    return {
        "pairs": int(pair_deltas.numel()),
        "records": len(pair_ids),
        "mean_delta": float(pair_deltas.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
    }


def pair_bootstrap_mean(
    *,
    values: Sequence[float],
    pair_ids: Sequence[str],
    seed: int,
    samples: int = 2000,
) -> dict[str, float | int]:
    """先在 pair 内平均两个方向，再对 pair 重采样连续效应。"""
    if len(values) != len(pair_ids) or not pair_ids:
        raise ValueError("pair bootstrap values/ids must have the same non-zero length")
    grouped: dict[str, list[float]] = {}
    for value, pair_id in zip(values, pair_ids, strict=True):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("pair bootstrap values must be finite")
        grouped.setdefault(str(pair_id), []).append(numeric)
    pair_means = torch.tensor(
        [sum(rows) / len(rows) for _, rows in sorted(grouped.items())], dtype=torch.float64
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(
        0,
        pair_means.numel(),
        (int(samples), pair_means.numel()),
        generator=generator,
    )
    draws = pair_means[indices].mean(dim=1)
    low, high = torch.quantile(
        draws, torch.tensor([0.025, 0.975], dtype=draws.dtype)
    ).tolist()
    return {
        "pairs": int(pair_means.numel()),
        "records": len(pair_ids),
        "mean": float(pair_means.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
    }


def aligned_effect_delta(
    rows_a: Sequence[dict], rows_b: Sequence[dict]
) -> list[dict[str, str | float]]:
    """按样本 ID 对齐两种干预，返回保留 pair 单位的逐样本效应差。"""
    by_a = {str(row["id"]): row for row in rows_a}
    by_b = {str(row["id"]): row for row in rows_b}
    if len(by_a) != len(rows_a) or len(by_b) != len(rows_b):
        raise ValueError("intervention rows contain duplicate sample IDs")
    if set(by_a) != set(by_b) or not by_a:
        raise ValueError("intervention sample IDs do not match")
    result = []
    for sample_id in sorted(by_a):
        row_a = by_a[sample_id]
        row_b = by_b[sample_id]
        pair_a = str(row_a["pair_id"])
        pair_b = str(row_b["pair_id"])
        if pair_a != pair_b:
            raise ValueError(f"pair ID mismatch for sample {sample_id}")
        effect_a = float(row_a["effect_vs_counterfactual"])
        effect_b = float(row_b["effect_vs_counterfactual"])
        if not math.isfinite(effect_a) or not math.isfinite(effect_b):
            raise ValueError("intervention effects must be finite")
        result.append(
            {
                "id": sample_id,
                "pair_id": pair_a,
                "effect_delta": effect_a - effect_b,
            }
        )
    return result


def pair_label_permutation_test(
    *,
    predictions: Sequence[int],
    labels: Sequence[int],
    pair_ids: Sequence[str],
    seed: int,
    samples: int = 2000,
) -> dict[str, float | int]:
    """在完整 a/b pair 间置换标签元组，构造无图像—标签关联的 null。"""
    if not (len(predictions) == len(labels) == len(pair_ids)) or not pair_ids:
        raise ValueError("pair permutation inputs must have the same non-zero length")
    grouped: dict[str, list[int]] = {}
    for index, pair_id in enumerate(pair_ids):
        grouped.setdefault(str(pair_id), []).append(index)
    ordered_pairs = sorted(grouped)
    if any(len(grouped[pair_id]) != 2 for pair_id in ordered_pairs):
        raise ValueError("pair label permutation requires exactly two records per pair")
    prediction_matrix = torch.tensor(
        [[int(predictions[index]) for index in grouped[pair_id]] for pair_id in ordered_pairs],
        dtype=torch.long,
    )
    label_matrix = torch.tensor(
        [[int(labels[index]) for index in grouped[pair_id]] for pair_id in ordered_pairs],
        dtype=torch.long,
    )
    observed = float(prediction_matrix.eq(label_matrix).float().mean())
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random_keys = torch.rand(
        int(samples), len(ordered_pairs), generator=generator, dtype=torch.float64
    )
    orders = random_keys.argsort(dim=1)
    permuted_labels = label_matrix[orders]
    null = prediction_matrix.unsqueeze(0).eq(permuted_labels).float().mean(dim=(1, 2)).double()
    low, high = torch.quantile(
        null, torch.tensor([0.025, 0.975], dtype=null.dtype)
    ).tolist()
    p_value = (1 + int((null >= observed).sum())) / (int(samples) + 1)
    return {
        "pairs": len(ordered_pairs),
        "records": len(pair_ids),
        "observed_accuracy": observed,
        "null_mean": float(null.mean()),
        "null_ci95_low": float(low),
        "null_ci95_high": float(high),
        "p_value": float(p_value),
        "permutation_samples": int(samples),
        "seed": int(seed),
    }


def patch_hidden_output(output: Any, donor: Tensor, token_mask: Tensor) -> Any:
    """把 decoder layer 输出中指定 token 替换成 donor，并保留其余返回值。"""
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, Tensor) or hidden.ndim != 3:
        raise ValueError("layer output must begin with a [batch, sequence, hidden] tensor")
    if donor.shape != hidden.shape:
        raise ValueError("donor hidden shape must match layer output")
    if token_mask.shape != hidden.shape[:2] or token_mask.dtype != torch.bool:
        raise ValueError("token mask must be bool [batch, sequence]")
    patched = hidden.clone()
    patched[token_mask] = donor.to(device=hidden.device, dtype=hidden.dtype)[token_mask]
    if isinstance(output, tuple):
        return (patched, *output[1:])
    return patched


def last_active_indices(attention_mask: Tensor) -> Tensor:
    """返回每行最后一个有效 token，兼容左填充和右填充。"""
    if attention_mask.ndim != 2:
        raise ValueError("attention mask must be [batch, sequence]")
    valid = attention_mask.to(dtype=torch.bool)
    if bool((valid.sum(dim=1) == 0).any()):
        raise ValueError("every attention row needs at least one active token")
    positions = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
    return positions.masked_fill(~valid, -1).max(dim=1).values


def masked_token_mean(hidden: Tensor, token_mask: Tensor) -> Tensor:
    """对逐样本 token mask 求均值，并拒绝空区域。"""
    if hidden.ndim != 3 or token_mask.shape != hidden.shape[:2]:
        raise ValueError("hidden/token mask shapes must be [batch, sequence, hidden]/[batch, sequence]")
    mask = token_mask.to(device=hidden.device, dtype=torch.bool)
    counts = mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("masked token mean received an empty sample region")
    return (hidden * mask.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)


def square_grid_region_mask(token_count: int, region: str) -> Tensor:
    """在固定正方形 token 网格上返回中心 40% 或其补集。"""
    side = math.isqrt(int(token_count))
    if side * side != token_count:
        raise ValueError("region intervention requires a square token grid")
    width = max(1, round(side * 0.4))
    start = (side - width) // 2
    mask = torch.zeros(side, side, dtype=torch.bool)
    mask[start : start + width, start : start + width] = True
    if region == "center":
        return mask.flatten()
    if region == "outer":
        return mask.logical_not().flatten()
    if region == "full":
        return torch.ones(token_count, dtype=torch.bool)
    raise ValueError(f"unknown grid region: {region}")
