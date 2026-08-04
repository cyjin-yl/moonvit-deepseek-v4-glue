"""Checkpoint 轨迹实验的纯聚合工具。"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from .metrics import normalize_answer


def ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value": (numerator / denominator if denominator else None),
    }


def derangement_indices(length: int, *, seed: int) -> list[int]:
    """返回确定性的 Sattolo 循环，保证没有固定点。"""

    if length < 2:
        raise ValueError("a derangement needs at least two records")
    values = list(range(length))
    rng = random.Random(seed)
    for index in range(length - 1, 0, -1):
        other = rng.randrange(index)
        values[index], values[other] = values[other], values[index]
    if any(index == other for index, other in enumerate(values)):
        raise RuntimeError("internal error: Sattolo cycle contains a fixed point")
    return values


def resolve_control_features(
    condition: str,
    sample_id: str,
    split: str,
    control: dict,
    cache_get,
):
    """解析单个视觉控制；permutation 只改变顺序并保留特征值。"""

    if condition == "blind":
        return None
    if condition == "vision":
        return cache_get(str(sample_id))
    if condition == "blank":
        return cache_get(str(control.get("blank_image_id", f"control:{split}:blank")))
    if condition == "same_image":
        return cache_get(f"control:{split}:same")
    if condition == "shuffled_image":
        shuffled_id = str(control["shuffled_image_id"])
        if shuffled_id == str(sample_id):
            raise ValueError(f"shuffle fixed point: {sample_id}")
        return cache_get(shuffled_id)
    if condition == "patch_permutation":
        import torch

        groups = cache_get(str(sample_id))
        seed = int(control["patch_permutation"]["seed"])
        permuted = []
        for group_index, group in enumerate(groups):
            generator = torch.Generator(device=group.device)
            generator.manual_seed((seed + group_index) % (2**63 - 1))
            order = torch.randperm(group.shape[0], generator=generator, device=group.device)
            permuted.append(group.index_select(0, order))
        return permuted
    raise ValueError(f"unknown control condition: {condition}")


def control_image_records(controls: Iterable[dict]) -> list[dict]:
    """为每个 split 构造两条固定控制图记录。"""

    by_split: dict[str, dict[str, str]] = {}
    for row in controls:
        split = str(row["split"])
        paths = {
            "blank": str(row["blank_image"]),
            "same": str(row["same_image"]),
        }
        previous = by_split.setdefault(split, paths)
        if previous != paths:
            raise ValueError(f"control images drift within split {split!r}")
    records = []
    for split in sorted(by_split):
        for kind in ("blank", "same"):
            records.append({
                "id": f"control:{split}:{kind}",
                "image": by_split[split][kind],
                "question": "control image",
                "answers": ["n/a"],
                "metric": "exact_match",
            })
    return records


def _summary(rows: list[dict]) -> dict:
    pairs: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pairs[str(row["pair_id"])].append(row)
    malformed = [pair_id for pair_id, pair in pairs.items() if len(pair) != 2]
    if malformed:
        raise ValueError(f"synthetic rows do not contain complete pairs: {malformed[:3]}")

    correct = sum(bool(row.get("correct", False)) for row in rows)
    failures = sum(bool(row.get("failure")) for row in rows)
    paired_correct = 0
    answer_flips = 0
    prediction_flips = 0
    for pair in pairs.values():
        predictions = [normalize_answer(str(row.get("prediction") or "")) for row in pair]
        flipped = predictions[0] != predictions[1]
        pair_correct = all(bool(row.get("correct", False)) for row in pair)
        paired_correct += pair_correct
        prediction_flips += flipped
        answer_flips += pair_correct and flipped
    return {
        "accuracy": ratio(correct, len(rows)),
        "paired_accuracy": ratio(paired_correct, len(pairs)),
        "answer_flip_accuracy": ratio(answer_flips, len(pairs)),
        "prediction_flip_rate": ratio(prediction_flips, len(pairs)),
        "failures": ratio(failures, len(rows)),
        "samples": len(rows),
        "pairs": len(pairs),
    }


def summarize_synthetic_rows(rows: Iterable[dict]) -> dict:
    """按总体和任务分别聚合完整最小对。"""

    materialized = list(rows)
    summary = _summary(materialized)
    tasks = sorted({str(row["task"]) for row in materialized})
    summary["by_task"] = {
        task: _summary([row for row in materialized if str(row["task"]) == task])
        for task in tasks
    }
    return summary
