"""为答案变化的最小图像对计算 teacher-forced 视觉证据。"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Iterable


def _ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value": numerator / denominator if denominator else None,
    }


def answer_logprob_stats(logits, labels, *, ignore_index: int = -100) -> list[dict]:
    """返回答案的因果 log-prob 总量与 token 归一化值。"""

    import torch

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits [batch, sequence, vocab] must align with rank-2 labels")
    shifted_logits = logits[:, :-1]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(ignore_index)
    if bool(((shifted_labels[mask] < 0) | (shifted_labels[mask] >= logits.shape[-1])).any()):
        raise ValueError("answer label is outside the logits vocabulary")
    output = []
    for row_logits, row_labels, row_mask in zip(
        shifted_logits, shifted_labels, mask
    ):
        count = int(row_mask.sum())
        if count == 0:
            raise ValueError("every sample needs at least one scored answer token")
        # 只对受监督 answer 位置计算 full-vocabulary log-softmax；长视觉前缀
        # 不参与分数，提前筛掉可显著降低 1024px grounding 诊断的显存与计算量。
        selected_logits = row_logits[row_mask].float()
        selected_labels = row_labels[row_mask]
        selected_logp = torch.log_softmax(selected_logits, dim=-1).gather(
            -1, selected_labels.unsqueeze(-1)
        ).squeeze(-1)
        total = float(selected_logp.sum())
        mean = total / count
        output.append(
            {
                "answer_tokens": count,
                "logp_sum": total,
                "logp_mean": mean,
                "token_normalized_nll": -mean,
            }
        )
    return output


def build_pair_index(records: Iterable[dict]) -> dict[str, dict[str, str]]:
    """把每个最小对样本映射到自身答案与成对反事实答案。"""

    pairs: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        pairs[str(record["pair_id"])].append(record)
    index: dict[str, dict[str, str]] = {}
    for pair_id, pair in pairs.items():
        if len(pair) != 2:
            raise ValueError(f"minimal pair must contain two samples: {pair_id}")
        first, second = sorted(pair, key=lambda row: str(row.get("pair_variant", row["id"])))
        if str(first["question"]) != str(second["question"]):
            raise ValueError(f"minimal pair question drift: {pair_id}")
        first_answer = str(first["answers"][0])
        second_answer = str(second["answers"][0])
        if first_answer == second_answer:
            raise ValueError(f"minimal pair answer did not change: {pair_id}")
        for record, other, correct, counterfactual in (
            (first, second, first_answer, second_answer),
            (second, first, second_answer, first_answer),
        ):
            sample_id = str(record["id"])
            if sample_id in index:
                raise ValueError(f"duplicate minimal-pair sample id: {sample_id}")
            index[sample_id] = {
                "pair_id": pair_id,
                "correct_answer": correct,
                "counterfactual_answer": counterfactual,
                "paired_image_id": str(other["id"]),
            }
    return index


def _summary(rows: list[dict]) -> dict:
    pairs: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pairs[str(row["pair_id"])].append(row)
    malformed = [pair_id for pair_id, pair in pairs.items() if len(pair) != 2]
    if malformed:
        raise ValueError(f"preference rows do not contain complete pairs: {malformed[:3]}")

    valid_margins = [
        float(row["correct_margin"])
        for row in rows
        if row.get("failure") is None
        and row.get("correct_margin") is not None
        and math.isfinite(float(row["correct_margin"]))
    ]
    sample_successes = sum(
        row.get("failure") is None
        and row.get("correct_margin") is not None
        and float(row["correct_margin"]) > 0
        for row in rows
    )
    pair_successes = sum(
        all(
            row.get("failure") is None
            and row.get("correct_margin") is not None
            and float(row["correct_margin"]) > 0
            for row in pair
        )
        for pair in pairs.values()
    )
    failures = sum(bool(row.get("failure")) for row in rows)
    return {
        "samples": len(rows),
        "pairs": len(pairs),
        "sample_preference_accuracy": _ratio(sample_successes, len(rows)),
        "paired_preference_accuracy": _ratio(pair_successes, len(pairs)),
        "mean_correct_margin": (
            statistics.fmean(valid_margins) if valid_margins else None
        ),
        "median_correct_margin": (
            statistics.median(valid_margins) if valid_margins else None
        ),
        "mean_correct_token_nll": (
            statistics.fmean(
                float(row["correct_token_nll"])
                for row in rows
                if row.get("failure") is None and row.get("correct_token_nll") is not None
            )
            if len(valid_margins)
            else None
        ),
        "mean_counterfactual_token_nll": (
            statistics.fmean(
                float(row["counterfactual_token_nll"])
                for row in rows
                if row.get("failure") is None
                and row.get("counterfactual_token_nll") is not None
            )
            if len(valid_margins)
            else None
        ),
        "failures": _ratio(failures, len(rows)),
    }


def summarize_preference_rows(rows: Iterable[dict]) -> dict:
    """按总体与任务汇总样本偏好和严格双图偏好。"""

    materialized = list(rows)
    summary = _summary(materialized)
    tasks = sorted({str(row["task"]) for row in materialized})
    summary["by_task"] = {
        task: _summary([row for row in materialized if str(row["task"]) == task])
        for task in tasks
    }
    return summary
