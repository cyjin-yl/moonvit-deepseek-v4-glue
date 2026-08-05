"""ScreenSpot 坐标答案的 teacher-forced 成对偏好诊断。"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable, Sequence

from .grounding_contract import format_click_action


def target_click_answer(sample: dict[str, Any]) -> str:
    """把冻结 bbox 中心转换为训练合同使用的 canonical click 文本。"""

    box = sample.get("bbox_999_xyxy")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("ScreenSpot sample needs bbox_999_xyxy")
    x1, y1, x2, y2 = (float(value) for value in box)
    if not (0.0 <= x1 <= x2 <= 999.0 and 0.0 <= y1 <= y2 <= 999.0):
        raise ValueError("ScreenSpot bbox falls outside [0, 999]")
    center = (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0)))
    return format_click_action(center)


def build_counterfactual_targets(
    manifest: dict[str, Any],
) -> dict[str, dict[str, str]]:
    """沿预冻结 shuffled-image derangement 绑定反事实坐标答案。"""

    samples = list(manifest.get("samples", []))
    by_id = {str(sample["sample_id"]): sample for sample in samples}
    if len(by_id) != len(samples):
        raise ValueError("ScreenSpot samples contain duplicate IDs")
    mapping_rows = list(
        manifest.get("shuffled_image_control", {}).get("mapping", [])
    )
    mapping: dict[str, dict[str, str]] = {}
    for row in mapping_rows:
        sample_id = str(row["sample_id"])
        shuffled_id = str(row["shuffled_image_sample_id"])
        if sample_id not in by_id:
            raise ValueError(f"unknown source sample in shuffled mapping: {sample_id}")
        if shuffled_id not in by_id:
            raise ValueError(f"unknown shuffled sample: {shuffled_id}")
        if sample_id == shuffled_id:
            raise ValueError(f"shuffled mapping contains a fixed point: {sample_id}")
        if sample_id in mapping:
            raise ValueError(f"duplicate shuffled mapping source: {sample_id}")
        correct = target_click_answer(by_id[sample_id])
        counterfactual = target_click_answer(by_id[shuffled_id])
        if correct == counterfactual:
            raise ValueError(
                f"counterfactual target did not change for sample: {sample_id}"
            )
        mapping[sample_id] = {
            "correct_answer": correct,
            "counterfactual_answer": counterfactual,
            "counterfactual_sample_id": shuffled_id,
        }
    if set(mapping) != set(by_id):
        missing = sorted(set(by_id) - set(mapping))
        raise ValueError(f"shuffled mapping does not cover every sample: {missing[:3]}")
    return mapping


def _validated_stats(name: str, value: dict[str, Any]) -> dict[str, float | int]:
    required = (
        "answer_tokens",
        "logp_sum",
        "logp_mean",
        "token_normalized_nll",
    )
    if any(key not in value for key in required):
        raise ValueError(f"{name} answer statistics are incomplete")
    tokens = int(value["answer_tokens"])
    numeric = {
        key: float(value[key])
        for key in ("logp_sum", "logp_mean", "token_normalized_nll")
    }
    if tokens <= 0 or not all(math.isfinite(item) for item in numeric.values()):
        raise ValueError(f"{name} answer statistics are invalid")
    return {"answer_tokens": tokens, **numeric}


def make_preference_row(
    *,
    sample: dict[str, Any],
    condition: str,
    input_image_sample_id: str | None,
    counterfactual_sample_id: str,
    correct_answer: str,
    counterfactual_answer: str,
    correct_stats: dict[str, Any],
    counterfactual_stats: dict[str, Any],
) -> dict[str, Any]:
    """保存一条同 prompt 下 correct/counterfactual 答案的可审计比较。"""

    if correct_answer == counterfactual_answer:
        raise ValueError("preference candidates must differ")
    correct = _validated_stats("correct", correct_stats)
    counterfactual = _validated_stats("counterfactual", counterfactual_stats)
    margin = float(correct["logp_mean"]) - float(counterfactual["logp_mean"])
    sum_margin = float(correct["logp_sum"]) - float(counterfactual["logp_sum"])
    return {
        "sample_id": str(sample["sample_id"]),
        "condition": str(condition),
        "platform": str(sample["platform"]),
        "target_type": str(sample["target_type"]),
        "input_image_sample_id": (
            str(input_image_sample_id) if input_image_sample_id is not None else None
        ),
        "counterfactual_sample_id": str(counterfactual_sample_id),
        "correct_answer": str(correct_answer),
        "counterfactual_answer": str(counterfactual_answer),
        "correct_answer_tokens": int(correct["answer_tokens"]),
        "counterfactual_answer_tokens": int(counterfactual["answer_tokens"]),
        "correct_logp_sum": float(correct["logp_sum"]),
        "counterfactual_logp_sum": float(counterfactual["logp_sum"]),
        "correct_logp_mean": float(correct["logp_mean"]),
        "counterfactual_logp_mean": float(counterfactual["logp_mean"]),
        "correct_token_nll": float(correct["token_normalized_nll"]),
        "counterfactual_token_nll": float(
            counterfactual["token_normalized_nll"]
        ),
        "correct_margin": margin,
        "correct_sum_margin": sum_margin,
        "correct_preferred": margin > 0.0,
        "tie": margin == 0.0,
    }


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _condition_summary(
    rows: Sequence[dict[str, Any]], *, require_unique_ids: bool = True
) -> dict[str, Any]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("preference summary needs at least one row")
    ids = [str(row["sample_id"]) for row in materialized]
    if require_unique_ids and len(ids) != len(set(ids)):
        raise ValueError("preference summary contains duplicate sample IDs")
    margins = [float(row["correct_margin"]) for row in materialized]
    if not all(math.isfinite(value) for value in margins):
        raise ValueError("preference summary contains non-finite margins")
    preferred = sum(bool(row["correct_preferred"]) for row in materialized)
    ties = sum(bool(row["tie"]) for row in materialized)
    return {
        "records": len(materialized),
        "preference_count": preferred,
        "paired_preference_accuracy": preferred / len(materialized),
        "tie_count": ties,
        "mean_correct_margin": statistics.fmean(margins),
        "median_correct_margin": statistics.median(margins),
        "mean_correct_sum_margin": _mean(materialized, "correct_sum_margin"),
        "mean_correct_logp": _mean(materialized, "correct_logp_mean"),
        "mean_counterfactual_logp": _mean(
            materialized, "counterfactual_logp_mean"
        ),
        "mean_correct_token_nll": _mean(materialized, "correct_token_nll"),
        "mean_counterfactual_token_nll": _mean(
            materialized, "counterfactual_token_nll"
        ),
    }


def _breakdowns(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[dict[str, Any]]] = {"overall": list(rows)}
    for row in rows:
        for key in (str(row["target_type"]), str(row["platform"])):
            values.setdefault(key, []).append(row)
    return {key: _condition_summary(value) for key, value in values.items()}


def summarize_preference_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """按 condition、平台和 text/icon-widget 汇总 paired preference。"""

    materialized = list(rows)
    if not materialized:
        raise ValueError("preference summary needs rows")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        grouped[str(row["condition"])].append(row)
    return {
        "schema_version": "grounding-teacher-forced-preference-summary-v1",
        "conditions": {
            condition: {"breakdowns": _breakdowns(condition_rows)}
            for condition, condition_rows in sorted(grouped.items())
        },
    }


def _metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    summary = _condition_summary(rows, require_unique_ids=False)
    return {
        key: float(summary[key])
        for key in (
            "paired_preference_accuracy",
            "mean_correct_margin",
            "median_correct_margin",
            "mean_correct_logp",
            "mean_correct_token_nll",
        )
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile needs values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_preference_bootstrap(
    first: Sequence[dict[str, Any]],
    second: Sequence[dict[str, Any]],
    *,
    samples: int = 2_000,
    seed: int = 20260805,
) -> dict[str, Any]:
    """对同一批样本的 teacher-forced 偏好做固定 seed paired bootstrap。"""

    first_rows = list(first)
    second_rows = list(second)
    first_ids = [str(row["sample_id"]) for row in first_rows]
    second_ids = [str(row["sample_id"]) for row in second_rows]
    if not first_rows or first_ids != second_ids:
        raise ValueError("paired bootstrap requires identical sample IDs in order")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    first_metrics = _metrics(first_rows)
    second_metrics = _metrics(second_rows)
    lower_is_better = {"mean_correct_token_nll"}
    distributions = {name: [] for name in first_metrics}
    rng = random.Random(seed)
    for _ in range(samples):
        indices = [rng.randrange(len(first_rows)) for _ in first_rows]
        first_draw = _metrics([first_rows[index] for index in indices])
        second_draw = _metrics([second_rows[index] for index in indices])
        for name in distributions:
            if name in lower_is_better:
                value = second_draw[name] - first_draw[name]
            else:
                value = first_draw[name] - second_draw[name]
            distributions[name].append(value)
    metrics = {}
    for name, first_value in first_metrics.items():
        second_value = second_metrics[name]
        improvement = (
            second_value - first_value
            if name in lower_is_better
            else first_value - second_value
        )
        distribution = distributions[name]
        metrics[name] = {
            "first": first_value,
            "second": second_value,
            "improvement": improvement,
            "ci95": [
                _percentile(distribution, 0.025),
                _percentile(distribution, 0.975),
            ],
            "bootstrap_samples_requested": samples,
            "bootstrap_samples_valid": len(distribution),
        }
    return {
        "schema_version": "paired-grounding-preference-bootstrap-v1",
        "sample_count": len(first_rows),
        "seed": seed,
        "confidence_level": 0.95,
        "method": "paired percentile bootstrap; type-7 quantiles",
        "orientation": "positive_means_first_condition_better",
        "metrics": metrics,
    }
