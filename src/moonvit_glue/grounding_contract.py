"""严格、可预注册的 ScreenSpot click-grounding 评分合同。

历史 ``moonvit_glue.metrics.extract_point`` 保留宽松兼容行为，供旧实验回读。
本模块只接受固定动作语法，避免评分器从自然语言或多个候选坐标中猜答案。
"""

from __future__ import annotations

import math
import random
import re
from statistics import median
from typing import Any, Callable, Iterable, Sequence

Point = tuple[int, int]

SCALE_MAX = 999.0
MAX_CENTER_L2 = math.sqrt(2.0) * SCALE_MAX
MAX_BBOX_L2 = MAX_CENTER_L2
MAX_BBOX_L1 = 2.0 * SCALE_MAX
THRESHOLDS = (50, 100, 200)

# 只容许首尾空白；动作内部必须与预注册的 canonical 文本完全一致。
_CLICK_ACTION = re.compile(
    r"\A[ \t\r\n]*click\(start_box=\[(?P<x>[0-9]{1,4}), (?P<y>[0-9]{1,4})\]\)[ \t\r\n]*\Z"
)


def parse_click_action(text: str) -> Point | None:
    """解析唯一合法格式 ``click(start_box=[x, y])``。

    坐标必须是 ``[0, 999]`` 内整数。任何前后解释、第二个坐标、浮点数、
    大小写变体或缺失的 canonical 空格都会计为 parse failure。
    """

    match = _CLICK_ACTION.fullmatch(text)
    if match is None:
        return None
    point = (int(match.group("x")), int(match.group("y")))
    if any(value < 0 or value > int(SCALE_MAX) for value in point):
        return None
    return point


def format_click_action(point: Sequence[int | float]) -> str:
    """把整数点格式化成合同唯一允许的输出文本。"""

    if len(point) != 2:
        raise ValueError("point must contain exactly two coordinates")
    values: list[int] = []
    for raw in point:
        value = int(raw)
        if float(raw) != value or not 0 <= value <= int(SCALE_MAX):
            raise ValueError("click coordinates must be integers in [0, 999]")
        values.append(value)
    return f"click(start_box=[{values[0]}, {values[1]}])"


def _validate_box(target_box: Sequence[float]) -> tuple[float, float, float, float]:
    if len(target_box) != 4:
        raise ValueError("target_box must contain [x1, y1, x2, y2]")
    x1, y1, x2, y2 = (float(value) for value in target_box)
    if not (0.0 <= x1 <= x2 <= SCALE_MAX and 0.0 <= y1 <= y2 <= SCALE_MAX):
        raise ValueError("target_box must be ordered and contained in [0, 999]")
    return x1, y1, x2, y2


def _distance_to_box(
    point: tuple[float, float], box: tuple[float, float, float, float]
) -> tuple[float, float]:
    x, y = point
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy), dx + dy


def score_click_prediction(
    *,
    sample_id: str,
    prediction: str,
    target_box: Sequence[float],
) -> dict[str, Any]:
    """为单条输出保存完整、无损的 grounding 原始评分字段。"""

    box = _validate_box(target_box)
    parsed = parse_click_action(prediction)
    result: dict[str, Any] = {
        "sample_id": str(sample_id),
        "prediction": prediction,
        "target_box_999_xyxy": list(box),
        "parse_ok": parsed is not None,
        "prediction_point_999": list(parsed) if parsed is not None else None,
        "accuracy_at_50": False,
        "accuracy_at_100": False,
        "accuracy_at_200": False,
        "click_in_box": False,
        "center_l2": None,
        "center_l2_penalized": MAX_CENTER_L2,
        "bbox_l2": None,
        "bbox_l2_penalized": MAX_BBOX_L2,
        "bbox_l1": None,
        "bbox_l1_penalized": MAX_BBOX_L1,
    }
    if parsed is None:
        return result

    point = (float(parsed[0]), float(parsed[1]))
    x1, y1, x2, y2 = box
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    center_l2 = math.dist(point, center)
    bbox_l2, bbox_l1 = _distance_to_box(point, box)
    result.update(
        {
            "accuracy_at_50": center_l2 <= 50.0,
            "accuracy_at_100": center_l2 <= 100.0,
            "accuracy_at_200": center_l2 <= 200.0,
            "click_in_box": x1 <= point[0] <= x2 and y1 <= point[1] <= y2,
            "center_l2": center_l2,
            "center_l2_penalized": center_l2,
            "bbox_l2": bbox_l2,
            "bbox_l2_penalized": bbox_l2,
            "bbox_l1": bbox_l1,
            "bbox_l1_penalized": bbox_l1,
        }
    )
    return result


def _percentile(values: Sequence[float], probability: float) -> float | None:
    """R/NumPy type-7 线性插值百分位，避免平台相关的统计库默认值。"""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _describe(values: Sequence[float]) -> dict[str, float | int | None]:
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": sum(numeric) / len(numeric) if numeric else None,
        "median": median(numeric) if numeric else None,
        "p90": _percentile(numeric, 0.90),
        "minimum": min(numeric) if numeric else None,
    }


def _rate_summary(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    parsed_count = sum(bool(row["parse_ok"]) for row in rows)
    hit_count = sum(bool(row[key]) for row in rows)
    return {
        "hit_count": hit_count,
        "parsed_denominator": parsed_count,
        "parsed_accuracy": hit_count / parsed_count if parsed_count else None,
        "all_denominator": len(rows),
        "all_accuracy": hit_count / len(rows) if rows else None,
    }


def summarize_click_scores(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """聚合格式、中心距离、阈值命中及 ScreenSpot 官方点击指标。"""

    records = list(rows)
    sample_ids = [str(row["sample_id"]) for row in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample IDs must be unique")
    parse_count = sum(bool(row["parse_ok"]) for row in records)
    parsed_center = [float(row["center_l2"]) for row in records if row["center_l2"] is not None]
    parsed_bbox_l2 = [float(row["bbox_l2"]) for row in records if row["bbox_l2"] is not None]
    parsed_bbox_l1 = [float(row["bbox_l1"]) for row in records if row["bbox_l1"] is not None]

    summary: dict[str, Any] = {
        "total_count": len(records),
        "parse_count": parse_count,
        "parse_rate": parse_count / len(records) if records else None,
        "center_distance": {
            "parsed": _describe(parsed_center),
            "all_penalized": _describe(
                [float(row["center_l2_penalized"]) for row in records]
            ),
            "unparsed_penalty": MAX_CENTER_L2,
        },
        "point_to_bbox_l2": {
            "parsed": _describe(parsed_bbox_l2),
            "all_penalized": _describe(
                [float(row["bbox_l2_penalized"]) for row in records]
            ),
            "unparsed_penalty": MAX_BBOX_L2,
        },
        "point_to_bbox_l1": {
            "parsed": _describe(parsed_bbox_l1),
            "all_penalized": _describe(
                [float(row["bbox_l1_penalized"]) for row in records]
            ),
            "unparsed_penalty": MAX_BBOX_L1,
        },
        "click_in_box_accuracy": _rate_summary(records, "click_in_box"),
    }
    for threshold in THRESHOLDS:
        summary[f"accuracy_at_{threshold}"] = _rate_summary(
            records, f"accuracy_at_{threshold}"
        )
    return summary


def _mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _condition_statistics(rows: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    count = len(rows)
    parsed = [row for row in rows if row["parse_ok"]]
    parsed_count = len(parsed)
    statistics: dict[str, float | None] = {
        "parse_rate": parsed_count / count if count else None,
        "click_in_box_parsed": (
            sum(bool(row["click_in_box"]) for row in rows) / parsed_count
            if parsed_count
            else None
        ),
        "click_in_box_all": (
            sum(bool(row["click_in_box"]) for row in rows) / count if count else None
        ),
    }
    for threshold in THRESHOLDS:
        key = f"accuracy_at_{threshold}"
        hits = sum(bool(row[key]) for row in rows)
        statistics[f"{key}_parsed"] = hits / parsed_count if parsed_count else None
        statistics[f"{key}_all"] = hits / count if count else None

    parsed_distances = [float(row["center_l2"]) for row in parsed]
    penalized_distances = [float(row["center_l2_penalized"]) for row in rows]
    statistics.update(
        {
            "mean_center_distance_parsed": _mean_or_none(parsed_distances),
            "median_center_distance_parsed": (
                median(parsed_distances) if parsed_distances else None
            ),
            "mean_center_distance_all_penalized": _mean_or_none(penalized_distances),
            "median_center_distance_all_penalized": (
                median(penalized_distances) if penalized_distances else None
            ),
        }
    )
    return statistics


def _improvement(first: float, second: float, *, lower_is_better: bool) -> float:
    return second - first if lower_is_better else first - second


def paired_bootstrap(
    first: Sequence[dict[str, Any]],
    second: Sequence[dict[str, Any]],
    *,
    samples: int = 2_000,
    seed: int = 20260805,
) -> dict[str, Any]:
    """对完全配对的逐样本结果做 percentile bootstrap 95% CI。

    所有 ``improvement`` 和 CI 都统一成“正值表示第一个条件更好”。距离越低
    越好，因此距离项使用 ``second - first``；其余项使用 ``first - second``。
    """

    first_rows = list(first)
    second_rows = list(second)
    first_ids = [row["sample_id"] for row in first_rows]
    second_ids = [row["sample_id"] for row in second_rows]
    if first_ids != second_ids:
        raise ValueError("paired bootstrap requires identical sample IDs in identical order")
    if not first_rows:
        raise ValueError("paired bootstrap requires at least one sample")
    if samples < 1:
        raise ValueError("samples must be positive")

    first_observed = _condition_statistics(first_rows)
    second_observed = _condition_statistics(second_rows)
    metric_names = tuple(first_observed)
    bootstrap_improvements: dict[str, list[float]] = {
        name: [] for name in metric_names
    }
    rng = random.Random(seed)
    count = len(first_rows)
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        first_stats = _condition_statistics([first_rows[index] for index in indices])
        second_stats = _condition_statistics([second_rows[index] for index in indices])
        for name in metric_names:
            first_value = first_stats[name]
            second_value = second_stats[name]
            if first_value is None or second_value is None:
                continue
            lower_is_better = "distance" in name
            bootstrap_improvements[name].append(
                _improvement(first_value, second_value, lower_is_better=lower_is_better)
            )

    metrics: dict[str, Any] = {}
    for name in metric_names:
        first_value = first_observed[name]
        second_value = second_observed[name]
        if first_value is None or second_value is None:
            raw_delta = None
            improvement = None
        else:
            raw_delta = first_value - second_value
            improvement = _improvement(
                first_value,
                second_value,
                lower_is_better="distance" in name,
            )
        values = bootstrap_improvements[name]
        metrics[name] = {
            "first": first_value,
            "second": second_value,
            "raw_first_minus_second": raw_delta,
            "improvement": improvement,
            "ci95": [
                _percentile(values, 0.025),
                _percentile(values, 0.975),
            ],
            "bootstrap_samples_requested": samples,
            "bootstrap_samples_valid": len(values),
        }

    return {
        "schema_version": "paired-grounding-bootstrap-v1",
        "sample_count": count,
        "seed": seed,
        "confidence_level": 0.95,
        "method": "paired percentile bootstrap; type-7 quantiles",
        "orientation": "positive_means_first_condition_better",
        "metrics": metrics,
    }
