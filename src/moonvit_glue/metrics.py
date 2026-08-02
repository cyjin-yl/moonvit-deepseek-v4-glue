"""Benchmark metrics for the MoonViT glue prototype.

Pure-Python scoring helpers shared by ``tools/eval_vlm.py`` and the tests.
They intentionally carry no torch dependency so the metric contract can be
verified on any machine, including ones without a PyTorch install.

Benchmark philosophy (mirrors the community GLM-5.2 vision experiment):

* Report the *blind* baseline (same model, no image) next to every number so
  language priors are not mistaken for visual grounding.
* For grounding, report the coordinate parse rate separately from accuracy;
  a model that cannot even emit parseable coordinates fails cheaply first.
* Grounding coordinates are compared on a normalized 0-999 scale, matching
  the Kimi/0xSero convention: Accuracy@threshold plus mean L2 click error.
"""

from __future__ import annotations

import math
import re
import string
from typing import Any, Iterable, Sequence

Point = tuple[float, float]

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_WHITESPACE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# Coordinate patterns, tried in order. First match wins.
_POINT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        # Tagged points: <point>(512, 341)</point>, <|point|>(512, 341),
        # <point_start>(512, 341)<point_end>
        r"<\|?point[^>]*\|?>\s*\(\s*(?P<x>-?\d+(?:\.\d+)?)\s*,\s*(?P<y>-?\d+(?:\.\d+)?)\s*\)",
        # JSON-ish: {"x": 512, "y": 341}
        r"[\"']x[\"']\s*:\s*(?P<x>-?\d+(?:\.\d+)?)\s*,\s*[\"']y[\"']\s*:\s*(?P<y>-?\d+(?:\.\d+)?)",
        # Plain parentheses: (512, 341)
        r"\(\s*(?P<x>-?\d+(?:\.\d+)?)\s*,\s*(?P<y>-?\d+(?:\.\d+)?)\s*\)",
        # Plain brackets: [512, 341]
        r"\[\s*(?P<x>-?\d+(?:\.\d+)?)\s*,\s*(?P<y>-?\d+(?:\.\d+)?)\s*\]",
    )
)


def normalize_answer(text: str) -> str:
    """VQA-style normalization: lowercase, strip punctuation/articles/extra spaces."""

    normalized = text.lower().translate(_PUNCT_TABLE)
    normalized = _ARTICLES.sub(" ", normalized)
    return _WHITESPACE.sub(" ", normalized).strip()


def exact_match(prediction: str, references: Sequence[str]) -> float:
    """1.0 when the normalized prediction equals any normalized reference."""

    normalized_prediction = normalize_answer(prediction)
    return float(
        any(normalize_answer(reference) == normalized_prediction for reference in references)
    )


def soft_vqa_accuracy(prediction: str, references: Sequence[str]) -> float:
    """Official VQA soft accuracy: min(1, #humans agreeing with prediction / 3)."""

    normalized_prediction = normalize_answer(prediction)
    if not normalized_prediction:
        return 0.0
    agreement = sum(
        1 for reference in references if normalize_answer(reference) == normalized_prediction
    )
    return min(1.0, agreement / 3.0)


def levenshtein(a: str, b: str) -> int:
    """Standard edit distance, O(len(a) * len(b)) with one rolling row."""

    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for index_a, char_a in enumerate(a, start=1):
        current = [index_a]
        for index_b, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[index_b] + 1,
                    current[index_b - 1] + 1,
                    previous[index_b - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def anls(prediction: str, references: Sequence[str], threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity (DocVQA/InfographicVQA metric)."""

    def similarity(candidate: str, reference: str) -> float:
        if not candidate and not reference:
            return 1.0
        longest = max(len(candidate), len(reference))
        if longest == 0:
            return 1.0
        return 1.0 - levenshtein(candidate, reference) / longest

    best = max((similarity(prediction, reference) for reference in references), default=0.0)
    return best if best >= threshold else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-tokens F1 after VQA normalization (cheap caption proxy)."""

    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    common: dict[str, int] = {}
    for token in prediction_tokens:
        if token in reference_tokens:
            common[token] = common.get(token, 0) + 1
    overlap = sum(
        min(count, reference_tokens.count(token)) for token, count in common.items()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_point(text: str) -> Point | None:
    """Extract the first (x, y) coordinate pair from generated text, or None."""

    for pattern in _POINT_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return (float(match.group("x")), float(match.group("y")))
    return None


def _to_scale(point: Point, scale: float) -> Point:
    """Fractions in [0, 1] are scaled up; larger values are assumed on-scale."""

    x, y = point
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return (x * scale, y * scale)
    return point


def grounding_metrics(
    prediction_text: str,
    *,
    gt_point: Point | None = None,
    gt_box: Sequence[float] | None = None,
    scale: float = 999.0,
    threshold: float = 50.0,
) -> dict[str, Any]:
    """Score one grounding prediction.

    ``gt_point``/``gt_box`` must already live on the normalized ``scale``
    (``tools/fetch_eval_data.py`` performs that conversion from pixels).
    ``correct`` means L2 distance to ``gt_point`` within ``threshold``
    (Accuracy@50 on the 999 scale in the 0xSero convention), or containment
    inside ``gt_box`` (the ScreenSpot convention). ``error`` is the L2
    distance to the ground-truth point, or to the box center for boxes.
    """

    result: dict[str, Any] = {
        "parse_ok": False,
        "correct": False,
        "error": None,
        "prediction_point": None,
    }
    if (gt_point is None) == (gt_box is None):
        raise ValueError("Provide exactly one of gt_point or gt_box")

    raw_point = extract_point(prediction_text)
    if raw_point is None:
        return result
    point = _to_scale(raw_point, scale)
    result["parse_ok"] = True
    result["prediction_point"] = point

    if gt_point is not None:
        target = gt_point
        error = math.dist(point, target)
        result["error"] = error
        result["correct"] = bool(error <= threshold)
    else:
        x1, y1, x2, y2 = (float(v) for v in gt_box)  # type: ignore[arg-type]
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        result["error"] = math.dist(point, center)
        result["correct"] = bool(x1 <= point[0] <= x2 and y1 <= point[1] <= y2)
    return result


def score_record(prediction: str, record: dict[str, Any]) -> dict[str, Any]:
    """Dispatch scoring for one benchmark record; see tools/eval_vlm.py for schema."""

    metric = record["metric"]
    if metric == "exact_match":
        return {"exact_match": exact_match(prediction, record["answers"])}
    if metric == "soft_vqa":
        return {"soft_vqa": soft_vqa_accuracy(prediction, record["answers"])}
    if metric == "anls":
        return {"anls": anls(prediction, record["answers"])}
    if metric == "token_f1":
        return {
            "token_f1": max(
                (token_f1(prediction, reference) for reference in record["answers"]),
                default=0.0,
            )
        }
    if metric == "grounding":
        scores = grounding_metrics(
            prediction,
            gt_point=(
                tuple(record["gt_point"]) if record.get("gt_point") is not None else None
            ),
            gt_box=record.get("gt_box"),
            scale=float(record.get("scale", 999.0)),
            threshold=float(record.get("threshold", 50.0)),
        )
        return {"grounding": scores}
    raise ValueError(f"Unknown metric: {metric!r}")


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-record scores into benchmark-level numbers."""

    records = list(records)
    summary: dict[str, Any] = {"count": len(records)}

    def mean(key: str, values: list[float]) -> None:
        if values:
            summary[key] = sum(values) / len(values)

    mean("exact_match", [r["exact_match"] for r in records if "exact_match" in r])
    mean("soft_vqa", [r["soft_vqa"] for r in records if "soft_vqa" in r])
    mean("anls", [r["anls"] for r in records if "anls" in r])
    mean("token_f1", [r["token_f1"] for r in records if "token_f1" in r])

    grounding = [r["grounding"] for r in records if "grounding" in r]
    if grounding:
        summary["grounding_count"] = len(grounding)
        summary["parse_rate"] = sum(g["parse_ok"] for g in grounding) / len(grounding)
        summary["accuracy"] = sum(g["correct"] for g in grounding) / len(grounding)
        errors = [g["error"] for g in grounding if g["error"] is not None]
        mean("mean_error", errors)
    return summary
