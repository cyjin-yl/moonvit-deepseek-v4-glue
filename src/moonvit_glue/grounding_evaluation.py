"""把固定 ScreenSpot manifest 与七种条件的生成结果组成正式评分包。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .grounding_contract import (
    paired_bootstrap,
    score_click_prediction,
    summarize_click_scores,
)
from .screenspot_contract import verify_manifest

REQUIRED_CONDITIONS = (
    "vision",
    "blind",
    "shuffled",
    "random_projector",
    "step0",
    "previous_best",
    "current_candidate",
)

PAIR_COMPARISONS = {
    "vision-minus-blind": ("vision", "blind"),
    "vision-minus-shuffled": ("vision", "shuffled"),
    "trained-minus-random-projector": ("current_candidate", "random_projector"),
    "current-candidate-minus-previous-best": ("current_candidate", "previous_best"),
}


def read_prediction_jsonl(
    path: str | Path, expected_sample_ids: Sequence[str]
) -> list[dict[str, str]]:
    """读取生成结果并强制要求与 manifest 完全相同的 ID 和顺序。"""

    source = Path(path)
    rows: list[dict[str, str]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            sample_id = raw.get("sample_id")
            prediction = raw.get("prediction")
            if not isinstance(sample_id, str) or not isinstance(prediction, str):
                raise ValueError(
                    f"{source}:{line_number} requires string sample_id and prediction"
                )
            rows.append({"sample_id": sample_id, "prediction": prediction})
    actual_ids = [row["sample_id"] for row in rows]
    if actual_ids != list(expected_sample_ids):
        raise ValueError(f"{source} predictions must follow exact manifest order")
    return rows


def _breakdowns(scored: Sequence[dict[str, Any]]) -> dict[str, Any]:
    target_types = ("text", "icon/widget")
    platforms = ("Android", "iOS", "Windows", "macOS", "Web")
    return {
        "overall": summarize_click_scores(scored),
        "target_type": {
            target_type: summarize_click_scores(
                row for row in scored if row["target_type"] == target_type
            )
            for target_type in target_types
        },
        "platform": {
            platform: summarize_click_scores(
                row for row in scored if row["platform"] == platform
            )
            for platform in platforms
        },
    }

def _validate_condition_rows(
    condition: str,
    predictions: Sequence[Mapping[str, Any]],
    expected_ids: Sequence[str],
) -> None:
    actual_ids = [row.get("sample_id") for row in predictions]
    if actual_ids != list(expected_ids):
        raise ValueError(
            f"condition {condition!r} must use identical sample IDs in exact manifest order"
        )
    for index, row in enumerate(predictions):
        if not isinstance(row.get("prediction"), str):
            raise ValueError(
                f"condition {condition!r} row {index} requires a string prediction"
            )


def evaluate_conditions(
    manifest: Mapping[str, Any],
    conditions: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 20260805,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """评分七种固定条件，并生成四个预注册 paired comparisons。"""

    if not verify_manifest(manifest):
        raise ValueError("ScreenSpot manifest SHA-256 verification failed")
    missing = [name for name in REQUIRED_CONDITIONS if name not in conditions]
    unknown = [name for name in conditions if name not in REQUIRED_CONDITIONS]
    if missing and not allow_partial:
        raise ValueError(f"missing required conditions: {missing}")
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")

    samples = list(manifest.get("samples", []))
    expected_ids = [str(sample["sample_id"]) for sample in samples]
    if [sample.get("evaluation_order") for sample in samples] != list(range(len(samples))):
        raise ValueError("manifest evaluation_order must be contiguous and zero-based")

    for name, predictions in conditions.items():
        _validate_condition_rows(name, predictions, expected_ids)
    if "vision" in conditions and "current_candidate" in conditions:
        vision_text = [row["prediction"] for row in conditions["vision"]]
        candidate_text = [row["prediction"] for row in conditions["current_candidate"]]
        if vision_text != candidate_text:
            raise ValueError(
                "current_candidate must alias the candidate's correct-image vision outputs"
            )

    condition_results: dict[str, Any] = {}
    scored_by_condition: dict[str, list[dict[str, Any]]] = {}
    for name in REQUIRED_CONDITIONS:
        if name not in conditions:
            continue
        predictions = conditions[name]
        scored: list[dict[str, Any]] = []
        for sample, prediction in zip(samples, predictions, strict=True):
            row = score_click_prediction(
                sample_id=sample["sample_id"],
                prediction=prediction["prediction"],
                target_box=sample["bbox_999_xyxy"],
            )
            row["platform"] = sample["platform"]
            row["target_type"] = sample["target_type"]
            row["evaluation_order"] = sample["evaluation_order"]
            scored.append(row)
        scored_by_condition[name] = scored
        condition_results[name] = {
            "breakdowns": _breakdowns(scored),
            "scores": scored,
        }

    comparisons: dict[str, Any] = {}
    for label, (first, second) in PAIR_COMPARISONS.items():
        if first not in scored_by_condition or second not in scored_by_condition:
            continue
        comparisons[label] = paired_bootstrap(
            scored_by_condition[first],
            scored_by_condition[second],
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )

    return {
        "schema_version": "community-grounding-evaluation-v1",
        "dataset_name": manifest.get("name"),
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "sample_count": len(samples),
        "formal_complete": not missing,
        "condition_order": [name for name in REQUIRED_CONDITIONS if name in conditions],
        "conditions": condition_results,
        "comparisons": comparisons,
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "scope": "overall; category point estimates are reported without category CIs",
        },
    }
