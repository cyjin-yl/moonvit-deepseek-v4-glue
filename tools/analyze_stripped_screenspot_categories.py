#!/usr/bin/env python3
"""汇总 stripped ScreenSpot 公共集结果，并补齐类别与逐样本配对置信区间。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from moonvit_glue.grounding_contract import summarize_click_scores


CONDITIONS = ("vision", "blind", "shuffled", "random_projector")
COMPARISONS = {
    "vision_minus_blind": ("vision", "blind"),
    "vision_minus_shuffled": ("vision", "shuffled"),
    "trained_minus_random_projector": ("vision", "random_projector"),
}
METRICS: dict[str, tuple[Callable[[dict[str, Any]], float], str, bool]] = {
    "parse_rate": (lambda row: float(bool(row["parse_ok"])), "mean", False),
    "accuracy_at_50": (lambda row: float(bool(row["accuracy_at_50"])), "mean", False),
    "accuracy_at_100": (lambda row: float(bool(row["accuracy_at_100"])), "mean", False),
    "accuracy_at_200": (lambda row: float(bool(row["accuracy_at_200"])), "mean", False),
    "click_in_box_accuracy": (lambda row: float(bool(row["click_in_box"])), "mean", False),
    "mean_center_distance": (lambda row: float(row["center_l2_penalized"]), "mean", True),
    "median_center_distance": (lambda row: float(row["center_l2_penalized"]), "median", True),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at line {line_number}") from exc
    return rows


def _stable_seed(base_seed: int, *parts: str) -> int:
    payload = "\0".join((str(base_seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def validate_and_index(
    manifest: dict[str, Any], rows: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, Any]]]:
    samples = {str(row["sample_id"]): row for row in manifest["samples"]}
    expected_ids = [str(row["sample_id"]) for row in manifest["samples"]]
    expected_shuffle = {
        str(row["sample_id"]): str(row["shuffled_image_sample_id"])
        for row in manifest["shuffled_image_control"]["mapping"]
    }
    indexed = {condition: {} for condition in CONDITIONS}
    for row in rows:
        condition = str(row.get("condition"))
        sample_id = str(row.get("sample_id"))
        if condition not in indexed:
            raise ValueError(f"unexpected condition: {condition}")
        if sample_id not in samples:
            raise ValueError(f"unexpected sample ID: {sample_id}")
        if sample_id in indexed[condition]:
            raise ValueError(f"duplicate row for {condition}/{sample_id}")
        if str(row.get("shuffled_sample_id")) != expected_shuffle[sample_id]:
            raise ValueError(f"shuffled provenance differs for {sample_id}")
        indexed[condition][sample_id] = row
    for condition in CONDITIONS:
        if list(indexed[condition]) != expected_ids:
            missing = [sample_id for sample_id in expected_ids if sample_id not in indexed[condition]]
            extra = [sample_id for sample_id in indexed[condition] if sample_id not in samples]
            raise ValueError(
                f"{condition} sample order/coverage differs; missing={missing[:3]}, extra={extra[:3]}"
            )
    return indexed, samples


def category_ids(manifest: dict[str, Any]) -> dict[str, list[str]]:
    samples = manifest["samples"]
    result = {"overall": [str(row["sample_id"]) for row in samples]}
    for target_type in ("text", "icon/widget"):
        result[target_type] = [
            str(row["sample_id"]) for row in samples if row["target_type"] == target_type
        ]
    for platform in ("Android", "iOS", "Windows", "macOS", "Web"):
        result[platform] = [
            str(row["sample_id"]) for row in samples if row["platform"] == platform
        ]
    if any(not ids for ids in result.values()):
        raise ValueError("one or more frozen ScreenSpot categories are empty")
    return result


def paired_bootstrap(
    current_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    metric_name: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    extractor, reducer, lower_is_better = METRICS[metric_name]
    current = np.asarray([extractor(row) for row in current_rows], dtype=np.float64)
    baseline = np.asarray([extractor(row) for row in baseline_rows], dtype=np.float64)
    if current.shape != baseline.shape or current.size == 0:
        raise ValueError("paired bootstrap rows must be non-empty and matched")

    reduce: Callable[[np.ndarray, int | None], np.ndarray | float]
    reduce = np.mean if reducer == "mean" else np.median
    raw_delta = float(reduce(current, None) - reduce(baseline, None))
    improvement = -raw_delta if lower_is_better else raw_delta
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, current.size, size=(bootstrap_samples, current.size))
    sampled_current = reduce(current[indices], 1)
    sampled_baseline = reduce(baseline[indices], 1)
    sampled_raw = sampled_current - sampled_baseline
    sampled_improvement = -sampled_raw if lower_is_better else sampled_raw
    lower, upper = np.quantile(sampled_improvement, [0.025, 0.975]).tolist()
    return {
        "sample_count": int(current.size),
        "current_minus_baseline": raw_delta,
        "improvement": improvement,
        "improvement_ci95_lower": float(lower),
        "improvement_ci95_upper": float(upper),
        "higher_improvement_is_better": True,
        "lower_raw_metric_is_better": lower_is_better,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
    }


def build_summary(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    indexed, _ = validate_and_index(manifest, rows)
    groups = category_ids(manifest)
    categories: dict[str, Any] = {}
    for category, ids in groups.items():
        condition_rows = {
            condition: [indexed[condition][sample_id] for sample_id in ids]
            for condition in CONDITIONS
        }
        paired: dict[str, Any] = {}
        for comparison, (current_name, baseline_name) in COMPARISONS.items():
            paired[comparison] = {}
            for metric_name in METRICS:
                metric_seed = _stable_seed(
                    bootstrap_seed, category, comparison, metric_name
                )
                paired[comparison][metric_name] = paired_bootstrap(
                    condition_rows[current_name],
                    condition_rows[baseline_name],
                    metric_name=metric_name,
                    bootstrap_samples=bootstrap_samples,
                    seed=metric_seed,
                )
        categories[category] = {
            "sample_count": len(ids),
            "conditions": {
                condition: summarize_click_scores(condition_rows[condition])
                for condition in CONDITIONS
            },
            "paired": paired,
        }
    return {
        "schema_version": "stripped-screenspot-public-category-analysis-v1",
        "status": "diagnostic_only",
        "dataset": manifest["name"],
        "sample_count": len(manifest["samples"]),
        "category_order": list(groups),
        "condition_order": list(CONDITIONS),
        "comparison_order": list(COMPARISONS),
        "metric_order": list(METRICS),
        "delta_semantics": {
            "current_minus_baseline": "raw metric difference",
            "improvement": "current-minus-baseline for rates; baseline-minus-current for distances",
            "promotion_rule": "positive lower CI is required for the named causal comparison",
        },
        "bootstrap": {"samples": bootstrap_samples, "base_seed": bootstrap_seed},
        "categories": categories,
        "capability_claim_allowed": False,
        "deepseek_transfer_label": "transferable_with_runtime_validation",
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260805)
    args = parser.parse_args()
    if args.bootstrap_samples < 2_000:
        raise ValueError("formal analysis requires at least 2,000 bootstrap samples")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = read_jsonl(args.rows)
    summary = build_summary(
        manifest,
        rows,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary["inputs"] = {
        "manifest": {
            "path": str(args.manifest),
            "bytes": args.manifest.stat().st_size,
            "sha256": sha256_file(args.manifest),
        },
        "rows": {
            "path": str(args.rows),
            "bytes": args.rows.stat().st_size,
            "sha256": sha256_file(args.rows),
        },
    }
    summary_path = args.out_dir / "CATEGORY_SUMMARY.json"
    write_json(summary_path, summary)
    write_json(
        args.out_dir / "ARTIFACT_MANIFEST.json",
        {
            "schema_version": "compact-artifact-manifest-v1",
            "files": [
                {
                    "path": summary_path.name,
                    "bytes": summary_path.stat().st_size,
                    "sha256": sha256_file(summary_path),
                }
            ],
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
