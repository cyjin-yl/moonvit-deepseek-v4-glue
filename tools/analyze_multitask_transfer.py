#!/usr/bin/env python3
"""联合分析多任务 teacher-forced preference 与自由生成迁移。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from moonvit_glue.trajectory_analysis import paired_gap_stats


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def pair_metric_rows(rows: list[dict], metric: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    output = []
    for pair_id, pair in sorted(grouped.items()):
        if len(pair) != 2 or any(row.get("failure") is not None for row in pair):
            raise ValueError(f"multitask analysis needs one valid complete pair: {pair_id}")
        if metric == "paired_preference":
            score = float(all(float(row["correct_margin"]) > 0 for row in pair))
        elif metric == "sample_preference":
            score = statistics.fmean(
                float(float(row["correct_margin"]) > 0) for row in pair
            )
        elif metric == "mean_margin":
            score = statistics.fmean(float(row["correct_margin"]) for row in pair)
        elif metric == "generation_paired":
            predictions = {str(row.get("normalized_prediction") or "") for row in pair}
            score = float(all(bool(row["correct"]) for row in pair) and len(predictions) == 2)
        elif metric == "generation_sample":
            score = statistics.fmean(float(bool(row["correct"])) for row in pair)
        elif metric == "prediction_flip":
            score = float(
                len({str(row.get("normalized_prediction") or "") for row in pair}) == 2
            )
        else:
            raise ValueError(f"unknown multitask pair metric: {metric}")
        output.append({"id": pair_id, "score": score})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preference-run", required=True, type=Path)
    parser.add_argument("--generation-run", required=True, type=Path)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite multitask analysis: {args.out}")
    args.out.mkdir(parents=True)

    preference_config = json.loads(
        (args.preference_run / "CONFIG.json").read_text(encoding="utf-8")
    )
    generation_config = json.loads(
        (args.generation_run / "CONFIG.json").read_text(encoding="utf-8")
    )
    preference = read_jsonl(args.preference_run / "preference_records.jsonl")
    generation = [
        row
        for row in read_jsonl(args.generation_run / "records.jsonl")
        if row["dataset"] == "synthetic"
    ]
    checkpoints = [str(row["id"]) for row in preference_config["checkpoints"]]
    generation_checkpoints = [
        str(row["id"]) for row in generation_config["checkpoints"]
    ]
    if checkpoints != generation_checkpoints:
        raise ValueError("preference/generation checkpoint order mismatch")
    if args.baseline_checkpoint not in checkpoints:
        raise ValueError("baseline checkpoint is absent")
    tasks = sorted({str(row["task"]) for row in preference})
    if tasks != sorted({str(row["task"]) for row in generation}):
        raise ValueError("preference/generation task set mismatch")

    modalities = {
        "preference": (
            preference,
            ("paired_preference", "sample_preference", "mean_margin"),
        ),
        "generation": (
            generation,
            ("generation_paired", "generation_sample", "prediction_flip"),
        ),
    }
    grouped: dict[tuple[str, str, str, str, str], list[dict]] = {}
    metric_rows = []
    for modality, (rows, metrics) in modalities.items():
        conditions = sorted({str(row["condition"]) for row in rows})
        for checkpoint in checkpoints:
            for condition in conditions:
                for task in ["overall", *tasks]:
                    cell = [
                        row
                        for row in rows
                        if str(row["checkpoint"]) == checkpoint
                        and str(row["condition"]) == condition
                        and (task == "overall" or str(row["task"]) == task)
                    ]
                    for metric in metrics:
                        pairs = pair_metric_rows(cell, metric)
                        grouped[(modality, checkpoint, condition, task, metric)] = pairs
                        metric_rows.append(
                            {
                                "modality": modality,
                                "checkpoint": checkpoint,
                                "condition": condition,
                                "task": task,
                                "metric": metric,
                                "pairs": len(pairs),
                                "mean": statistics.fmean(row["score"] for row in pairs),
                            }
                        )

    contrasts = []
    contrast_index: dict[tuple[str, str, str, str, str], dict] = {}
    counter = 0

    def add_contrast(
        family: str,
        modality: str,
        metric: str,
        checkpoint_a: str,
        condition_a: str,
        checkpoint_b: str,
        condition_b: str,
        task: str,
    ) -> None:
        nonlocal counter
        counter += 1
        stats = paired_gap_stats(
            grouped[(modality, checkpoint_a, condition_a, task, metric)],
            grouped[(modality, checkpoint_b, condition_b, task, metric)],
            bootstrap_samples=2000,
            seed=args.seed + counter,
        )
        row = {
            "family": family,
            "modality": modality,
            "metric": metric,
            "checkpoint_a": checkpoint_a,
            "condition_a": condition_a,
            "checkpoint_b": checkpoint_b,
            "condition_b": condition_b,
            "task": task,
            **stats,
        }
        contrasts.append(row)
        contrast_index[(family, modality, metric, checkpoint_a, task)] = row

    for modality, (_, metrics) in modalities.items():
        conditions = sorted(
            {str(row["condition"]) for row in modalities[modality][0]}
        )
        for checkpoint in checkpoints:
            for task in ["overall", *tasks]:
                for metric in metrics:
                    for condition in conditions:
                        if condition != "vision":
                            add_contrast(
                                "vision_minus_condition",
                                modality,
                                metric,
                                checkpoint,
                                "vision",
                                checkpoint,
                                condition,
                                task,
                            )
                    if checkpoint != args.baseline_checkpoint:
                        add_contrast(
                            "checkpoint_minus_baseline",
                            modality,
                            metric,
                            checkpoint,
                            "vision",
                            args.baseline_checkpoint,
                            "vision",
                            task,
                        )

    latest = checkpoints[-1]
    validated_tasks = []
    negative_transfer_tasks = []
    generation_improved_tasks = []
    task_decisions = {}
    for task in tasks:
        improvement = contrast_index[
            ("checkpoint_minus_baseline", "preference", "paired_preference", latest, task)
        ]
        causal = next(
            row
            for row in contrasts
            if row["family"] == "vision_minus_condition"
            and row["modality"] == "preference"
            and row["metric"] == "paired_preference"
            and row["checkpoint_a"] == latest
            and row["condition_b"] == "shuffled_image"
            and row["task"] == task
        )
        generation_improvement = contrast_index[
            ("checkpoint_minus_baseline", "generation", "generation_paired", latest, task)
        ]
        validated = improvement["ci95_low"] > 0 and causal["ci95_low"] > 0
        negative = improvement["ci95_high"] < 0
        generation_improved = generation_improvement["ci95_low"] > 0
        if validated:
            validated_tasks.append(task)
        if negative:
            negative_transfer_tasks.append(task)
        if generation_improved:
            generation_improved_tasks.append(task)
        task_decisions[task] = {
            "paired_preference_improvement": improvement,
            "vision_minus_shuffled_paired_preference": causal,
            "paired_generation_improvement": generation_improvement,
            "validated_preference_transfer": validated,
            "negative_preference_transfer": negative,
            "validated_generation_transfer": generation_improved,
        }
    non_shape_validated = [task for task in validated_tasks if task != "shape"]
    decisions = {
        "status": "valid",
        "baseline_checkpoint": args.baseline_checkpoint,
        "latest_checkpoint": latest,
        "validated_preference_transfer_tasks": validated_tasks,
        "validated_generation_transfer_tasks": generation_improved_tasks,
        "negative_preference_transfer_tasks": negative_transfer_tasks,
        "broad_non_shape_transfer_supported": len(non_shape_validated) >= 3,
        "shape_specific_supported": (
            "shape" in validated_tasks and len(non_shape_validated) < 2
        ),
        "decision_rule": "broad transfer requires positive paired-bootstrap lower bounds for checkpoint-minus-baseline and vision-minus-shuffle on at least three of five non-shape tasks",
        "tasks": task_decisions,
        "interpretation_limits": [
            "all bootstrap resampling units are complete counterfactual pairs",
            "shape-projector-step50 was trained only on shape records",
            "synthetic selection is disjoint from all adaptation train IDs and pairs",
            "final odd halves remain unscored",
        ],
    }

    metric_path = args.out / "transfer_metrics.csv"
    contrast_path = args.out / "transfer_contrasts.csv"
    with metric_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    contrast_fields = list(contrasts[0])
    with contrast_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=contrast_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(contrasts)
    decision_path = args.out / "DECISIONS.json"
    decision_path.write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "valid",
        "format_version": "multitask-transfer-analysis-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preference_summary_sha256": sha256(args.preference_run / "SUMMARY.json"),
        "generation_summary_sha256": sha256(args.generation_run / "SUMMARY.json"),
        "preference_rows": len(preference),
        "generation_rows": len(generation),
        "metric_rows": len(metric_rows),
        "contrast_rows": len(contrasts),
        "bootstrap_samples": 2000,
        "seed": args.seed,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (metric_path, contrast_path, decision_path)
        },
        "final_half_scored": False,
    }
    if any(
        not math.isfinite(float(row["mean_gap"]))
        for row in contrasts
    ):
        raise ValueError("multitask contrasts contain non-finite values")
    (args.out / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
