#!/usr/bin/env python3
"""分析同序列 projector 续训与顶部 LoRA 的成对差异。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from analyze_multitask_transfer import pair_metric_rows
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


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def endpoint_direction(decisions: dict[str, dict]) -> str:
    lora_wins = sum(
        row["generation_lora_minus_projector_ci95_low"] > 0
        for row in decisions.values()
    )
    projector_wins = sum(
        row["generation_lora_minus_projector_ci95_high"] < 0
        for row in decisions.values()
    )
    if lora_wins >= 2:
        return "language_upper_stack_supported"
    if projector_wins >= 2:
        return "additional_projector_training_supported"
    return "mixed_or_underpowered_expand_trajectory"


def latest_adaptation_state(states: list[str], prefix: str) -> str:
    candidates = [state for state in states if state.startswith(prefix)]
    if not candidates:
        raise ValueError(f"adaptation state prefix is absent: {prefix}")
    return max(candidates, key=lambda state: int(state.removeprefix(prefix)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite adaptation analysis: {args.out}")
    args.out.mkdir(parents=True)

    preference = read_jsonl(args.eval_run / "preference_records.jsonl")
    generation = read_jsonl(args.eval_run / "generation_records.jsonl")
    tasks = sorted({str(row["task"]) for row in preference})
    if tasks != sorted({str(row["task"]) for row in generation}):
        raise ValueError("adaptation preference/generation task sets differ")
    states = list(dict.fromkeys(str(row["state"]) for row in preference))
    if states != list(dict.fromkeys(str(row["state"]) for row in generation)):
        raise ValueError("adaptation preference/generation state order differs")
    frozen = next(state for state in states if state.startswith("frozen"))
    lora = latest_adaptation_state(states, "lora-step")
    projector = latest_adaptation_state(states, "projector-step")

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
        for state in states:
            for condition in conditions:
                for task in ["overall", *tasks]:
                    cell = [
                        row
                        for row in rows
                        if str(row["state"]) == state
                        and str(row["condition"]) == condition
                        and (task == "overall" or str(row["task"]) == task)
                    ]
                    for metric in metrics:
                        pairs = pair_metric_rows(cell, metric)
                        grouped[(modality, state, condition, task, metric)] = pairs
                        metric_rows.append(
                            {
                                "modality": modality,
                                "state": state,
                                "condition": condition,
                                "task": task,
                                "metric": metric,
                                "pairs": len(pairs),
                                "mean": statistics.fmean(
                                    float(row["score"]) for row in pairs
                                ),
                            }
                        )

    contrasts = []
    contrast_index = {}
    counter = 0

    def add_contrast(
        family: str,
        modality: str,
        metric: str,
        state_a: str,
        condition_a: str,
        state_b: str,
        condition_b: str,
        task: str,
    ) -> None:
        nonlocal counter
        counter += 1
        stats = paired_gap_stats(
            grouped[(modality, state_a, condition_a, task, metric)],
            grouped[(modality, state_b, condition_b, task, metric)],
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + counter,
        )
        row = {
            "family": family,
            "modality": modality,
            "metric": metric,
            "state_a": state_a,
            "condition_a": condition_a,
            "state_b": state_b,
            "condition_b": condition_b,
            "task": task,
            **stats,
        }
        contrasts.append(row)
        contrast_index[(family, modality, metric, state_a, task)] = row

    for modality, (rows, metrics) in modalities.items():
        conditions = sorted({str(row["condition"]) for row in rows})
        for state in (lora, projector):
            for task in ["overall", *tasks]:
                for metric in metrics:
                    add_contrast(
                        "endpoint_minus_frozen",
                        modality,
                        metric,
                        state,
                        "vision",
                        frozen,
                        "vision",
                        task,
                    )
        for task in ["overall", *tasks]:
            for metric in metrics:
                add_contrast(
                    "lora_minus_projector",
                    modality,
                    metric,
                    lora,
                    "vision",
                    projector,
                    "vision",
                    task,
                )
        if "shuffled_image" in conditions:
            for state in states:
                for task in ["overall", *tasks]:
                    for metric in metrics:
                        add_contrast(
                            "vision_minus_shuffle",
                            modality,
                            metric,
                            state,
                            "vision",
                            state,
                            "shuffled_image",
                            task,
                        )

    task_decisions = {}
    for task in tasks:
        preference_delta = contrast_index[
            ("lora_minus_projector", "preference", "paired_preference", lora, task)
        ]
        generation_delta = contrast_index[
            ("lora_minus_projector", "generation", "generation_paired", lora, task)
        ]
        task_decisions[task] = {
            "preference_lora_minus_projector": preference_delta["mean_gap"],
            "preference_lora_minus_projector_ci95_low": preference_delta["ci95_low"],
            "preference_lora_minus_projector_ci95_high": preference_delta["ci95_high"],
            "generation_lora_minus_projector": generation_delta["mean_gap"],
            "generation_lora_minus_projector_ci95_low": generation_delta["ci95_low"],
            "generation_lora_minus_projector_ci95_high": generation_delta["ci95_high"],
        }
    decisions = {
        "status": "valid",
        "frozen_state": frozen,
        "lora_state": lora,
        "projector_state": projector,
        "tasks": task_decisions,
        "selected_direction": endpoint_direction(task_decisions),
        "decision_rule": (
            "at least two task-level paired-generation confidence intervals must "
            "exclude zero in the same arm direction"
        ),
    }

    metrics_path = args.out / "adaptation_metrics.csv"
    contrasts_path = args.out / "adaptation_contrasts.csv"
    decisions_path = args.out / "DECISIONS.json"
    write_csv(metrics_path, metric_rows)
    write_csv(contrasts_path, contrasts)
    decisions_path.write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "valid",
        "format_version": "balanced-adaptation-comparison-analysis-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "states": states,
        "tasks": tasks,
        "metric_rows": len(metric_rows),
        "contrasts": len(contrasts),
        "bootstrap_samples": args.bootstrap_samples,
        "eval_summary_sha256": sha256(args.eval_run / "SUMMARY.json"),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (metrics_path, contrasts_path, decisions_path)
        },
        "final_half_scored": False,
    }
    (args.out / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
