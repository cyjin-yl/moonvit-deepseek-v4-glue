#!/usr/bin/env python3
"""分析 step-50/100 projector 线性插值的多任务 Pareto 行为。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
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


def select_interpolation_candidate(
    states: dict[str, dict],
    *,
    retention_tolerance: float = 0.05,
    macro_tolerance: float = 0.02,
) -> dict:
    by_alpha = {float(row["alpha"]): (state, row) for state, row in states.items()}
    if 0.0 not in by_alpha or 1.0 not in by_alpha:
        raise ValueError("interpolation decision requires alpha 0 and 1 endpoints")
    early_state, early = by_alpha[0.0]
    late_state, late = by_alpha[1.0]

    def enrich(state: str, row: dict) -> dict:
        preference = {task: float(value) for task, value in row["preference"].items()}
        macro = statistics.fmean(preference.values())
        worst = min(preference.values())
        target = {
            task: max(float(early["preference"][task]), float(late["preference"][task]))
            for task in preference
        }
        regret = sum(max(0.0, target[task] - preference[task]) for task in preference)
        retention = (
            preference["count"] >= float(early["preference"]["count"]) - retention_tolerance
            and preference["shape"]
            >= float(early["preference"]["shape"]) - retention_tolerance
        )
        acquisition = (
            preference["coordinate"] > float(early["preference"]["coordinate"])
            and preference["spatial"] > float(early["preference"]["spatial"])
        )
        return {
            "state": state,
            "alpha": float(row["alpha"]),
            "preference_macro": macro,
            "preference_worst_task": worst,
            "generation_macro": float(row["generation_macro"]),
            "endpoint_regret": regret,
            "retains_count_shape": retention,
            "gains_coordinate_spatial": acquisition,
        }

    enriched = {
        state: enrich(state, row)
        for state, row in states.items()
    }
    endpoint_worst = max(
        enriched[early_state]["preference_worst_task"],
        enriched[late_state]["preference_worst_task"],
    )
    endpoint_macro = max(
        enriched[early_state]["preference_macro"],
        enriched[late_state]["preference_macro"],
    )
    candidates = []
    for state, row in enriched.items():
        if row["alpha"] in (0.0, 1.0):
            continue
        row = {
            **row,
            "improves_endpoint_worst_task": row["preference_worst_task"] > endpoint_worst,
            "meets_macro_floor": row["preference_macro"] >= endpoint_macro - macro_tolerance,
        }
        row["targeted_merge_pass"] = bool(
            row["retains_count_shape"]
            and row["gains_coordinate_spatial"]
            and row["improves_endpoint_worst_task"]
            and row["meets_macro_floor"]
        )
        candidates.append(row)
    candidates.sort(
        key=lambda row: (
            bool(row["targeted_merge_pass"]),
            float(row["preference_worst_task"]),
            -float(row["endpoint_regret"]),
            float(row["preference_macro"]),
            float(row["generation_macro"]),
        ),
        reverse=True,
    )
    selected = next((row for row in candidates if row["targeted_merge_pass"]), None)
    return {
        "selected_state": selected["state"] if selected else None,
        "targeted_merge_pass": selected is not None,
        "best_balanced_diagnostic_state": candidates[0]["state"] if candidates else None,
        "early_state": early_state,
        "late_state": late_state,
        "retention_tolerance": retention_tolerance,
        "macro_tolerance": macro_tolerance,
        "candidates": candidates,
        "decision_rule": (
            "a non-endpoint must retain count/shape within 0.05 of alpha 0, "
            "strictly gain coordinate/spatial over alpha 0, improve the better "
            "endpoint worst-task preference, and remain within 0.02 of the better "
            "endpoint macro preference"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite interpolation analysis: {args.out}")
    args.out.mkdir(parents=True)

    preference = read_jsonl(args.evaluation / "preference_records.jsonl")
    generation = read_jsonl(args.evaluation / "generation_records.jsonl")
    tasks = sorted({str(row["task"]) for row in preference})
    if tasks != sorted({str(row["task"]) for row in generation}):
        raise ValueError("interpolation preference/generation task sets differ")
    states = list(dict.fromkeys(str(row["state"]) for row in preference))
    if states != list(dict.fromkeys(str(row["state"]) for row in generation)):
        raise ValueError("interpolation preference/generation state order differs")
    frozen = next(state for state in states if state.startswith("frozen"))
    interpolation_states = [state for state in states if state.startswith("projector-interp")]
    if len(interpolation_states) < 3:
        raise ValueError("interpolation analysis requires endpoints and a middle state")

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
    grouped = {}
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
                                "interpolation_alpha": cell[0].get("interpolation_alpha"),
                                "condition": condition,
                                "task": task,
                                "metric": metric,
                                "pairs": len(pairs),
                                "mean": statistics.fmean(float(row["score"]) for row in pairs),
                            }
                        )

    contrasts = []
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
        contrasts.append(
            {
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
        )

    alpha_by_state = {
        state: float(
            next(
                row["interpolation_alpha"]
                for row in preference
                if str(row["state"]) == state
            )
        )
        for state in interpolation_states
    }
    early = next(state for state, alpha in alpha_by_state.items() if alpha == 0.0)
    late = next(state for state, alpha in alpha_by_state.items() if alpha == 1.0)
    for modality, (rows, metrics) in modalities.items():
        conditions = sorted({str(row["condition"]) for row in rows})
        for state in interpolation_states:
            for task in ["overall", *tasks]:
                for metric in metrics:
                    add_contrast(
                        "interpolation_minus_frozen",
                        modality,
                        metric,
                        state,
                        "vision",
                        frozen,
                        "vision",
                        task,
                    )
                    if state != early:
                        add_contrast(
                            "interpolation_minus_alpha0",
                            modality,
                            metric,
                            state,
                            "vision",
                            early,
                            "vision",
                            task,
                        )
                    if state != late:
                        add_contrast(
                            "interpolation_minus_alpha1",
                            modality,
                            metric,
                            state,
                            "vision",
                            late,
                            "vision",
                            task,
                        )
        if modality == "preference" and "shuffled_image" in conditions:
            for state in interpolation_states:
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

    state_metrics = {}
    for state in interpolation_states:
        state_metrics[state] = {
            "alpha": alpha_by_state[state],
            "preference": {
                task: next(
                    float(row["mean"])
                    for row in metric_rows
                    if row["modality"] == "preference"
                    and row["state"] == state
                    and row["condition"] == "vision"
                    and row["task"] == task
                    and row["metric"] == "paired_preference"
                )
                for task in tasks
            },
            "generation_macro": next(
                float(row["mean"])
                for row in metric_rows
                if row["modality"] == "generation"
                and row["state"] == state
                and row["condition"] == "vision"
                and row["task"] == "overall"
                and row["metric"] == "generation_paired"
            ),
        }
    decision = select_interpolation_candidate(state_metrics)
    decision.update(
        {
            "status": "valid",
            "state_metrics": state_metrics,
            "final_half_scored": False,
        }
    )

    metrics_path = args.out / "interpolation_metrics.csv"
    contrasts_path = args.out / "interpolation_contrasts.csv"
    decisions_path = args.out / "DECISIONS.json"
    write_csv(metrics_path, metric_rows)
    write_csv(contrasts_path, contrasts)
    decisions_path.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "valid",
        "format_version": "projector-interpolation-analysis-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "states": states,
        "tasks": tasks,
        "metric_rows": len(metric_rows),
        "contrasts": len(contrasts),
        "bootstrap_samples": args.bootstrap_samples,
        "evaluation_summary_sha256": sha256(args.evaluation / "SUMMARY.json"),
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
