#!/usr/bin/env python3
"""分析固定预算 ordinary、fixed replay 与 triggered replay 轨迹。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized_trapezoid_auc(points: list[tuple[int, float]]) -> float:
    points = sorted(points)
    width = points[-1][0] - points[0][0]
    if width <= 0:
        raise ValueError("trajectory needs a positive step range")
    area = sum(
        (right_step - left_step) * (left_value + right_value) / 2
        for (left_step, left_value), (right_step, right_value) in zip(points, points[1:])
    )
    return area / width


def forgetting_auc(points: list[tuple[int, float]], *, reference: float) -> float:
    deficits = [(step, max(0.0, reference - value)) for step, value in points]
    return normalized_trapezoid_auc(deficits)


def recovered_within(*, endpoint: float, reference: float, tolerance: float) -> bool:
    if tolerance < 0:
        raise ValueError("recovery tolerance must be non-negative")
    return endpoint >= reference - tolerance


def policy_support(primary: dict, donors: dict, *, maximum_donor_cost: float) -> str:
    if float(primary["ci95_low"]) <= 0:
        return "target_effect_underpowered"
    if float(donors["mean_gap"]) < -maximum_donor_cost:
        return "donor_cost_exceeded"
    return "supported"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--sentinel", required=True, type=Path)
    parser.add_argument("--ordinary-run", required=True, type=Path)
    parser.add_argument("--fixed-run", required=True, type=Path)
    parser.add_argument("--triggered-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--recovery-tolerance", type=float, default=0.05)
    parser.add_argument("--maximum-donor-cost", type=float, default=0.05)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite matched replay analysis: {args.out}")

    preference_path = args.evaluation / "preference_records.jsonl"
    generation_path = args.evaluation / "generation_records.jsonl"
    preference = read_jsonl(preference_path)
    generation = read_jsonl(generation_path)
    states = list(dict.fromkeys(str(row["state"]) for row in preference))
    expected_states = [
        "frozen-base",
        "exchange-step50",
        "ordinary-step75",
        "ordinary-step100",
        "fixed-step75",
        "fixed-step100",
        "triggered-step100",
    ]
    generation_states = list(dict.fromkeys(str(row["state"]) for row in generation))
    if (
        len(states) != len(expected_states)
        or set(states) != set(expected_states)
        or len(generation_states) != len(expected_states)
        or set(generation_states) != set(expected_states)
    ):
        raise ValueError(f"matched replay state order drifted: {states}")
    tasks = sorted({str(row["task"]) for row in preference})
    if tasks != sorted({str(row["task"]) for row in generation}):
        raise ValueError("preference and generation task sets differ")

    modalities = {
        "preference": (preference, ("paired_preference", "sample_preference", "mean_margin")),
        "generation": (generation, ("generation_paired", "generation_sample", "prediction_flip")),
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
                                "condition": condition,
                                "task": task,
                                "metric": metric,
                                "pairs": len(pairs),
                                "mean": statistics.fmean(float(row["score"]) for row in pairs),
                            }
                        )

    contrasts = []
    counter = 0

    def rows_for(
        modality: str,
        state: str,
        condition: str,
        metric: str,
        selected_tasks: set[str] | None,
    ) -> list[dict]:
        rows = modalities[modality][0]
        cell = [
            row
            for row in rows
            if str(row["state"]) == state
            and str(row["condition"]) == condition
            and (selected_tasks is None or str(row["task"]) in selected_tasks)
        ]
        return pair_metric_rows(cell, metric)

    def add_contrast(
        family: str,
        modality: str,
        metric: str,
        state_a: str,
        state_b: str,
        task_label: str,
        selected_tasks: set[str] | None,
        condition_a: str = "vision",
        condition_b: str = "vision",
    ) -> dict:
        nonlocal counter
        counter += 1
        stats = paired_gap_stats(
            rows_for(modality, state_a, condition_a, metric, selected_tasks),
            rows_for(modality, state_b, condition_b, metric, selected_tasks),
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
            "task": task_label,
            **stats,
        }
        contrasts.append(row)
        return row

    families = [
        ("fixed_minus_ordinary_step75", "fixed-step75", "ordinary-step75"),
        ("fixed_minus_ordinary_step100", "fixed-step100", "ordinary-step100"),
        ("triggered_minus_ordinary_step100", "triggered-step100", "ordinary-step100"),
        ("fixed_minus_triggered_step100", "fixed-step100", "triggered-step100"),
    ]
    for family, state_a, state_b in families:
        for modality, (_, metrics) in modalities.items():
            for metric in metrics:
                for task in ["overall", *tasks]:
                    add_contrast(
                        family,
                        modality,
                        metric,
                        state_a,
                        state_b,
                        task,
                        None if task == "overall" else {task},
                    )
    for state in states:
        for task in ["overall", *tasks]:
            add_contrast(
                "vision_minus_shuffle",
                "preference",
                "paired_preference",
                state,
                state,
                task,
                None if task == "overall" else {task},
                condition_a="vision",
                condition_b="shuffled_image",
            )

    fixed_targets = {"count", "shape"}
    triggered_targets = {"count"}
    fixed_primary = add_contrast(
        "fixed_targets_minus_ordinary_endpoint",
        "preference",
        "paired_preference",
        "fixed-step100",
        "ordinary-step100",
        "count+shape",
        fixed_targets,
    )
    fixed_donors = add_contrast(
        "fixed_donors_minus_ordinary_endpoint",
        "preference",
        "paired_preference",
        "fixed-step100",
        "ordinary-step100",
        "donors",
        set(tasks) - fixed_targets,
    )
    triggered_primary = add_contrast(
        "triggered_target_minus_ordinary_endpoint",
        "preference",
        "paired_preference",
        "triggered-step100",
        "ordinary-step100",
        "count",
        triggered_targets,
    )
    triggered_donors = add_contrast(
        "triggered_donors_minus_ordinary_endpoint",
        "preference",
        "paired_preference",
        "triggered-step100",
        "ordinary-step100",
        "donors",
        set(tasks) - triggered_targets,
    )
    fixed_generation_targets = add_contrast(
        "fixed_targets_minus_ordinary_endpoint",
        "generation",
        "generation_paired",
        "fixed-step100",
        "ordinary-step100",
        "count+shape",
        fixed_targets,
    )
    triggered_generation_target = add_contrast(
        "triggered_target_minus_ordinary_endpoint",
        "generation",
        "generation_paired",
        "triggered-step100",
        "ordinary-step100",
        "count",
        triggered_targets,
    )

    def metric_mean(modality: str, state: str, condition: str, task: str, metric: str) -> float:
        return next(
            float(row["mean"])
            for row in metric_rows
            if row["modality"] == modality
            and row["state"] == state
            and row["condition"] == condition
            and row["task"] == task
            and row["metric"] == metric
        )

    state_metrics = {}
    for state in states:
        preference_tasks = {
            task: metric_mean("preference", state, "vision", task, "paired_preference")
            for task in tasks
        }
        generation_tasks = {
            task: metric_mean("generation", state, "vision", task, "generation_paired")
            for task in tasks
        }
        state_metrics[state] = {
            "preference": preference_tasks,
            "preference_macro": statistics.fmean(preference_tasks.values()),
            "preference_worst_task": min(preference_tasks.values()),
            "generation": generation_tasks,
            "generation_macro": statistics.fmean(generation_tasks.values()),
            "vision_shuffle_overall": metric_mean(
                "preference", state, "vision", "overall", "paired_preference"
            )
            - metric_mean(
                "preference", state, "shuffled_image", "overall", "paired_preference"
            ),
        }

    trajectory_rows = []
    for task in tasks:
        reference = state_metrics["exchange-step50"]["preference"][task]
        policy_states = {
            "ordinary": ("ordinary-step75", "ordinary-step100"),
            "fixed": ("fixed-step75", "fixed-step100"),
            "triggered": ("ordinary-step75", "triggered-step100"),
        }
        for policy, (middle, endpoint) in policy_states.items():
            points = [
                (50, reference),
                (75, state_metrics[middle]["preference"][task]),
                (100, state_metrics[endpoint]["preference"][task]),
            ]
            endpoint_value = points[-1][1]
            trajectory_rows.append(
                {
                    "task": task,
                    "policy": policy,
                    "step50": points[0][1],
                    "step75": points[1][1],
                    "step100": endpoint_value,
                    "ability_auc": normalized_trapezoid_auc(points),
                    "forgetting_auc": forgetting_auc(points, reference=reference),
                    "endpoint_minus_exchange": endpoint_value - reference,
                    "recovered_within_tolerance": recovered_within(
                        endpoint=endpoint_value,
                        reference=reference,
                        tolerance=args.recovery_tolerance,
                    ),
                }
            )

    sentinel = json.loads(args.sentinel.read_text(encoding="utf-8"))
    ordinary_summary = json.loads((args.ordinary_run / "SUMMARY.json").read_text(encoding="utf-8"))
    fixed_summary = json.loads((args.fixed_run / "SUMMARY.json").read_text(encoding="utf-8"))
    triggered_summary = json.loads((args.triggered_run / "SUMMARY.json").read_text(encoding="utf-8"))
    triggered_config = json.loads((args.triggered_run / "CONFIG.json").read_text(encoding="utf-8"))
    trigger_decision = triggered_config["arms"]["triggered_replay"]["trigger_decision"]
    if sha256(args.sentinel) != trigger_decision["sha256"]:
        raise ValueError("trigger decision SHA-256 drifted")
    if sentinel["trigger_tasks"] != triggered_summary["replay_policy"]["replay_tasks"]:
        raise ValueError("triggered run tasks differ from sentinel decision")
    if ordinary_summary["fixed_training_budget"] != {"steps": 50, "examples": 1200}:
        raise ValueError("ordinary training budget drifted")
    if fixed_summary["fixed_training_budget"] != {"steps": 50, "examples": 1200}:
        raise ValueError("fixed replay training budget drifted")
    if triggered_summary["fixed_training_budget"] != {"steps": 25, "examples": 600}:
        raise ValueError("triggered replay remaining budget drifted")

    fixed_status = policy_support(
        fixed_primary, fixed_donors, maximum_donor_cost=args.maximum_donor_cost
    )
    triggered_status = policy_support(
        triggered_primary,
        triggered_donors,
        maximum_donor_cost=args.maximum_donor_cost,
    )
    fixed_count_recovery = next(
        row for row in trajectory_rows if row["policy"] == "fixed" and row["task"] == "count"
    )
    fixed_shape_recovery = next(
        row for row in trajectory_rows if row["policy"] == "fixed" and row["task"] == "shape"
    )
    triggered_count_recovery = next(
        row for row in trajectory_rows if row["policy"] == "triggered" and row["task"] == "count"
    )
    direct_fixed_triggered = next(
        row
        for row in contrasts
        if row["family"] == "fixed_minus_triggered_step100"
        and row["modality"] == "preference"
        and row["metric"] == "paired_preference"
        and row["task"] == "overall"
    )
    if (
        fixed_status == "supported"
        and float(direct_fixed_triggered["ci95_low"]) > 0
        and float(fixed_generation_targets["mean_gap"]) > 0
    ):
        recommendation = "fixed_preventive_replay"
    elif triggered_status == "supported":
        recommendation = "triggered_replay"
    else:
        recommendation = "replay_not_supported"

    decisions = {
        "status": "valid",
        "format_version": "fixed-budget-matched-replay-analysis-v1",
        "recommendation": recommendation,
        "ranking_note": "policy ranking synthesizes preregistered target and donor tests with direct endpoint and generation evidence",
        "fixed_policy": {
            "status": fixed_status,
            "target_preference": fixed_primary,
            "donor_preference": fixed_donors,
            "target_generation": fixed_generation_targets,
            "count_recovery": fixed_count_recovery,
            "shape_recovery": fixed_shape_recovery,
            "reallocated_examples": len(fixed_summary["replay_policy"]["added_ids"]),
            "extra_training_examples": 0,
        },
        "triggered_policy": {
            "status": triggered_status,
            "target_preference": triggered_primary,
            "donor_preference": triggered_donors,
            "target_generation": triggered_generation_target,
            "count_recovery": triggered_count_recovery,
            "trigger_tasks": sentinel["trigger_tasks"],
            "reallocated_examples": len(triggered_summary["replay_policy"]["added_ids"]),
            "extra_training_examples": 0,
        },
        "fixed_minus_triggered_overall_preference": direct_fixed_triggered,
        "state_metrics": state_metrics,
        "training_budgets": {
            "ordinary": ordinary_summary["fixed_training_budget"],
            "fixed": fixed_summary["fixed_training_budget"],
            "triggered_prefix_shared_with_ordinary": {"steps": 25, "examples": 600},
            "triggered_remaining": triggered_summary["fixed_training_budget"],
            "total_examples_per_policy": 1200,
        },
        "thresholds": {
            "recovery_tolerance": args.recovery_tolerance,
            "maximum_mean_donor_cost": args.maximum_donor_cost,
        },
        "sources": {
            "preference_records_sha256": sha256(preference_path),
            "generation_records_sha256": sha256(generation_path),
            "sentinel_decision_sha256": sha256(args.sentinel),
            "ordinary_summary_sha256": sha256(args.ordinary_run / "SUMMARY.json"),
            "fixed_summary_sha256": sha256(args.fixed_run / "SUMMARY.json"),
            "triggered_summary_sha256": sha256(args.triggered_run / "SUMMARY.json"),
        },
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.seed},
        "final_half_scored": False,
    }
    args.out.mkdir(parents=True)
    write_csv(args.out / "metrics.csv", metric_rows)
    write_csv(args.out / "contrasts.csv", contrasts)
    write_csv(args.out / "trajectories.csv", trajectory_rows)
    (args.out / "DECISIONS.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decisions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
