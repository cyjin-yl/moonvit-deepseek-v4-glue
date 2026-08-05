#!/usr/bin/env python3
"""分析 task-conditioned projector 输出锚定是否缓解能力遗忘。"""

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
from verify_projector_interpolation import endpoint_equivalence


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


def select_retention_candidate(
    states: dict[str, dict],
    *,
    retention_tolerance: float = 0.05,
    macro_tolerance: float = 0.02,
) -> dict:
    """按预注册的保留、获取和 macro 下限选择锚定端点。"""
    if "frozen-base" not in states or "resume-control" not in states:
        raise ValueError("retention decision requires frozen-base and resume-control")
    baseline = states["frozen-base"]
    control = states["resume-control"]

    def macro(row: dict) -> float:
        return statistics.fmean(float(value) for value in row["preference"].values())

    endpoint_macro = max(macro(baseline), macro(control))
    endpoint_target = {
        task: max(
            float(baseline["preference"][task]),
            float(control["preference"][task]),
        )
        for task in baseline["preference"]
    }
    candidates = []
    for state, row in states.items():
        if not state.startswith("anchor-"):
            continue
        preference = {
            task: float(value) for task, value in row["preference"].items()
        }
        preference_macro = statistics.fmean(preference.values())
        result = {
            "state": state,
            "weight": float(row["weight"]),
            "preference_macro": preference_macro,
            "preference_worst_task": min(preference.values()),
            "generation_macro": float(row["generation_macro"]),
            "endpoint_regret": sum(
                max(0.0, endpoint_target[task] - preference[task])
                for task in preference
            ),
            "retains_count_shape": (
                preference["count"]
                >= float(baseline["preference"]["count"]) - retention_tolerance
                and preference["shape"]
                >= float(baseline["preference"]["shape"]) - retention_tolerance
            ),
            "gains_coordinate_spatial": (
                preference["coordinate"]
                > float(baseline["preference"]["coordinate"])
                and preference["spatial"]
                > float(baseline["preference"]["spatial"])
            ),
            "meets_macro_floor": preference_macro
            >= endpoint_macro - macro_tolerance,
        }
        result["targeted_retention_pass"] = bool(
            result["retains_count_shape"]
            and result["gains_coordinate_spatial"]
            and result["meets_macro_floor"]
        )
        candidates.append(result)
    candidates.sort(
        key=lambda row: (
            bool(row["targeted_retention_pass"]),
            float(row["preference_worst_task"]),
            -float(row["endpoint_regret"]),
            float(row["preference_macro"]),
            float(row["generation_macro"]),
        ),
        reverse=True,
    )
    selected = next(
        (row for row in candidates if row["targeted_retention_pass"]), None
    )
    return {
        "selected_state": selected["state"] if selected else None,
        "targeted_retention_pass": selected is not None,
        "best_diagnostic_state": candidates[0]["state"] if candidates else None,
        "baseline_state": "frozen-base",
        "control_state": "resume-control",
        "retention_tolerance": retention_tolerance,
        "macro_tolerance": macro_tolerance,
        "candidates": candidates,
        "decision_rule": (
            "an anchor must retain count/shape within 0.05 of step 50, "
            "strictly gain coordinate/spatial over step 50, and remain within "
            "0.02 of the better endpoint macro paired-preference accuracy"
        ),
    }


def verify_reference_endpoints(
    evaluation: Path,
    reference_evaluation: Path,
) -> dict:
    current_preference = read_jsonl(evaluation / "preference_records.jsonl")
    current_generation = read_jsonl(evaluation / "generation_records.jsonl")
    reference_preference = read_jsonl(
        reference_evaluation / "preference_records.jsonl"
    )
    reference_generation = read_jsonl(
        reference_evaluation / "generation_records.jsonl"
    )
    preference_fields = (
        "visual_source_id",
        "correct_answer",
        "counterfactual_answer",
        "correct_logp",
        "counterfactual_logp",
        "correct_margin",
        "correct_token_nll",
        "counterfactual_token_nll",
        "failure",
    )
    generation_structural_fields = ("visual_source_id", "answers", "failure")
    results = {}
    for current, reference in (
        ("frozen-base", "projector-interp000"),
        ("resume-control", "projector-interp100"),
    ):
        generation_rows = {
            (str(row["condition"]), str(row["id"])): row
            for row in current_generation
            if str(row["state"]) == current
        }
        reference_generation_rows = {
            (str(row["condition"]), str(row["id"])): row
            for row in reference_generation
            if str(row["state"]) == reference
        }
        structurally_matched = endpoint_equivalence(
            current_generation,
            reference_generation,
            interpolation_state=current,
            reference_state=reference,
            value_fields=generation_structural_fields,
        )
        generation_differences = {
            field: sum(
                generation_rows[key].get(field)
                != reference_generation_rows[key].get(field)
                for key in generation_rows
            )
            for field in ("prediction", "normalized_prediction", "correct")
        }
        results[current] = {
            "reference_state": reference,
            "preference_rows_exact": endpoint_equivalence(
                current_preference,
                reference_preference,
                interpolation_state=current,
                reference_state=reference,
                value_fields=preference_fields,
            ),
            "generation_rows_structurally_exact": structurally_matched,
            "generation_value_differences": generation_differences,
            "generation_strict_exact": not any(generation_differences.values()),
            "interpretation": (
                "teacher-forced values are bit-exact; free generation is reported "
                "with observed cross-run GPU decode variability and is not used by "
                "the candidate selection rule"
            ),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--reference-evaluation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite retention analysis: {args.out}")
    args.out.mkdir(parents=True)

    preference = read_jsonl(args.evaluation / "preference_records.jsonl")
    generation = read_jsonl(args.evaluation / "generation_records.jsonl")
    tasks = sorted({str(row["task"]) for row in preference})
    if tasks != sorted({str(row["task"]) for row in generation}):
        raise ValueError("retention preference/generation task sets differ")
    states = list(dict.fromkeys(str(row["state"]) for row in preference))
    if states != list(dict.fromkeys(str(row["state"]) for row in generation)):
        raise ValueError("retention preference/generation state order differs")
    if "frozen-base" not in states or "resume-control" not in states:
        raise ValueError("retention endpoints are absent")
    anchor_states = [state for state in states if state.startswith("anchor-")]
    if len(anchor_states) < 3:
        raise ValueError("retention analysis requires three anchor strengths")

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
                                "anchor_weight": (
                                    float(state.removeprefix("anchor-"))
                                    if state.startswith("anchor-")
                                    else None
                                ),
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

    adapted_states = [state for state in states if state != "frozen-base"]
    for modality, (rows, metrics) in modalities.items():
        conditions = sorted({str(row["condition"]) for row in rows})
        for state in adapted_states:
            for task in ["overall", *tasks]:
                for metric in metrics:
                    add_contrast(
                        "state_minus_step50",
                        modality,
                        metric,
                        state,
                        "vision",
                        "frozen-base",
                        "vision",
                        task,
                    )
                    if state.startswith("anchor-"):
                        add_contrast(
                            "anchor_minus_control",
                            modality,
                            metric,
                            state,
                            "vision",
                            "resume-control",
                            "vision",
                            task,
                        )
        if modality == "preference" and "shuffled_image" in conditions:
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

    state_metrics = {}
    for state in states:
        state_metrics[state] = {
            **(
                {"weight": float(state.removeprefix("anchor-"))}
                if state.startswith("anchor-")
                else {}
            ),
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
    decision = select_retention_candidate(state_metrics)
    endpoint_reproduction = verify_reference_endpoints(
        args.evaluation, args.reference_evaluation
    )
    decision.update(
        {
            "status": "valid",
            "state_metrics": state_metrics,
            "endpoint_evaluation_reproduction": endpoint_reproduction,
            "final_half_scored": False,
        }
    )

    metrics_path = args.out / "retention_metrics.csv"
    contrasts_path = args.out / "retention_contrasts.csv"
    decisions_path = args.out / "DECISIONS.json"
    write_csv(metrics_path, metric_rows)
    write_csv(contrasts_path, contrasts)
    decisions_path.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "valid",
        "format_version": "projector-retention-analysis-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "states": states,
        "tasks": tasks,
        "metric_rows": len(metric_rows),
        "contrasts": len(contrasts),
        "bootstrap_samples": args.bootstrap_samples,
        "evaluation_summary_sha256": sha256(args.evaluation / "SUMMARY.json"),
        "reference_evaluation_summary_sha256": sha256(
            args.reference_evaluation / "SUMMARY.json"
        ),
        "endpoint_evaluation_reproduction": endpoint_reproduction,
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
