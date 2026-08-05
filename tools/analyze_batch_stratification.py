#!/usr/bin/env python3
"""分析严格配对的分层平衡批次与全局随机批次轨迹。"""

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


def classify_batch_effect(endpoint_rows: list[dict]) -> str:
    """用端点成对置信区间判断批次分层是否产生可辨别效应。"""
    overall = next(row for row in endpoint_rows if row["task"] == "overall")
    if float(overall["ci95_low"]) > 0:
        return "balanced_batch_effect_supported"
    if float(overall["ci95_high"]) < 0:
        return "global_random_effect_supported"
    task_rows = [row for row in endpoint_rows if row["task"] != "overall"]
    positive = sum(float(row["ci95_low"]) > 0 for row in task_rows)
    negative = sum(float(row["ci95_high"]) < 0 for row in task_rows)
    if positive >= 2 and negative == 0:
        return "balanced_batch_effect_supported"
    if negative >= 2 and positive == 0:
        return "global_random_effect_supported"
    return "mixed_or_underpowered"


def state_step(state: str, prefix: str) -> int:
    if not state.startswith(prefix):
        raise ValueError(f"state does not start with {prefix}: {state}")
    return int(state.removeprefix(prefix))


def trapezoid_auc(points: list[tuple[int, float]]) -> float:
    points = sorted(points)
    width = points[-1][0] - points[0][0]
    if width <= 0:
        raise ValueError("trajectory AUC requires a positive step range")
    area = sum(
        (right_step - left_step) * (left_value + right_value) / 2
        for (left_step, left_value), (right_step, right_value) in zip(points, points[1:])
    )
    return area / width


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {args.out}")
    verification = json.loads((args.verification / "VERIFICATION.json").read_text(encoding="utf-8"))
    if verification.get("status") != "valid" or not all(verification["checks"].values()):
        raise ValueError("batch-order invariants are not verified")

    preference = read_jsonl(args.evaluation / "preference_records.jsonl")
    generation = read_jsonl(args.evaluation / "generation_records.jsonl")
    tasks = sorted({str(row["task"]) for row in preference})
    states = list(dict.fromkeys(str(row["state"]) for row in preference))
    if states != list(dict.fromkeys(str(row["state"]) for row in generation)):
        raise ValueError("preference and generation state orders differ")
    frozen = "frozen-base"
    if frozen not in states:
        raise ValueError("frozen start state is absent")
    arm_prefixes = ("stratified-step", "global-step")
    arm_states = {
        prefix: sorted(
            [state for state in states if state.startswith(prefix)],
            key=lambda state: state_step(state, prefix),
        )
        for prefix in arm_prefixes
    }
    arm_steps = {
        prefix: [state_step(state, prefix) for state in values]
        for prefix, values in arm_states.items()
    }
    if arm_steps[arm_prefixes[0]] != arm_steps[arm_prefixes[1]] or arm_steps[arm_prefixes[0]] != [25, 50, 100]:
        raise ValueError("matched arms must expose steps 25, 50, and 100")

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
                        row for row in rows
                        if str(row["state"]) == state
                        and str(row["condition"]) == condition
                        and (task == "overall" or str(row["task"]) == task)
                    ]
                    for metric in metrics:
                        pairs = pair_metric_rows(cell, metric)
                        grouped[(modality, state, condition, task, metric)] = pairs
                        metric_rows.append({
                            "modality": modality,
                            "state": state,
                            "condition": condition,
                            "task": task,
                            "metric": metric,
                            "pairs": len(pairs),
                            "mean": statistics.fmean(float(row["score"]) for row in pairs),
                        })

    contrasts = []
    counter = 0

    def add_contrast(family: str, modality: str, metric: str, state_a: str, condition_a: str, state_b: str, condition_b: str, task: str) -> None:
        nonlocal counter
        counter += 1
        stats = paired_gap_stats(
            grouped[(modality, state_a, condition_a, task, metric)],
            grouped[(modality, state_b, condition_b, task, metric)],
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + counter,
        )
        contrasts.append({
            "family": family,
            "modality": modality,
            "metric": metric,
            "state_a": state_a,
            "condition_a": condition_a,
            "state_b": state_b,
            "condition_b": condition_b,
            "task": task,
            **stats,
        })

    for step in (25, 50, 100):
        stratified = f"stratified-step{step}"
        global_random = f"global-step{step}"
        for modality, (rows, metrics) in modalities.items():
            for task in ["overall", *tasks]:
                for metric in metrics:
                    add_contrast("stratified_minus_global", modality, metric, stratified, "vision", global_random, "vision", task)
                    add_contrast("stratified_minus_start", modality, metric, stratified, "vision", frozen, "vision", task)
                    add_contrast("global_minus_start", modality, metric, global_random, "vision", frozen, "vision", task)
    for state in states:
        for task in ["overall", *tasks]:
            for metric in modalities["preference"][1]:
                add_contrast("vision_minus_shuffle", "preference", metric, state, "vision", state, "shuffled_image", task)

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
        task_preference = {
            task: metric_mean("preference", state, "vision", task, "paired_preference")
            for task in tasks
        }
        state_metrics[state] = {
            "preference": task_preference,
            "preference_macro": statistics.fmean(task_preference.values()),
            "preference_worst_task": min(task_preference.values()),
            "generation_macro": statistics.fmean(
                metric_mean("generation", state, "vision", task, "generation_paired")
                for task in tasks
            ),
            "vision_shuffle_overall": metric_mean("preference", state, "vision", "overall", "paired_preference")
            - metric_mean("preference", state, "shuffled_image", "overall", "paired_preference"),
        }

    endpoint_rows = [
        row for row in contrasts
        if row["family"] == "stratified_minus_global"
        and row["modality"] == "preference"
        and row["metric"] == "paired_preference"
        and row["state_a"] == "stratified-step100"
    ]
    trajectory = {}
    for task in tasks:
        trajectory[task] = {}
        start = metric_mean("preference", frozen, "vision", task, "paired_preference")
        for label, prefix in (("stratified", "stratified-step"), ("global", "global-step")):
            points = [(0, start)] + [
                (step, metric_mean("preference", f"{prefix}{step}", "vision", task, "paired_preference"))
                for step in (25, 50, 100)
            ]
            peak_step, peak_value = max(points, key=lambda item: (item[1], -item[0]))
            trajectory[task][label] = {
                "points": [{"step": step, "value": value} for step, value in points],
                "auc": trapezoid_auc(points),
                "peak_step": peak_step,
                "peak_value": peak_value,
                "endpoint": points[-1][1],
                "forgetting_from_peak": peak_value - points[-1][1],
            }
        trajectory[task]["auc_stratified_minus_global"] = trajectory[task]["stratified"]["auc"] - trajectory[task]["global"]["auc"]

    decision = {
        "status": "valid",
        "format_version": "matched-batch-order-analysis-v1",
        "batch_effect": classify_batch_effect(endpoint_rows),
        "decision_rule": "endpoint overall paired-preference CI excludes zero, or at least two task CIs agree with no significant task reversal",
        "steps": [25, 50, 100],
        "state_metrics": state_metrics,
        "endpoint_paired_preference_contrasts": endpoint_rows,
        "trajectory": trajectory,
        "sources": {
            "preference_records_sha256": sha256(args.evaluation / "preference_records.jsonl"),
            "generation_records_sha256": sha256(args.evaluation / "generation_records.jsonl"),
            "verification_sha256": sha256(args.verification / "VERIFICATION.json"),
        },
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.seed},
        "final_half_scored": False,
    }
    args.out.mkdir(parents=True)
    write_csv(args.out / "metrics.csv", metric_rows)
    write_csv(args.out / "contrasts.csv", contrasts)
    (args.out / "DECISIONS.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
