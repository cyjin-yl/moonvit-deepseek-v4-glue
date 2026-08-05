#!/usr/bin/env python3
"""复用完整 paired rows，标定遗忘 sentinel 的最小可靠分母。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import time
from pathlib import Path

from analyze_multitask_transfer import pair_metric_rows
from analyze_replay_sentinel import select_trigger_tasks
from moonvit_glue.trajectory_analysis import paired_gap_stats


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def subsample_metric_rows(
    current: list[dict], reference: list[dict], *, pairs: int, seed: int
) -> tuple[list[dict], list[dict]]:
    """从两个 state 同步抽取相同 pair IDs。"""

    current_by_id = {str(row["id"]): row for row in current}
    reference_by_id = {str(row["id"]): row for row in reference}
    if not current_by_id or set(current_by_id) != set(reference_by_id):
        raise ValueError("sentinel subsampling requires identical non-empty pair IDs")
    if pairs <= 0 or pairs > len(current_by_id):
        raise ValueError(
            f"requested pairs must be within available pairs: {pairs} > {len(current_by_id)}"
        )
    identifiers = sorted(current_by_id)
    selected = sorted(random.Random(seed).sample(identifiers, pairs))
    return (
        [current_by_id[pair_id] for pair_id in selected],
        [reference_by_id[pair_id] for pair_id in selected],
    )


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """返回二项比例的 95% Wilson 区间。"""

    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Wilson interval needs 0 <= successes <= positive trials")
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4 * trials * trials)
        )
        / denominator
    )
    low = max(0.0, centre - radius)
    high = min(1.0, centre + radius)
    if successes == 0:
        low = 0.0
    if successes == trials:
        high = 1.0
    return low, high


def percentile(values: list[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile needs non-empty values and a bounded quantile")
    ordered = sorted(float(value) for value in values)
    location = (len(ordered) - 1) * quantile
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def candidate_passes(row: dict, criteria: dict) -> bool:
    return bool(
        float(row["count_recall"]) >= float(criteria["minimum_count_recall"])
        and float(row["count_recall_ci95_low"])
        >= float(criteria["minimum_count_recall_ci95_low"])
        and float(row["exact_decision_rate"])
        >= float(criteria["minimum_exact_decision_rate"])
        and float(row["exact_decision_ci95_low"])
        >= float(criteria["minimum_exact_decision_ci95_low"])
        and float(row["familywise_false_trigger_rate"])
        <= float(criteria["maximum_familywise_false_trigger_rate"])
        and float(row["familywise_false_trigger_ci95_high"])
        <= float(criteria["maximum_familywise_false_trigger_ci95_high"])
    )


def select_minimum_candidate(rows: list[dict], criteria: dict) -> dict | None:
    passing = [row for row in rows if candidate_passes(row, criteria)]
    return min(passing, key=lambda row: int(row["pairs_per_task"])) if passing else None


def paired_metrics_by_task(
    rows: list[dict], *, state: str, condition: str
) -> dict[str, list[dict]]:
    tasks = sorted(
        {
            str(row["task"])
            for row in rows
            if str(row["state"]) == state and str(row["condition"]) == condition
        }
    )
    output = {}
    for task in tasks:
        selected = [
            row
            for row in rows
            if str(row["state"]) == state
            and str(row["condition"]) == condition
            and str(row["task"]) == task
        ]
        output[task] = pair_metric_rows(selected, "paired_preference")
    return output


def contrast_row(
    current: list[dict],
    reference: list[dict],
    *,
    task: str,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    return {
        "task": task,
        **paired_gap_stats(
            current,
            reference,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
    }


def run(config: dict, evaluation: Path, out: Path) -> dict:
    started = time.time()
    source = evaluation / "preference_records.jsonl"
    if sha256(source) != str(config["source"]["preference_records_sha256"]):
        raise ValueError("sentinel preference source hash drifted")
    rows = read_jsonl(source)
    reference_state = str(config["states"]["reference"])
    current_state = str(config["states"]["current"])
    condition = str(config["condition"])
    current = paired_metrics_by_task(rows, state=current_state, condition=condition)
    reference = paired_metrics_by_task(rows, state=reference_state, condition=condition)
    tasks = sorted(current)
    if tasks != sorted(reference) or tasks != sorted(config["tasks"]):
        raise ValueError("sentinel task sets drifted")
    for task in tasks:
        if {row["id"] for row in current[task]} != {row["id"] for row in reference[task]}:
            raise ValueError(f"sentinel pair IDs drifted for task: {task}")

    bootstrap_samples = int(config["bootstrap_samples"])
    minimum_drop = float(config["trigger_rule"]["minimum_absolute_drop"])
    max_tasks = int(config["trigger_rule"]["maximum_tasks"])
    target_task = str(config["target_task"])
    expected_triggers = [str(task) for task in config["expected_full_trigger_tasks"]]
    full_contrasts = [
        contrast_row(
            current[task],
            reference[task],
            task=task,
            bootstrap_samples=bootstrap_samples,
            seed=int(config["bootstrap_seed"]) + task_index,
        )
        for task_index, task in enumerate(tasks)
    ]
    full_triggers = select_trigger_tasks(
        full_contrasts, minimum_drop=minimum_drop, max_tasks=max_tasks
    )
    if full_triggers != expected_triggers:
        raise ValueError(
            f"full sentinel decision drifted: {full_triggers} != {expected_triggers}"
        )

    trial_rows: list[dict] = []
    task_rows: list[dict] = []
    summary_rows: list[dict] = []
    trials = int(config["trials"])
    for pairs_per_task in [int(value) for value in config["pairs_per_task"]]:
        count_successes = 0
        exact_successes = 0
        false_successes = 0
        count_ci_highs = []
        count_mean_gaps = []
        false_by_task = {task: 0 for task in tasks if task not in expected_triggers}
        for trial in range(trials):
            contrasts = []
            for task_index, task in enumerate(tasks):
                sample_seed = (
                    int(config["sampling_seed"])
                    + pairs_per_task * 1_000_000
                    + trial * 100
                    + task_index
                )
                bootstrap_seed = (
                    int(config["bootstrap_seed"])
                    + pairs_per_task * 1_000_000
                    + trial * 100
                    + task_index
                )
                sampled_current, sampled_reference = subsample_metric_rows(
                    current[task],
                    reference[task],
                    pairs=pairs_per_task,
                    seed=sample_seed,
                )
                contrast = contrast_row(
                    sampled_current,
                    sampled_reference,
                    task=task,
                    bootstrap_samples=bootstrap_samples,
                    seed=bootstrap_seed,
                )
                contrasts.append(contrast)
            triggers = select_trigger_tasks(
                contrasts, minimum_drop=minimum_drop, max_tasks=max_tasks
            )
            false_tasks = [task for task in triggers if task not in expected_triggers]
            count_hit = target_task in triggers
            exact_hit = triggers == expected_triggers
            false_hit = bool(false_tasks)
            count_successes += int(count_hit)
            exact_successes += int(exact_hit)
            false_successes += int(false_hit)
            for task in false_tasks:
                false_by_task[task] += 1
            count_row = next(row for row in contrasts if row["task"] == target_task)
            count_ci_highs.append(float(count_row["ci95_high"]))
            count_mean_gaps.append(float(count_row["mean_gap"]))
            trial_rows.append(
                {
                    "pairs_per_task": pairs_per_task,
                    "trial": trial,
                    "trigger_tasks": "|".join(triggers),
                    "target_triggered": count_hit,
                    "exact_decision": exact_hit,
                    "familywise_false_trigger": false_hit,
                    "false_trigger_tasks": "|".join(false_tasks),
                }
            )
            for row in contrasts:
                task_rows.append(
                    {
                        "pairs_per_task": pairs_per_task,
                        "trial": trial,
                        "task": row["task"],
                        "mean_gap": row["mean_gap"],
                        "ci95_low": row["ci95_low"],
                        "ci95_high": row["ci95_high"],
                        "triggered": row["task"] in triggers,
                        "sampling_seed": (
                            int(config["sampling_seed"])
                            + pairs_per_task * 1_000_000
                            + trial * 100
                            + tasks.index(str(row["task"]))
                        ),
                        "bootstrap_seed": row["seed"],
                    }
                )
        count_low, count_high = wilson_interval(count_successes, trials)
        exact_low, exact_high = wilson_interval(exact_successes, trials)
        false_low, false_high = wilson_interval(false_successes, trials)
        summary_rows.append(
            {
                "pairs_per_task": pairs_per_task,
                "teacher_records_per_state": pairs_per_task * 2 * len(tasks),
                "trials": trials,
                "count_recall": count_successes / trials,
                "count_recall_ci95_low": count_low,
                "count_recall_ci95_high": count_high,
                "exact_decision_rate": exact_successes / trials,
                "exact_decision_ci95_low": exact_low,
                "exact_decision_ci95_high": exact_high,
                "familywise_false_trigger_rate": false_successes / trials,
                "familywise_false_trigger_ci95_low": false_low,
                "familywise_false_trigger_ci95_high": false_high,
                "count_mean_gap_median": statistics.median(count_mean_gaps),
                "count_mean_gap_p05": percentile(count_mean_gaps, 0.05),
                "count_mean_gap_p95": percentile(count_mean_gaps, 0.95),
                "count_ci95_high_median": statistics.median(count_ci_highs),
                "count_ci95_high_p95": percentile(count_ci_highs, 0.95),
                "false_trigger_rates_by_task": json.dumps(
                    {
                        task: false_by_task[task] / trials
                        for task in sorted(false_by_task)
                    },
                    sort_keys=True,
                ),
            }
        )

    criteria = dict(config["selection_criteria"])
    for row in summary_rows:
        row["passes_preregistered_criteria"] = candidate_passes(row, criteria)
    selected = select_minimum_candidate(summary_rows, criteria)
    candidates = [int(value) for value in config["pairs_per_task"]]
    tiny = int(selected["pairs_per_task"]) if selected else None
    medium = (
        next((value for value in candidates if tiny is not None and value > tiny), tiny)
        if tiny is not None
        else None
    )
    output = {
        "status": "valid" if selected else "underpowered",
        "format_version": "sentinel-power-analysis-v1",
        "source_git_sha": git_sha(),
        "reference_state": reference_state,
        "current_state": current_state,
        "condition": condition,
        "tasks": tasks,
        "full_source_pairs_per_task": {
            task: len(current[task]) for task in tasks
        },
        "full_source_trigger_tasks": full_triggers,
        "candidate_summary": summary_rows,
        "selection_criteria": criteria,
        "recommended_tiny_pairs_per_task": tiny,
        "recommended_medium_pairs_per_task": medium,
        "timing_protocol": config["timing_protocol"],
        "source": {
            "preference_records": str(source),
            "preference_records_sha256": sha256(source),
        },
        "analysis": {
            "trials_per_candidate": trials,
            "bootstrap_samples_per_contrast": bootstrap_samples,
            "sampling_seed": int(config["sampling_seed"]),
            "bootstrap_seed": int(config["bootstrap_seed"]),
            "wall_seconds": time.time() - started,
        },
        "final_half_scored": False,
    }
    out.mkdir(parents=True)
    write_csv(out / "full_contrasts.csv", full_contrasts)
    write_csv(out / "candidate_summary.csv", summary_rows)
    write_csv(out / "trial_decisions.csv", trial_rows)
    write_csv(out / "task_trials.csv", task_rows)
    (out / "DECISIONS.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite sentinel power analysis: {args.out}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = run(config, args.evaluation, args.out)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
