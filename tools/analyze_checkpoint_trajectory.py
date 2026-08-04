#!/usr/bin/env python3
"""从轨迹 run 计算成对不确定区间与证据决策。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from moonvit_glue.trajectory_analysis import paired_gap_stats
from moonvit_glue.trajectory_data import configured_conditions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def pair_success(sample_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in sample_rows:
        grouped[str(row["pair_id"])].append(row)
    output = []
    for pair_id, pair in sorted(grouped.items()):
        if len(pair) != 2:
            raise ValueError(f"incomplete pair in analysis: {pair_id}")
        predictions = {str(row.get("normalized_prediction") or "") for row in pair}
        output.append({
            "id": pair_id,
            "score": float(all(bool(row["correct"]) for row in pair) and len(predictions) == 2),
        })
    return output


def pair_mean_score(sample_rows: list[dict]) -> list[dict]:
    """合并同一最小对的两条样本，使 bootstrap 始终重采样完整 pair。"""

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in sample_rows:
        grouped[str(row["pair_id"])].append(row)
    output = []
    for pair_id, pair in sorted(grouped.items()):
        if len(pair) != 2:
            raise ValueError(f"incomplete pair in analysis: {pair_id}")
        output.append(
            {"id": pair_id, "score": statistics.fmean(float(row["score"]) for row in pair)}
        )
    return output


def bootstrap_gap(a: list[dict], b: list[dict], seed: int) -> dict:
    return paired_gap_stats(a, b, bootstrap_samples=2000, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite trajectory analysis: {args.out}")
    args.out.mkdir(parents=True)
    config = json.loads((args.run / "CONFIG.json").read_text(encoding="utf-8"))
    summary = json.loads((args.run / "SUMMARY.json").read_text(encoding="utf-8"))
    raw = rows(args.run / "records.jsonl")
    shuffle = rows(args.run / "shuffle_loss_records.jsonl")
    checkpoints = [str(row["id"]) for row in config["checkpoints"]]
    examples_seen = {str(row["id"]): int(row["examples_seen"]) for row in config["checkpoints"]}
    synthetic = [row for row in raw if row["dataset"] == "synthetic"]
    benchmark_rows = [row for row in raw if row["dataset"] == "benchmarks"]
    tasks = sorted({str(row["task"]) for row in synthetic})
    conditions = next(row for row in config["datasets"] if row["name"] == "synthetic")["conditions"]
    grouped = {
        (checkpoint, condition, task): [
            row for row in synthetic
            if row["checkpoint"] == checkpoint
            and row["condition"] == condition
            and (task == "overall" or row["task"] == task)
        ]
        for checkpoint in checkpoints
        for condition in conditions
        for task in ["overall", *tasks]
    }

    gaps = []
    gap_index: dict[tuple[str, str, str, str], dict] = {}
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        for task_index, task in enumerate(["overall", *tasks]):
            vision = grouped[(checkpoint, "vision", task)]
            for condition_index, condition in enumerate(conditions):
                if condition == "vision":
                    continue
                comparison = grouped[(checkpoint, condition, task)]
                stats = bootstrap_gap(
                    pair_mean_score(vision),
                    pair_mean_score(comparison),
                    args.seed + checkpoint_index * 100 + task_index * 10 + condition_index,
                )
                stats.update({
                    "checkpoint": checkpoint,
                    "examples_seen": examples_seen[checkpoint],
                    "task": task,
                    "a": "vision",
                    "b": condition,
                    "metric": "sample_accuracy",
                })
                gaps.append(stats)
                gap_index[(checkpoint, task, "sample", condition)] = stats

            vision_pairs = pair_success(vision)
            blind_pairs = pair_success(grouped[(checkpoint, "blind", task)])
            pair_stats = bootstrap_gap(
                vision_pairs,
                blind_pairs,
                args.seed + 10000 + checkpoint_index * 100 + task_index,
            )
            pair_stats.update({
                "checkpoint": checkpoint,
                "examples_seen": examples_seen[checkpoint],
                "task": task,
                "a": "vision",
                "b": "blind",
                "metric": "paired_answer_flip_accuracy",
            })
            gaps.append(pair_stats)
            gap_index[(checkpoint, task, "pair", "blind")] = pair_stats

    checkpoint_deltas = []
    for index in range(1, len(checkpoints)):
        previous, current = checkpoints[index - 1], checkpoints[index]
        for task_index, task in enumerate(["overall", *tasks]):
            current_pairs = pair_success(grouped[(current, "vision", task)])
            previous_pairs = pair_success(grouped[(previous, "vision", task)])
            stats = bootstrap_gap(
                current_pairs,
                previous_pairs,
                args.seed + 20000 + index * 100 + task_index,
            )
            stats.update({
                "previous_checkpoint": previous,
                "checkpoint": current,
                "previous_examples_seen": examples_seen[previous],
                "examples_seen": examples_seen[current],
                "task": task,
                "metric": "paired_answer_flip_accuracy",
            })
            checkpoint_deltas.append(stats)

    benchmark_gaps = []
    benchmark_names = sorted({str(row["benchmark"]) for row in benchmark_rows})
    benchmark_config = next(
        row for row in config["datasets"] if row["name"] == "benchmarks"
    )
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        for benchmark_index, benchmark in enumerate(benchmark_names):
            vision = [
                {"id": row["id"], "score": row["score"]}
                for row in benchmark_rows
                if row["checkpoint"] == checkpoint
                and row["benchmark"] == benchmark
                and row["condition"] == "vision"
            ]
            for condition_index, condition in enumerate(
                configured_conditions(benchmark_config, checkpoint)
            ):
                if condition == "vision":
                    continue
                comparison = [
                    {"id": row["id"], "score": row["score"]}
                    for row in benchmark_rows
                    if row["checkpoint"] == checkpoint
                    and row["benchmark"] == benchmark
                    and row["condition"] == condition
                ]
                stats = bootstrap_gap(
                    vision,
                    comparison,
                    args.seed
                    + 40_000
                    + checkpoint_index * 1000
                    + benchmark_index * 100
                    + condition_index,
                )
                stats.update(
                    {
                        "checkpoint": checkpoint,
                        "examples_seen": examples_seen[checkpoint],
                        "benchmark": benchmark,
                        "metric": "primary_score",
                        "a": "vision",
                        "b": condition,
                    }
                )
                benchmark_gaps.append(stats)

    shuffle_stats = {}
    for index, checkpoint in enumerate(checkpoints):
        checkpoint_rows = [row for row in shuffle if row["checkpoint"] == checkpoint]
        delta_rows = [{"id": row["id"], "score": float(row["delta"])} for row in checkpoint_rows]
        zeros = [{"id": row["id"], "score": 0.0} for row in checkpoint_rows]
        stats = bootstrap_gap(delta_rows, zeros, args.seed + 30000 + index)
        stats["checkpoint"] = checkpoint
        stats["examples_seen"] = examples_seen[checkpoint]
        shuffle_stats[checkpoint] = stats

    def first_significant(metric: str, task: str = "overall"):
        for checkpoint in checkpoints:
            if metric == "shuffle":
                stats = shuffle_stats[checkpoint]
            elif metric == "pair":
                stats = gap_index[(checkpoint, task, "pair", "blind")]
            else:
                stats = gap_index[(checkpoint, task, "sample", "blind")]
            if stats["ci95_low"] > 0:
                return checkpoint
        return None

    task_onsets = {task: first_significant("sample", task) for task in tasks}
    pair_onset = first_significant("pair")
    shuffle_onset = first_significant("shuffle")
    onset_position = {checkpoint: index for index, checkpoint in enumerate(checkpoints)}
    shuffle_earlier = (
        shuffle_onset is not None
        and (pair_onset is None or onset_position[shuffle_onset] < onset_position[pair_onset])
    )

    overall_deltas = [row for row in checkpoint_deltas if row["task"] == "overall"]
    final_delta = overall_deltas[-1]
    improvements = [abs(float(row["mean_gap"])) for row in overall_deltas]
    median_improvement = statistics.median(improvements) if improvements else 0.0
    largest = max(overall_deltas, key=lambda row: float(row["mean_gap"]), default=None)
    grokking = bool(
        largest
        and largest["ci95_low"] > 0
        and largest["mean_gap"] >= 0.05
        and largest["mean_gap"] > 2 * median_improvement
    )

    paired_values = {
        checkpoint: gap_index[(checkpoint, "overall", "pair", "blind")]["mean_a"]
        for checkpoint in checkpoints
    }
    best_checkpoint = max(checkpoints, key=lambda checkpoint: (paired_values[checkpoint], examples_seen[checkpoint]))
    latest = checkpoints[-1]
    latest_gap = gap_index[(latest, "overall", "sample", "blind")]
    latest_shuffled_gap = gap_index[(latest, "overall", "sample", "shuffled_image")]
    latest_same_gap = gap_index[(latest, "overall", "sample", "same_image")]
    latest_blank_gap = gap_index[(latest, "overall", "sample", "blank")]
    decisions = {
        "status": "valid",
        "checkpoint_order": checkpoints,
        "shuffle_delta_onset": shuffle_onset,
        "paired_answer_flip_onset": pair_onset,
        "shuffle_delta_earlier_than_paired_generation": shuffle_earlier,
        "task_vision_gap_onsets": task_onsets,
        "step_2000_still_rising": {
            "supported": final_delta["ci95_low"] > 0,
            "mean_pair_accuracy_change_vs_step_1500": final_delta["mean_gap"],
            "ci95": [final_delta["ci95_low"], final_delta["ci95_high"]],
        },
        "sudden_grokking": {
            "supported": grokking,
            "largest_transition": (
                [largest["previous_checkpoint"], largest["checkpoint"]] if largest else None
            ),
            "largest_pair_accuracy_change": largest["mean_gap"] if largest else None,
            "rule": "positive CI, absolute jump >=0.05, and >2x median absolute checkpoint jump",
        },
        "best_checkpoint_by_paired_answer_flip": best_checkpoint,
        "best_is_last_unique_checkpoint": best_checkpoint == latest,
        "latest_vision_minus_blind": latest_gap,
        "latest_vision_minus_blank": latest_blank_gap,
        "latest_vision_minus_same_image": latest_same_gap,
        "latest_vision_minus_shuffled_image": latest_shuffled_gap,
        "latest_content_specific_gap_supported": (
            latest_same_gap["ci95_low"] > 0 and latest_shuffled_gap["ci95_low"] > 0
        ),
        "stop_condition_no_significant_synthetic_vision_gap": latest_gap["ci95_low"] <= 0,
        "interpretation_limits": [
            "bootstrap intervals are descriptive paired resampling, not proof of mechanism",
            "checkpoint selection uses synthetic/benchmark selection data only; final odd halves remain unscored",
            "current-final is a bit-identical alias of step-002000",
        ],
    }

    statistics_payload = {
        "format_version": "trajectory-analysis-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(args.run.resolve()),
        "run_summary_sha256": sha256(args.run / "SUMMARY.json"),
        "raw_records_sha256": sha256(args.run / "records.jsonl"),
        "shuffle_records_sha256": sha256(args.run / "shuffle_loss_records.jsonl"),
        "bootstrap_seed": args.seed,
        "gaps": gaps,
        "checkpoint_deltas": checkpoint_deltas,
        "benchmark_gaps": benchmark_gaps,
        "shuffle_delta_stats": shuffle_stats,
    }
    (args.out / "STATISTICS.json").write_text(
        json.dumps(statistics_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "DECISIONS.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out / "paired_gaps.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["checkpoint", "examples_seen", "task", "metric", "a", "b", "sum_a", "sum_b", "denominator", "mean_a", "mean_b", "mean_gap", "ci95_low", "ci95_high", "a_only_better", "b_only_better", "equal"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(gaps)
    with (args.out / "checkpoint_deltas.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["previous_checkpoint", "checkpoint", "previous_examples_seen", "examples_seen", "task", "metric", "sum_a", "sum_b", "denominator", "mean_a", "mean_b", "mean_gap", "ci95_low", "ci95_high"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(checkpoint_deltas)
    with (args.out / "benchmark_paired_gaps.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "checkpoint", "examples_seen", "benchmark", "metric", "a", "b",
            "sum_a", "sum_b", "denominator", "mean_a", "mean_b", "mean_gap",
            "ci95_low", "ci95_high", "a_only_better", "b_only_better", "equal",
            "bootstrap_samples", "seed",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(benchmark_gaps)
    print(json.dumps(decisions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
