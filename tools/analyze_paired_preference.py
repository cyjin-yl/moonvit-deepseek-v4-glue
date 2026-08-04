#!/usr/bin/env python3
"""对 teacher-forced paired-preference 轨迹做 bootstrap 与制表。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from moonvit_glue.paired_preference import summarize_preference_rows
from moonvit_glue.trajectory_analysis import paired_gap_stats


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def pair_metric_rows(rows: list[dict], metric: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    output = []
    for pair_id, pair in sorted(grouped.items()):
        if len(pair) != 2 or any(row.get("failure") is not None for row in pair):
            raise ValueError(f"preference analysis needs a valid complete pair: {pair_id}")
        margins = [float(row["correct_margin"]) for row in pair]
        if metric == "paired_preference":
            score = float(all(value > 0 for value in margins))
        elif metric == "mean_margin":
            score = statistics.fmean(margins)
        else:
            raise ValueError(f"unknown pair metric: {metric}")
        output.append({"id": pair_id, "score": score})
    return output


def against_zero(rows: list[dict], *, seed: int) -> dict:
    zeros = [{"id": row["id"], "score": 0.0} for row in rows]
    return paired_gap_stats(rows, zeros, bootstrap_samples=2000, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite preference analysis: {args.out}")
    args.out.mkdir(parents=True)

    config = json.loads((args.run / "CONFIG.json").read_text(encoding="utf-8"))
    raw = read_jsonl(args.run / "preference_records.jsonl")
    checkpoints = [str(row["id"]) for row in config["checkpoints"]]
    examples_seen = {str(row["id"]): int(row["examples_seen"]) for row in config["checkpoints"]}
    conditions = [str(value) for value in config["synthetic"]["conditions"]]
    tasks = sorted({str(row["task"]) for row in raw})
    grouped = {
        (checkpoint, condition, task): [
            row
            for row in raw
            if str(row["checkpoint"]) == checkpoint
            and str(row["condition"]) == condition
            and (task == "overall" or str(row["task"]) == task)
        ]
        for checkpoint in checkpoints
        for condition in conditions
        for task in ["overall", *tasks]
    }

    curve_rows = []
    interval_rows = []
    gap_rows = []
    checkpoint_delta_rows = []
    gap_index: dict[tuple[str, str, str, str], dict] = {}
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        for condition_index, condition in enumerate(conditions):
            for task_index, task in enumerate(["overall", *tasks]):
                rows = grouped[(checkpoint, condition, task)]
                summary = summarize_preference_rows(rows)
                curve_rows.append(
                    {
                        "checkpoint": checkpoint,
                        "optimizer_steps": next(
                            int(value["optimizer_steps"])
                            for value in config["checkpoints"]
                            if str(value["id"]) == checkpoint
                        ),
                        "examples_seen": examples_seen[checkpoint],
                        "task": task,
                        "condition": condition,
                        "samples": summary["samples"],
                        "pairs": summary["pairs"],
                        "sample_preference_accuracy": summary["sample_preference_accuracy"]["value"],
                        "paired_preference_accuracy": summary["paired_preference_accuracy"]["value"],
                        "mean_correct_margin": summary["mean_correct_margin"],
                        "median_correct_margin": summary["median_correct_margin"],
                        "mean_correct_token_nll": summary["mean_correct_token_nll"],
                        "mean_counterfactual_token_nll": summary["mean_counterfactual_token_nll"],
                    }
                )
                for metric_index, metric in enumerate(("paired_preference", "mean_margin")):
                    pair_rows = pair_metric_rows(rows, metric)
                    stats = against_zero(
                        pair_rows,
                        seed=(
                            args.seed
                            + checkpoint_index * 1000
                            + condition_index * 100
                            + task_index * 10
                            + metric_index
                        ),
                    )
                    stats.update(
                        {
                            "checkpoint": checkpoint,
                            "examples_seen": examples_seen[checkpoint],
                            "condition": condition,
                            "task": task,
                            "metric": metric,
                        }
                    )
                    interval_rows.append(stats)

        for task_index, task in enumerate(["overall", *tasks]):
            vision = grouped[(checkpoint, "vision", task)]
            for condition_index, condition in enumerate(conditions):
                if condition == "vision":
                    continue
                comparison = grouped[(checkpoint, condition, task)]
                for metric_index, metric in enumerate(("paired_preference", "mean_margin")):
                    stats = paired_gap_stats(
                        pair_metric_rows(vision, metric),
                        pair_metric_rows(comparison, metric),
                        bootstrap_samples=2000,
                        seed=(
                            args.seed
                            + 100_000
                            + checkpoint_index * 1000
                            + condition_index * 100
                            + task_index * 10
                            + metric_index
                        ),
                    )
                    stats.update(
                        {
                            "checkpoint": checkpoint,
                            "examples_seen": examples_seen[checkpoint],
                            "task": task,
                            "metric": metric,
                            "a": "vision",
                            "b": condition,
                        }
                    )
                    gap_rows.append(stats)
                    gap_index[(checkpoint, task, metric, condition)] = stats

    for checkpoint_index in range(1, len(checkpoints)):
        previous = checkpoints[checkpoint_index - 1]
        current = checkpoints[checkpoint_index]
        for task_index, task in enumerate(["overall", *tasks]):
            for metric_index, metric in enumerate(("paired_preference", "mean_margin")):
                stats = paired_gap_stats(
                    pair_metric_rows(grouped[(current, "vision", task)], metric),
                    pair_metric_rows(grouped[(previous, "vision", task)], metric),
                    bootstrap_samples=2000,
                    seed=(
                        args.seed
                        + 200_000
                        + checkpoint_index * 100
                        + task_index * 10
                        + metric_index
                    ),
                )
                stats.update(
                    {
                        "previous_checkpoint": previous,
                        "checkpoint": current,
                        "previous_examples_seen": examples_seen[previous],
                        "examples_seen": examples_seen[current],
                        "task": task,
                        "condition": "vision",
                        "metric": metric,
                    }
                )
                checkpoint_delta_rows.append(stats)

    random_checkpoint = next(
        str(row["id"]) for row in config["checkpoints"] if row["kind"] == "random"
    )
    trained_checkpoints = [
        str(row["id"]) for row in config["checkpoints"] if row["kind"] == "trained"
    ]
    validated_signals = {
        task: {
            checkpoint: (
                gap_index[
                    (checkpoint, task, "paired_preference", "shuffled_image")
                ]["ci95_low"]
                > 0
                and gap_index[
                    (
                        checkpoint,
                        task,
                        "mean_margin",
                        "paired_counterfactual_image",
                    )
                ]["ci95_low"]
                > 0
            )
            for checkpoint in trained_checkpoints
        }
        for task in tasks
    }
    onset = {
        task: next(
            (
                checkpoint
                for checkpoint in trained_checkpoints
                if validated_signals[task][checkpoint]
            ),
            None,
        )
        for task in tasks
    }
    latest = checkpoints[-1]
    task_peaks = {}
    for task_index, task in enumerate(tasks):
        pair_rows_by_checkpoint = {
            checkpoint: pair_metric_rows(
                grouped[(checkpoint, "vision", task)], "paired_preference"
            )
            for checkpoint in checkpoints
        }
        values = {
            checkpoint: statistics.fmean(row["score"] for row in pair_rows)
            for checkpoint, pair_rows in pair_rows_by_checkpoint.items()
        }
        best = max(
            trained_checkpoints,
            key=lambda checkpoint: (values[checkpoint], -checkpoints.index(checkpoint)),
        )
        best_vs_latest = paired_gap_stats(
            pair_rows_by_checkpoint[best],
            pair_rows_by_checkpoint[latest],
            bootstrap_samples=2000,
            seed=args.seed + 300_000 + task_index,
        )
        best_vs_random = paired_gap_stats(
            pair_rows_by_checkpoint[best],
            pair_rows_by_checkpoint[random_checkpoint],
            bootstrap_samples=2000,
            seed=args.seed + 310_000 + task_index,
        )
        task_peaks[task] = {
            "best_trained_checkpoint": best,
            "best_trained_value": values[best],
            "random_value": values[random_checkpoint],
            "latest_value": values[latest],
            "best_vs_random": best_vs_random,
            "best_vs_latest": best_vs_latest,
            "vision_minus_shuffle_at_best": gap_index[
                (best, task, "paired_preference", "shuffled_image")
            ],
            "vision_minus_paired_image_margin_at_best": gap_index[
                (best, task, "mean_margin", "paired_counterfactual_image")
            ],
            "causally_validated_at_best": validated_signals[task][best],
            "transient_peak_supported": (
                best != latest and best_vs_latest["ci95_low"] > 0
            ),
        }
    decisions = {
        "status": "valid",
        "checkpoint_order": checkpoints,
        "task_teacher_forced_vision_onsets": onset,
        "task_paired_preference_peaks": task_peaks,
        "causally_validated_task_checkpoint_signals": validated_signals,
        "latest_vision_minus_blind_paired_preference": gap_index[
            (latest, "overall", "paired_preference", "blind")
        ],
        "latest_vision_minus_blind_margin": gap_index[
            (latest, "overall", "mean_margin", "blind")
        ],
        "latest_vision_minus_paired_image_margin": gap_index[
            (latest, "overall", "mean_margin", "paired_counterfactual_image")
        ],
        "latest_background_shift_margin": gap_index[
            (latest, "overall", "mean_margin", "background_matched_aux")
        ],
        "internal_visual_signal_supported": any(
            supported
            for task in validated_signals.values()
            for supported in task.values()
        ),
        "latest_strict_paired_preference": statistics.fmean(
            row["score"]
            for row in pair_metric_rows(
                grouped[(latest, "vision", "overall")], "paired_preference"
            )
        ),
        "latest_strict_signal_vs_shuffle_supported": gap_index[
            (latest, "overall", "paired_preference", "shuffled_image")
        ]["ci95_low"]
        > 0,
        "interpretation_limits": [
            "paired bootstrap resamples complete minimal pairs",
            "the matched random projector is a seed-0 configuration control, not a saved historical step-0 tensor",
            "the background-matched auxiliary is diagnostic-only and never used for training",
            "the final odd evaluation half remains unscored",
        ],
    }

    def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    write_csv(
        args.out / "preference_curve.csv",
        curve_rows,
        [
            "checkpoint",
            "optimizer_steps",
            "examples_seen",
            "task",
            "condition",
            "samples",
            "pairs",
            "sample_preference_accuracy",
            "paired_preference_accuracy",
            "mean_correct_margin",
            "median_correct_margin",
            "mean_correct_token_nll",
            "mean_counterfactual_token_nll",
        ],
    )
    gap_fields = [
        "checkpoint",
        "examples_seen",
        "task",
        "metric",
        "a",
        "b",
        "denominator",
        "mean_a",
        "mean_b",
        "mean_gap",
        "ci95_low",
        "ci95_high",
        "a_only_better",
        "b_only_better",
        "equal",
        "bootstrap_samples",
        "seed",
    ]
    write_csv(args.out / "preference_intervals.csv", interval_rows, gap_fields)
    write_csv(args.out / "preference_gaps.csv", gap_rows, gap_fields)
    write_csv(
        args.out / "preference_checkpoint_deltas.csv",
        checkpoint_delta_rows,
        [
            "previous_checkpoint",
            "checkpoint",
            "previous_examples_seen",
            "examples_seen",
            "task",
            "condition",
            "metric",
            "denominator",
            "mean_a",
            "mean_b",
            "mean_gap",
            "ci95_low",
            "ci95_high",
            "bootstrap_samples",
            "seed",
        ],
    )
    payload = {
        "format_version": "paired-preference-analysis-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(args.run.resolve()),
        "run_summary_sha256": sha256(args.run / "SUMMARY.json"),
        "raw_records_sha256": sha256(args.run / "preference_records.jsonl"),
        "bootstrap_seed": args.seed,
        "bootstrap_samples": 2000,
        "intervals": interval_rows,
        "gaps": gap_rows,
        "checkpoint_deltas": checkpoint_delta_rows,
    }
    (args.out / "STATISTICS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "DECISIONS.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decisions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
