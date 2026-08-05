#!/usr/bin/env python3
"""汇总 Tiny/Medium sentinel 重复计时并求固定开销下的最小间隔。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def achieved_overhead(
    evaluation_seconds: float, training_step_seconds: float, interval_steps: int
) -> float:
    if evaluation_seconds <= 0 or training_step_seconds <= 0 or interval_steps <= 0:
        raise ValueError("overhead inputs must be positive")
    return evaluation_seconds / (
        interval_steps * training_step_seconds + evaluation_seconds
    )


def minimum_interval_steps(
    *,
    evaluation_seconds: float,
    training_step_seconds: float,
    maximum_overhead: float,
) -> int:
    """解 t_eval / (K*t_step + t_eval) <= overhead。"""

    if not 0 < maximum_overhead < 1:
        raise ValueError("maximum overhead must be between zero and one")
    if evaluation_seconds <= 0 or training_step_seconds <= 0:
        raise ValueError("timing inputs must be positive")
    required = evaluation_seconds * (1.0 - maximum_overhead) / (
        maximum_overhead * training_step_seconds
    )
    return max(1, math.ceil(required - 1e-12))


def next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("power-of-two rounding needs a positive integer")
    return 1 << (value - 1).bit_length()


def validate_repeat_summaries(
    summaries: list[dict],
    *,
    expected_repeats: int,
    expected_git_sha: str,
    expected_records_per_state: int,
) -> None:
    if len(summaries) != expected_repeats:
        raise ValueError("timing repeat count drifted")
    preference_hashes = set()
    for summary in summaries:
        if summary.get("status") != "valid" or summary.get("final_half_scored") is not False:
            raise ValueError("timing summary is invalid")
        if str(summary["metadata"]["git_sha"]) != expected_git_sha:
            raise ValueError("timing source Git SHA drifted")
        if int(summary["states"]) != 2 or summary["teacher_conditions"] != ["vision"]:
            raise ValueError("timing state or condition contract drifted")
        if summary.get("generation_skipped") is not True or int(summary["generation_rows"]) != 0:
            raise ValueError("timing run unexpectedly executed generation")
        if int(summary["teacher_forced_records_per_cell"]) != expected_records_per_state:
            raise ValueError("timing teacher denominator drifted")
        if int(summary["preference_rows"]) != expected_records_per_state * 2:
            raise ValueError("timing preference row count drifted")
        preference_hashes.add(
            str(summary["files"]["preference_records.jsonl"]["sha256"])
        )
    if len(preference_hashes) != 1:
        raise ValueError("timing preference hash drifted across repeats")


def verify_declared_files(run: Path, summary: dict) -> None:
    for name, metadata in summary["files"].items():
        path = run / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(metadata["bytes"]) or sha256(path) != str(
            metadata["sha256"]
        ):
            raise ValueError(f"timing artifact drifted: {path}")


def summarize_profile(
    profile: str,
    pairs_per_task: int,
    runs: list[Path],
    *,
    expected_repeats: int,
    expected_git_sha: str,
    tasks: int,
) -> dict:
    summaries = [read_json(run / "SUMMARY.json") for run in runs]
    expected_records = pairs_per_task * 2 * tasks
    validate_repeat_summaries(
        summaries,
        expected_repeats=expected_repeats,
        expected_git_sha=expected_git_sha,
        expected_records_per_state=expected_records,
    )
    for run, summary in zip(runs, summaries, strict=True):
        verify_declared_files(run, summary)
    fields = (
        "wall_seconds",
        "setup_seconds",
        "state_load_seconds",
        "teacher_forced_seconds",
        "generation_seconds",
    )
    timings = {
        field: [float(summary["metadata"][field]) for summary in summaries]
        for field in fields
    }
    return {
        "profile": profile,
        "pairs_per_task": pairs_per_task,
        "teacher_records_per_state": expected_records,
        "states": 2,
        "preference_rows_per_repeat": expected_records * 2,
        "repeats": len(runs),
        **{
            f"{field}_{statistic}": value
            for field, values in timings.items()
            for statistic, value in (
                ("median", statistics.median(values)),
                ("min", min(values)),
                ("max", max(values)),
            )
        },
        "peak_gpu_memory_bytes": max(
            int(summary["metadata"]["peak_gpu_memory_bytes"])
            for summary in summaries
        ),
        "preference_records_sha256": summaries[0]["files"][
            "preference_records.jsonl"
        ]["sha256"],
        "run_summaries": [str(run / "SUMMARY.json") for run in runs],
        "run_summary_sha256": [sha256(run / "SUMMARY.json") for run in runs],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--power", required=True, type=Path)
    parser.add_argument("--tiny-run", required=True, action="append", type=Path)
    parser.add_argument("--medium-run", required=True, action="append", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite sentinel timing analysis: {args.out}")
    config = read_json(args.config)
    power = read_json(args.power)
    if power.get("status") != "valid" or power.get("final_half_scored") is not False:
        raise ValueError("sentinel power decision is invalid")
    tiny_pairs = int(power["recommended_tiny_pairs_per_task"])
    medium_pairs = int(power["recommended_medium_pairs_per_task"])
    protocol = config["timing_protocol"]
    expected_repeats = int(protocol["repeats"])
    expected_git_sha = str(power["source_git_sha"])
    tasks = len(config["tasks"])
    profiles = [
        summarize_profile(
            "tiny",
            tiny_pairs,
            args.tiny_run,
            expected_repeats=expected_repeats,
            expected_git_sha=expected_git_sha,
            tasks=tasks,
        ),
        summarize_profile(
            "medium",
            medium_pairs,
            args.medium_run,
            expected_repeats=expected_repeats,
            expected_git_sha=expected_git_sha,
            tasks=tasks,
        ),
    ]
    training_step_seconds = float(protocol["training_step_seconds_reference"])
    overhead_rows = []
    for profile in profiles:
        for time_basis, field in (
            ("model_resident_teacher", "teacher_forced_seconds_median"),
            ("separate_process_end_to_end", "wall_seconds_median"),
        ):
            evaluation_seconds = float(profile[field])
            for maximum_overhead in [float(value) for value in protocol["overhead_targets"]]:
                minimum = minimum_interval_steps(
                    evaluation_seconds=evaluation_seconds,
                    training_step_seconds=training_step_seconds,
                    maximum_overhead=maximum_overhead,
                )
                rounded = next_power_of_two(minimum)
                overhead_rows.append(
                    {
                        "profile": profile["profile"],
                        "time_basis": time_basis,
                        "evaluation_seconds": evaluation_seconds,
                        "training_step_seconds": training_step_seconds,
                        "maximum_overhead": maximum_overhead,
                        "minimum_interval_steps": minimum,
                        "rounded_power_of_two_interval_steps": rounded,
                        "overhead_at_minimum": achieved_overhead(
                            evaluation_seconds, training_step_seconds, minimum
                        ),
                        "overhead_at_rounded_interval": achieved_overhead(
                            evaluation_seconds, training_step_seconds, rounded
                        ),
                    }
                )
    tiny_model_resident = [
        row
        for row in overhead_rows
        if row["profile"] == "tiny" and row["time_basis"] == "model_resident_teacher"
    ]
    decision = {
        "status": "valid",
        "format_version": "sentinel-timing-analysis-v1",
        "power_source_git_sha": expected_git_sha,
        "recommended_profile": "tiny",
        "tiny_pairs_per_task": tiny_pairs,
        "medium_pairs_per_task": medium_pairs,
        "profiles": profiles,
        "overhead_contract": {
            "formula": "t_eval / (K * t_step + t_eval) <= maximum_overhead",
            "training_step_seconds_reference": training_step_seconds,
            "model_resident_tiny": {
                str(row["maximum_overhead"]): {
                    "minimum_interval_steps": row["minimum_interval_steps"],
                    "rounded_power_of_two_interval_steps": row[
                        "rounded_power_of_two_interval_steps"
                    ],
                }
                for row in tiny_model_resident
            },
            "recompute_after_target_runtime_calibration": True,
        },
        "training_action": {
            "use_fixed_preventive_replay": True,
            "run_tiny_sentinel_as_sparse_checkpoint_audit": True,
            "run_tiny_sentinel_every_25_small_model_steps": False,
            "reason": "Tiny teacher cost is comparable to one 25-step V100 training window",
            "training_examples_changed": False,
            "training_steps_changed": False,
        },
        "sources": {
            "config": str(args.config),
            "config_sha256": sha256(args.config),
            "power": str(args.power),
            "power_sha256": sha256(args.power),
        },
        "final_half_scored": False,
    }
    args.out.mkdir(parents=True)
    write_csv(args.out / "timing_profiles.csv", profiles)
    write_csv(args.out / "overhead_budget.csv", overhead_rows)
    (args.out / "DECISIONS.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
