#!/usr/bin/env python3
"""按预注册 paired-bootstrap 规则机械选择 matched replay 任务。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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


def select_trigger_tasks(
    contrasts: list[dict], *, minimum_drop: float, max_tasks: int
) -> list[str]:
    """保留绝对下降达阈值、且 paired CI 上界仍小于零的任务。"""

    if minimum_drop <= 0 or max_tasks <= 0:
        raise ValueError("trigger threshold and maximum task count must be positive")
    eligible = [
        row
        for row in contrasts
        if str(row["task"]) != "overall"
        and float(row["mean_gap"]) <= -minimum_drop
        and float(row["ci95_high"]) < 0
    ]
    eligible.sort(key=lambda row: (float(row["mean_gap"]), str(row["task"])))
    return [str(row["task"]) for row in eligible[:max_tasks]]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--reference-state", required=True)
    parser.add_argument("--current-state", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--minimum-drop", type=float, default=0.10)
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite replay sentinel: {args.out}")

    source = args.evaluation / "preference_records.jsonl"
    rows = read_jsonl(source)
    states = {str(row["state"]) for row in rows}
    required_states = {args.reference_state, args.current_state}
    if not required_states <= states:
        raise ValueError(f"replay sentinel states are absent: {sorted(required_states - states)}")
    tasks = sorted({str(row["task"]) for row in rows})
    contrasts = []
    for index, task in enumerate(["overall", *tasks]):
        def cell(state: str) -> list[dict]:
            selected = [
                row
                for row in rows
                if str(row["state"]) == state
                and str(row["condition"]) == "vision"
                and (task == "overall" or str(row["task"]) == task)
            ]
            return pair_metric_rows(selected, "paired_preference")

        stats = paired_gap_stats(
            cell(args.current_state),
            cell(args.reference_state),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + index,
        )
        contrasts.append(
            {
                "task": task,
                "current_state": args.current_state,
                "reference_state": args.reference_state,
                **stats,
            }
        )
    selected = select_trigger_tasks(
        contrasts,
        minimum_drop=args.minimum_drop,
        max_tasks=args.max_tasks,
    )
    output = {
        "status": "valid",
        "format_version": "matched-replay-sentinel-v1",
        "reference_state": args.reference_state,
        "current_state": args.current_state,
        "decision_rule": {
            "metric": "vision paired_preference",
            "minimum_absolute_drop": args.minimum_drop,
            "paired_ci_requirement": "current-minus-reference ci95_high < 0",
            "maximum_tasks": args.max_tasks,
            "ranking": "most negative mean gap, then task name",
        },
        "trigger_tasks": selected,
        "action": "replay" if selected else "base_distribution",
        "contrasts": contrasts,
        "source": {
            "preference_records": str(source),
            "preference_records_sha256": sha256(source),
        },
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.seed},
        "final_half_scored": False,
    }
    args.out.mkdir(parents=True)
    write_csv(args.out / "contrasts.csv", contrasts)
    (args.out / "DECISION.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
