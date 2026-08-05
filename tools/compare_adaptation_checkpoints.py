#!/usr/bin/env python3
"""跨独立评测 run 比较同一适配臂的两个 checkpoint。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from analyze_multitask_transfer import pair_metric_rows
from moonvit_glue.trajectory_analysis import paired_gap_stats


MODALITIES = {
    "preference": (
        "preference_records.jsonl",
        ("paired_preference", "sample_preference", "mean_margin"),
    ),
    "generation": (
        "generation_records.jsonl",
        ("generation_paired", "generation_sample", "prediction_flip"),
    ),
}


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


def _cell(
    rows: list[dict], *, state: str, condition: str, task: str
) -> list[dict]:
    selected = [
        row
        for row in rows
        if str(row["state"]) == state
        and str(row["condition"]) == condition
        and (task == "overall" or str(row["task"]) == task)
    ]
    if not selected:
        raise ValueError(
            f"empty evaluation cell: state={state}, condition={condition}, task={task}"
        )
    return selected


def paired_run_metric_rows(
    early_rows: list[dict],
    late_rows: list[dict],
    *,
    early_state: str,
    late_state: str,
    condition: str,
    task: str,
    metric: str,
) -> tuple[list[dict], list[dict]]:
    """严格校验逐样本身份后，返回两个 run 的完整 minimal-pair 指标。"""

    early = _cell(
        early_rows, state=early_state, condition=condition, task=task
    )
    late = _cell(late_rows, state=late_state, condition=condition, task=task)
    key = lambda row: (
        str(row["id"]),
        str(row["pair_id"]),
        str(row.get("pair_variant") or ""),
        str(row["task"]),
        str(row["condition"]),
    )
    if {key(row) for row in early} != {key(row) for row in late}:
        raise ValueError("cross-run comparison requires identical sample identities")
    return pair_metric_rows(early, metric), pair_metric_rows(late, metric)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early-eval", required=True, type=Path)
    parser.add_argument("--late-eval", required=True, type=Path)
    parser.add_argument("--early-lora-state", required=True)
    parser.add_argument("--late-lora-state", required=True)
    parser.add_argument("--early-projector-state", required=True)
    parser.add_argument("--late-projector-state", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint comparison: {args.out}")
    args.out.mkdir(parents=True)

    arm_states = {
        "lora": (args.early_lora_state, args.late_lora_state),
        "projector": (args.early_projector_state, args.late_projector_state),
    }
    rows_by_modality = {}
    for modality, (filename, _) in MODALITIES.items():
        early_rows = read_jsonl(args.early_eval / filename)
        late_rows = read_jsonl(args.late_eval / filename)
        early_tasks = sorted({str(row["task"]) for row in early_rows})
        late_tasks = sorted({str(row["task"]) for row in late_rows})
        if early_tasks != late_tasks:
            raise ValueError(f"{modality} task sets differ across runs")
        early_conditions = sorted({str(row["condition"]) for row in early_rows})
        late_conditions = sorted({str(row["condition"]) for row in late_rows})
        if early_conditions != late_conditions:
            raise ValueError(f"{modality} condition sets differ across runs")
        rows_by_modality[modality] = (
            early_rows,
            late_rows,
            early_tasks,
            early_conditions,
        )

    contrasts = []
    counter = 0
    for arm, (early_state, late_state) in arm_states.items():
        for modality, (_, metrics) in MODALITIES.items():
            early_rows, late_rows, tasks, conditions = rows_by_modality[modality]
            for condition in conditions:
                for task in ["overall", *tasks]:
                    for metric in metrics:
                        counter += 1
                        early_pairs, late_pairs = paired_run_metric_rows(
                            early_rows,
                            late_rows,
                            early_state=early_state,
                            late_state=late_state,
                            condition=condition,
                            task=task,
                            metric=metric,
                        )
                        stats = paired_gap_stats(
                            late_pairs,
                            early_pairs,
                            bootstrap_samples=args.bootstrap_samples,
                            seed=args.seed + counter,
                        )
                        contrasts.append(
                            {
                                "family": "late_minus_early_cross_run",
                                "arm": arm,
                                "modality": modality,
                                "metric": metric,
                                "late_state": late_state,
                                "early_state": early_state,
                                "condition": condition,
                                "task": task,
                                **stats,
                            }
                        )

    index = {
        (row["arm"], row["modality"], row["metric"], row["condition"], row["task"]): row
        for row in contrasts
    }
    tasks = rows_by_modality["preference"][2]
    decisions = {"status": "valid", "arms": {}}
    for arm in arm_states:
        decisions["arms"][arm] = {
            task: {
                "preference_late_minus_early": index[
                    (arm, "preference", "paired_preference", "vision", task)
                ],
                "generation_late_minus_early": index[
                    (arm, "generation", "generation_paired", "vision", task)
                ],
            }
            for task in tasks
        }

    contrasts_path = args.out / "checkpoint_contrasts.csv"
    decisions_path = args.out / "DECISIONS.json"
    write_csv(contrasts_path, contrasts)
    decisions_path.write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    source_files = {
        "early": {
            filename: sha256(args.early_eval / filename)
            for filename, _ in MODALITIES.values()
        },
        "late": {
            filename: sha256(args.late_eval / filename)
            for filename, _ in MODALITIES.values()
        },
    }
    summary = {
        "status": "valid",
        "format_version": "cross-run-adaptation-checkpoint-comparison-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm_states": arm_states,
        "tasks": tasks,
        "contrasts": len(contrasts),
        "bootstrap_samples": args.bootstrap_samples,
        "source_files": source_files,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (contrasts_path, decisions_path)
        },
        "final_half_scored": False,
    }
    (args.out / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
