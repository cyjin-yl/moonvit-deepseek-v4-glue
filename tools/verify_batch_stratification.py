#!/usr/bin/env python3
"""验证分层平衡批次与全局随机批次只在批次构造上不同。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from tools_common import load_records


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


def summarize_order_rows(
    rows: list[dict],
    *,
    task_by_id: dict[str, str],
    tasks: list[str],
    batch_size: int,
) -> dict:
    """汇总训练顺序，并显式计数每个批次是否严格按任务平衡。"""
    expected_quota = batch_size // len(tasks)
    flattened: list[str] = []
    task_totals = Counter()
    balanced_batches = 0
    max_task_count = 0
    batch_task_counts = []
    for expected_step, row in enumerate(rows, start=1):
        if int(row["step"]) != expected_step:
            raise ValueError("training order steps are not contiguous from one")
        ids = [str(sample_id) for sample_id in row["ids"]]
        if len(ids) != batch_size:
            raise ValueError("training order batch size drifted")
        missing = [sample_id for sample_id in ids if sample_id not in task_by_id]
        if missing:
            raise ValueError(f"training order contains unknown IDs: {missing[:3]}")
        counts = Counter(task_by_id[sample_id] for sample_id in ids)
        counts_row = {task: int(counts[task]) for task in tasks}
        if all(counts_row[task] == expected_quota for task in tasks):
            balanced_batches += 1
        max_task_count = max(max_task_count, *counts_row.values())
        task_totals.update(counts)
        batch_task_counts.append({"step": expected_step, **counts_row})
        flattened.extend(ids)
    return {
        "steps": len(rows),
        "records": len(flattened),
        "unique_ids": len(set(flattened)),
        "duplicate_records": len(flattened) - len(set(flattened)),
        "balanced_batches": balanced_batches,
        "unbalanced_batches": len(rows) - balanced_batches,
        "max_task_count_in_batch": max_task_count,
        "task_totals": {task: int(task_totals[task]) for task in tasks},
        "ordered_ids": flattened,
        "batch_task_counts": batch_task_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stratified-run", required=True, type=Path)
    parser.add_argument("--global-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite verification: {args.out}")

    experiment_config = json.loads(args.config.read_text(encoding="utf-8"))
    stratified_summary_path = args.stratified_run / "SUMMARY.json"
    global_summary_path = args.global_run / "SUMMARY.json"
    stratified_summary = json.loads(stratified_summary_path.read_text(encoding="utf-8"))
    global_summary = json.loads(global_summary_path.read_text(encoding="utf-8"))
    if (
        stratified_summary.get("status") != "valid"
        or global_summary.get("status") != "valid"
    ):
        raise ValueError("matched training summaries must both be valid")

    tasks = [str(task) for task in experiment_config["dataset"]["tasks"]]
    train_rows = [
        row
        for row in load_records(Path(experiment_config["dataset"]["train_data"]))
        if str(row.get("task")) in tasks
    ]
    task_by_id = {str(row["id"]): str(row["task"]) for row in train_rows}
    if len(task_by_id) != int(experiment_config["dataset"]["expected_train_records"]):
        raise ValueError("training manifest ID denominator drifted")
    batch_size = int(experiment_config["training"]["batch_size"])
    stratified_rows = read_jsonl(args.stratified_run / "training_order.jsonl")
    global_rows = read_jsonl(args.global_run / "training_order.jsonl")
    stratified_order = summarize_order_rows(
        stratified_rows,
        task_by_id=task_by_id,
        tasks=tasks,
        batch_size=batch_size,
    )
    global_order = summarize_order_rows(
        global_rows,
        task_by_id=task_by_id,
        tasks=tasks,
        batch_size=batch_size,
    )

    expected_records = int(experiment_config["dataset"]["expected_train_records"])
    expected_steps = int(experiment_config["training"]["steps"])
    expected_task_total = expected_records // len(tasks)
    checks = {
        "both_runs_valid": stratified_summary.get("status") == global_summary.get("status") == "valid",
        "same_base_projector_sha256": stratified_summary["base_projector_sha256"]
        == global_summary["base_projector_sha256"],
        "same_initial_projector_tensors": stratified_summary["checkpoints"]["step-000000"]["weights_tensor_sha256"]
        == global_summary["checkpoints"]["step-000000"]["weights_tensor_sha256"],
        "same_restored_optimizer_source": stratified_summary["optimizer_resume"]["source_sha256"]
        == global_summary["optimizer_resume"]["source_sha256"],
        "optimizer_restored_in_both": bool(stratified_summary["optimizer_resume"]["restored"])
        and bool(global_summary["optimizer_resume"]["restored"]),
        "same_seed": int(json.loads((args.stratified_run / "CONFIG.json").read_text(encoding="utf-8"))["seed"])
        == int(json.loads((args.global_run / "CONFIG.json").read_text(encoding="utf-8"))["seed"]),
        "same_training_hyperparameters": all(
            json.loads((args.stratified_run / "CONFIG.json").read_text(encoding="utf-8"))["training"][key]
            == json.loads((args.global_run / "CONFIG.json").read_text(encoding="utf-8"))["training"][key]
            for key in ("steps", "batch_size", "learning_rate", "weight_decay", "gradient_clip")
        ),
        "same_record_set": set(stratified_order["ordered_ids"])
        == set(global_order["ordered_ids"])
        == set(task_by_id),
        "each_record_used_once": stratified_order["records"]
        == stratified_order["unique_ids"]
        == global_order["records"]
        == global_order["unique_ids"]
        == expected_records,
        "same_task_totals": stratified_order["task_totals"]
        == global_order["task_totals"]
        == {task: expected_task_total for task in tasks},
        "same_step_count": stratified_order["steps"]
        == global_order["steps"]
        == expected_steps,
        "stratified_every_batch_balanced": stratified_order["balanced_batches"]
        == expected_steps,
        "global_has_unbalanced_batches": global_order["unbalanced_batches"] > 0,
        "orders_differ": stratified_order["ordered_ids"] != global_order["ordered_ids"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"matched batch-order verification failed: {failed}")

    def compact(summary: dict) -> dict:
        return {key: value for key, value in summary.items() if key not in {"ordered_ids", "batch_task_counts"}}

    output = {
        "status": "valid",
        "format_version": "matched-batch-order-verification-v1",
        "checks": checks,
        "stratified_order": compact(stratified_order),
        "global_order": compact(global_order),
        "batch_task_counts": {
            "stratified": stratified_order["batch_task_counts"],
            "global": global_order["batch_task_counts"],
        },
        "sources": {
            "config": {"path": str(args.config), "sha256": sha256(args.config)},
            "stratified_summary": {"path": str(stratified_summary_path), "sha256": sha256(stratified_summary_path)},
            "global_summary": {"path": str(global_summary_path), "sha256": sha256(global_summary_path)},
            "stratified_order_sha256": sha256(args.stratified_run / "training_order.jsonl"),
            "global_order_sha256": sha256(args.global_run / "training_order.jsonl"),
        },
    }
    args.out.mkdir(parents=True)
    (args.out / "VERIFICATION.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
