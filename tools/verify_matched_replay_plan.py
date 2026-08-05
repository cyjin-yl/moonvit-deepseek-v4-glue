#!/usr/bin/env python3
"""在加载 GPU 模型前验证 fixed-budget matched replay 的完整采样账本。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from replay_order import transform_replay_batches
from tools_common import load_records
from train_shape_adaptation import read_training_order_window, validate_fixed_training_budget


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def counts(records: list[dict], batches: list[list[int]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(records[index]["task"])
                for batch in batches
                for index in batch
            ).items()
        )
    )


def verify(config: dict) -> dict:
    training = config["training"]
    budget = validate_fixed_training_budget(
        training,
        initial_step=int(training["initial_step"]),
        final_step=int(training["steps"]),
        batch_size=int(training["batch_size"]),
    )
    records = load_records(Path(config["dataset"]["train_data"]))
    index_by_id = {str(record["id"]): index for index, record in enumerate(records)}
    configured_tasks = [str(task) for task in config["dataset"]["tasks"]]
    records = [record for record in records if str(record["task"]) in set(configured_tasks)]
    if len(records) != int(config["dataset"]["expected_train_records"]):
        raise ValueError("matched replay train denominator drifted")
    if len(index_by_id) != len(records):
        # 当前冻结 manifest 正好只含六任务；若未来扩展，需重新构建局部索引。
        index_by_id = {str(record["id"]): index for index, record in enumerate(records)}

    source_path = Path(training["source_training_order"])
    source_hash = sha256(source_path)
    if source_hash != str(training["source_training_order_sha256"]):
        raise ValueError("matched replay source order SHA-256 mismatch")
    ordinary, ordinary_source = read_training_order_window(
        source_path,
        records,
        start_step=int(training["initial_step"]),
        end_step=int(training["steps"]),
        batch_size=int(training["batch_size"]),
        expected_sha256=training["source_training_order_sha256"],
    )
    policy = config["arms"]["fixed_replay"]["replay_policy"]
    history, history_source = read_training_order_window(
        source_path,
        records,
        start_step=int(policy["history_start_step_exclusive"]),
        end_step=int(policy["history_end_step_inclusive"]),
        batch_size=int(training["batch_size"]),
        expected_sha256=policy["history_training_order_sha256"],
    )
    fixed, replay = transform_replay_batches(
        records,
        source_batches=ordinary,
        history_batches=history,
        tasks=configured_tasks,
        replay_tasks=[str(task) for task in policy["tasks"]],
        pairs_per_task_per_window=int(policy["pairs_per_task_per_window"]),
        window_batch_count=int(policy["window_steps"]),
        seed=int(policy["seed"]),
    )
    ordinary_counts = counts(records, ordinary)
    fixed_counts = counts(records, fixed)
    expected_counts = {
        str(task): int(value)
        for task, value in config["matched_replay"]["fixed_policy_expected_counts"].items()
    }
    batch_size = int(training["batch_size"])
    checks = {
        "fixed_step_budget": len(ordinary) == len(fixed) == budget["steps"],
        "fixed_example_budget": sum(map(len, ordinary)) == sum(map(len, fixed)) == budget["examples"],
        "batch_size_unchanged": all(len(batch) == batch_size for batch in [*ordinary, *fixed]),
        "ordinary_balanced": len(set(ordinary_counts.values())) == 1,
        "fixed_counts_exact": fixed_counts == expected_counts,
        "examples_preserved": bool(replay["examples_preserved"]),
        "complete_pair_replacements": bool(replay["all_replacements_are_complete_pairs"]),
        "added_removed_cardinality": len(replay["added_ids"]) == len(replay["removed_ids"]),
        "no_extra_optimizer_steps": int(config["matched_replay"]["fixed_training_budget"]["extra_optimizer_steps_allowed"]) == 0,
        "no_extra_training_examples": int(config["matched_replay"]["fixed_training_budget"]["extra_training_examples_allowed"]) == 0,
    }
    if not all(checks.values()):
        raise ValueError(f"matched replay plan checks failed: {checks}")

    base = Path(config["base_projector"])
    base_projector_hash = sha256(base / "projector.safetensors")
    optimizer_hash = sha256(base / "training_state.pt")
    for arm_name in ("ordinary_continuation", "fixed_replay"):
        arm = config["arms"][arm_name]
        if base_projector_hash != arm["expected_base_projector_sha256"]:
            raise ValueError(f"{arm_name} base projector SHA-256 mismatch")
        if optimizer_hash != arm["expected_optimizer_sha256"]:
            raise ValueError(f"{arm_name} optimizer SHA-256 mismatch")
    return {
        "status": "valid",
        "format_version": "fixed-budget-matched-replay-plan-v1",
        "checks": checks,
        "budget_per_arm": budget,
        "ordinary_records_by_task": ordinary_counts,
        "fixed_records_by_task": fixed_counts,
        "base": {
            "projector_sha256": base_projector_hash,
            "optimizer_sha256": optimizer_hash,
            "optimizer_step": config["arms"]["ordinary_continuation"]["expected_optimizer_step"],
        },
        "ordinary_source": ordinary_source,
        "history_source": history_source,
        "replay": replay,
        "final_half_scored": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite matched replay plan: {args.out}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = verify(config)
    result["config"] = str(args.config)
    result["config_sha256"] = sha256(args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
