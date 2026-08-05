#!/usr/bin/env python3
"""从 sentinel 决策机械生成剩余固定预算的 triggered replay 配置。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_triggered_config(
    base: dict,
    decision: dict,
    ordinary_summary: dict,
    *,
    ordinary_run: Path,
    decision_path: Path,
    decision_sha256: str,
) -> dict:
    """冻结 step75 起点、触发任务和剩余 25-step 预算。"""

    if decision.get("status") != "valid":
        raise ValueError("trigger decision is not valid")
    if ordinary_summary.get("status") != "valid":
        raise ValueError("ordinary continuation is not valid")
    trigger_tasks = [str(task) for task in decision["trigger_tasks"]]
    configured_tasks = {str(task) for task in base["dataset"]["tasks"]}
    maximum = int(base["matched_replay"]["trigger_rule"]["maximum_tasks"])
    if len(set(trigger_tasks)) != len(trigger_tasks):
        raise ValueError("trigger decision repeats tasks")
    if not set(trigger_tasks) <= configured_tasks or len(trigger_tasks) > maximum:
        raise ValueError("trigger decision violates the preregistered task universe")
    key = "step-000075"
    if key not in ordinary_summary["checkpoints"]:
        raise ValueError("ordinary continuation is missing the step75 checkpoint")
    manifest = ordinary_summary["checkpoints"][key]
    files = manifest["files"]
    projector_hash = files["projector.safetensors"]["sha256"]
    optimizer_hash = files["training_state.pt"]["sha256"]
    checkpoint = ordinary_run / "checkpoints" / key

    derived = copy.deepcopy(base)
    derived["run_id"] = f"{base['run_id']}-triggered"
    derived["training_format_version"] = "fixed-budget-triggered-replay-v1"
    derived["base_projector"] = str(checkpoint)
    derived["training"].update(
        {
            "initial_step": 75,
            "steps": 100,
            "checkpoint_steps": [75, 100],
            "fixed_continuation_steps": 25,
            "fixed_continuation_examples": 600,
        }
    )
    replay_policy = copy.deepcopy(
        base["matched_replay"]["triggered_policy_template"]
    )
    replay_policy["tasks"] = trigger_tasks
    derived["arms"] = {
        "triggered_replay": {
            "kind": "projector",
            "resume_optimizer": True,
            "expected_base_projector_sha256": projector_hash,
            "expected_optimizer_sha256": optimizer_hash,
            "expected_optimizer_step": 75,
            "replay_policy": replay_policy,
            "trigger_decision": {
                "path": str(decision_path),
                "sha256": decision_sha256,
                "reference_state": decision["reference_state"],
                "current_state": decision["current_state"],
                "tasks": trigger_tasks,
            },
        }
    }
    derived["trigger_derivation"] = {
        "ordinary_run": str(ordinary_run),
        "ordinary_step": 75,
        "ordinary_projector_sha256": projector_hash,
        "ordinary_optimizer_sha256": optimizer_hash,
        "decision_path": str(decision_path),
        "decision_sha256": decision_sha256,
        "remaining_steps": 25,
        "remaining_examples": 600,
        "extra_training_examples": 0,
    }
    return derived


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--ordinary-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite triggered replay config: {args.out}")
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    summary = json.loads(
        (args.ordinary_run / "SUMMARY.json").read_text(encoding="utf-8")
    )
    derived = build_triggered_config(
        base,
        decision,
        summary,
        ordinary_run=args.ordinary_run,
        decision_path=args.decision,
        decision_sha256=sha256(args.decision),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(derived, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(derived, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
