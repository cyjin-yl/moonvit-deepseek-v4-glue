#!/usr/bin/env python3
"""独立重验固定 4k Qwen3B 训练历史、checkpoint 清单与最终状态。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


EXPECTED_CHECKPOINT_FILES = {
    "history.json",
    "projector.safetensors",
    "projector_bf16.safetensors",
    "projector_config.json",
    "training_state.pt",
}


def git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def git_tracked_worktree_clean() -> bool:
    unstaged = subprocess.run(["git", "diff", "--quiet", "--"], check=False)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--"], check=False
    )
    return unstaged.returncode == 0 and staged.returncode == 0


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_training_history(
    history: list[dict[str, Any]],
    supervision: list[dict[str, Any]],
    order_records: list[dict[str, Any]],
    *,
    gradient_accumulation: int,
) -> dict[str, Any]:
    """逐 step 重建样本区间、ID hash 与 answer-token 累计。"""

    if len(supervision) != len(order_records):
        raise ValueError("supervision and training-order counts differ")
    answer_tokens_seen = 0
    losses = []
    for zero_based_step, row in enumerate(history):
        one_based_step = zero_based_step + 1
        start = zero_based_step * gradient_accumulation
        end = start + gradient_accumulation
        if int(row.get("step", -1)) != one_based_step:
            raise ValueError("training history steps are not contiguous")
        if (
            int(row.get("optimizer_steps", -1)) != one_based_step
            or int(row.get("batch_start_index", -1)) != start
            or int(row.get("batch_end_index", -1)) != end - 1
            or int(row.get("examples_seen", -1)) != end
        ):
            raise ValueError(f"training history cursor differs at step {one_based_step}")
        expected_ids = [str(item["id"]) for item in order_records[start:end]]
        if row.get("batch_record_ids_sha256") != canonical_sha256(expected_ids):
            raise ValueError(f"training batch ID hash differs at step {one_based_step}")
        supervision_ids = [str(item["id"]) for item in supervision[start:end]]
        if supervision_ids != expected_ids:
            raise ValueError(f"supervision order differs at step {one_based_step}")
        answer_tokens_seen += sum(
            int(item["answer_tokens"]) for item in supervision[start:end]
        )
        if int(row.get("answer_tokens_seen", -1)) != answer_tokens_seen:
            raise ValueError(f"answer-token count differs at step {one_based_step}")
        loss = float(row["loss"])
        if not torch.isfinite(torch.tensor(loss)):
            raise ValueError(f"training loss is non-finite at step {one_based_step}")
        losses.append(loss)
    return {
        "optimizer_steps": len(history),
        "examples_seen": len(history) * gradient_accumulation,
        "answer_tokens_seen": answer_tokens_seen,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "loss_mean": sum(losses) / len(losses),
    }


def verify_checkpoint_inventory(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "CHECKPOINT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed = sorted(manifest["files"], key=lambda row: str(row["path"]))
    actual_paths = sorted(
        path for path in directory.iterdir() if path.name != manifest_path.name
    )
    if {path.name for path in actual_paths} != EXPECTED_CHECKPOINT_FILES:
        raise ValueError(f"checkpoint file inventory differs: {directory}")
    actual = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in actual_paths
    ]
    if listed != actual:
        raise ValueError(f"checkpoint hashes differ: {directory}")
    if int(manifest.get("file_count", -1)) != len(actual):
        raise ValueError(f"checkpoint file count differs: {directory}")
    if int(manifest.get("total_bytes", -1)) != sum(row["bytes"] for row in actual):
        raise ValueError(f"checkpoint byte count differs: {directory}")
    return {
        "directory": str(directory),
        "manifest": manifest,
        "manifest_file_sha256": sha256_file(manifest_path),
        "files": actual,
    }


def build_checkpoint_binding(
    *,
    contract: dict[str, Any],
    order: dict[str, Any],
    cache: dict[str, Any],
    contract_path: Path,
    order_path: Path,
    cache_manifest_path: Path,
    runner_git_sha: str,
) -> dict[str, Any]:
    """从冻结输入重建 checkpoint 身份；不复用预算摘要字段。"""

    return {
        "runner_git_sha": runner_git_sha,
        "contract_file_sha256": sha256_file(contract_path),
        "training_order_manifest_file_sha256": sha256_file(order_path),
        "training_order_manifest_sha256": order["manifest_sha256"],
        "training_order_records_sha256": order["records_sha256"],
        "feature_cache_manifest_file_sha256": sha256_file(cache_manifest_path),
        "feature_cache_records_sha256": cache["records_sha256"],
        "feature_cache_runner_git_sha": cache["git_sha"],
        "initial_projector_sha256": contract["canonical_projector"][
            "initialization_contract"
        ]["step0"]["weights_sha256"],
        "proxy_receiver_sha256": contract["qwen_proxy_receiver"]["buffer_sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--training-order-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--expected-runner-git-sha", required=True)
    parser.add_argument("--step0-projector", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.require_clean_git and not git_tracked_worktree_clean():
        raise RuntimeError("tracked Git worktree is dirty; verification is refused")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    order = json.loads(args.training_order_manifest.read_text(encoding="utf-8"))
    cache_manifest_path = args.feature_cache / "MANIFEST.json"
    cache = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    attempt = json.loads((args.run / "ATTEMPT.json").read_text(encoding="utf-8"))
    run_config = json.loads((args.run / "RUN_CONFIG.json").read_text(encoding="utf-8"))
    summary = json.loads((args.run / "SUMMARY.json").read_text(encoding="utf-8"))
    supervision_path = args.run / "SUPERVISION_RECORDS.jsonl"
    supervision = read_jsonl(supervision_path)
    history_path = args.run / "TRAINING_HISTORY.jsonl"
    history = read_jsonl(history_path)
    budget = contract["training_budget"]
    expected_steps = int(budget["optimizer_steps_checkpoints"][0])
    expected_examples = int(budget["examples_seen_checkpoints"][0])
    accumulation = int(budget["gradient_accumulation"])
    if not (
        attempt["formal_result_allowed"] is True
        and attempt["git_tracked_worktree_clean"] is True
        and attempt["git_sha"] == args.expected_runner_git_sha
        and run_config["formal_run"] is True
        and run_config["git_tracked_worktree_clean"] is True
        and summary["status"] == "valid"
        and summary["formal_training_complete"] is True
        and summary["runner_git_sha"] == args.expected_runner_git_sha
        and summary["capability_claim_allowed"] is False
    ):
        raise ValueError("formal training provenance or capability boundary differs")
    if (
        sha256_file(args.contract) != run_config["contract_file_sha256"]
        or sha256_file(args.training_order_manifest)
        != run_config["training_order_manifest_file_sha256"]
        or sha256_file(cache_manifest_path)
        != run_config["feature_cache_manifest_file_sha256"]
    ):
        raise ValueError("formal training input file binding differs")
    budget_binding = run_config["binding"]
    if (
        int(budget_binding["optimizer_steps"]) != expected_steps
        or int(budget_binding["examples_seen"]) != expected_examples
        or int(budget_binding["gradient_accumulation"]) != accumulation
    ):
        raise ValueError("formal training budget binding differs")
    checkpoint_binding = build_checkpoint_binding(
        contract=contract,
        order=order,
        cache=cache,
        contract_path=args.contract,
        order_path=args.training_order_manifest,
        cache_manifest_path=cache_manifest_path,
        runner_git_sha=args.expected_runner_git_sha,
    )
    if len(supervision) != expected_examples or sha256_file(supervision_path) != summary[
        "supervision_records_sha256"
    ]:
        raise ValueError("formal supervision records differ")
    history_result = validate_training_history(
        history,
        supervision,
        order["records"],
        gradient_accumulation=accumulation,
    )
    expected_summary = {
        "optimizer_steps": expected_steps,
        "examples_seen": expected_examples,
        "answer_tokens_seen": history_result["answer_tokens_seen"],
        "loss_first": history_result["loss_first"],
        "loss_last": history_result["loss_last"],
        "loss_min": history_result["loss_min"],
        "loss_mean": history_result["loss_mean"],
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"formal summary differs from history: {key}")
    if int(summary["peak_gpu_memory_bytes"]) >= 32 * 1024**3:
        raise ValueError("formal training peak memory exceeds the V100 capacity")

    checkpoint_steps = list(range(100, expected_steps + 1, 100))
    if sorted(path.name for path in (args.run / "checkpoints").iterdir()) != [
        f"step-{step:06d}" for step in checkpoint_steps
    ]:
        raise ValueError("formal checkpoint schedule differs")
    binding_keys = (
        "runner_git_sha",
        "contract_file_sha256",
        "training_order_manifest_file_sha256",
        "training_order_manifest_sha256",
        "training_order_records_sha256",
        "feature_cache_manifest_file_sha256",
        "feature_cache_records_sha256",
        "feature_cache_runner_git_sha",
        "initial_projector_sha256",
        "proxy_receiver_sha256",
    )
    checkpoints = []
    for step in checkpoint_steps:
        verified = verify_checkpoint_inventory(
            args.run / "checkpoints" / f"step-{step:06d}"
        )
        manifest = verified["manifest"]
        if int(manifest["step"]) != step:
            raise ValueError(f"checkpoint step differs: {step}")
        if any(
            manifest.get(key) != checkpoint_binding.get(key) for key in binding_keys
        ):
            raise ValueError(f"checkpoint binding differs: {step}")
        history_payload = json.loads(
            (args.run / "checkpoints" / f"step-{step:06d}" / "history.json").read_text(
                encoding="utf-8"
            )
        )
        if history_payload != {"step": step, "history": history[:step]}:
            raise ValueError(f"checkpoint history prefix differs: {step}")
        checkpoints.append(
            {
                "step": step,
                "manifest_file_sha256": verified["manifest_file_sha256"],
                "file_count": len(verified["files"]),
                "total_bytes": sum(row["bytes"] for row in verified["files"]),
                "files": verified["files"],
            }
        )

    final_dir = args.run / "checkpoints" / f"step-{expected_steps:06d}"
    state = torch.load(
        final_dir / "training_state.pt", map_location="cpu", weights_only=False
    )
    if int(state["step"]) != expected_steps or state["history"] != history:
        raise ValueError("final serialized training state differs")
    for key in ("optimizer", "python_rng", "torch_rng", "cuda_rng"):
        if key not in state:
            raise ValueError(f"final serialized training state is missing: {key}")
    fp32 = load_file(str(final_dir / "projector.safetensors"), device="cpu")
    bf16 = load_file(str(final_dir / "projector_bf16.safetensors"), device="cpu")
    if set(fp32) != set(bf16) or not all(
        torch.isfinite(value).all() for value in fp32.values()
    ):
        raise ValueError("final projector tensor inventory or finiteness differs")
    if not all(torch.equal(bf16[key], fp32[key].to(torch.bfloat16)) for key in fp32):
        raise ValueError("final BF16 projector is not the exact FP32 cast")
    final_projector_sha = sha256_file(final_dir / "projector.safetensors")
    step0_sha = sha256_file(args.step0_projector / "projector.safetensors")
    if final_projector_sha == step0_sha:
        raise ValueError("formal projector did not change from step0")

    result = {
        "format_version": "qwen3b-fixed-budget-training-verification-v1",
        "status": "verified",
        "capability_claim_allowed": False,
        "visual_ability_established": False,
        "verifier_git_sha": git_sha(),
        "verifier_git_tracked_worktree_clean": git_tracked_worktree_clean(),
        "verifier_file_sha256": sha256_file(Path(__file__).resolve()),
        "runner_git_sha": args.expected_runner_git_sha,
        "contract_file_sha256": sha256_file(args.contract),
        "training_order_manifest_file_sha256": sha256_file(
            args.training_order_manifest
        ),
        "feature_cache_manifest_file_sha256": sha256_file(cache_manifest_path),
        "feature_cache_records_sha256": cache["records_sha256"],
        "supervision_records_file_sha256": sha256_file(supervision_path),
        "training_history_file_sha256": sha256_file(history_path),
        "history": history_result,
        "checkpoints": checkpoints,
        "checkpoint_total_bytes": sum(row["total_bytes"] for row in checkpoints),
        "final_projector_sha256": final_projector_sha,
        "step0_projector_sha256": step0_sha,
        "final_projector_changed": True,
        "final_training_state_step": int(state["step"]),
        "final_optimizer_parameter_states": len(state["optimizer"]["state"]),
        "peak_gpu_memory_bytes": int(summary["peak_gpu_memory_bytes"]),
        "paid_resources_used": False,
        "transfer_label": "transferable_with_runtime_validation",
    }
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
