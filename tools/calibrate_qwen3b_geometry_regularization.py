#!/usr/bin/env python3
"""用冻结 step100 batch 标定 projector 几何辅助项的固定 λ。"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

import moonvit_glue.geometry_regularization as geometry_module
from moonvit_glue import FeatureCache
from moonvit_glue.geometry_regularization import (
    geometry_payload,
    geometry_regularization_loss,
    global_gradient_norm,
    pool_projector_batch,
)
from moonvit_glue.projector import PatchMergerProjector
from moonvit_glue.training_order import verify_training_order_manifest

from train_qwen3b_proxy import (
    _Tee,
    canonical_sha256,
    git_sha,
    git_tracked_worktree_clean,
    sha256_file,
    verify_bound_checkpoint,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-contract", type=Path, required=True)
    parser.add_argument("--screen-contract", type=Path, required=True)
    parser.add_argument("--training-order-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--reference-projector", type=Path, required=True)
    parser.add_argument("--checkpoint-projector", type=Path, required=True)
    parser.add_argument("--training-history", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-dirty-development-run", action="store_true")
    return parser.parse_args()


def runtime_source_files() -> list[dict[str, Any]]:
    paths = (Path(__file__).resolve(), Path(geometry_module.__file__).resolve())
    return [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ]


def _geometry_kwargs(screen_contract: dict[str, Any]) -> dict[str, float]:
    objective = screen_contract["geometry_objective"]
    weights = objective["component_weights"]
    return {
        "scale_weight": float(weights["scale"]),
        "relative_spread_weight": float(weights["relative_spread"]),
        "centered_gram_weight": float(weights["centered_gram"]),
        "epsilon": float(objective["epsilon"]),
    }


def _verify_inputs(args: argparse.Namespace) -> dict[str, Any]:
    core = json.loads(args.core_contract.read_text(encoding="utf-8"))
    screen = json.loads(args.screen_contract.read_text(encoding="utf-8"))
    order = json.loads(args.training_order_manifest.read_text(encoding="utf-8"))
    cache_manifest_path = args.feature_cache / "MANIFEST.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    calibration = screen["calibration"]
    if sha256_file(args.core_contract) != screen["core_contract_file_sha256"]:
        raise ValueError("core contract differs from geometry screen contract")
    if (
        sha256_file(args.training_order_manifest)
        != screen["training_order_manifest_file_sha256"]
        or not verify_training_order_manifest(order)
        or order["manifest_sha256"] != screen["training_order_manifest_sha256"]
        or order["records_sha256"] != screen["training_order_records_sha256"]
    ):
        raise ValueError("training order differs from geometry screen contract")
    if sha256_file(cache_manifest_path) != screen["feature_cache_manifest_file_sha256"]:
        raise ValueError("feature cache differs from geometry screen contract")
    if (
        cache_manifest["training_order_manifest_sha256"] != order["manifest_sha256"]
        or cache_manifest["training_order_records_sha256"] != order["records_sha256"]
    ):
        raise ValueError("feature cache training-order binding differs")
    reference_weights = args.reference_projector / "projector.safetensors"
    if (
        sha256_file(reference_weights) != screen["reference_projector_sha256"]
        or screen["reference_projector_sha256"]
        != core["canonical_projector"]["initialization_contract"]["step0"][
            "weights_sha256"
        ]
    ):
        raise ValueError("reference projector differs from exact step0")
    checkpoint_manifest_path = args.checkpoint_projector / "CHECKPOINT_MANIFEST.json"
    if (
        sha256_file(checkpoint_manifest_path)
        != calibration["checkpoint_manifest_sha256"]
        or sha256_file(args.checkpoint_projector / "projector.safetensors")
        != calibration["checkpoint_projector_sha256"]
    ):
        raise ValueError("calibration checkpoint differs from geometry contract")
    checkpoint_manifest = verify_bound_checkpoint(
        args.checkpoint_projector, expected_binding={}
    )
    if int(checkpoint_manifest["step"]) != int(calibration["checkpoint_step"]):
        raise ValueError("calibration checkpoint step differs")
    if sha256_file(args.training_history) != calibration["training_history_sha256"]:
        raise ValueError("calibration training history differs")
    history = [
        json.loads(line)
        for line in args.training_history.read_text(encoding="utf-8").splitlines()
    ]
    history_row = history[int(calibration["optimizer_batch_step"]) - 1]
    if (
        int(history_row["step"]) != int(calibration["optimizer_batch_step"])
        or history_row["batch_record_ids_sha256"]
        != calibration["batch_record_ids_sha256"]
        or float(history_row["gradient_norm_before_clip"])
        != float(calibration["recorded_ce_gradient_norm_before_clip"])
    ):
        raise ValueError("calibration history row differs")
    indices = [int(value) for value in calibration["record_indices"]]
    record_ids = [str(order["records"][index]["id"]) for index in indices]
    if canonical_sha256(record_ids) != calibration["batch_record_ids_sha256"]:
        raise ValueError("calibration batch IDs differ")
    return {
        "core": core,
        "screen": screen,
        "order": order,
        "cache_manifest": cache_manifest,
        "record_ids": record_ids,
        "checkpoint_manifest": checkpoint_manifest,
        "history_row": history_row,
    }


def compute_calibration(
    *,
    screen_contract: dict[str, Any],
    reference_projector: PatchMergerProjector,
    checkpoint_projector: PatchMergerProjector,
    cache: FeatureCache,
    record_ids: list[str],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    feature_batches = [
        cache.get(record_id, device=device, dtype=torch.float32)
        for record_id in record_ids
    ]
    with torch.no_grad():
        reference_outputs = [reference_projector(groups) for groups in feature_batches]
        reference_pooled = pool_projector_batch(reference_outputs)
    current_outputs = [checkpoint_projector(groups) for groups in feature_batches]
    current_pooled = pool_projector_batch(current_outputs)
    result = geometry_regularization_loss(
        current_pooled, reference_pooled, **_geometry_kwargs(screen_contract)
    )
    parameters = tuple(checkpoint_projector.parameters())
    gradients = torch.autograd.grad(result.total, parameters)
    gradient_norm = float(global_gradient_norm(gradients))
    if gradient_norm <= 0.0:
        raise ValueError("unweighted geometry auxiliary gradient is zero")
    calibration = screen_contract["calibration"]
    ce_gradient_norm = float(calibration["recorded_ce_gradient_norm_before_clip"])
    derived_arms = {"control": {"target_gradient_ratio": 0.0, "lambda": 0.0}}
    for arm in screen_contract["screen"]["arms"]:
        name = str(arm["name"])
        ratio = float(arm["target_gradient_ratio"])
        if name == "control":
            continue
        derived_arms[name] = {
            "target_gradient_ratio": ratio,
            "lambda": ratio * ce_gradient_norm / gradient_norm,
        }
    per_parameter = []
    for (name, parameter), gradient in zip(
        checkpoint_projector.named_parameters(), gradients, strict=True
    ):
        per_parameter.append(
            {
                "name": name,
                "numel": parameter.numel(),
                "gradient_norm": float(torch.linalg.vector_norm(gradient.detach())),
                "gradient_nonzero": int(torch.count_nonzero(gradient.detach())),
            }
        )
    payload = {
        "geometry": geometry_payload(result),
        "unweighted_auxiliary_gradient_norm": gradient_norm,
        "recorded_ce_gradient_norm_before_clip": ce_gradient_norm,
        "derived_arms": derived_arms,
        "per_parameter_gradients": per_parameter,
    }
    tensors = {
        "current_pooled": current_pooled.detach().to(device="cpu", dtype=torch.float64),
        "reference_pooled": reference_pooled.detach().to(
            device="cpu", dtype=torch.float64
        ),
    }
    return payload, tensors


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    tracked_clean = git_tracked_worktree_clean()
    if not tracked_clean and not args.allow_dirty_development_run:
        raise RuntimeError("tracked Git worktree is dirty; formal calibration is refused")
    verified = _verify_inputs(args)
    screen = verified["screen"]
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal geometry calibration requires the local V100")
    torch.manual_seed(20260805)
    torch.cuda.manual_seed_all(20260805)
    torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = PatchMergerProjector.from_pretrained(
        args.reference_projector, device=device, dtype=torch.float32
    ).requires_grad_(False).eval()
    checkpoint = PatchMergerProjector.from_pretrained(
        args.checkpoint_projector, device=device, dtype=torch.float32
    ).eval()
    payload, tensors = compute_calibration(
        screen_contract=screen,
        reference_projector=reference,
        checkpoint_projector=checkpoint,
        cache=FeatureCache(args.feature_cache),
        record_ids=verified["record_ids"],
        device=device,
    )
    torch.cuda.synchronize(device)
    pooled_path = args.out / "CALIBRATION_POOLED.safetensors"
    save_file(
        tensors,
        str(pooled_path),
        metadata={"format": "qwen3b-geometry-calibration-pooled-v1"},
    )
    run_config = {
        "format_version": "qwen3b-geometry-calibration-run-v1",
        "runner_git_sha": git_sha(),
        "git_tracked_worktree_clean": tracked_clean,
        "formal_run": tracked_clean and not args.allow_dirty_development_run,
        "core_contract_file_sha256": sha256_file(args.core_contract),
        "screen_contract_file_sha256": sha256_file(args.screen_contract),
        "screen_contract": screen,
        "training_order_manifest_file_sha256": sha256_file(
            args.training_order_manifest
        ),
        "feature_cache_manifest_file_sha256": sha256_file(
            args.feature_cache / "MANIFEST.json"
        ),
        "reference_projector_sha256": sha256_file(
            args.reference_projector / "projector.safetensors"
        ),
        "checkpoint_projector_sha256": sha256_file(
            args.checkpoint_projector / "projector.safetensors"
        ),
        "checkpoint_manifest_sha256": sha256_file(
            args.checkpoint_projector / "CHECKPOINT_MANIFEST.json"
        ),
        "training_history_sha256": sha256_file(args.training_history),
        "record_ids": verified["record_ids"],
        "runtime_source_files": runtime_source_files(),
        "paid_resources_used": False,
        "final_half_scored": False,
    }
    write_json(args.out / "RUN_CONFIG.json", run_config)
    summary = {
        "format_version": "qwen3b-geometry-calibration-summary-v1",
        "status": "valid",
        "formal_calibration_complete": run_config["formal_run"],
        **payload,
        "pooled": {
            "path": str(pooled_path),
            "bytes": pooled_path.stat().st_size,
            "sha256": sha256_file(pooled_path),
            "tensors": {name: list(tensor.shape) for name, tensor in tensors.items()},
        },
        "gpu_name": torch.cuda.get_device_name(device),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "wall_seconds": time.perf_counter() - started,
        "capability_claim_allowed": False,
        "visual_ability_established": False,
        "paid_resources_used": False,
        "final_half_scored": False,
    }
    write_json(args.out / "SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {args.out}")
    args.out.mkdir(parents=True)
    log_path = args.out / "run.log"
    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            sys.stdout = _Tee(original_stdout, log)
            sys.stderr = _Tee(original_stderr, log)
            _run(args)
    except Exception as exc:
        write_json(
            args.out / "FAILURE.json",
            {
                "format_version": "qwen3b-geometry-calibration-failure-v1",
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "paid_resources_used": False,
                "final_half_scored": False,
            },
        )
        raise
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    main()
