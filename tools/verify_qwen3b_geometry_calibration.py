#!/usr/bin/env python3
"""独立重算 Qwen3B 几何辅助项、梯度范数与派生 λ。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from moonvit_glue import FeatureCache
from moonvit_glue.geometry_regularization import (
    geometry_payload,
    geometry_regularization_loss,
    global_gradient_norm,
    pool_projector_batch,
)
from moonvit_glue.projector import PatchMergerProjector
from train_qwen3b_proxy import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--core-contract", type=Path, required=True)
    parser.add_argument("--screen-contract", type=Path, required=True)
    parser.add_argument("--training-order-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--reference-projector", type=Path, required=True)
    parser.add_argument("--checkpoint-projector", type=Path, required=True)
    parser.add_argument("--training-history", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _assert_close(actual: Any, expected: Any, *, path: str = "root") -> None:
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"key set differs: {path}")
        for key, value in expected.items():
            _assert_close(actual[key], value, path=f"{path}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"list length differs: {path}")
        for index, value in enumerate(expected):
            _assert_close(actual[index], value, path=f"{path}[{index}]")
    elif isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"float differs: {path}: {actual} != {expected}")
    elif actual != expected:
        raise ValueError(f"value differs: {path}: {actual} != {expected}")


def main() -> None:
    args = parse_args()
    config = json.loads((args.run / "RUN_CONFIG.json").read_text(encoding="utf-8"))
    summary = json.loads((args.run / "SUMMARY.json").read_text(encoding="utf-8"))
    screen = json.loads(args.screen_contract.read_text(encoding="utf-8"))
    bindings = {
        "core_contract_file_sha256": sha256_file(args.core_contract),
        "screen_contract_file_sha256": sha256_file(args.screen_contract),
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
    }
    for key, expected in bindings.items():
        if config[key] != expected:
            raise ValueError(f"calibration input binding differs: {key}")
    if config["screen_contract"] != screen:
        raise ValueError("embedded screen contract differs")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("independent calibration verification requires the V100")
    torch.manual_seed(20260805)
    torch.cuda.manual_seed_all(20260805)
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = PatchMergerProjector.from_pretrained(
        args.reference_projector, device=device, dtype=torch.float32
    ).requires_grad_(False).eval()
    checkpoint = PatchMergerProjector.from_pretrained(
        args.checkpoint_projector, device=device, dtype=torch.float32
    ).eval()
    cache = FeatureCache(args.feature_cache)
    features = [
        cache.get(record_id, device=device, dtype=torch.float32)
        for record_id in config["record_ids"]
    ]
    with torch.no_grad():
        reference_pooled = pool_projector_batch(
            [reference(groups) for groups in features]
        )
    current_pooled = pool_projector_batch([checkpoint(groups) for groups in features])
    objective = screen["geometry_objective"]
    weights = objective["component_weights"]
    result = geometry_regularization_loss(
        current_pooled,
        reference_pooled,
        scale_weight=float(weights["scale"]),
        relative_spread_weight=float(weights["relative_spread"]),
        centered_gram_weight=float(weights["centered_gram"]),
        epsilon=float(objective["epsilon"]),
    )
    parameters = tuple(checkpoint.parameters())
    gradients = torch.autograd.grad(result.total, parameters)
    gradient_norm = float(global_gradient_norm(gradients))
    ce_gradient_norm = float(
        screen["calibration"]["recorded_ce_gradient_norm_before_clip"]
    )
    derived = {"control": {"target_gradient_ratio": 0.0, "lambda": 0.0}}
    for arm in screen["screen"]["arms"]:
        name = str(arm["name"])
        ratio = float(arm["target_gradient_ratio"])
        if name != "control":
            derived[name] = {
                "target_gradient_ratio": ratio,
                "lambda": ratio * ce_gradient_norm / gradient_norm,
            }
    per_parameter = []
    for (name, parameter), gradient in zip(
        checkpoint.named_parameters(), gradients, strict=True
    ):
        per_parameter.append(
            {
                "name": name,
                "numel": parameter.numel(),
                "gradient_norm": float(torch.linalg.vector_norm(gradient.detach())),
                "gradient_nonzero": int(torch.count_nonzero(gradient.detach())),
            }
        )
    recomputed = {
        "geometry": geometry_payload(result),
        "unweighted_auxiliary_gradient_norm": gradient_norm,
        "recorded_ce_gradient_norm_before_clip": ce_gradient_norm,
        "derived_arms": derived,
        "per_parameter_gradients": per_parameter,
    }
    declared = {
        key: summary[key]
        for key in (
            "geometry",
            "unweighted_auxiliary_gradient_norm",
            "recorded_ce_gradient_norm_before_clip",
            "derived_arms",
            "per_parameter_gradients",
        )
    }
    _assert_close(recomputed, declared)

    pooled_path = args.run / "CALIBRATION_POOLED.safetensors"
    if sha256_file(pooled_path) != summary["pooled"]["sha256"]:
        raise ValueError("calibration pooled SHA-256 differs")
    tensors = load_file(str(pooled_path), device="cpu")
    expected_tensors = {
        "current_pooled": current_pooled.detach().cpu().to(torch.float64),
        "reference_pooled": reference_pooled.detach().cpu().to(torch.float64),
    }
    if set(tensors) != set(expected_tensors) or not all(
        torch.equal(tensors[name], expected_tensors[name]) for name in tensors
    ):
        raise ValueError("calibration pooled tensors differ from recomputation")

    verification = {
        "format_version": "qwen3b-geometry-calibration-verification-v1",
        "status": "verified",
        "runner_git_sha": config["runner_git_sha"],
        "screen_contract_file_sha256": sha256_file(args.screen_contract),
        "pooled_sha256": sha256_file(pooled_path),
        **recomputed,
        "capability_claim_allowed": False,
        "visual_ability_established": False,
        "paid_resources_used": False,
        "final_half_scored": False,
    }
    args.out.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
