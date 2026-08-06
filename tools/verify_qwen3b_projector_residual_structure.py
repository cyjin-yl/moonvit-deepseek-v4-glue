#!/usr/bin/env python3
"""独立校验 Package 15R 残差 projector 的初始化与迁移边界。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from moonvit_glue.projector import PatchMergerProjector


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(
    *, root: Path, contract_path: Path, arm_name: str, projector_dir: Path, base_dir: Path
) -> dict[str, Any]:
    contract = load_json(contract_path)
    arms = {str(row["name"]): row for row in contract["arms"]}
    if arm_name not in arms:
        raise ValueError(f"unknown residual structure arm: {arm_name}")
    arm = arms[arm_name]
    config_path = projector_dir / "projector_config.json"
    weights_path = projector_dir / "projector.safetensors"
    base_config_path = base_dir / "projector_config.json"
    base_weights_path = base_dir / "projector.safetensors"
    for path in (config_path, weights_path, base_config_path, base_weights_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(root / contract["projector_source_file"]) != str(
        contract["projector_source_file_sha256"]
    ):
        raise ValueError("projector source hash differs from residual contract")
    config_sha = sha256_file(config_path)
    if config_sha != str(arm["config_sha256"]):
        raise ValueError(f"residual config SHA differs: {config_sha}")
    weights_sha = sha256_file(weights_path)
    if weights_sha != str(arm["weights_sha256"]):
        raise ValueError(f"residual weights SHA differs: {weights_sha}")
    base_sha = sha256_file(base_weights_path)
    if base_sha != str(contract["base_projector"]["weights_sha256"]):
        raise ValueError("base step0 projector SHA differs")
    config = load_json(config_path)
    if config.get("output_norm", "none") != "none":
        raise ValueError("residual screen must keep output_norm=none")
    if config.get("residual_mode") != arm["residual_mode"]:
        raise ValueError("residual mode differs from frozen arm")
    for key, value in {"language_width": 4096, "vision_width": 1024, "merge_factor": 4}.items():
        if config.get(key) != value:
            raise ValueError(f"residual config field differs: {key}")

    projector = PatchMergerProjector.from_pretrained(projector_dir, device="cpu")
    base = PatchMergerProjector.from_pretrained(base_dir, device="cpu")
    parameter_count = sum(parameter.numel() for parameter in projector.parameters())
    if parameter_count != int(arm["parameter_count"]):
        raise ValueError("residual parameter count differs")
    if sum(parameter.numel() for parameter in base.parameters()) != int(
        contract["base_projector"]["parameter_count"]
    ):
        raise ValueError("base parameter count differs")
    base_state = base.state_dict()
    variant_state = projector.state_dict()
    for key, value in base_state.items():
        if key not in variant_state or not value.equal(variant_state[key]):
            raise ValueError(f"base initialization tensor differs: {key}")
    if arm["residual_mode"] == "zero_init":
        if int(torch.count_nonzero(projector.residual.weight)) != 0:
            raise ValueError("zero-init residual branch is nonzero")
    elif projector.residual_gate.item() != 0.0:
        raise ValueError("gated residual gate is not zero")
    if arm["residual_mode"] == "gated" and int(torch.count_nonzero(projector.residual.weight)) == 0:
        raise ValueError("gated residual branch unexpectedly has zero weights")

    torch.manual_seed(20260805)
    features = [torch.randn(2, 4, 1024)]
    with torch.no_grad():
        base_output = base(features)[0]
        variant_output = projector(features)[0]
    if not torch.equal(base_output, variant_output):
        raise ValueError("residual initialization changes the step0 output")
    result = {
        "format_version": "qwen3b-projector-residual-verification-v1",
        "status": "verified",
        "arm": arm_name,
        "residual_mode": arm["residual_mode"],
        "config_sha256": config_sha,
        "weights_sha256": weights_sha,
        "base_weights_sha256": base_sha,
        "parameter_count": parameter_count,
        "base_state_keys_match": True,
        "base_initialization_match": True,
        "initial_output_matches_base": True,
        "paid_resources_used": False,
        "capability_claim_allowed": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--projector-dir", type=Path, required=True)
    parser.add_argument("--base-projector-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = verify(
        root=root,
        contract_path=(root / args.contract).resolve()
        if not args.contract.is_absolute()
        else args.contract,
        arm_name=args.arm,
        projector_dir=args.projector_dir.resolve(),
        base_dir=args.base_projector_dir.resolve(),
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
