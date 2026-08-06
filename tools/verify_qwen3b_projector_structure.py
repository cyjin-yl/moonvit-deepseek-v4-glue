#!/usr/bin/env python3
"""独立校验 Qwen3B projector 结构候选的权重、边界和参数不变量。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from moonvit_glue.projector import PatchMergerProjector


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def effective_output_norm(config: dict[str, Any]) -> str:
    """兼容历史配置：省略 output_norm 等价于显式的 none。"""

    return str(config.get("output_norm", "none"))


def verify(
    *, root: Path, contract_path: Path, arm_name: str, projector_dir: Path, base_dir: Path
) -> dict[str, Any]:
    contract = load_json(contract_path)
    arms = {str(row["name"]): row for row in contract["arms"]}
    if arm_name not in arms:
        raise ValueError(f"unknown structure arm: {arm_name}")
    arm = arms[arm_name]
    config_path = projector_dir / "projector_config.json"
    weights_path = projector_dir / "projector.safetensors"
    base_config_path = base_dir / "projector_config.json"
    base_weights_path = base_dir / "projector.safetensors"
    for path in (config_path, weights_path, base_config_path, base_weights_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config_sha = sha256_file(config_path)
    if config_sha != str(arm["config_sha256"]):
        raise ValueError(f"structure config SHA differs: {config_sha}")
    weights_sha = sha256_file(weights_path)
    if weights_sha != str(contract["base_projector"]["weights_sha256"]):
        raise ValueError("structure arm weights differ from frozen step0")
    if sha256_file(base_weights_path) != weights_sha:
        raise ValueError("base and structure-arm weight bytes differ")
    if sha256_file(root / contract["projector_source_file"]) != str(
        contract["projector_source_file_sha256"]
    ):
        raise ValueError("projector source hash differs from the frozen structure contract")

    config = load_json(config_path)
    expected = {
        "language_width": 4096,
        "merge_factor": 4,
        "vision_width": 1024,
        "output_norm": arm["output_norm"],
    }
    for key, value in expected.items():
        actual = effective_output_norm(config) if key == "output_norm" else config.get(key)
        if actual != value:
            raise ValueError(f"structure config field differs: {key}")
    projector = PatchMergerProjector.from_pretrained(projector_dir, device="cpu")
    base = PatchMergerProjector.from_pretrained(base_dir, device="cpu")
    if sum(parameter.numel() for parameter in projector.parameters()) != int(
        contract["base_projector"]["parameter_count"]
    ):
        raise ValueError("structure arm parameter count differs")
    if sum(parameter.numel() for parameter in projector.parameters()) != sum(
        parameter.numel() for parameter in base.parameters()
    ):
        raise ValueError("structure arm added trainable parameters")
    if projector.state_dict().keys() != base.state_dict().keys():
        raise ValueError("structure arm state keys differ from base projector")
    for key, value in base.state_dict().items():
        if not value.equal(projector.state_dict()[key]):
            raise ValueError(f"structure arm changed frozen initialization tensor: {key}")
    result = {
        "format_version": "qwen3b-projector-structure-verification-v1",
        "status": "verified",
        "arm": arm_name,
        "output_norm": arm["output_norm"],
        "config_sha256": config_sha,
        "weights_sha256": weights_sha,
        "parameter_count": sum(parameter.numel() for parameter in projector.parameters()),
        "state_keys_match": True,
        "weights_match_step0": True,
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
