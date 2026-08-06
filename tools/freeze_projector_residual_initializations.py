#!/usr/bin/env python3
"""从冻结 step0 MLP 权重构造 Package 15R 的残差结构初始状态。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from moonvit_glue.projector import PatchMergerProjector, ProjectorConfig, seeded_projector


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().contiguous().cpu()
        header = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        # reshape(-1) 兼容 gated residual 的 0-d scalar gate 参数。
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def load_config(path: Path) -> ProjectorConfig:
    return ProjectorConfig(**json.loads(path.read_text(encoding="utf-8")))


def copy_base_state(base: PatchMergerProjector, variant: PatchMergerProjector) -> None:
    base_state = base.state_dict()
    variant_state = variant.state_dict()
    for key, value in base_state.items():
        if key not in variant_state:
            raise ValueError(f"residual variant is missing base state key: {key}")
        if tuple(variant_state[key].shape) != tuple(value.shape):
            raise ValueError(f"base state shape differs for {key}")
        variant_state[key].copy_(value)


def freeze_arm(
    *,
    arm: str,
    config_path: Path,
    config: ProjectorConfig,
    base: PatchMergerProjector,
    seed: int,
    destination: Path,
) -> dict[str, Any]:
    if config.residual_mode not in {"zero_init", "gated"}:
        raise ValueError(f"{arm} must be a residual mode")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    with torch.random.fork_rng(devices=[]):
        variant = seeded_projector(config, seed=seed)
    with torch.no_grad():
        copy_base_state(base, variant)
        if config.residual_mode == "zero_init":
            variant.residual.weight.zero_()
        else:
            variant.residual_gate.zero_()
    base_state = base.state_dict()
    variant_state = variant.state_dict()
    for key, value in base_state.items():
        if not value.equal(variant_state[key]):
            raise RuntimeError(f"base initialization changed for {arm}: {key}")
    if config.residual_mode == "zero_init" and torch.count_nonzero(
        variant.residual.weight
    ):
        raise RuntimeError("zero-init residual branch is nonzero")
    if config.residual_mode == "gated" and variant.residual_gate.item() != 0.0:
        raise RuntimeError("gated residual gate is not zero")
    torch.manual_seed(20260805)
    probe_features = [torch.randn(1, config.merge_factor, config.vision_width)]
    with torch.no_grad():
        base_output = base(probe_features)
        variant_output = variant(probe_features)
    if not all(
        left.equal(right)
        for left, right in zip(base_output, variant_output, strict=True)
    ):
        raise RuntimeError(f"initial output differs from base projector for {arm}")
    variant.save_pretrained(destination)
    return {
        "name": arm,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "seed": seed,
        "residual_mode": config.residual_mode,
        "parameter_count": sum(parameter.numel() for parameter in variant.parameters()),
        "tensor_state_sha256": tensor_state_sha256(variant.state_dict()),
        "base_tensor_state_sha256": tensor_state_sha256(base_state),
        "weights_sha256": sha256_file(destination / "projector.safetensors"),
        "config_file_sha256": sha256_file(destination / "projector_config.json"),
        "base_weight_sha256": sha256_file(destination / "projector.safetensors"),
        "base_state_keys_preserved": True,
        "initial_output_matches_base": True,
        "paid_resources_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-projector-dir", type=Path, required=True)
    parser.add_argument("--zero-config", type=Path, required=True)
    parser.add_argument("--gated-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = PatchMergerProjector.from_pretrained(args.base_projector_dir, device="cpu")
    if base.config.residual_mode != "none":
        raise ValueError("base projector must be the residual-free step0 projector")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arms = []
    for index, (name, path) in enumerate(
        (("zero_init_residual", args.zero_config), ("gated_residual", args.gated_config))
    ):
        arms.append(
            freeze_arm(
                arm=name,
                config_path=path,
                config=load_config(path),
                base=base,
                seed=args.seed + index + 1,
                destination=args.out_dir / name,
            )
        )
    payload = {
        "format_version": "qwen3b-projector-residual-initializations-v1",
        "base_projector_dir": str(args.base_projector_dir),
        "base_projector_weights_sha256": sha256_file(
            args.base_projector_dir / "projector.safetensors"
        ),
        "base_projector_tensor_state_sha256": tensor_state_sha256(base.state_dict()),
        "base_parameter_count": sum(parameter.numel() for parameter in base.parameters()),
        "arms": arms,
        "same_base_mlp_weights": True,
        "initial_output_matches_base": True,
        "paid_resources_used": False,
    }
    (args.out_dir / "MANIFEST.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
