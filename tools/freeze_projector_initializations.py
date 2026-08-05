"""冻结 Qwen3B/DeepSeek 共用的 canonical step0 与 random-projector 权重。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from moonvit_glue.projector import ProjectorConfig, seeded_projector
from moonvit_glue.screenspot_contract import seal_manifest, verify_manifest


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtype, shape and raw contiguous bytes canonically."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().contiguous().cpu()
        header = json.dumps(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = tensor.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _freeze_role(
    *,
    role: str,
    seed: int,
    config: ProjectorConfig,
    destination: Path,
) -> dict[str, Any]:
    projector = seeded_projector(config, seed=seed)
    parameter_count = sum(parameter.numel() for parameter in projector.parameters())
    if any(parameter.dtype != torch.float32 for parameter in projector.parameters()):
        raise ValueError("canonical initialization must be serialized in float32")
    state_hash = tensor_state_sha256(projector.state_dict())
    regenerated_hash = tensor_state_sha256(
        seeded_projector(config, seed=seed).state_dict()
    )
    if regenerated_hash != state_hash:
        raise RuntimeError(f"seeded regeneration differs for {role}")

    projector.save_pretrained(destination)
    restored = type(projector).from_pretrained(destination, device="cpu")
    restored_hash = tensor_state_sha256(restored.state_dict())
    if restored_hash != state_hash:
        raise RuntimeError(f"save/restore differs for {role}")

    files = []
    for path in sorted(destination.iterdir()):
        if path.is_file():
            files.append(
                {
                    "path": f"{role}/{path.name}",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "role": role,
        "seed": seed,
        "dtype": "float32",
        "parameter_count": parameter_count,
        "tensor_state_sha256": state_hash,
        "regenerated_tensor_state_sha256": regenerated_hash,
        "save_restore_tensor_state_sha256": restored_hash,
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projector-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--step0-seed", type=int, default=20260805)
    parser.add_argument("--random-projector-seed", type=int, default=20260806)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.step0_seed == args.random_projector_seed:
        raise ValueError("step0 and random-projector seeds must differ")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.out_dir}")
    raw_config = json.loads(args.projector_config.read_text(encoding="utf-8"))
    config = ProjectorConfig(**raw_config)
    if config.language_width != 4096:
        raise ValueError("canonical contract projector output width must be 4096")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    roles = [
        _freeze_role(
            role="step0",
            seed=args.step0_seed,
            config=config,
            destination=args.out_dir / "step0",
        ),
        _freeze_role(
            role="random_projector",
            seed=args.random_projector_seed,
            config=config,
            destination=args.out_dir / "random_projector",
        ),
    ]
    if roles[0]["tensor_state_sha256"] == roles[1]["tensor_state_sha256"]:
        raise RuntimeError("step0 and random-projector states unexpectedly match")
    manifest = seal_manifest(
        {
            "schema_version": "canonical-projector-initializations-v1",
            "frozen_on": "2026-08-05",
            "torch_version": torch.__version__,
            "construction_device": "cpu",
            "config_source": str(args.projector_config),
            "config_source_sha256": sha256_file(args.projector_config),
            "config": raw_config,
            "comparison_rule": "every horizontal method comparison must load the same exact step0 file; random_projector always uses the separately frozen control file",
            "roles": roles,
        }
    )
    manifest_path = args.out_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not verify_manifest(json.loads(manifest_path.read_text(encoding="utf-8"))):
        raise RuntimeError("projector initialization manifest failed self-verification")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
