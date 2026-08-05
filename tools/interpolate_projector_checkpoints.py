#!/usr/bin/env python3
"""在线性权重路径上构造可审计的 projector checkpoint。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        tensor = state[name].detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def interpolate_state_dict(
    early: dict[str, torch.Tensor],
    late: dict[str, torch.Tensor],
    *,
    alpha: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("interpolation alpha must be within [0, 1]")
    if set(early) != set(late) or not early:
        raise ValueError("interpolation endpoints need identical non-empty tensor keys")
    output = {}
    for name in sorted(early):
        left = early[name].detach().cpu().contiguous()
        right = late[name].detach().cpu().contiguous()
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"interpolation endpoint tensor contract differs: {name}")
        if not (left.is_floating_point() or left.is_complex()):
            raise ValueError(f"projector interpolation requires floating tensors: {name}")
        if alpha == 0.0:
            mixed = left.clone()
        elif alpha == 1.0:
            mixed = right.clone()
        else:
            mixed = torch.lerp(left, right, alpha)
        output[name] = mixed.contiguous()
    return output


def git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def alpha_label(alpha: float) -> str:
    return f"{alpha:.3f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--early-step", required=True, type=int)
    parser.add_argument("--late-step", required=True, type=int)
    parser.add_argument("--alphas", required=True, type=float, nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite interpolation run: {args.out}")
    alphas = sorted(set(args.alphas))
    if not alphas or alphas[0] < 0.0 or alphas[-1] > 1.0:
        raise ValueError("all interpolation alphas must be within [0, 1]")
    if 0.0 not in alphas or 1.0 not in alphas:
        raise ValueError("endpoint alphas 0 and 1 are required for reproduction checks")

    source_summary_path = args.source_run / "SUMMARY.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if source_summary.get("status") != "valid":
        raise ValueError("source projector run is not valid")

    def source_checkpoint(step: int) -> tuple[Path, dict]:
        key = f"step-{step:06d}"
        manifest = source_summary["checkpoints"].get(key)
        if manifest is None or int(manifest["step"]) != step:
            raise ValueError(f"source checkpoint is absent: {key}")
        path = args.source_run / "checkpoints" / key / "projector.safetensors"
        expected = manifest["files"]["projector.safetensors"]["sha256"]
        if sha256(path) != expected:
            raise ValueError(f"source checkpoint file hash mismatch: {key}")
        return path, manifest

    early_path, early_manifest = source_checkpoint(args.early_step)
    late_path, late_manifest = source_checkpoint(args.late_step)
    early = load_file(str(early_path), device="cpu")
    late = load_file(str(late_path), device="cpu")
    # 先独立走一次合同检查，避免第一个输出目录创建后才发现端点不兼容。
    interpolate_state_dict(early, late, alpha=0.0)

    args.out.mkdir(parents=True)
    checkpoints = {}
    endpoint_reproduction = {}
    for alpha in alphas:
        label = alpha_label(alpha)
        relative = Path("checkpoints") / f"alpha-{label}"
        directory = args.out / relative
        directory.mkdir(parents=True)
        state = interpolate_state_dict(early, late, alpha=alpha)
        weights_path = directory / "projector.safetensors"
        save_file(state, str(weights_path))
        config_source = early_path.parent / "projector_config.json"
        config_path = directory / "projector_config.json"
        shutil.copy2(config_source, config_path)
        reloaded = load_file(str(weights_path), device="cpu")
        if tensor_state_hash(reloaded) != tensor_state_hash(state):
            raise ValueError(f"saved interpolation tensor hash mismatch: alpha={alpha}")
        if alpha in (0.0, 1.0):
            source = early if alpha == 0.0 else late
            exact = all(torch.equal(reloaded[name], source[name]) for name in source)
            if not exact:
                raise ValueError(f"interpolation endpoint did not reproduce: alpha={alpha}")
            endpoint_reproduction[str(alpha)] = {
                "exact_tensor_equality": True,
                "source_tensor_sha256": tensor_state_hash(source),
                "output_tensor_sha256": tensor_state_hash(reloaded),
            }
        state_id = f"projector-interp{int(round(alpha * 100)):03d}"
        manifest = {
            "status": "valid",
            "kind": "projector",
            "step": int(round(alpha * 100)),
            "examples_seen": None,
            "evaluation_state_id": state_id,
            "interpolation_alpha": alpha,
            "relative_path": relative.as_posix(),
            "source_early_step": args.early_step,
            "source_late_step": args.late_step,
            "weights_tensor_sha256": tensor_state_hash(reloaded),
            "files": {
                "projector.safetensors": {
                    "bytes": weights_path.stat().st_size,
                    "sha256": sha256(weights_path),
                },
                "projector_config.json": {
                    "bytes": config_path.stat().st_size,
                    "sha256": sha256(config_path),
                },
            },
        }
        (directory / "MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        checkpoints[f"alpha-{label}"] = manifest

    summary = {
        "status": "valid",
        "format_version": "projector-linear-interpolation-v1",
        "metadata": {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha(),
            "host": platform.node(),
            "torch": torch.__version__,
            "final_half_scored": False,
        },
        "kind": "projector_interpolation",
        "formula": "(1 - alpha) * projector_step_early + alpha * projector_step_late",
        "source_run": str(args.source_run.resolve()),
        "source_summary_sha256": sha256(source_summary_path),
        "early_step": args.early_step,
        "late_step": args.late_step,
        "early_source_manifest": early_manifest,
        "late_source_manifest": late_manifest,
        "alphas": alphas,
        "endpoint_reproduction": endpoint_reproduction,
        "checkpoints": checkpoints,
        "final_half_scored": False,
    }
    (args.out / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
