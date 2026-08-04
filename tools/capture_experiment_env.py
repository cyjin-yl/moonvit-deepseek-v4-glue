"""Capture reproducible host/software/GPU context without recording secrets."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command(*args: str) -> dict:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return {
        "argv": list(args),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite environment capture: {args.out}")
    args.out.mkdir(parents=True)
    git = command("git", "rev-parse", "HEAD")
    git_status = command("git", "status", "--short")
    freeze = command(sys.executable, "-m", "pip", "freeze")
    (args.out / "pip_freeze.txt").write_text(freeze["stdout"], encoding="utf-8")
    torch_context: dict = {"available": False}
    try:
        import torch

        torch_context = {
            "available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            torch_context.update({
                "device_name": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_bytes": properties.total_memory,
                "memory_info_bytes": list(torch.cuda.mem_get_info(0)),
                "device_count": torch.cuda.device_count(),
            })
    except Exception as exc:
        torch_context["error"] = f"{type(exc).__name__}: {exc}"
    result = {
        "run_id": args.run_id,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git["stdout"].strip() or None,
        "git_status_short": git_status["stdout"].splitlines(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "packages": {
            name: package_version(name)
            for name in (
                "torch", "transformers", "safetensors", "pillow", "numpy",
                "pyarrow", "pytest", "scikit-learn",
            )
        },
        "cuda": torch_context,
        "nvidia_smi": command("nvidia-smi"),
        "gpu_device_users": command("sh", "-lc", "fuser -v /dev/nvidia0 2>&1 || true"),
        "pip_freeze_file": "pip_freeze.txt",
    }
    (args.out / "environment.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
