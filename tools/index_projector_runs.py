#!/usr/bin/env python3
"""把多个独立 projector run 索引成 eval_shape_adaptation 可读取的比较集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--state",
        action="append",
        required=True,
        metavar="ID=RUN:STEP",
        help="可重复；STEP 是源 run 的全局 checkpoint 步数",
    )
    return parser.parse_args()


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def parse_state(value: str) -> tuple[str, Path, int]:
    state_id, separator, source = value.partition("=")
    if not separator or not state_id:
        raise ValueError(f"invalid state specification: {value}")
    run_text, separator, step_text = source.rpartition(":")
    if not separator or not run_text:
        raise ValueError(f"state specification is missing a step: {value}")
    return state_id, Path(run_text), int(step_text)


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite projector index: {args.out}")
    checkpoints = {}
    source_runs = {}
    seen_ids = set()
    for value in args.state:
        state_id, source_run, step = parse_state(value)
        if state_id in seen_ids:
            raise ValueError(f"duplicate evaluation state ID: {state_id}")
        seen_ids.add(state_id)
        summary_path = source_run / "SUMMARY.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "valid":
            raise ValueError(f"source projector run is invalid: {source_run}")
        key = f"step-{step:06d}"
        if key not in summary["checkpoints"]:
            raise ValueError(f"source projector checkpoint is absent: {source_run} {key}")
        source_manifest = summary["checkpoints"][key]
        checkpoint = source_run / "checkpoints" / key
        weights = checkpoint / "projector.safetensors"
        expected_sha256 = source_manifest["files"][weights.name]["sha256"]
        actual_sha256 = sha256(weights)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"source projector checkpoint SHA-256 mismatch: {weights}")
        checkpoints[state_id] = {
            "status": "valid",
            "kind": "projector",
            "step": step,
            "examples_seen": source_manifest.get("examples_seen"),
            "evaluation_state_id": state_id,
            "relative_path": str(checkpoint.resolve()),
            "source_run": str(source_run.resolve()),
            "source_summary_sha256": sha256(summary_path),
            "weights_tensor_sha256": source_manifest["weights_tensor_sha256"],
            "files": {
                weights.name: {
                    "bytes": weights.stat().st_size,
                    "sha256": actual_sha256,
                }
            },
        }
        source_runs[state_id] = {
            "run": str(source_run.resolve()),
            "summary_sha256": sha256(summary_path),
            "step": step,
        }
    args.out.mkdir(parents=True)
    output = {
        "status": "valid",
        "format_version": "projector-comparison-index-v1",
        "metadata": {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha(),
            "host": platform.node(),
            "final_half_scored": False,
        },
        "source_runs": source_runs,
        "checkpoints": checkpoints,
        "final_half_scored": False,
    }
    (args.out / "SUMMARY.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
