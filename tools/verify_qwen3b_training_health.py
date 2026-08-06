#!/usr/bin/env python3
"""独立复核 Qwen3B 在线 health 日志、guards、止损与 checkpoint 产物。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from moonvit_glue.training_health import (
    evaluate_guards,
    probe_due,
    validate_health_contract,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def checkpoint_files(directory: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "CHECKPOINT_MANIFEST.json"
    ]


def verify_checkpoint(directory: Path) -> dict[str, Any]:
    manifest = load_json(directory / "CHECKPOINT_MANIFEST.json")
    actual = checkpoint_files(directory)
    if sorted(manifest["files"], key=lambda row: str(row["path"])) != actual:
        raise ValueError(f"checkpoint file inventory differs: {directory}")
    if int(manifest["file_count"]) != len(actual):
        raise ValueError(f"checkpoint file count differs: {directory}")
    if int(manifest["total_bytes"]) != sum(row["bytes"] for row in actual):
        raise ValueError(f"checkpoint total bytes differ: {directory}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--health-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _finite_fields(row: dict[str, Any], fields: tuple[str, ...]) -> None:
    for field in fields:
        value = float(row[field])
        if not math.isfinite(value):
            raise ValueError(f"non-finite health field: {field}")


def main() -> None:
    args = parse_args()
    contract = load_json(args.health_contract)
    validate_health_contract(contract)
    run_config = load_json(args.run / "RUN_CONFIG.json")
    if run_config["health_setup"]["contract_file_sha256"] != sha256_file(
        args.health_contract
    ):
        raise ValueError("run is bound to a different health contract")
    health_rows = load_jsonl(args.run / "train_health.jsonl")
    if not health_rows or health_rows[0].get("event") != "run_start":
        raise ValueError("health log does not begin with run_start")
    step_rows = [row for row in health_rows if row.get("event") != "run_start"]
    steps = [int(row["optimizer_step"]) for row in step_rows]
    expected_steps = list(range(steps[0], steps[-1] + 1)) if steps else []
    if steps != expected_steps:
        raise ValueError("health optimizer steps are not contiguous")
    required = (
        "projector_output_rms",
        "receiver_output_rms",
        "between_image_rms",
        "within_image_token_rms",
        "relative_spread",
        "mean_direction_fraction",
        "projector_gradient_norm_before_clip",
        "projector_gradient_norm_after_clip",
        "ce_loss",
        "geometry_loss",
        "total_loss",
        "learning_rate",
    )
    for row in step_rows:
        _finite_fields(row, required)
        if bool(row["has_nan_or_inf"]):
            raise ValueError("completed health row reports NaN/Inf")

    probes = load_jsonl(args.run / "probe_metrics.jsonl")
    probe_steps = [int(row["step"]) for row in probes]
    if probe_steps != sorted(set(probe_steps)):
        raise ValueError("health probe steps are duplicated or unordered")
    max_step = steps[-1] if steps else 0
    expected_probe_steps = [
        step
        for step in range(0, max_step + 1)
        if probe_due(
            step,
            max_step=max_step,
            every_after=int(contract["probe_schedule"]["every_after_step"]),
        )
    ]
    if probe_steps != expected_probe_steps:
        raise ValueError("health probe schedule differs from the frozen contract")
    state: dict[str, int] = {}
    previous = None
    recomputed_guards = []
    for probe in probes:
        recorded = probe.get("guards")
        if not isinstance(recorded, dict):
            raise ValueError("health probe is missing guard decision")
        recomputed = evaluate_guards(
            probe,
            previous=previous,
            state=state,
            contract=contract,
        )
        if recorded != recomputed:
            raise ValueError(f"health guard decision differs at step {probe['step']}")
        recomputed_guards.append(recomputed)
        previous = probe

    artifact = load_json(args.run / "HEALTH_ARTIFACT_MANIFEST.json")
    for row in artifact["files"]:
        path = args.run / row["path"]
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"health artifact byte count differs: {row['path']}")
        if sha256_file(path) != str(row["sha256"]):
            raise ValueError(f"health artifact SHA-256 differs: {row['path']}")
    checkpoint_dirs = sorted(
        path
        for path in (args.run / "health_snapshots" / "checkpoints").glob("*")
        if path.is_dir()
    )
    checkpoint_manifests = [verify_checkpoint(path) for path in checkpoint_dirs]
    failure_path = args.run / "FAILURE.json"
    stopped = failure_path.is_file()
    if stopped:
        failure = load_json(failure_path)
        rollback = load_json(args.run / "ROLLBACK.json")
        if failure["status"] != "auto_stopped_by_projector_health_guard":
            raise ValueError("health failure status differs")
        if not recomputed_guards[-1]["stop"]:
            raise ValueError("failure exists without a recomputed stop guard")
        failure_checkpoints = [
            row for row in checkpoint_manifests if row.get("health_checkpoint_role") == "failure"
        ]
        if len(failure_checkpoints) != 1:
            raise ValueError("auto-stop needs exactly one failure checkpoint")
        if rollback["status"] != "rolled_back_to_last_healthy_checkpoint":
            raise ValueError("auto-stop did not restore a healthy checkpoint")
    elif any(row["stop"] for row in recomputed_guards):
        raise ValueError("critical guard exists without auto-stop artifacts")

    result = {
        "status": "verified",
        "run": str(args.run.resolve()),
        "optimizer_step_rows": len(step_rows),
        "probe_steps": probe_steps,
        "checkpoint_count": len(checkpoint_dirs),
        "auto_stopped": stopped,
        "artifact_file_count": int(artifact["file_count"]),
        "artifact_total_bytes": int(artifact["total_bytes"]),
        "guards_recomputed": len(recomputed_guards),
        "paid_resources_used": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
