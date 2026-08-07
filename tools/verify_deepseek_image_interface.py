#!/usr/bin/env python3
"""Independent verifier for the DeepSeek image-interface screen pointer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(pointer_path: Path) -> dict[str, object]:
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    summary_info = pointer["raw_files"]["summary"]
    tensors_info = pointer["raw_files"]["merged_tensors"]
    summary_path = Path(summary_info["path"])
    tensors_path = Path(tensors_info["path"])
    checks["pointer_schema"] = pointer.get("schema_version") == "deepseek-v4-image-interface-screen-result-pointer-v2"
    checks["summary_exists"] = summary_path.is_file()
    checks["merged_tensors_exists"] = tensors_path.is_file()
    checks["summary_sha256"] = checks["summary_exists"] and sha256(summary_path) == summary_info["sha256"]
    checks["merged_tensors_sha256"] = checks["merged_tensors_exists"] and sha256(tensors_path) == tensors_info["sha256"]
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if checks["summary_exists"] else {}
    checks["summary_schema"] = summary.get("schema_version") == "deepseek-v4-image-interface-screen-v2"
    checks["software_pass"] = summary.get("status") == "software_interface_pass_hardware_pending"
    checks["gate_d_no_go"] = summary.get("gate_d_status") == "NO-GO"
    merged_checks = summary.get("checks", {})
    required_checks = (
        "routing_placeholder_repeated",
        "position_ids_contiguous",
        "image_labels_ignored",
        "text_labels_preserved",
        "projector_gradient_finite_nonzero",
        "language_gradient_all_none",
        "routing_ids_are_consumed",
        "position_ids_are_consumed",
    )
    checks["merge_and_gradient_checks"] = all(merged_checks.get(key) is True for key in required_checks)
    causal = summary.get("causal_interface_screen", {})
    checks["positive_causal_deltas"] = (
        float(causal.get("routing_id_ablation_max_abs_logit_delta", 0.0)) > 0
        and float(causal.get("position_id_ablation_max_abs_logit_delta", 0.0)) > 0
    )
    checks["synthetic_table_declared"] = "synthetic" in str(summary.get("contract", {}).get("tiny_hash_route_table", "")) or "mod num_experts" in str(summary.get("contract", {}).get("tiny_hash_route_table", ""))
    verified = all(checks.values())
    result = {
        "schema_version": "deepseek-v4-image-interface-screen-verifier-v1",
        "pointer": str(pointer_path),
        "verified": verified,
        "checks": checks,
        "interpretation": "Independent artifact and invariant verification only; Gate D remains NO-GO and full 0731 runtime is pending.",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.pointer)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
