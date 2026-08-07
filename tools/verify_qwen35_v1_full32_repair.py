#!/usr/bin/env python3
"""Verify the repaired Qwen3.5-4B MoonViT V1 full32 ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify(repo: Path, pointer_path: Path) -> dict:
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    root = pointer_path.parent / "qwen35_4b_moonvit_v1_full32_repair_20260808"
    expected_indices = list(range(32))
    checks: dict[str, bool] = {
        "schema": pointer.get("schema_version") == "qwen35-4b-moonvit-v1-full32-repair-pointer-v1",
        "diagnostic_reject": pointer.get("status") == "verified_diagnostic_reject",
        "capability_blocked": pointer.get("promotion", {}).get("capability_claim_allowed") is False,
        "full_public_blocked": pointer.get("promotion", {}).get("expand_full_public") is False,
        "pointer_indices_full32": pointer.get("training", {}).get("sample_indices") == expected_indices,
    }
    repair = json.loads((repo / pointer["repair_contract"]).read_text(encoding="utf-8"))
    parent = json.loads((repo / pointer["parent_contract"]).read_text(encoding="utf-8"))
    checks["repair_preregistered"] = repair.get("status") == "preregistered_before_repair_results"
    checks["repair_indices_full32"] = repair.get("training", {}).get("sample_indices") == expected_indices
    checks["parent_contract_count32"] = parent.get("training", {}).get("sample_count") == 32

    old_root = pointer_path.parent / "qwen35_4b_moonvit_v1_v2_ablation_20260808"
    for arm in ("train_ceonly", "train_margin05"):
        old_config = json.loads((old_root / arm / "RUN_CONFIG.json").read_text(encoding="utf-8"))
        checks[f"old_{arm}_is_eight_row_pilot"] = old_config.get("sample_indices") == list(range(8))

    v2_root = pointer_path.parent / "qwen35_4b_external_moonvit_ablation_20260808"
    for arm in ("train_ceonly", "train_margin05"):
        v2_config = json.loads((v2_root / arm / "RUN_CONFIG.json").read_text(encoding="utf-8"))
        checks[f"v2_{arm}_full32"] = v2_config.get("sample_indices") == expected_indices

    train_artifacts = {
        "ce_config": (root / "train_ceonly/RUN_CONFIG.json", pointer["training"]["ce_only"]["run_config_sha256"]),
        "ce_summary": (root / "train_ceonly/SUMMARY.json", pointer["training"]["ce_only"]["summary_sha256"]),
        "ce_health": (root / "train_ceonly/train_health.jsonl", pointer["training"]["ce_only"]["health_sha256"]),
        "margin_config": (root / "train_margin05/RUN_CONFIG.json", pointer["training"]["paired_margin05"]["run_config_sha256"]),
        "margin_summary": (root / "train_margin05/SUMMARY.json", pointer["training"]["paired_margin05"]["summary_sha256"]),
        "margin_health": (root / "train_margin05/train_health.jsonl", pointer["training"]["paired_margin05"]["health_sha256"]),
    }
    for name, (path, expected_sha) in train_artifacts.items():
        checks[f"{name}_sha"] = path.is_file() and sha256(path) == expected_sha
    for arm in ("train_ceonly", "train_margin05"):
        config = json.loads((root / arm / "RUN_CONFIG.json").read_text(encoding="utf-8"))
        summary = json.loads((root / arm / "SUMMARY.json").read_text(encoding="utf-8"))
        checks[f"{arm}_indices_full32"] = config.get("sample_indices") == expected_indices
        checks[f"{arm}_feature_rows32"] = len(config.get("token_selection", {}).get("feature_rows", [])) == 32
        trajectory = summary.get("trajectory", [])
        checks[f"{arm}_finite_steps0to3"] = [row.get("optimizer_step") for row in trajectory] == [0, 1, 2, 3] and all(not row.get("nan_or_inf", True) for row in trajectory)
        checks[f"{arm}_native_calls_zero"] = summary.get("native_vision_forward_calls") == 0

    role_dirs = {
        "step0": "eval_step0",
        "ce_only_full32": "eval_ceonly",
        "paired_margin05_full32": "eval_margin05",
    }
    expected_conditions = {"vision", "blind", "shuffled", "random_projector"}
    vision_orders = []
    for role, directory in role_dirs.items():
        role_pointer = pointer["evaluation"]["roles"][role]
        summary_path = root / directory / "SUMMARY.json"
        rows_path = root / directory / "generation_rows.jsonl"
        checks[f"{role}_summary_sha"] = sha256(summary_path) == role_pointer["summary_sha256"]
        checks[f"{role}_rows_sha"] = sha256(rows_path) == role_pointer["rows_sha256"]
        rows = json_rows(rows_path)
        checks[f"{role}_rows200"] = len(rows) == 200
        checks[f"{role}_conditions50_each"] = {row.get("condition") for row in rows} == expected_conditions and all(sum(row.get("condition") == condition for row in rows) == 50 for condition in expected_conditions)
        vision = [row for row in rows if row.get("condition") == "vision"]
        checks[f"{role}_vision_order"] = [row.get("sample_index") for row in vision] == list(range(50))
        vision_orders.append(tuple(str(row.get("sample_id")) for row in vision))
        ci = role_pointer["paired_click_ci"]
        checks[f"{role}_blind_gate_failed"] = ci["vision_minus_blind"][0] <= 0
        checks[f"{role}_shuffle_gate_failed"] = ci["vision_minus_shuffled"][0] <= 0
    checks["same_screen_order"] = len(set(vision_orders)) == 1

    verified = all(checks.values())
    return {
        "schema_version": "qwen35-4b-moonvit-v1-full32-repair-verifier-v1",
        "pointer": str(pointer_path),
        "verified": verified,
        "checks": checks,
        "interpretation": "The repair uses all 32 ordered training records for both V1 arms, matches the existing V2 budget, preserves four-condition ScreenSpot order, and remains a causal rejection rather than a visual-capability claim."
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--pointer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.repo, args.pointer)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
