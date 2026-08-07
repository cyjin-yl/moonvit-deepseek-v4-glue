#!/usr/bin/env python3
"""独立核验 Qwen3.5-4B MoonViT V1 完整四条件对照。"""

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


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify(pointer_path: Path) -> dict[str, object]:
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    root = pointer_path.parent / "qwen35_4b_moonvit_v1_v2_ablation_20260808"
    checks = {
        "schema": pointer.get("schema_version") == "qwen35-4b-moonvit-v1-ablation-pointer-v1",
        "diagnostic_reject": pointer.get("status") == "verified_diagnostic_reject",
        "capability_blocked": pointer.get("promotion", {}).get("capability_claim_allowed") is False,
        "native_path_bypassed": pointer.get("native_vision_bypassed") is True and pointer.get("native_vision_forward_calls") == 0,
        "sample_count": pointer.get("evaluation", {}).get("sample_count") == 50,
        "bootstrap": pointer.get("evaluation", {}).get("bootstrap_samples") == 2000,
    }
    artifacts = {
        "train_ce_summary": (root / "train_ceonly/SUMMARY.json", pointer["training"]["ce_only"]["summary_sha256"]),
        "train_ce_health": (root / "train_ceonly/train_health.jsonl", pointer["training"]["ce_only"]["health_sha256"]),
        "train_margin_summary": (root / "train_margin05/SUMMARY.json", pointer["training"]["paired_margin05"]["summary_sha256"]),
        "train_margin_health": (root / "train_margin05/train_health.jsonl", pointer["training"]["paired_margin05"]["health_sha256"]),
        "step0_summary": (root / "eval_step0/SUMMARY.json", pointer["evaluation"]["roles"]["step0"]["summary_sha256"]),
        "step0_rows": (root / "eval_step0/generation_rows.jsonl", pointer["evaluation"]["roles"]["step0"]["rows_sha256"]),
        "ce_summary": (root / "eval_ceonly/SUMMARY.json", pointer["evaluation"]["roles"]["ce_only"]["summary_sha256"]),
        "ce_rows": (root / "eval_ceonly/generation_rows.jsonl", pointer["evaluation"]["roles"]["ce_only"]["rows_sha256"]),
        "margin_summary": (root / "eval_margin05/SUMMARY.json", pointer["evaluation"]["roles"]["paired_margin05"]["summary_sha256"]),
        "margin_rows": (root / "eval_margin05/generation_rows.jsonl", pointer["evaluation"]["roles"]["paired_margin05"]["rows_sha256"]),
    }
    for name, (path, expected) in artifacts.items():
        checks[f"{name}_exists"] = path.is_file()
        checks[f"{name}_sha256"] = checks[f"{name}_exists"] and sha256(path) == expected

    for name in ("train_ceonly", "train_margin05"):
        summary = json.loads((root / name / "SUMMARY.json").read_text(encoding="utf-8"))
        trajectory = summary.get("trajectory", [])
        checks[f"{name}_finite_0_to_3"] = [r.get("optimizer_step") for r in trajectory] == [0, 1, 2, 3] and all(not r.get("nan_or_inf", True) for r in trajectory)
        checks[f"{name}_native_calls_zero"] = summary.get("native_vision_forward_calls") == 0

    expected_conditions = {"vision", "blind", "shuffled", "random_projector"}
    orders = []
    for role, directory in (("step0", "eval_step0"), ("ce_only", "eval_ceonly"), ("paired_margin05", "eval_margin05")):
        data = rows(root / f"{directory}/generation_rows.jsonl")
        checks[f"{role}_200_rows"] = len(data) == 200
        checks[f"{role}_four_conditions"] = {str(r.get("condition")) for r in data} == expected_conditions and all(sum(r.get("condition") == c for r in data) == 50 for c in expected_conditions)
        vision = [r for r in data if r.get("condition") == "vision"]
        orders.append(tuple(str(r.get("sample_id")) for r in vision))
        checks[f"{role}_vision_index_order"] = [r.get("sample_index") for r in vision] == list(range(50))
    checks["same_vision_order"] = len(set(orders)) == 1
    for role in ("step0", "ce_only", "paired_margin05"):
        ci = pointer["evaluation"]["roles"][role]["paired_click_ci"]
        checks[f"{role}_vision_blind_not_positive_ci"] = ci["vision_minus_blind"][1] <= 0
        checks[f"{role}_vision_shuffled_not_positive_ci"] = ci["vision_minus_shuffled"][1] <= 0

    verified = all(checks.values())
    return {
        "schema_version": "qwen35-4b-moonvit-v1-ablation-verifier-v1",
        "pointer": str(pointer_path),
        "verified": verified,
        "checks": checks,
        "interpretation": "Verified compact hashes, finite training, four-condition sample order, native-path bypass and diagnostic rejection. This is not a visual-capability claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.pointer)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
