#!/usr/bin/env python3
"""独立核验 Qwen3.5-4B 外接 MoonViT 的固定四条件对照产物。"""

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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify(pointer_path: Path) -> dict[str, object]:
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    root = pointer_path.parent / "qwen35_4b_external_moonvit_ablation_20260808"
    checks: dict[str, bool] = {
        "pointer_schema": pointer.get("schema_version") == "qwen35-4b-external-moonvit-ablation-pointer-v1",
        "diagnostic_reject": pointer.get("status") == "verified_diagnostic_reject",
        "capability_claim_blocked": pointer.get("promotion", {}).get("capability_claim_allowed") is False,
        "native_visual_bypassed": pointer.get("native_vision_bypassed") is True and pointer.get("native_vision_forward_calls") == 0,
        "fixed_sample_count": pointer.get("evaluation", {}).get("sample_count") == 50,
        "bootstrap_2000": pointer.get("evaluation", {}).get("bootstrap_samples") == 2000,
    }

    mappings = {
        "train_ceonly_summary": (root / "train_ceonly/SUMMARY.json", pointer["training"]["ce_only"]["summary_sha256"]),
        "train_ceonly_health": (root / "train_ceonly/train_health.jsonl", pointer["training"]["ce_only"]["health_sha256"]),
        "train_margin_summary": (root / "train_margin05/SUMMARY.json", pointer["training"]["paired_margin05"]["summary_sha256"]),
        "train_margin_health": (root / "train_margin05/train_health.jsonl", pointer["training"]["paired_margin05"]["health_sha256"]),
    }
    role_dirs = {"step0": "eval_step0", "ce_only": "eval_ceonly", "paired_margin05": "eval_margin05"}
    for role, directory in role_dirs.items():
        entry = pointer["evaluation"]["roles"][role]
        mappings[f"{role}_rows"] = (root / f"{directory}/generation_rows.jsonl", entry["rows_sha256"])
        mappings[f"{role}_summary"] = (root / f"{directory}/SUMMARY.json", entry["summary_sha256"])
        category_dir = {"step0": "categories_step0", "ce_only": "categories_ceonly", "paired_margin05": "categories_margin05"}[role]
        mappings[f"{role}_category"] = (root / f"{category_dir}/CATEGORY_SUMMARY.json", entry["category_summary_sha256"])
    for name, (path, expected) in mappings.items():
        checks[f"{name}_exists"] = path.is_file()
        checks[f"{name}_sha256"] = checks[f"{name}_exists"] and sha256(path) == expected

    # 每个训练臂必须包含 step 0..3，且没有 NaN/Inf。
    for name in ("train_ceonly", "train_margin05"):
        summary = json.loads((root / name / "SUMMARY.json").read_text(encoding="utf-8"))
        trajectory = summary.get("trajectory", [])
        checks[f"{name}_finite_steps"] = [row.get("optimizer_step") for row in trajectory] == [0, 1, 2, 3] and all(not row.get("nan_or_inf", True) for row in trajectory)
        checks[f"{name}_native_calls_zero"] = summary.get("native_vision_forward_calls") == 0

    # 四条件必须共用相同 50 条样本顺序；随机 projector 与错误图像不能混入 vision 顺序。
    expected_conditions = {"vision", "blind", "shuffled", "random_projector"}
    condition_orders: dict[str, list[str]] = {}
    sample_orders: dict[str, list[str]] = {}
    for role, directory in role_dirs.items():
        rows = read_jsonl(root / f"{directory}/generation_rows.jsonl")
        condition_orders[role] = sorted({str(row.get("condition")) for row in rows})
        vision_rows = [row for row in rows if row.get("condition") == "vision"]
        sample_orders[role] = [str(row.get("sample_id")) for row in vision_rows]
        checks[f"{role}_four_conditions"] = len(rows) == 200 and set(condition_orders[role]) == expected_conditions and all(sum(row.get("condition") == c for row in rows) == 50 for c in expected_conditions)
        checks[f"{role}_vision_ordered"] = len(sample_orders[role]) == 50 and [row.get("sample_index") for row in vision_rows] == list(range(50))
    checks["same_vision_sample_order"] = len({tuple(order) for order in sample_orders.values()}) == 1

    # 复核摘要中的硬性结论：所有候选都没有 paired CI 下界同时大于零。
    for role in ("step0", "ce_only", "paired_margin05"):
        entry = pointer["evaluation"]["roles"][role]
        for comparison in ("vision_minus_blind", "vision_minus_shuffled"):
            checks[f"{role}_{comparison}_not_positive_ci"] = entry["paired_click_ci"][comparison][1] <= 0

    verified = all(checks.values())
    return {
        "schema_version": "qwen35-4b-external-moonvit-ablation-verifier-v1",
        "pointer": str(pointer_path),
        "verified": verified,
        "checks": checks,
        "interpretation": "Verified hashes, four-condition 50-row order, finite three-step training, zero native visual calls and preregistered diagnostic rejection. This artifact does not claim usable visual grounding or DeepSeek capability.",
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
