#!/usr/bin/env python3
"""Independent verifier for the matched Qwen2.5-7B V1 architecture screen."""

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
    root = pointer_path.parent / "qwen25_7b_v1_community_screen_20260808"
    checks: dict[str, bool] = {
        "pointer_schema": pointer.get("schema_version") == "qwen25-7b-v1-community-screen-result-pointer-v1",
        "decision_rejects_promotion": pointer.get("decision", {}).get("promote_v1_to_previous_best") is False,
    }
    artifact_map = {
        "train_ceonly_summary_sha256": root / "train_ceonly/SUMMARY.json",
        "train_ceonly_health_sha256": root / "train_ceonly/train_health.jsonl",
        "train_margin05_summary_sha256": root / "train_margin05/SUMMARY.json",
        "train_margin05_health_sha256": root / "train_margin05/train_health.jsonl",
        "probe_ceonly_summary_sha256": root / "probe_ceonly/SUMMARY.json",
        "probe_ceonly_rows_sha256": root / "probe_ceonly/probe_metrics.jsonl",
        "probe_margin05_summary_sha256": root / "probe_margin05/SUMMARY.json",
        "probe_margin05_rows_sha256": root / "probe_margin05/probe_metrics.jsonl",
        "bootstrap_all_sha256": root / "qwen25_7b_v1_v2_matched_bootstrap_20260808.json",
        "bootstrap_v1_margin_vs_ce_sha256": root / "qwen25_7b_v1_margin_vs_ce_bootstrap_20260808.json",
        "bootstrap_v1_vs_v2_margin05_sha256": root / "qwen25_7b_v1_vs_v2_margin05_bootstrap_20260808.json",
    }
    for key, path in artifact_map.items():
        checks[f"{key}_exists"] = path.is_file()
        checks[key] = checks[f"{key}_exists"] and sha256(path) == pointer["compact_artifacts"][key]

    manifest = json.loads((pointer_path.parent / "qwen25_7b_real_probe32_manifest.json").read_text(encoding="utf-8")) if (pointer_path.parent / "qwen25_7b_real_probe32_manifest.json").is_file() else None
    if manifest is None:
        checks["sample_order"] = False
    else:
        expected_ids = [str(row["id"]) for row in manifest["records"]]
        order_checks = []
        for name in ("probe_ceonly", "probe_margin05"):
            rows = [json.loads(line) for line in (root / name / "probe_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            order_checks.append(len(rows) == 32 and [str(row["sample_id"]) for row in rows] == expected_ids)
        checks["sample_order"] = all(order_checks)

    for name in ("train_ceonly", "train_margin05"):
        summary = json.loads((root / name / "SUMMARY.json").read_text(encoding="utf-8"))
        trajectory = summary.get("trajectory", [])
        checks[f"{name}_four_finite_steps"] = [row.get("optimizer_step") for row in trajectory] == [0, 1, 2, 3] and all(not row.get("nan_or_inf", True) for row in trajectory)
        checks[f"{name}_native_vision_bypassed"] = summary.get("native_vision_forward_calls") == 0

    bootstrap = json.loads((root / "qwen25_7b_v1_v2_matched_bootstrap_20260808.json").read_text(encoding="utf-8"))
    v1_margin = bootstrap["runs"]["v1_margin05"]["vision_minus_shuffle"]
    checks["bootstrap_2000"] = bootstrap.get("bootstrap_samples") == 2000
    checks["v1_shuffle_ci_crosses_zero"] = v1_margin["ci95_lower"] < 0 < v1_margin["ci95_upper"]
    checks["v1_vs_v2_margin_ci_below_zero"] = bootstrap["paired_token_condition_deltas"].get("v1_margin05_minus_v2_ce", {}).get("ci95_upper", 1.0) < 0 or json.loads((root / "qwen25_7b_v1_vs_v2_margin05_bootstrap_20260808.json").read_text(encoding="utf-8"))["paired_token_condition_deltas"]["v1_margin05_minus_v2_margin05"]["ci95_upper"] < 0
    verified = all(checks.values())
    return {
        "schema_version": "qwen25-7b-v1-community-screen-verifier-v1",
        "pointer": str(pointer_path),
        "verified": verified,
        "checks": checks,
        "interpretation": "Verified raw compact artifacts, sample order, finite training, 2,000-bootstrap screen and the preregistered V1 rejection. This remains a Qwen teacher-forced diagnostic and does not claim DeepSeek capability.",
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
