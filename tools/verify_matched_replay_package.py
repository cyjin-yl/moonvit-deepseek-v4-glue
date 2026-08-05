#!/usr/bin/env python3
"""验证 Package 13 的固定预算 replay 产物、统计分母与决策链。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def verify_declared_files(directory: Path, declared: dict) -> None:
    for name, metadata in declared.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(metadata["bytes"]) or sha256(path) != metadata["sha256"]:
            raise ValueError(f"declared artifact drifted: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root
    raw = root / "raw" / "adaptation"
    plan = read_json(raw / "matched_replay_plan_v1" / "PLAN_VERIFICATION.json")
    ordinary_dir = raw / "matched_replay_ordinary_v1"
    fixed_dir = raw / "matched_replay_fixed_v1"
    triggered_dir = raw / "matched_replay_triggered_v1"
    ordinary = read_json(ordinary_dir / "SUMMARY.json")
    fixed = read_json(fixed_dir / "SUMMARY.json")
    triggered = read_json(triggered_dir / "SUMMARY.json")
    sentinel_dir = raw / "matched_replay_sentinel_eval_v1"
    sentinel_summary = read_json(sentinel_dir / "SUMMARY.json")
    sentinel_decision_path = raw / "matched_replay_sentinel_analysis_v1" / "DECISION.json"
    sentinel = read_json(sentinel_decision_path)
    final_dir = raw / "matched_replay_final_eval_v1"
    final_summary = read_json(final_dir / "SUMMARY.json")
    analysis_dir = raw / "matched_replay_analysis_v1"
    decision = read_json(analysis_dir / "DECISIONS.json")
    charts = read_json(root / "charts" / "CHARTS.json")

    forbidden = [path for path in root.rglob("*") if path.suffix in {".safetensors", ".pt", ".bin"}]
    if forbidden:
        raise ValueError(f"package contains large checkpoint weights: {forbidden[:3]}")
    if plan.get("status") != "valid" or not all(plan["checks"].values()):
        raise ValueError("matched replay plan is not fully verified")
    if plan["budget_per_arm"] != {"steps": 50, "examples": 1200}:
        raise ValueError("plan budget drifted")
    if plan["fixed_records_by_task"] != {
        "color": 180,
        "coordinate": 180,
        "count": 240,
        "ocr": 180,
        "shape": 240,
        "spatial": 180,
    }:
        raise ValueError("fixed replay plan allocation drifted")

    for summary in (ordinary, fixed, triggered):
        if summary.get("status") != "valid" or summary.get("final_half_scored") is not False:
            raise ValueError("a replay training summary is invalid")
        verify_declared_files(
            ordinary_dir if summary is ordinary else fixed_dir if summary is fixed else triggered_dir,
            summary["files"],
        )
    if ordinary["fixed_training_budget"] != {"steps": 50, "examples": 1200}:
        raise ValueError("ordinary budget drifted")
    if fixed["fixed_training_budget"] != {"steps": 50, "examples": 1200}:
        raise ValueError("fixed replay budget drifted")
    if triggered["fixed_training_budget"] != {"steps": 25, "examples": 600}:
        raise ValueError("triggered remaining budget drifted")
    if ordinary["exact_reproduction"]["status"] != "exact":
        raise ValueError("ordinary control did not exactly reproduce the historical endpoint")
    if fixed["replay_policy"]["added_records_by_task"] != {"count": 40, "shape": 40}:
        raise ValueError("fixed replay additions drifted")
    if sum(fixed["replay_policy"]["final_records_by_task"].values()) != 1200:
        raise ValueError("fixed replay changed the training example budget")
    if sentinel["trigger_tasks"] != ["count"]:
        raise ValueError("sentinel trigger decision drifted")
    if triggered["replay_policy"]["added_records_by_task"] != {"count": 20}:
        raise ValueError("triggered replay additions drifted")
    if sum(triggered["replay_policy"]["final_records_by_task"].values()) != 600:
        raise ValueError("triggered replay changed the remaining example budget")
    triggered_config = read_json(triggered_dir / "CONFIG.json")
    trigger_source = triggered_config["arms"]["triggered_replay"]["trigger_decision"]
    if trigger_source["sha256"] != sha256(sentinel_decision_path):
        raise ValueError("trigger decision provenance drifted")

    arm_checkpoint_manifests = sum(
        len(list((directory / "checkpoints").glob("step-*/MANIFEST.json")))
        for directory in (ordinary_dir, fixed_dir, triggered_dir)
    )
    if arm_checkpoint_manifests != 8:
        raise ValueError("replay checkpoint manifest count drifted")

    verify_declared_files(sentinel_dir, sentinel_summary["files"])
    sentinel_preference_rows = jsonl_rows(sentinel_dir / "preference_records.jsonl")
    sentinel_generation_rows = jsonl_rows(sentinel_dir / "generation_records.jsonl")
    if (sentinel_preference_rows, sentinel_generation_rows) != (21600, 3600):
        raise ValueError("sentinel evaluation denominator drifted")
    verify_declared_files(final_dir, final_summary["files"])
    preference_rows = jsonl_rows(final_dir / "preference_records.jsonl")
    generation_rows = jsonl_rows(final_dir / "generation_records.jsonl")
    if (preference_rows, generation_rows) != (50400, 8400) or int(final_summary["states"]) != 7:
        raise ValueError("final matched replay evaluation denominator drifted")

    metric_rows = csv_rows(analysis_dir / "metrics.csv")
    contrast_rows = csv_rows(analysis_dir / "contrasts.csv")
    trajectory_rows = csv_rows(analysis_dir / "trajectories.csv")
    if (metric_rows, contrast_rows, trajectory_rows) != (735, 223, 18):
        raise ValueError("matched replay analysis row counts drifted")
    if decision["recommendation"] != "fixed_preventive_replay":
        raise ValueError("matched replay recommendation drifted")
    fixed_policy = decision["fixed_policy"]
    triggered_policy = decision["triggered_policy"]
    if fixed_policy["status"] != "supported" or triggered_policy["status"] != "supported":
        raise ValueError("replay primary effect status drifted")
    if float(fixed_policy["target_preference"]["ci95_low"]) <= 0:
        raise ValueError("fixed target preference CI no longer supports replay")
    if abs(float(fixed_policy["donor_preference"]["mean_gap"])) > 0.05:
        raise ValueError("fixed donor cost exceeded the preregistered bound")
    if float(fixed_policy["target_generation"]["ci95_low"]) <= 0:
        raise ValueError("fixed target generation CI no longer supports replay")
    if not bool(fixed_policy["count_recovery"]["recovered_within_tolerance"]):
        raise ValueError("fixed count recovery status drifted")
    if bool(triggered_policy["count_recovery"]["recovered_within_tolerance"]):
        raise ValueError("triggered count unexpectedly meets the recovery criterion")
    if float(decision["fixed_minus_triggered_overall_preference"]["ci95_low"]) <= 0:
        raise ValueError("fixed-versus-triggered endpoint preference CI drifted")
    if decision["training_budgets"]["total_examples_per_policy"] != 1200:
        raise ValueError("analysis training budget drifted")
    if sha256(final_dir / "preference_records.jsonl") != decision["sources"]["preference_records_sha256"]:
        raise ValueError("analysis preference source drifted")
    if sha256(final_dir / "generation_records.jsonl") != decision["sources"]["generation_records_sha256"]:
        raise ValueError("analysis generation source drifted")

    verify_declared_files(root / "charts", charts["charts"])
    if charts.get("status") != "valid" or len(charts["charts"]) != 4:
        raise ValueError("matched replay chart manifest is invalid")
    if not (root / "logs" / "analysis.failed-state-order.log").is_file():
        raise ValueError("corrected analysis failure log is missing")

    output = {
        "status": "valid",
        "format_version": "matched-replay-package-verification-v1",
        "source_git_sha": ordinary["metadata"]["git_sha"],
        "training": {
            "total_examples_per_policy": 1200,
            "ordinary_steps": 50,
            "fixed_steps": 50,
            "triggered_shared_steps": 25,
            "triggered_replay_steps": 25,
            "fixed_reallocated_examples": 80,
            "triggered_reallocated_examples": 20,
            "extra_training_examples": 0,
            "checkpoint_manifests": arm_checkpoint_manifests,
            "ordinary_exact_reproduction": True,
        },
        "evaluation": {
            "sentinel_preference_rows": sentinel_preference_rows,
            "sentinel_generation_rows": sentinel_generation_rows,
            "final_states": 7,
            "preference_rows": preference_rows,
            "generation_rows": generation_rows,
            "metric_rows": metric_rows,
            "contrast_rows": contrast_rows,
            "trajectory_rows": trajectory_rows,
        },
        "decision": {
            "trigger_tasks": ["count"],
            "recommendation": decision["recommendation"],
            "fixed_target_preference_gap": fixed_policy["target_preference"]["mean_gap"],
            "fixed_target_preference_ci95": [
                fixed_policy["target_preference"]["ci95_low"],
                fixed_policy["target_preference"]["ci95_high"],
            ],
            "fixed_target_generation_gap": fixed_policy["target_generation"]["mean_gap"],
            "fixed_minus_triggered_overall_gap": decision["fixed_minus_triggered_overall_preference"]["mean_gap"],
        },
        "large_checkpoint_weights_in_git": False,
        "failed_attempt_preserved": True,
        "final_half_scored": False,
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    output_path = args.out or root / "PACKAGE_VERIFICATION.json"
    output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
