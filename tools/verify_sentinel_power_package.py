#!/usr/bin/env python3
"""验证 Package 14 sentinel 功效、V100 timing 与预算合同。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED_GIT_SHA = "d7eb18e97e5981563c397c705c933921e2a558ec"
TASKS = ["color", "coordinate", "count", "ocr", "shape", "spatial"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def jsonl_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def verify_declared_files(directory: Path, declared: dict) -> None:
    for name, metadata in declared.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(metadata["bytes"]) or sha256(path) != str(
            metadata["sha256"]
        ):
            raise ValueError(f"declared artifact drifted: {path}")


def close(actual: float, expected: float, tolerance: float = 1e-12) -> bool:
    return abs(float(actual) - float(expected)) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root
    raw = root / "raw" / "adaptation"
    power_dir = raw / "sentinel_power_analysis_v1"
    timing_dir = raw / "sentinel_timing_analysis_v1"
    power = read_json(power_dir / "DECISIONS.json")
    timing = read_json(timing_dir / "DECISIONS.json")
    if power.get("status") != "valid" or timing.get("status") != "valid":
        raise ValueError("sentinel power or timing decision is invalid")
    if power["source_git_sha"] != EXPECTED_GIT_SHA or timing["power_source_git_sha"] != EXPECTED_GIT_SHA:
        raise ValueError("Package 14 source Git SHA drifted")
    if power["tasks"] != TASKS or power["full_source_trigger_tasks"] != ["count"]:
        raise ValueError("full sentinel task or trigger set drifted")
    if set(power["full_source_pairs_per_task"].values()) != {200}:
        raise ValueError("full sentinel pair denominator drifted")
    candidates = {int(row["pairs_per_task"]): row for row in power["candidate_summary"]}
    if sorted(candidates) != [8, 16, 25, 50, 100]:
        raise ValueError("sentinel candidate grid drifted")
    if power["recommended_tiny_pairs_per_task"] != 25 or power["recommended_medium_pairs_per_task"] != 50:
        raise ValueError("Tiny/Medium sentinel selection drifted")
    expected_power = {
        8: (0.375, 0.36, 0.015, False),
        16: (0.76, 0.72, 0.045, False),
        25: (0.975, 0.935, 0.04, True),
        50: (1.0, 0.965, 0.035, True),
        100: (1.0, 1.0, 0.0, True),
    }
    for size, expected in expected_power.items():
        row = candidates[size]
        actual = (
            float(row["count_recall"]),
            float(row["exact_decision_rate"]),
            float(row["familywise_false_trigger_rate"]),
            bool(row["passes_preregistered_criteria"]),
        )
        if not all(close(a, b) for a, b in zip(actual[:3], expected[:3], strict=True)) or actual[3] != expected[3]:
            raise ValueError(f"sentinel power result drifted at {size} pairs")
    if len(csv_rows(power_dir / "full_contrasts.csv")) != 6:
        raise ValueError("full contrast row count drifted")
    if len(csv_rows(power_dir / "candidate_summary.csv")) != 5:
        raise ValueError("candidate summary row count drifted")
    if len(csv_rows(power_dir / "trial_decisions.csv")) != 1000:
        raise ValueError("trial decision row count drifted")
    if len(csv_rows(power_dir / "task_trials.csv")) != 6000:
        raise ValueError("task trial row count drifted")

    profiles = {row["profile"]: row for row in timing["profiles"]}
    if timing["recommended_profile"] != "tiny" or set(profiles) != {"tiny", "medium"}:
        raise ValueError("timing recommendation drifted")
    if not close(profiles["tiny"]["teacher_forced_seconds_median"], 22.501351576764137):
        raise ValueError("Tiny teacher timing drifted")
    if not close(profiles["medium"]["teacher_forced_seconds_median"], 43.880532410461456):
        raise ValueError("Medium teacher timing drifted")
    model_resident = timing["overhead_contract"]["model_resident_tiny"]
    if model_resident != {
        "0.05": {"minimum_interval_steps": 476, "rounded_power_of_two_interval_steps": 512},
        "0.1": {"minimum_interval_steps": 226, "rounded_power_of_two_interval_steps": 256},
    }:
        raise ValueError("Tiny overhead interval contract drifted")
    if timing["training_action"] != {
        "use_fixed_preventive_replay": True,
        "run_tiny_sentinel_as_sparse_checkpoint_audit": True,
        "run_tiny_sentinel_every_25_small_model_steps": False,
        "reason": "Tiny teacher cost is comparable to one 25-step V100 training window",
        "training_examples_changed": False,
        "training_steps_changed": False,
    }:
        raise ValueError("training action drifted")
    if len(csv_rows(timing_dir / "timing_profiles.csv")) != 2 or len(
        csv_rows(timing_dir / "overhead_budget.csv")
    ) != 8:
        raise ValueError("timing analysis row counts drifted")

    repeated_hashes = {"tiny": set(), "medium": set()}
    timing_rows = 0
    for profile, pairs in (("tiny", 25), ("medium", 50)):
        for repeat in (1, 2, 3):
            directory = raw / f"sentinel_timing_{profile}_rep{repeat}_v1"
            summary = read_json(directory / "SUMMARY.json")
            config = read_json(directory / "CONFIG.json")
            if summary.get("status") != "valid" or summary.get("final_half_scored") is not False:
                raise ValueError(f"invalid timing repeat: {directory}")
            if summary["metadata"]["git_sha"] != EXPECTED_GIT_SHA:
                raise ValueError("timing repeat Git SHA drifted")
            if summary["states"] != 2 or summary["teacher_conditions"] != ["vision"]:
                raise ValueError("timing repeat state/condition drifted")
            if summary["generation_skipped"] is not True or summary["generation_rows"] != 0:
                raise ValueError("timing repeat ran generation")
            expected_records = pairs * 2 * len(TASKS)
            if summary["teacher_forced_records_per_cell"] != expected_records or summary["preference_rows"] != expected_records * 2:
                raise ValueError("timing repeat denominator drifted")
            override = config["evaluation_override"]
            if override["limit_per_task"] != pairs * 2:
                raise ValueError("timing per-task record limit drifted")
            if override["state_ids"] != ["exchange-step50", "ordinary-step75"]:
                raise ValueError("timing state IDs drifted")
            if override["teacher_conditions"] != ["vision"] or override["skip_generation"] is not True:
                raise ValueError("timing teacher-only override drifted")
            verify_declared_files(directory, summary["files"])
            count = jsonl_count(directory / "preference_records.jsonl")
            if count != expected_records * 2:
                raise ValueError("timing raw preference row count drifted")
            timing_rows += count
            repeated_hashes[profile].add(
                summary["files"]["preference_records.jsonl"]["sha256"]
            )
            if (directory / "FAILURE.json").exists():
                raise ValueError("timing repeat retained an unexpected failure")
            log = root / "logs" / f"sentinel_timing_{profile}_rep{repeat}_v1.log"
            text = log.read_text(encoding="utf-8")
            if "completed exchange-step50" not in text or "completed ordinary-step75" not in text:
                raise ValueError("timing log is incomplete")
    if any(len(values) != 1 for values in repeated_hashes.values()):
        raise ValueError("timing raw preference hashes differ across repeats")

    charts = read_json(root / "charts" / "CHARTS.json")
    verify_declared_files(root / "charts", charts["charts"])
    if charts.get("status") != "valid" or len(charts["charts"]) != 3:
        raise ValueError("sentinel chart manifest is invalid")
    forbidden = [
        path
        for path in root.rglob("*")
        if path.suffix in {".safetensors", ".pt", ".bin"}
    ]
    if forbidden:
        raise ValueError(f"package contains checkpoint weights: {forbidden[:3]}")
    output = {
        "status": "valid",
        "format_version": "sentinel-power-package-verification-v1",
        "source_git_sha": EXPECTED_GIT_SHA,
        "power": {
            "candidate_pairs_per_task": [8, 16, 25, 50, 100],
            "trials_per_candidate": 200,
            "bootstrap_samples_per_contrast": 2000,
            "trial_rows": 1000,
            "task_trial_rows": 6000,
            "tiny_pairs_per_task": 25,
            "medium_pairs_per_task": 50,
            "tiny_count_recall": 0.975,
            "tiny_exact_decision_rate": 0.935,
            "tiny_familywise_false_trigger_rate": 0.04,
        },
        "timing": {
            "repeats_per_profile": 3,
            "raw_preference_rows": timing_rows,
            "tiny_teacher_seconds_median": 22.501351576764137,
            "medium_teacher_seconds_median": 43.880532410461456,
            "tiny_minimum_interval_steps_5pct": 476,
            "tiny_minimum_interval_steps_10pct": 226,
            "peak_gpu_memory_bytes": 6886173184,
            "repeated_preference_hashes_exact": True,
        },
        "decision": {
            "recommended_profile": "tiny",
            "fixed_preventive_replay_default": True,
            "sentinel_role": "sparse checkpoint audit",
            "every_25_small_model_steps": False,
        },
        "large_checkpoint_weights_in_git": False,
        "final_half_scored": False,
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    output_path = args.out or root / "PACKAGE_VERIFICATION.json"
    output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
