#!/usr/bin/env python3
"""独立验证 projector retention 实验包的来源、分母、哈希与失效记录。"""

from __future__ import annotations

import argparse
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


def verify_bound_files(directory: Path, summary: dict) -> int:
    count = 0
    for name, expected in summary["files"].items():
        path = directory / name
        if path.stat().st_size != int(expected["bytes"]) or sha256(path) != expected[
            "sha256"
        ]:
            raise ValueError(f"summary-bound file mismatch: {path}")
        count += 1
    return count


def count_jsonl(path: Path) -> tuple[int, int]:
    rows = 0
    failures = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            failures += row.get("failure") is not None
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.package.resolve()
    output_path = args.out or root / "PACKAGE_VERIFICATION.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite package verification: {output_path}")

    raw = root / "raw" / "adaptation"
    training_names = (
        "projector_retention_resume_control_v2",
        "projector_retention_anchor_1e-4_v2",
        "projector_retention_anchor_1e-3_v2",
        "projector_retention_anchor_1e-2_v2",
    )
    training = {name: read_json(raw / name / "SUMMARY.json") for name in training_names}
    for name, summary in training.items():
        if summary.get("status") != "valid":
            raise ValueError(f"invalid retention training run: {name}")
        verify_bound_files(raw / name, summary)
        if summary["optimizer_resume"]["source_sha256"] != (
            "57e9ddbcd4a3b60594aa169c6d4e4b4adf2a5803d22a9da41e4fad4c23ac2f32"
        ):
            raise ValueError(f"optimizer source drifted: {name}")
        if summary["training_order_resume"]["source_sha256"] != (
            "a0929326fc8b494f85b82fc504855cd23f1ddfa74348ccc9b364a13ada39f2f5"
        ):
            raise ValueError(f"training order source drifted: {name}")
    first_batches = {
        tuple(summary["training_order_resume"]["first_batch_ids"])
        for summary in training.values()
    }
    if len(first_batches) != 1:
        raise ValueError("retention arms do not share the same first continuation batch")
    control = training["projector_retention_resume_control_v2"]
    exact = control.get("exact_reproduction")
    if exact is None or exact.get("status") != "exact" or exact["matched_tensors"] != 6:
        raise ValueError("unregularized continuation did not reproduce exactly")

    screen_dir = raw / "projector_retention_screen_v2"
    screen = read_json(screen_dir / "SUMMARY.json")
    if screen.get("status") != "valid":
        raise ValueError("retention screen is invalid")
    screen_files = verify_bound_files(screen_dir, screen)
    preference_rows, preference_failures = count_jsonl(
        screen_dir / "preference_records.jsonl"
    )
    generation_rows, generation_failures = count_jsonl(
        screen_dir / "generation_records.jsonl"
    )
    if (preference_rows, generation_rows) != (9000, 6000):
        raise ValueError("retention screen denominator mismatch")
    if preference_failures or generation_failures:
        raise ValueError("retention screen contains failed rows")

    analysis_dir = raw / "projector_retention_analysis_v2"
    analysis = read_json(analysis_dir / "SUMMARY.json")
    decisions = read_json(analysis_dir / "DECISIONS.json")
    if analysis.get("status") != "valid" or decisions.get("status") != "valid":
        raise ValueError("retention analysis is invalid")
    analysis_files = verify_bound_files(analysis_dir, analysis)
    if analysis["evaluation_summary_sha256"] != sha256(screen_dir / "SUMMARY.json"):
        raise ValueError("retention analysis is not bound to the packaged screen")
    for endpoint in ("frozen-base", "resume-control"):
        reproduced = decisions["endpoint_evaluation_reproduction"][endpoint]
        if reproduced["preference_rows_exact"] != 1800:
            raise ValueError(f"teacher-forced endpoint did not reproduce: {endpoint}")

    invalidations = sorted(raw.glob("invalid-*/INVALIDATION.json"))
    if len(invalidations) != 7:
        raise ValueError("retention package must preserve seven invalidation records")
    for path in invalidations:
        invalidation = read_json(path)
        if (
            invalidation.get("status") != "invalid"
            or invalidation.get("old_results_must_not_be_used") is not True
        ):
            raise ValueError(f"ambiguous invalidation record: {path}")

    output = {
        "status": "valid",
        "training_runs": len(training),
        "shared_first_batch_ids": len(next(iter(first_batches))),
        "exact_control_reproduction": exact,
        "screen": {
            "states": screen["states"],
            "preference_rows": preference_rows,
            "generation_rows": generation_rows,
            "preference_failures": preference_failures,
            "generation_failures": generation_failures,
            "files_verified": screen_files,
        },
        "analysis": {
            "metric_rows": analysis["metric_rows"],
            "contrasts": analysis["contrasts"],
            "files_verified": analysis_files,
            "selected_state": decisions["selected_state"],
            "targeted_retention_pass": decisions["targeted_retention_pass"],
            "endpoint_evaluation_reproduction": decisions[
                "endpoint_evaluation_reproduction"
            ],
        },
        "invalidations_verified": len(invalidations),
        "final_half_scored": False,
    }
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
