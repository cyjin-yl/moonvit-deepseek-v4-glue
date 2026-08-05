#!/usr/bin/env python3
"""独立重读 package 6 的原始行数、摘要绑定和联合分析哈希。"""

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


def jsonl_rows(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(bool(line.strip()) for line in stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.package.resolve()
    preference = root / "preference"
    generation = root / "generation"
    analysis = root / "analysis"

    preference_summary = json.loads(
        (preference / "SUMMARY.json").read_text(encoding="utf-8")
    )
    generation_summary = json.loads(
        (generation / "SUMMARY.json").read_text(encoding="utf-8")
    )
    analysis_summary = json.loads(
        (analysis / "SUMMARY.json").read_text(encoding="utf-8")
    )
    preference_verification = json.loads(
        (preference / "VERIFICATION.json").read_text(encoding="utf-8")
    )
    generation_verification = json.loads(
        (generation / "VERIFICATION.json").read_text(encoding="utf-8")
    )
    if any(
        row.get("status") != "valid"
        for row in (
            preference_summary,
            generation_summary,
            analysis_summary,
            preference_verification,
            generation_verification,
        )
    ):
        raise ValueError("package contains a non-valid canonical summary")
    if analysis_summary["preference_summary_sha256"] != sha256(
        preference / "SUMMARY.json"
    ):
        raise ValueError("preference summary hash mismatch")
    if analysis_summary["generation_summary_sha256"] != sha256(
        generation / "SUMMARY.json"
    ):
        raise ValueError("generation summary hash mismatch")

    preference_rows = jsonl_rows(preference / "preference_records.jsonl")
    generation_rows = jsonl_rows(generation / "records.jsonl")
    if preference_rows != analysis_summary["preference_rows"]:
        raise ValueError("preference raw-row count mismatch")
    if generation_rows != analysis_summary["generation_rows"]:
        raise ValueError("generation raw-row count mismatch")
    if preference_rows != preference_verification["rows_verified"]:
        raise ValueError("preference verifier count mismatch")
    if generation_rows != generation_verification["generation_rows_verified"]:
        raise ValueError("generation verifier count mismatch")

    for filename, expected in analysis_summary["files"].items():
        path = analysis / filename
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"analysis artifact mismatch: {filename}")
    decisions = json.loads((analysis / "DECISIONS.json").read_text(encoding="utf-8"))
    if decisions["latest_checkpoint"] != "shape-projector-step50":
        raise ValueError("unexpected transfer checkpoint")

    payload = {
        "status": "valid",
        "preference_rows_verified": preference_rows,
        "generation_rows_verified": generation_rows,
        "preference_cells_verified": preference_verification["cells_verified"],
        "generation_checkpoints_verified": generation_verification[
            "unique_checkpoints_verified"
        ],
        "metric_rows_verified": analysis_summary["metric_rows"],
        "contrast_rows_verified": analysis_summary["contrast_rows"],
        "bootstrap_samples": analysis_summary["bootstrap_samples"],
        "validated_preference_transfer_tasks": decisions[
            "validated_preference_transfer_tasks"
        ],
        "validated_generation_transfer_tasks": decisions[
            "validated_generation_transfer_tasks"
        ],
        "broad_non_shape_transfer_supported": decisions[
            "broad_non_shape_transfer_supported"
        ],
        "shape_specific_supported": decisions["shape_specific_supported"],
        "final_half_scored": False,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
