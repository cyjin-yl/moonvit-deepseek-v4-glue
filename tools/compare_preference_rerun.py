#!/usr/bin/env python3
"""确认仅修 provenance 的 preference 重跑没有改变任何数值。"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import zip_longest
from pathlib import Path


NUMERIC_FIELDS = (
    "correct_answer_tokens",
    "correct_logp_sum",
    "correct_logp_mean",
    "correct_token_nll",
    "counterfactual_answer_tokens",
    "counterfactual_logp_sum",
    "counterfactual_logp_mean",
    "counterfactual_token_nll",
    "correct_margin",
    "preference_correct",
)
KEY_FIELDS = ("checkpoint", "condition", "id", "pair_id", "pair_variant", "task")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invalid-run", required=True, type=Path)
    parser.add_argument("--replacement-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    old_path = args.invalid_run / "preference_records.jsonl"
    new_path = args.replacement_run / "preference_records.jsonl"
    compared = 0
    source_corrections = 0
    for line_number, pair in enumerate(
        zip_longest(rows(old_path), rows(new_path)), start=1
    ):
        old, new = pair
        if old is None or new is None:
            raise ValueError("preference rerun row count drift")
        if tuple(old[field] for field in KEY_FIELDS) != tuple(
            new[field] for field in KEY_FIELDS
        ):
            raise ValueError(f"preference rerun key drift at row {line_number}")
        for field in NUMERIC_FIELDS:
            if old[field] != new[field]:
                raise ValueError(
                    f"preference rerun numeric drift at row {line_number}: {field}"
                )
        source_corrections += old.get("visual_source_id") != new.get("visual_source_id")
        compared += 1
    payload = {
        "status": "valid",
        "rows_compared": compared,
        "numeric_fields_bit_identical": True,
        "visual_source_rows_corrected": source_corrections,
        "invalid_run_sha256": sha256(old_path),
        "replacement_run_sha256": sha256(new_path),
        "invalid_run_invalidation_sha256": sha256(
            args.invalid_run / "INVALIDATION.json"
        ),
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
