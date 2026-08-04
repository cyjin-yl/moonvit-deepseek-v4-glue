#!/usr/bin/env python3
"""校验 shape-matched benchmark blank 特征与控制分配。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    blanks = read_jsonl(args.controls_dir / "blank_records.jsonl")
    controls = read_jsonl(args.controls_dir / "controls.jsonl")
    cached = read_jsonl(args.cache_dir / "cache_records.jsonl")
    expected = {
        str(row["id"]): tuple(int(value) for value in row["expected_feature_shape"])
        for row in blanks
    }
    actual = {
        str(row["id"]): tuple(int(value) for value in row["feature_shape"])
        for row in cached
        if row.get("status") == "ok"
    }
    if expected != actual:
        mismatch = {
            key: {"expected": expected.get(key), "actual": actual.get(key)}
            for key in set(expected) | set(actual)
            if expected.get(key) != actual.get(key)
        }
        raise ValueError(f"blank feature-grid mismatch: {list(mismatch.items())[:3]}")
    assignments = Counter(str(row["blank_image_id"]) for row in controls)
    if set(assignments) != set(expected):
        raise ValueError("blank control assignments do not cover the cached blank IDs")
    failures = read_jsonl(args.cache_dir / "failures.jsonl")
    if failures:
        raise ValueError(f"benchmark blank cache has failures: {len(failures)}")
    payload = {
        "status": "valid",
        "controls": len(controls),
        "unique_feature_shapes": len(expected),
        "cached": len(actual),
        "failures": 0,
        "assignment_min": min(assignments.values()),
        "assignment_max": max(assignments.values()),
        "final_half_scored": False,
        "hashes": {
            "controls_manifest": sha256(args.controls_dir / "MANIFEST.json"),
            "controls": sha256(args.controls_dir / "controls.jsonl"),
            "blank_records": sha256(args.controls_dir / "blank_records.jsonl"),
            "cache_manifest": sha256(args.cache_dir / "MANIFEST.json"),
            "cache_records": sha256(args.cache_dir / "cache_records.jsonl"),
        },
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
