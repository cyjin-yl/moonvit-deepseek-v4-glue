#!/usr/bin/env python3
"""冻结任务均衡且 pair 完整的 synthetic 自由生成子集。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from moonvit_glue.trajectory_data import (
    _sha256,
    _write_jsonl,
    make_pair_stratified_subset,
)
from tools_common import load_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pairs-per-task", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--logical-dataset-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite generation subset: {args.out}")
    args.out.mkdir(parents=True)
    records = load_records(args.data)
    selected = make_pair_stratified_subset(
        records,
        pairs_per_task=args.pairs_per_task,
        seed=args.seed,
    )
    data_path = args.out / "synthetic_generation_selection.jsonl"
    _write_jsonl(data_path, selected)
    pair_counts = Counter((str(row["task"]), str(row["pair_id"])) for row in selected)
    manifest = {
        "format_version": "synthetic-generation-selection-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "free-generation diagnostic only; teacher-forced evaluation remains full-selection",
        "selection_rule": "per-task SHA-256 ranking of seed:task:pair_id; retain both a/b variants",
        "seed": args.seed,
        "pairs_per_task": args.pairs_per_task,
        "logical_dataset_sha256": args.logical_dataset_sha256,
        "source": {
            "path": str(args.data.resolve()),
            "bytes": args.data.stat().st_size,
            "sha256": _sha256(args.data),
            "records": len(records),
        },
        "selection": {
            "records": len(selected),
            "pairs": len(pair_counts),
            "records_by_task": dict(sorted(Counter(str(row["task"]) for row in selected).items())),
            "pairs_by_task": dict(sorted(Counter(task for task, _ in pair_counts).items())),
            "sample_ids": [str(row["id"]) for row in selected],
            "pair_ids": sorted({str(row["pair_id"]) for row in selected}),
        },
        "files": {
            data_path.name: {
                "bytes": data_path.stat().st_size,
                "sha256": _sha256(data_path),
            }
        },
    }
    (args.out / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
