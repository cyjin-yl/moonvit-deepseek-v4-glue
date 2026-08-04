#!/usr/bin/env python3
"""准备固定 benchmark selection 与历史 heldout 轨迹 JSONL。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moonvit_glue.trajectory_data import prepare_trajectory_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", action="append", required=True, type=Path, dest="eval_files")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--heldout-count", type=int, default=32)
    args = parser.parse_args()
    manifest = prepare_trajectory_data(
        eval_files=args.eval_files,
        train_file=args.train,
        output_dir=args.out,
        heldout_count=args.heldout_count,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
