#!/usr/bin/env python3
"""为冻结特征缓存固化唯一的 blank 与 same-image 记录。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moonvit_glue.trajectory_metrics import control_image_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite control records: {args.out}")
    rows = [json.loads(line) for line in args.controls.read_text(encoding="utf-8").splitlines() if line]
    records = control_image_records(rows)
    with args.out.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"status": "valid", "records": len(records), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
