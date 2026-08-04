#!/usr/bin/env python3
"""给预注册 benchmark 控制补充逐特征网格匹配的 blank 图。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from moonvit_glue.trajectory_data import (
    _sha256,
    _write_jsonl,
    make_shape_matched_blank_controls,
)
from tools_common import load_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--feature-cache-records", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    updated, blank_records = make_shape_matched_blank_controls(
        load_records(args.data),
        load_records(args.controls),
        load_records(args.feature_cache_records),
        output_dir=args.out,
    )
    controls_path = args.out / "controls.jsonl"
    blank_path = args.out / "blank_records.jsonl"
    _write_jsonl(controls_path, updated)
    _write_jsonl(blank_path, blank_records)
    manifest = {
        "format_version": "benchmark-shape-matched-blank-controls-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "blank_rgb": [255, 255, 255],
        "records": len(updated),
        "unique_feature_shapes": len(blank_records),
        "final_half_materialized": False,
        "final_half_scored": False,
        "sources": {
            "data": {"path": str(args.data.resolve()), "sha256": _sha256(args.data)},
            "controls": {
                "path": str(args.controls.resolve()),
                "sha256": _sha256(args.controls),
            },
            "feature_cache_records": {
                "path": str(args.feature_cache_records.resolve()),
                "sha256": _sha256(args.feature_cache_records),
            },
        },
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (controls_path, blank_path)
        },
    }
    (args.out / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
