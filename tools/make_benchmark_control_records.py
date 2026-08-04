#!/usr/bin/env python3
"""创建预注册的 benchmark 内部因果控制分配。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from moonvit_glue.trajectory_data import _sha256, _write_jsonl, make_stratified_control_records
from tools_common import load_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--split", default="selection")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite benchmark controls: {args.out}")
    args.out.mkdir(parents=True)

    source = args.data.resolve()
    rows = make_stratified_control_records(
        load_records(source), seed=args.seed, split=args.split
    )
    controls_path = args.out / "controls.jsonl"
    _write_jsonl(controls_path, rows)
    manifest = {
        "format_version": "benchmark-causal-controls-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "split": args.split,
        "shuffle_rule": "deterministic Sattolo derangement within benchmark",
        "patch_rule": "deterministic torch.randperm seed per sample",
        "final_half_materialized": False,
        "final_half_scored": False,
        "source": {"path": str(source), "sha256": _sha256(source)},
        "records": len(rows),
        "files": {
            controls_path.name: {
                "bytes": controls_path.stat().st_size,
                "sha256": _sha256(controls_path),
            }
        },
    }
    manifest_path = args.out / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
