#!/usr/bin/env python3
"""为实验 run 写入不可含混的 invalidation 记录。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--impact", required=True)
    parser.add_argument("--replacement-run", required=True)
    args = parser.parse_args()
    output = args.run / "INVALIDATION.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite invalidation: {output}")
    files = {}
    for name in (
        "CONFIG.json",
        "SUMMARY.json",
        "SUMMARY.partial.json",
        "preference_records.jsonl",
        "records.jsonl",
        "shuffle_loss_records.jsonl",
        "patching_records.jsonl",
        "patching_curve.csv",
        "probe_metrics.csv",
        "probe_intervals.csv",
        "probe_predictions.jsonl",
        "PROBES.safetensors",
        "DECISIONS.json",
        "failures.jsonl",
    ):
        path = args.run / name
        if path.exists():
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    payload = {
        "status": "invalid",
        "invalidated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": args.reason,
        "impact": args.impact,
        "replacement_run": args.replacement_run,
        "old_results_must_not_be_used": True,
        "files": files,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
