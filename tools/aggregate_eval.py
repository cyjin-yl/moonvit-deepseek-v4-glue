"""Aggregate per-benchmark eval reports into SUMMARY.json (+ optional HF upload).

Scans ``--results-dir`` for eval report JSONs written by ``tools/eval_vlm.py``
(generation mode), builds a benchmark × condition matrix with the blind gap
per metric, marks benchmarks that are in-domain by construction (ScreenSpot:
GUI grounding data was part of training), and optionally uploads the whole
results directory — raw per-record predictions included — to the HF repo.

Example::

    python tools/aggregate_eval.py --results-dir eval_results/run1 \
        --upload-repo cyjin-yl/DeepSeek-V4-Flash-0731-Vision --run-tag gated-2100
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

# Benchmarks whose training data shares domain with the eval split. Their
# numbers read as in-domain generalization, never as held-out capability.
IN_DOMAIN_BENCHMARKS = frozenset({"screenspot"})


def aggregate_reports(reports: list[dict]) -> dict:
    benchmarks: dict[str, dict] = {}
    skipped: list[str] = []
    for report in reports:
        metadata = report.get("metadata") or {}
        name = Path(metadata.get("data", "unknown")).stem
        if report.get("mode") != "generation":
            skipped.append(name)
            continue
        vision = report.get("summary") or {}
        entry: dict = {"vision": vision}
        blind = report.get("blind_summary")
        if blind:
            entry["blind"] = blind
            entry["gap"] = {
                key: round(value - blind[key], 6)
                for key, value in vision.items()
                if key != "count"
                and isinstance(value, (int, float))
                and isinstance(blind.get(key), (int, float))
            }
        if name in IN_DOMAIN_BENCHMARKS:
            entry["in_domain"] = True
        benchmarks[name] = entry
    return {
        "benchmarks": benchmarks,
        "skipped_non_generation": skipped,
        "notes": "in_domain=True: GUI grounding data (ShowUI-desktop) was in the "
                 "train mix; read as in-domain, not held-out.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--upload-repo", default=None)
    parser.add_argument("--run-tag", default=None, help="HF path segment; default UTC timestamp")
    args = parser.parse_args()

    reports = []
    for path in sorted(args.results_dir.glob("*.json")):
        if path.name == "SUMMARY.json":
            continue
        try:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"skip unparseable {path.name}")
    summary = aggregate_reports(reports)
    out_path = args.results_dir / "SUMMARY.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.upload_repo:
        from moonvit_glue.checkpointing import CheckpointUploader

        tag = args.run_tag or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        uploader = CheckpointUploader(args.upload_repo)
        uploader.upload_async(args.results_dir, f"eval/{tag}")
        uploader.wait()
        for error in uploader.errors:
            print(f"[upload] ERROR {error}")
        print(f"[upload] {args.results_dir} -> {args.upload_repo}:eval/{tag}", flush=True)


if __name__ == "__main__":
    main()
