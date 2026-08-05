#!/usr/bin/env python3
"""为 sentinel 功效、V100 时延和开销间隔生成确定性 SVG。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from moonvit_glue.svg_charts import line_chart_svg


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power", required=True, type=Path)
    parser.add_argument("--timing", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite sentinel charts: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    power_path = args.power / "DECISIONS.json"
    timing_path = args.timing / "DECISIONS.json"
    power = json.loads(power_path.read_text(encoding="utf-8"))
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    if power.get("status") != "valid" or timing.get("status") != "valid":
        raise ValueError("sentinel power or timing analysis is invalid")

    candidates = power["candidate_summary"]
    pairs = [int(row["pairs_per_task"]) for row in candidates]
    charts = {
        "01-power-curves.svg": line_chart_svg(
            title="Sentinel subsampling power across 200 trials",
            x_label="complete pairs per task",
            y_label="trial rate",
            x_values=pairs,
            series={
                "count recall": [float(row["count_recall"]) for row in candidates],
                "exact count-only decision": [
                    float(row["exact_decision_rate"]) for row in candidates
                ],
                "familywise false trigger": [
                    float(row["familywise_false_trigger_rate"]) for row in candidates
                ],
            },
            y_bounds=(0.0, 1.0),
        ),
        "02-v100-timing.svg": line_chart_svg(
            title="Teacher-only sentinel cost on V100",
            x_label="complete pairs per task",
            y_label="median seconds across three repeats",
            x_values=[int(row["pairs_per_task"]) for row in timing["profiles"]],
            series={
                "synchronized teacher": [
                    float(row["teacher_forced_seconds_median"])
                    for row in timing["profiles"]
                ],
                "separate-process wall": [
                    float(row["wall_seconds_median"]) for row in timing["profiles"]
                ],
            },
            y_bounds=(0.0, 60.0),
        ),
    }
    with (args.timing / "overhead_budget.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        overhead_rows = list(csv.DictReader(stream))
    model_resident = [
        row
        for row in overhead_rows
        if row["time_basis"] == "model_resident_teacher"
    ]
    overheads = sorted({float(row["maximum_overhead"]) for row in model_resident})
    charts["03-required-interval.svg"] = line_chart_svg(
        title="Minimum interval under a fixed evaluation overhead",
        x_label="maximum evaluation overhead (%)",
        y_label="minimum training steps between sentinels",
        x_values=[value * 100 for value in overheads],
        series={
            profile: [
                int(
                    next(
                        row["minimum_interval_steps"]
                        for row in model_resident
                        if row["profile"] == profile
                        and float(row["maximum_overhead"]) == overhead
                    )
                )
                for overhead in overheads
            ]
            for profile in ("tiny", "medium")
        },
        y_bounds=(0.0, 1000.0),
    )

    for name, content in charts.items():
        (args.out / name).write_text(content, encoding="utf-8")
    manifest = {
        "status": "valid",
        "recommended_profile": timing["recommended_profile"],
        "source_files": {
            str(power_path): sha256(power_path),
            str(timing_path): sha256(timing_path),
            str(args.timing / "overhead_budget.csv"): sha256(
                args.timing / "overhead_budget.csv"
            ),
        },
        "charts": {
            name: {
                "bytes": (args.out / name).stat().st_size,
                "sha256": sha256(args.out / name),
            }
            for name in sorted(charts)
        },
        "final_half_scored": False,
    }
    (args.out / "CHARTS.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
