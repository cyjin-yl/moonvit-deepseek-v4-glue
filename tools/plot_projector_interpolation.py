#!/usr/bin/env python3
"""为 projector checkpoint 插值 screen 生成确定性曲线。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from moonvit_glue.svg_charts import line_chart_svg


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite interpolation charts: {args.out}")
    args.out.mkdir(parents=True)

    decision_path = args.analysis / "DECISIONS.json"
    preference_path = args.evaluation / "preference_curve.csv"
    generation_path = args.evaluation / "generation_curve.csv"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    preference = read_csv(preference_path)
    generation = read_csv(generation_path)
    states = sorted(
        decision["state_metrics"],
        key=lambda state: float(decision["state_metrics"][state]["alpha"]),
    )
    alphas = [float(decision["state_metrics"][state]["alpha"]) for state in states]
    tasks = sorted(next(iter(decision["state_metrics"].values()))["preference"])

    def curve_value(rows: list[dict], state: str, task: str, key: str) -> float:
        return float(
            next(
                row[key]
                for row in rows
                if row["state"] == state
                and row["condition"] == "vision"
                and row["task"] == task
            )
        )

    charts = {
        "01-task-preference.svg": line_chart_svg(
            title="Projector interpolation: task strict paired preference",
            x_label="alpha (step 50 to step 100)",
            y_label="paired preference",
            x_values=alphas,
            series={
                task: [
                    float(decision["state_metrics"][state]["preference"][task])
                    for state in states
                ]
                for task in tasks
            },
            y_bounds=(0.0, 1.0),
        ),
        "02-task-generation.svg": line_chart_svg(
            title="Projector interpolation: task paired generation",
            x_label="alpha (step 50 to step 100)",
            y_label="paired generation",
            x_values=alphas,
            series={
                task: [curve_value(generation, state, task, "paired_accuracy") for state in states]
                for task in tasks
            },
            y_bounds=(0.0, 1.0),
        ),
        "03-balance-summary.svg": line_chart_svg(
            title="Projector interpolation: balance summary",
            x_label="alpha (step 50 to step 100)",
            y_label="score",
            x_values=alphas,
            series={
                "macro preference": [
                    statistics.fmean(
                        float(value)
                        for value in decision["state_metrics"][state]["preference"].values()
                    )
                    for state in states
                ],
                "worst-task preference": [
                    min(
                        float(value)
                        for value in decision["state_metrics"][state]["preference"].values()
                    )
                    for state in states
                ],
                "macro generation": [
                    float(decision["state_metrics"][state]["generation_macro"])
                    for state in states
                ],
            },
            y_bounds=(0.0, 1.0),
        ),
    }
    for name, content in charts.items():
        (args.out / name).write_text(content, encoding="utf-8")
    manifest = {
        "status": "valid",
        "source_files": {
            str(decision_path): sha256(decision_path),
            str(preference_path): sha256(preference_path),
            str(generation_path): sha256(generation_path),
        },
        "charts": {
            name: {
                "bytes": (args.out / name).stat().st_size,
                "sha256": sha256(args.out / name),
            }
            for name in sorted(charts)
        },
    }
    (args.out / "CHARTS.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
