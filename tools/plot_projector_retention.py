#!/usr/bin/env python3
"""为 projector 保留目标 screen 生成确定性曲线。"""

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
        raise FileExistsError(f"refusing to overwrite retention charts: {args.out}")
    args.out.mkdir(parents=True)

    decision_path = args.analysis / "DECISIONS.json"
    preference_path = args.evaluation / "preference_curve.csv"
    generation_path = args.evaluation / "generation_curve.csv"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    generation = read_csv(generation_path)
    states = [
        "frozen-base",
        "resume-control",
        "anchor-1e-4",
        "anchor-1e-3",
        "anchor-1e-2",
    ]
    if set(states) != set(decision["state_metrics"]):
        raise ValueError("retention chart state set differs from the preregistered screen")
    x_values = list(range(len(states)))
    tasks = sorted(next(iter(decision["state_metrics"].values()))["preference"])

    def generation_value(state: str, task: str) -> float:
        return float(
            next(
                row["paired_accuracy"]
                for row in generation
                if row["state"] == state
                and row["condition"] == "vision"
                and row["task"] == task
            )
        )

    charts = {
        "01-task-preference.svg": line_chart_svg(
            title="Projector retention: task strict paired preference",
            x_label="state 0=P50, 1=control, 2=1e-4, 3=1e-3, 4=1e-2",
            y_label="paired preference",
            x_values=x_values,
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
            title="Projector retention: task paired generation",
            x_label="state 0=P50, 1=control, 2=1e-4, 3=1e-3, 4=1e-2",
            y_label="paired generation",
            x_values=x_values,
            series={
                task: [generation_value(state, task) for state in states]
                for task in tasks
            },
            y_bounds=(0.0, 1.0),
        ),
        "03-balance-summary.svg": line_chart_svg(
            title="Projector retention: balance summary",
            x_label="state 0=P50, 1=control, 2=1e-4, 3=1e-3, 4=1e-2",
            y_label="score",
            x_values=x_values,
            series={
                "macro preference": [
                    statistics.fmean(
                        float(value)
                        for value in decision["state_metrics"][state][
                            "preference"
                        ].values()
                    )
                    for state in states
                ],
                "worst-task preference": [
                    min(
                        float(value)
                        for value in decision["state_metrics"][state][
                            "preference"
                        ].values()
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
        "state_order": states,
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
