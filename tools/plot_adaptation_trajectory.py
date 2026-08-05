#!/usr/bin/env python3
"""为 balanced LoRA/projector 多 checkpoint screen 生成任务热图。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from moonvit_glue.svg_charts import heatmap_svg


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite trajectory charts: {args.out}")
    args.out.mkdir(parents=True)
    preference_path = args.evaluation / "preference_curve.csv"
    generation_path = args.evaluation / "generation_curve.csv"
    preference = read_csv(preference_path)
    generation = read_csv(generation_path)
    tasks = sorted({row["task"] for row in preference if row["task"] != "overall"})
    states = list(dict.fromkeys(row["state"] for row in preference))
    labels = [
        "base" if state == "frozen-base" else state.replace("projector-step", "P").replace("lora-step", "L")
        for state in states
    ]

    def value(rows: list[dict], state: str, task: str, condition: str, key: str) -> float:
        return float(
            next(
                row[key]
                for row in rows
                if row["state"] == state
                and row["task"] == task
                and row["condition"] == condition
            )
        )

    charts = (
        (
            "01-trajectory-paired-preference.svg",
            "Balanced trajectory: strict paired preference",
            preference,
            "vision",
            "paired_preference_accuracy",
            "accuracy",
            (0.0, 1.0),
        ),
        (
            "02-trajectory-paired-generation.svg",
            "Balanced trajectory: paired free generation",
            generation,
            "vision",
            "paired_accuracy",
            "accuracy",
            (0.0, 1.0),
        ),
    )
    for filename, title, rows, condition, key, value_label, bounds in charts:
        (args.out / filename).write_text(
            heatmap_svg(
                title=title,
                row_labels=tasks,
                column_labels=labels,
                values=[
                    [value(rows, state, task, condition, key) for state in states]
                    for task in tasks
                ],
                value_label=value_label,
                bounds=bounds,
            ),
            encoding="utf-8",
        )
    (args.out / "03-trajectory-vision-minus-shuffle.svg").write_text(
        heatmap_svg(
            title="Balanced trajectory: vision minus shuffled preference",
            row_labels=tasks,
            column_labels=labels,
            values=[
                [
                    value(preference, state, task, "vision", "paired_preference_accuracy")
                    - value(
                        preference,
                        state,
                        task,
                        "shuffled_image",
                        "paired_preference_accuracy",
                    )
                    for state in states
                ]
                for task in tasks
            ],
            value_label="accuracy difference",
            bounds=(-1.0, 1.0),
        ),
        encoding="utf-8",
    )
    chart_paths = sorted(args.out.glob("*.svg"))
    manifest = {
        "status": "valid",
        "source_files": {
            str(preference_path): sha256(preference_path),
            str(generation_path): sha256(generation_path),
        },
        "charts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in chart_paths
        },
    }
    (args.out / "CHARTS.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
