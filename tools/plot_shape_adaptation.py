#!/usr/bin/env python3
"""从 package 5 聚合表生成确定性 SVG 适配轨迹图。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from moonvit_glue.svg_charts import heatmap_svg, line_chart_svg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--base-probes", required=True, type=Path)
    parser.add_argument("--lora-probes", required=True, type=Path)
    parser.add_argument("--projector-probes", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite adaptation charts: {args.out}")
    args.out.mkdir(parents=True)
    preference_path = args.evaluation / "preference_curve.csv"
    generation_path = args.evaluation / "generation_curve.csv"
    preference = read_csv(preference_path)
    generation = read_csv(generation_path)

    examples = [0, 400, 800, 1600]

    def trajectory(rows: list[dict], kind: str, metric: str) -> list[float]:
        frozen = next(
            row
            for row in rows
            if row["kind"] == "frozen" and row["condition"] == "vision"
        )
        values = [float(frozen[metric])]
        for seen in examples[1:]:
            row = next(
                row
                for row in rows
                if row["kind"] == kind
                and row["condition"] == "vision"
                and int(row["adaptation_examples_seen"]) == seen
            )
            values.append(float(row[metric]))
        return values

    charts = (
        (
            "01-paired-preference.svg",
            "Shape paired preference after adaptation",
            "strict paired preference accuracy",
            preference,
            "paired_preference_accuracy",
            (0.0, 1.0),
        ),
        (
            "02-paired-generation.svg",
            "Shape paired free-generation after adaptation",
            "paired generation accuracy",
            generation,
            "paired_accuracy",
            (0.0, 1.0),
        ),
        (
            "03-correct-margin.svg",
            "Teacher-forced correct-answer margin",
            "mean correct margin",
            preference,
            "mean_correct_margin",
            None,
        ),
    )
    for filename, title, y_label, rows, metric, bounds in charts:
        (args.out / filename).write_text(
            line_chart_svg(
                title=title,
                x_label="shape adaptation examples seen",
                y_label=y_label,
                x_values=examples,
                series={
                    "top-12 LoRA": trajectory(rows, "lora", metric),
                    "projector continuation": trajectory(rows, "projector", metric),
                },
                y_bounds=bounds,
            ),
            encoding="utf-8",
        )

    selected_states = ["frozen-step1500", "lora-step100", "projector-step50"]
    conditions = ["vision", "shuffled_image"]
    (args.out / "04-vision-shuffle-control.svg").write_text(
        heatmap_svg(
            title="Paired preference follows the visual source",
            row_labels=selected_states,
            column_labels=["vision", "shuffled image"],
            values=[
                [
                    float(
                        next(
                            row["paired_preference_accuracy"]
                            for row in preference
                            if row["state"] == state and row["condition"] == condition
                        )
                    )
                    for condition in conditions
                ]
                for state in selected_states
            ],
            value_label="strict paired preference accuracy",
            bounds=(0.0, 1.0),
        ),
        encoding="utf-8",
    )

    probe_paths = {
        "frozen step 1500": args.base_probes / "probe_metrics.csv",
        "top-12 LoRA step 100": args.lora_probes / "probe_metrics.csv",
        "projector continuation step 50": args.projector_probes / "probe_metrics.csv",
    }
    checkpoints = {
        "frozen step 1500": "step-001500",
        "top-12 LoRA step 100": "lora-step100",
        "projector continuation step 50": "projector-step50",
    }
    layers = list(range(25))
    probe_rows = {label: read_csv(path) for label, path in probe_paths.items()}
    (args.out / "05-assistant-layerwise.svg").write_text(
        line_chart_svg(
            title="Assistant-position shape recoverability",
            x_label="hidden-state index",
            y_label="balanced accuracy",
            x_values=layers,
            series={
                label: [
                    float(
                        next(
                            row["target_balanced_accuracy"]
                            for row in probe_rows[label]
                            if row["checkpoint"] == checkpoints[label]
                            and row["condition"] == "vision"
                            and row["site"] == f"layer_{layer:02d}_assistant"
                        )
                    )
                    for layer in layers
                ]
                for label in probe_paths
            },
            y_bounds=(0.0, 1.0),
        ),
        encoding="utf-8",
    )

    source_paths = [preference_path, generation_path, *probe_paths.values()]
    chart_paths = sorted(args.out.glob("*.svg"))
    manifest = {
        "status": "valid",
        "source_csv": {str(path): sha256(path) for path in source_paths},
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
