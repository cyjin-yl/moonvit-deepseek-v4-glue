#!/usr/bin/env python3
"""为 balanced projector/LoRA endpoint 对照生成确定性 SVG。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from moonvit_glue.svg_charts import heatmap_svg, line_chart_svg


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--lora-training", required=True, type=Path)
    parser.add_argument("--projector-training", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite adaptation charts: {args.out}")
    args.out.mkdir(parents=True)

    preference_path = args.evaluation / "preference_curve.csv"
    generation_path = args.evaluation / "generation_curve.csv"
    contrasts_path = args.analysis / "adaptation_contrasts.csv"
    lora_history_path = args.lora_training / "train_history.jsonl"
    projector_history_path = args.projector_training / "train_history.jsonl"
    preference = read_csv(preference_path)
    generation = read_csv(generation_path)
    contrasts = read_csv(contrasts_path)
    lora_history = read_jsonl(lora_history_path)
    projector_history = read_jsonl(projector_history_path)
    tasks = sorted(
        {row["task"] for row in preference if row["task"] != "overall"}
    )
    states = ["frozen-base", "lora-step100", "projector-step100"]
    state_labels = ["balanced step100", "top-12 LoRA", "extra projector epoch"]

    def metric(rows: list[dict], state: str, task: str, key: str) -> float:
        return float(
            next(
                row[key]
                for row in rows
                if row["state"] == state
                and row["condition"] == "vision"
                and row["task"] == task
            )
        )

    (args.out / "00-training-loss.svg").write_text(
        line_chart_svg(
            title="Matched balanced adaptation loss",
            x_label="relative optimizer step",
            y_label="teacher-forced loss",
            x_values=[int(row["step"]) for row in lora_history],
            series={
                "top-12 LoRA": [float(row["loss"]) for row in lora_history],
                "extra projector epoch": [
                    float(row["loss"]) for row in projector_history
                ],
            },
        ),
        encoding="utf-8",
    )
    for filename, title, rows, key in (
        (
            "01-endpoint-paired-preference.svg",
            "Strict paired preference by task",
            preference,
            "paired_preference_accuracy",
        ),
        (
            "02-endpoint-paired-generation.svg",
            "Paired free generation by task",
            generation,
            "paired_accuracy",
        ),
    ):
        (args.out / filename).write_text(
            heatmap_svg(
                title=title,
                row_labels=tasks,
                column_labels=state_labels,
                values=[
                    [metric(rows, state, task, key) for state in states]
                    for task in tasks
                ],
                value_label="accuracy",
                bounds=(0.0, 1.0),
            ),
            encoding="utf-8",
        )

    delta_values = []
    for task in tasks:
        values = []
        for modality, metric_name in (
            ("preference", "paired_preference"),
            ("generation", "generation_paired"),
        ):
            row = next(
                row
                for row in contrasts
                if row["family"] == "lora_minus_projector"
                and row["modality"] == modality
                and row["metric"] == metric_name
                and row["task"] == task
            )
            values.append(float(row["mean_gap"]))
        delta_values.append(values)
    (args.out / "03-lora-minus-projector.svg").write_text(
        heatmap_svg(
            title="Top-12 LoRA minus extra projector epoch",
            row_labels=tasks,
            column_labels=["paired preference", "paired generation"],
            values=delta_values,
            value_label="accuracy difference",
            bounds=(-1.0, 1.0),
        ),
        encoding="utf-8",
    )

    sources = (
        preference_path,
        generation_path,
        contrasts_path,
        lora_history_path,
        projector_history_path,
    )
    chart_paths = sorted(args.out.glob("*.svg"))
    manifest = {
        "status": "valid",
        "source_files": {str(path): sha256(path) for path in sources},
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
