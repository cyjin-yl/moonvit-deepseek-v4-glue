#!/usr/bin/env python3
"""从 package 6 联合分析表生成确定性 SVG 图。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from moonvit_glue.svg_charts import heatmap_svg


TASKS = ("color", "coordinate", "count", "ocr", "shape", "spatial")
CHECKPOINTS = ("step-000000", "step-001500", "shape-projector-step50")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_matrix(
    rows: list[dict], *, modality: str, metric: str, condition: str
) -> list[list[float]]:
    """按固定 task/checkpoint 次序提取可直接审计的矩阵。"""

    return [
        [
            float(
                next(
                    row["mean"]
                    for row in rows
                    if row["modality"] == modality
                    and row["metric"] == metric
                    and row["condition"] == condition
                    and row["task"] == task
                    and row["checkpoint"] == checkpoint
                )
            )
            for checkpoint in CHECKPOINTS
        ]
        for task in TASKS
    ]


def contrast_vector(
    rows: list[dict], *, modality: str, metric: str, checkpoint: str
) -> list[float]:
    """提取同一 checkpoint 的 vision-minus-shuffle 配对差值。"""

    return [
        float(
            next(
                row["mean_gap"]
                for row in rows
                if row["family"] == "vision_minus_condition"
                and row["modality"] == modality
                and row["metric"] == metric
                and row["checkpoint_a"] == checkpoint
                and row["condition_b"] == "shuffled_image"
                and row["task"] == task
            )
        )
        for task in TASKS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite transfer charts: {args.out}")
    args.out.mkdir(parents=True)

    metrics_path = args.analysis / "transfer_metrics.csv"
    contrasts_path = args.analysis / "transfer_contrasts.csv"
    metrics = read_csv(metrics_path)
    contrasts = read_csv(contrasts_path)
    charts = {
        "01-paired-preference.svg": heatmap_svg(
            title="Teacher-forced paired preference by task",
            row_labels=TASKS,
            column_labels=CHECKPOINTS,
            values=metric_matrix(
                metrics,
                modality="preference",
                metric="paired_preference",
                condition="vision",
            ),
            value_label="strict paired preference accuracy",
            bounds=(0.0, 1.0),
        ),
        "02-paired-generation.svg": heatmap_svg(
            title="Free-generation paired accuracy by task",
            row_labels=TASKS,
            column_labels=CHECKPOINTS,
            values=metric_matrix(
                metrics,
                modality="generation",
                metric="generation_paired",
                condition="vision",
            ),
            value_label="strict paired generation accuracy",
            bounds=(0.0, 1.0),
        ),
        "03-vision-minus-shuffle.svg": heatmap_svg(
            title="Visual causal effect after shape-only continuation",
            row_labels=TASKS,
            column_labels=("paired preference", "paired generation"),
            values=[
                [preference, generation]
                for preference, generation in zip(
                    contrast_vector(
                        contrasts,
                        modality="preference",
                        metric="paired_preference",
                        checkpoint="shape-projector-step50",
                    ),
                    contrast_vector(
                        contrasts,
                        modality="generation",
                        metric="generation_paired",
                        checkpoint="shape-projector-step50",
                    ),
                )
            ],
            value_label="vision minus shuffled-image paired accuracy",
            bounds=(-1.0, 1.0),
        ),
    }
    for filename, svg in charts.items():
        (args.out / filename).write_text(svg, encoding="utf-8")

    chart_paths = sorted(args.out.glob("*.svg"))
    payload = {
        "status": "valid",
        "source_csv": {
            str(metrics_path): sha256(metrics_path),
            str(contrasts_path): sha256(contrasts_path),
        },
        "charts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in chart_paths
        },
    }
    (args.out / "CHARTS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
