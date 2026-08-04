#!/usr/bin/env python3
"""从已提交 CSV 渲染 package 3 预注册的十张核心图。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from moonvit_glue.svg_charts import heatmap_svg, line_chart_svg, scatter_chart_svg


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-run", required=True, type=Path)
    parser.add_argument("--preference-analysis", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite package charts: {args.out}")
    args.out.mkdir(parents=True)

    generation_config = json.loads(
        (args.generation_run / "CONFIG.json").read_text(encoding="utf-8")
    )
    checkpoints = [str(row["id"]) for row in generation_config["checkpoints"]]
    examples_seen = [int(row["examples_seen"]) for row in generation_config["checkpoints"]]
    generation = read_csv(args.generation_run / "synthetic_curve.csv")
    benchmark = read_csv(args.generation_run / "benchmark_curve.csv")
    shuffle = read_csv(args.generation_run / "shuffle_loss_curve.csv")
    preference = read_csv(args.preference_analysis / "preference_curve.csv")
    preference_gaps = read_csv(args.preference_analysis / "preference_gaps.csv")

    generation_index = {
        (row["checkpoint"], row["condition"], row["task"]): row for row in generation
    }
    preference_index = {
        (row["checkpoint"], row["condition"], row["task"]): row for row in preference
    }

    def generation_values(condition: str, task: str, field: str) -> list[float]:
        return [
            float(generation_index[(checkpoint, condition, task)][field])
            for checkpoint in checkpoints
        ]

    def preference_values(condition: str, task: str, field: str) -> list[float]:
        return [
            float(preference_index[(checkpoint, condition, task)][field])
            for checkpoint in checkpoints
        ]

    write(
        args.out / "01-generation-accuracy.svg",
        line_chart_svg(
            title="Checkpoint → synthetic generation accuracy",
            x_label="examples seen",
            y_label="sample accuracy",
            x_values=examples_seen,
            series={
                "vision": generation_values("vision", "overall", "accuracy"),
                "blind": generation_values("blind", "overall", "accuracy"),
            },
            y_bounds=(0.0, 1.0),
        ),
    )
    write(
        args.out / "02-paired-preference.svg",
        line_chart_svg(
            title="Checkpoint → paired preference accuracy",
            x_label="examples seen",
            y_label="strict pair accuracy",
            x_values=examples_seen,
            series={
                "vision": preference_values(
                    "vision", "overall", "paired_preference_accuracy"
                ),
                "blind": preference_values(
                    "blind", "overall", "paired_preference_accuracy"
                ),
                "paired image": preference_values(
                    "paired_counterfactual_image",
                    "overall",
                    "paired_preference_accuracy",
                ),
            },
            y_bounds=(0.0, 1.0),
        ),
    )
    write(
        args.out / "03-answer-flip.svg",
        line_chart_svg(
            title="Checkpoint → generation answer-flip accuracy",
            x_label="examples seen",
            y_label="pair accuracy",
            x_values=examples_seen,
            series={
                "vision": generation_values(
                    "vision", "overall", "answer_flip_accuracy"
                ),
                "paired counterfactual": generation_values(
                    "paired_counterfactual_image", "overall", "answer_flip_accuracy"
                ),
            },
            y_bounds=(0.0, 1.0),
        ),
    )
    write(
        args.out / "04-correct-margin.svg",
        line_chart_svg(
            title="Checkpoint → teacher-forced correct margin",
            x_label="examples seen",
            y_label="mean token-normalized log-prob margin",
            x_values=examples_seen,
            series={
                "vision": preference_values("vision", "overall", "mean_correct_margin"),
                "blind": preference_values("blind", "overall", "mean_correct_margin"),
                "paired image": preference_values(
                    "paired_counterfactual_image", "overall", "mean_correct_margin"
                ),
            },
        ),
    )
    shuffle_index = {
        row["checkpoint"]: row
        for row in shuffle
        if row.get("source", "overall") == "overall"
    }
    write(
        args.out / "05-true-shuffled-loss.svg",
        line_chart_svg(
            title="Checkpoint → held-out true/shuffled loss",
            x_label="examples seen",
            y_label="token loss",
            x_values=examples_seen,
            series={
                "true": [
                    float(shuffle_index[checkpoint]["mean_true_loss"])
                    for checkpoint in checkpoints
                ],
                "shuffled": [
                    float(shuffle_index[checkpoint]["mean_shuffled_loss"])
                    for checkpoint in checkpoints
                ],
            },
        ),
    )
    shuffle_sources = sorted(
        {row["source"] for row in shuffle if row.get("source") not in (None, "", "overall")}
    )
    shuffle_source_index = {
        (row["checkpoint"], row["source"]): float(row["mean_delta"])
        for row in shuffle
        if row.get("source") not in (None, "")
    }
    write(
        args.out / "05b-shuffle-delta-by-source.svg",
        line_chart_svg(
            title="Checkpoint → held-out shuffle delta by source",
            x_label="examples seen",
            y_label="shuffled loss minus true loss",
            x_values=examples_seen,
            series={
                source: [
                    shuffle_source_index[(checkpoint, source)]
                    for checkpoint in checkpoints
                ]
                for source in shuffle_sources
            },
        ),
    )
    benchmark_index = {
        (row["checkpoint"], row["condition"], row["benchmark"]): float(row["score"])
        for row in benchmark
    }
    benchmarks = sorted({row["benchmark"] for row in benchmark})
    write(
        args.out / "06-benchmark-vision-minus-blind.svg",
        line_chart_svg(
            title="Checkpoint → benchmark vision-minus-blind gap",
            x_label="examples seen",
            y_label="raw metric gap",
            x_values=examples_seen,
            series={
                name: [
                    benchmark_index[(checkpoint, "vision", name)]
                    - benchmark_index[(checkpoint, "blind", name)]
                    for checkpoint in checkpoints
                ]
                for name in benchmarks
            },
        ),
    )
    tasks = sorted(
        {row["task"] for row in preference if row["task"] != "overall"}
    )
    write(
        args.out / "07-task-checkpoint-heatmap.svg",
        heatmap_svg(
            title="Task × checkpoint paired preference",
            row_labels=tasks,
            column_labels=checkpoints,
            values=[
                preference_values("vision", task, "paired_preference_accuracy")
                for task in tasks
            ],
            value_label="strict paired preference accuracy",
            bounds=(0.0, 1.0),
        ),
    )
    controls = [
        "blind",
        "blank",
        "same_image",
        "shuffled_image",
        "patch_permutation",
        "paired_counterfactual_image",
        "background_matched_aux",
    ]
    gap_index = {
        (row["checkpoint"], row["task"], row["metric"], row["b"]): float(
            row["mean_gap"]
        )
        for row in preference_gaps
    }
    gap_values = [
        [
            gap_index[(checkpoint, "overall", "mean_margin", condition)]
            for checkpoint in checkpoints
        ]
        for condition in controls
    ]
    absolute_gap = max(abs(value) for row in gap_values for value in row) or 1.0
    write(
        args.out / "08-control-checkpoint-heatmap.svg",
        heatmap_svg(
            title="Control × checkpoint causal margin gap",
            row_labels=controls,
            column_labels=checkpoints,
            values=gap_values,
            value_label="vision minus control pair-mean margin",
            bounds=(-absolute_gap, absolute_gap),
        ),
    )
    write(
        args.out / "09-background-comparison.svg",
        line_chart_svg(
            title="Authoritative vs background-matched selection",
            x_label="examples seen",
            y_label="accuracy",
            x_values=examples_seen,
            series={
                "generation authoritative": generation_values(
                    "vision", "overall", "accuracy"
                ),
                "generation background-matched": generation_values(
                    "background_matched_aux", "overall", "accuracy"
                ),
                "preference authoritative": preference_values(
                    "vision", "overall", "paired_preference_accuracy"
                ),
                "preference background-matched": preference_values(
                    "background_matched_aux",
                    "overall",
                    "paired_preference_accuracy",
                ),
            },
            y_bounds=(0.0, 1.0),
        ),
    )

    scatter_rows = [
        {
            "checkpoint": checkpoint,
            "examples_seen": examples_seen[index],
            "paired_preference_accuracy": preference_values(
                "vision", "overall", "paired_preference_accuracy"
            )[index],
            "generation_accuracy": generation_values("vision", "overall", "accuracy")[
                index
            ],
        }
        for index, checkpoint in enumerate(checkpoints)
    ]
    scatter_path = args.out / "10-evidence-vs-generation.csv"
    with scatter_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scatter_rows[0]))
        writer.writeheader()
        writer.writerows(scatter_rows)
    write(
        args.out / "10-evidence-vs-generation.svg",
        scatter_chart_svg(
            title="Teacher-forced evidence vs generation accuracy",
            x_label="paired preference accuracy",
            y_label="generation sample accuracy",
            points=[
                (
                    row["checkpoint"],
                    row["paired_preference_accuracy"],
                    row["generation_accuracy"],
                )
                for row in scatter_rows
            ],
            x_bounds=(0.0, 1.0),
            y_bounds=(0.0, 1.0),
        ),
    )

    paired_scatter_rows = [
        {
            "checkpoint": checkpoint,
            "task": task,
            "examples_seen": examples_seen[index],
            "paired_preference_accuracy": preference_values(
                "vision", task, "paired_preference_accuracy"
            )[index],
            "generation_paired_accuracy": generation_values(
                "vision", task, "paired_accuracy"
            )[index],
        }
        for task in tasks
        for index, checkpoint in enumerate(checkpoints)
    ]
    paired_scatter_path = args.out / "10b-paired-evidence-vs-paired-generation.csv"
    with paired_scatter_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(paired_scatter_rows[0])
        )
        writer.writeheader()
        writer.writerows(paired_scatter_rows)
    write(
        args.out / "10b-paired-evidence-vs-paired-generation.svg",
        scatter_chart_svg(
            title="Task evidence vs paired generation",
            x_label="teacher-forced paired preference",
            y_label="generation paired accuracy",
            points=[
                (
                    f"{row['checkpoint']}/{row['task']}",
                    row["paired_preference_accuracy"],
                    row["generation_paired_accuracy"],
                )
                for row in paired_scatter_rows
                if row["paired_preference_accuracy"] > 0
                or row["generation_paired_accuracy"] > 0
            ],
            x_bounds=(0.0, 0.2),
            y_bounds=(0.0, 0.2),
        ),
    )
    charts = sorted(args.out.glob("*.svg"))
    sources = [
        args.generation_run / "synthetic_curve.csv",
        args.generation_run / "benchmark_curve.csv",
        args.generation_run / "shuffle_loss_curve.csv",
        args.preference_analysis / "preference_curve.csv",
        args.preference_analysis / "preference_gaps.csv",
        scatter_path,
        paired_scatter_path,
    ]
    manifest = {
        "status": "valid",
        "sources": {path.name: sha256(path) for path in sources},
        "charts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in charts
        },
    }
    (args.out / "CHARTS.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
