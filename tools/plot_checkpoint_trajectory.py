#!/usr/bin/env python3
"""从已提交的轨迹 CSV 渲染无额外依赖的 SVG 曲线。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from moonvit_glue.svg_charts import line_chart_svg


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_svg(path: Path, **kwargs) -> None:
    path.write_text(line_chart_svg(**kwargs), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite trajectory charts: {args.out}")
    args.out.mkdir(parents=True)
    config = json.loads((args.run / "CONFIG.json").read_text(encoding="utf-8"))
    checkpoints = [str(row["id"]) for row in config["checkpoints"]]
    x_values = [int(row["examples_seen"]) for row in config["checkpoints"]]
    synthetic = read_csv(args.run / "synthetic_curve.csv")
    benchmark = read_csv(args.run / "benchmark_curve.csv")
    shuffle = read_csv(args.run / "shuffle_loss_curve.csv")

    def synthetic_values(condition: str, task: str, field: str) -> list[float]:
        index = {
            (row["checkpoint"], row["condition"], row["task"]): float(row[field])
            for row in synthetic
        }
        return [index[(checkpoint, condition, task)] for checkpoint in checkpoints]

    write_svg(
        args.out / "synthetic_overall.svg",
        title="Synthetic capability trajectory",
        x_label="examples seen",
        y_label="accuracy",
        x_values=x_values,
        series={
            "vision sample": synthetic_values("vision", "overall", "accuracy"),
            "blind sample": synthetic_values("blind", "overall", "accuracy"),
            "vision paired": synthetic_values("vision", "overall", "paired_accuracy"),
            "vision answer-flip": synthetic_values("vision", "overall", "answer_flip_accuracy"),
        },
        y_bounds=(0.0, 1.0),
    )
    tasks = sorted({row["task"] for row in synthetic if row["task"] != "overall"})
    write_svg(
        args.out / "synthetic_tasks.svg",
        title="Synthetic vision accuracy by task",
        x_label="examples seen",
        y_label="accuracy",
        x_values=x_values,
        series={task: synthetic_values("vision", task, "accuracy") for task in tasks},
        y_bounds=(0.0, 1.0),
    )

    shuffle_index = {
        row["checkpoint"]: row
        for row in shuffle
        if row.get("source", "overall") == "overall"
    }
    write_svg(
        args.out / "shuffle_loss.svg",
        title="Held-out true and shuffled answer loss",
        x_label="examples seen",
        y_label="loss",
        x_values=x_values,
        series={
            "true loss": [float(shuffle_index[checkpoint]["mean_true_loss"]) for checkpoint in checkpoints],
            "shuffled loss": [float(shuffle_index[checkpoint]["mean_shuffled_loss"]) for checkpoint in checkpoints],
        },
    )
    write_svg(
        args.out / "shuffle_delta.svg",
        title="Held-out shuffle delta",
        x_label="examples seen",
        y_label="shuffled loss - true loss",
        x_values=x_values,
        series={
            "shuffle delta": [float(shuffle_index[checkpoint]["mean_delta"]) for checkpoint in checkpoints],
        },
    )

    score_index = {
        (row["checkpoint"], row["condition"], row["benchmark"]): float(row["score"])
        for row in benchmark
    }
    benchmarks = sorted({row["benchmark"] for row in benchmark})
    write_svg(
        args.out / "benchmark_gaps.svg",
        title="Benchmark vision-minus-blind gaps (selection only)",
        x_label="examples seen",
        y_label="raw metric gap",
        x_values=x_values,
        series={
            name: [
                score_index[(checkpoint, "vision", name)] - score_index[(checkpoint, "blind", name)]
                for checkpoint in checkpoints
            ]
            for name in benchmarks
        },
    )
    chart_paths = sorted(args.out.glob("*.svg"))
    manifest = {
        "status": "valid",
        "source_csv": {
            path.name: file_sha256(path)
            for path in (
                args.run / "synthetic_curve.csv",
                args.run / "benchmark_curve.csv",
                args.run / "shuffle_loss_curve.csv",
            )
        },
        "charts": {
            path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in chart_paths
        },
    }
    (args.out / "CHARTS.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
