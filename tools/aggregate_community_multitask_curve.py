#!/usr/bin/env python3
"""Join per-checkpoint community multi-task CSVs and render SVG growth curves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from moonvit_glue.svg_charts import line_chart_svg


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Directory containing per-checkpoint outputs")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    rows = []
    source_files = []
    for path in sorted(args.root.glob("*/CURVE.csv")):
        source_files.append(path)
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no per-checkpoint CURVE.csv under {args.root}")
    for row in rows:
        for key in ("optimizer_step", "examples_seen", "count"):
            row[key] = int(row[key])
        row["score"] = float(row["score"])
    args.out.mkdir(parents=True)
    rows.sort(key=lambda row: (row["dataset"], row["metric"], row["condition"], row["examples_seen"]))
    with (args.out / "curve.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["checkpoint", "optimizer_step", "examples_seen", "dataset", "condition", "metric", "score", "count", "source_report"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["metric"])].append(row)
    charts = []
    for (dataset, metric), group in sorted(grouped.items()):
        x_values = sorted({row["examples_seen"] for row in group})
        series = {}
        for condition in ("vision", "blind", "shuffled", "random_projector"):
            index = {(row["examples_seen"], row["condition"]): row["score"] for row in group}
            values = [index[(x, condition)] for x in x_values if (x, condition) in index]
            if len(values) == len(x_values):
                series[condition] = values
        if not series:
            continue
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{dataset}_{metric}")
        chart = args.out / f"{safe}.svg"
        chart.write_text(
            line_chart_svg(
                title=f"{dataset}: {metric} by examples seen",
                x_label="examples seen",
                y_label=metric,
                x_values=x_values,
                series=series,
            ),
            encoding="utf-8",
        )
        charts.append(chart)
    manifest = {
        "schema_version": "community-multitask-curves-v1",
        "status": "valid",
        "source_csv": {str(path): sha256(path) for path in source_files},
        "curve_csv": sha256(args.out / "curve.csv"),
        "charts": {path.name: sha256(path) for path in charts},
    }
    (args.out / "CHARTS.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
