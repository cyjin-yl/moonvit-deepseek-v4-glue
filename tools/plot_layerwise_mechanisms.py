#!/usr/bin/env python3
"""从 package 4 聚合表生成确定性 SVG 机制图。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from moonvit_glue.svg_charts import heatmap_svg, line_chart_svg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--patching", required=True, type=Path)
    parser.add_argument("--contrasts", required=True, type=Path)
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
        raise FileExistsError(f"refusing to overwrite mechanism charts: {args.out}")
    args.out.mkdir(parents=True)
    metrics_path = args.analysis / "probe_metrics.csv"
    curve_path = args.patching / "patching_curve.csv"
    contrasts_path = args.contrasts / "patching_contrasts.csv"
    metrics = read_csv(metrics_path)
    curves = read_csv(curve_path)
    contrasts = read_csv(contrasts_path)
    checkpoints = ["step-000000", "step-001500", "step-002000"]
    checkpoint_labels = ["random", "step 1500", "step 2000"]

    def metric(checkpoint: str, condition: str, site: str, field: str) -> float:
        rows = [
            row
            for row in metrics
            if row["checkpoint"] == checkpoint
            and row["condition"] == condition
            and row["site"] == site
        ]
        if len(rows) != 1:
            raise ValueError(f"probe metric cell mismatch: {checkpoint}/{condition}/{site}")
        return float(rows[0][field])

    def curve(checkpoint: str, intervention: str, layer: int, field: str) -> float:
        rows = [
            row
            for row in curves
            if row["checkpoint"] == checkpoint
            and row["intervention"] == intervention
            and int(row["layer_index"]) == layer
        ]
        if len(rows) != 1:
            raise ValueError(
                f"patching curve cell mismatch: {checkpoint}/{intervention}/{layer}"
            )
        return float(rows[0][field])

    layers = list(range(25))
    for site_suffix, filename, title in (
        ("assistant", "01-assistant-probe.svg", "Shape probe at assistant position"),
        ("image_mean", "02-image-span-probe.svg", "Shape probe over image-span states"),
    ):
        (args.out / filename).write_text(
            line_chart_svg(
                title=title,
                x_label="hidden-state index",
                y_label="balanced accuracy",
                x_values=layers,
                series={
                    label: [
                        metric(
                            checkpoint,
                            "vision",
                            f"layer_{layer:02d}_{site_suffix}",
                            "target_balanced_accuracy",
                        )
                        for layer in layers
                    ]
                    for checkpoint, label in zip(checkpoints, checkpoint_labels, strict=True)
                },
                y_bounds=(0.0, 1.0),
            ),
            encoding="utf-8",
        )

    pooling_sites = [
        f"{family}_{pooling}"
        for family in ("tower", "projector")
        for pooling in ("global_mean", "center_mean", "spatial_2x2")
    ]
    (args.out / "03-tower-projector-probes.svg").write_text(
        heatmap_svg(
            title="Frozen tower and projector shape recoverability",
            row_labels=pooling_sites,
            column_labels=checkpoint_labels,
            values=[
                [
                    metric(checkpoint, "vision", site, "target_balanced_accuracy")
                    for checkpoint in checkpoints
                ]
                for site in pooling_sites
            ],
            value_label="selection balanced accuracy",
            bounds=(0.0, 1.0),
        ),
        encoding="utf-8",
    )

    patch_layers = list(range(24))
    for intervention, filename, title in (
        ("correct_image_span", "04-image-span-patching.svg", "Correct-image causal recovery"),
        ("correct_assistant", "05-assistant-patching.svg", "Assistant-state positive control"),
    ):
        (args.out / filename).write_text(
            line_chart_svg(
                title=title,
                x_label="patched decoder layer",
                y_label="margin effect vs paired-counterfactual",
                x_values=patch_layers,
                series={
                    label: [
                        curve(checkpoint, intervention, layer, "mean_effect")
                        for layer in patch_layers
                    ]
                    for checkpoint, label in zip(checkpoints, checkpoint_labels, strict=True)
                },
            ),
            encoding="utf-8",
        )

    negative_layers = [0, 5, 11, 17, 23]
    (args.out / "06-content-specific-patching.svg").write_text(
        line_chart_svg(
            title="Correct image donor minus wrong-label donor",
            x_label="patched decoder layer",
            y_label="paired margin-effect delta",
            x_values=negative_layers,
            series={
                label: [
                    float(
                        next(
                            row["mean_delta"]
                            for row in contrasts
                            if row["family"]
                            == "correct_image_minus_wrong_label_donor"
                            and row["checkpoint_a"] == checkpoint
                            and int(row["layer_index"]) == layer
                        )
                    )
                    for layer in negative_layers
                ]
                for checkpoint, label in zip(checkpoints, checkpoint_labels, strict=True)
            },
            y_bounds=(-0.30, 0.35),
        ),
        encoding="utf-8",
    )

    region_sites = ["input_clean_center", "input_clean_outer", "input_clean_full"]
    (args.out / "07-input-token-regions.svg").write_text(
        heatmap_svg(
            title="Input projector-token replacement",
            row_labels=["center token positions", "outer token positions", "full span"],
            column_labels=checkpoint_labels,
            values=[
                [curve(checkpoint, site, -1, "mean_effect") for checkpoint in checkpoints]
                for site in region_sites
            ],
            value_label="margin effect; token positions are globally contextualized",
            bounds=(-0.4, 0.4),
        ),
        encoding="utf-8",
    )

    decisive = [("step-001500", 12), ("step-002000", 14)]
    conditions = [
        "vision",
        "paired_counterfactual_image",
        "shuffled_image",
        "patch_permutation",
    ]
    row_labels = []
    values = []
    for checkpoint, layer in decisive:
        for condition in conditions:
            site = f"layer_{layer:02d}_assistant"
            row_labels.append(f"{checkpoint[5:]} L{layer} {condition}")
            values.append(
                [
                    metric(checkpoint, condition, site, "target_accuracy"),
                    metric(checkpoint, condition, site, "source_accuracy"),
                ]
            )
    (args.out / "08-probe-source-controls.svg").write_text(
        heatmap_svg(
            title="Probe follows the actual visual source",
            row_labels=row_labels,
            column_labels=["target label", "visual-source label"],
            values=values,
            value_label="selection accuracy",
            bounds=(0.0, 1.0),
        ),
        encoding="utf-8",
    )

    charts = sorted(args.out.glob("*.svg"))
    manifest = {
        "status": "valid",
        "source_csv": {
            str(metrics_path): sha256(metrics_path),
            str(curve_path): sha256(curve_path),
            str(contrasts_path): sha256(contrasts_path),
        },
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
