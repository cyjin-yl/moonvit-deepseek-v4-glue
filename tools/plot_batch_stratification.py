#!/usr/bin/env python3
"""为严格匹配的 batch-order 消融生成确定性 SVG 图表。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from moonvit_glue.svg_charts import line_chart_svg


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--gradients", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite batch-stratification charts: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    decision_path = args.analysis / "DECISIONS.json"
    gradient_path = args.gradients / "SUMMARY.json"
    verification_path = args.verification / "VERIFICATION.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    gradients = json.loads(gradient_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if decision.get("status") != gradients.get("status") or decision.get("status") != "valid":
        raise ValueError("analysis and gradient diagnostics must both be valid")
    if not all(verification["checks"].values()):
        raise ValueError("matched batch-order verification is incomplete")

    steps = [0, 25, 50, 100]
    base = decision["state_metrics"]["frozen-base"]

    def state_values(arm: str, key: str) -> list[float]:
        return [float(base[key])] + [
            float(decision["state_metrics"][f"{arm}-step{step}"][key])
            for step in steps[1:]
        ]

    tasks = sorted(decision["trajectory"])
    charts = {
        "01-summary-trajectories.svg": line_chart_svg(
            title="Matched batch order: capability trajectories",
            x_label="continuation step",
            y_label="score",
            x_values=steps,
            series={
                "stratified macro pref": state_values("stratified", "preference_macro"),
                "global macro pref": state_values("global", "preference_macro"),
                "stratified worst pref": state_values("stratified", "preference_worst_task"),
                "global worst pref": state_values("global", "preference_worst_task"),
                "stratified macro gen": state_values("stratified", "generation_macro"),
                "global macro gen": state_values("global", "generation_macro"),
            },
            y_bounds=(0.0, 0.6),
        ),
        "02-task-preference-delta.svg": line_chart_svg(
            title="Stratified minus global: task paired preference",
            x_label="continuation step",
            y_label="paired-preference gap",
            x_values=steps,
            series={
                task: [0.0] + [
                    float(decision["state_metrics"][f"stratified-step{step}"]["preference"][task])
                    - float(decision["state_metrics"][f"global-step{step}"]["preference"][task])
                    for step in steps[1:]
                ]
                for task in tasks
            },
            y_bounds=(-0.35, 0.35),
        ),
    }

    gradient_states = gradients["state_summaries"]

    def gradient_values(arm: str, key: str) -> list[float]:
        base_value = gradient_states["frozen-base"][key]
        return [float(base_value)] + [
            float(gradient_states[f"{arm}-step{step}"][key])
            for step in steps[1:]
        ]

    charts["03-gradient-conflict.svg"] = line_chart_svg(
        title="Fixed-task projector gradient conflict",
        x_label="continuation step",
        y_label="mean cosine / negative-pair fraction",
        x_values=steps,
        series={
            "stratified mean cosine": gradient_values("stratified", "mean_task_cosine"),
            "global mean cosine": gradient_values("global", "mean_task_cosine"),
            "stratified negative fraction": [
                value / 15.0 for value in gradient_values("stratified", "negative_cosine_pairs")
            ],
            "global negative fraction": [
                value / 15.0 for value in gradient_values("global", "negative_cosine_pairs")
            ],
        },
        y_bounds=(0.0, 0.6),
    )

    batch_counts = verification["batch_task_counts"]
    batch_steps = [int(row["step"]) for row in batch_counts["stratified"]]
    charts["04-batch-imbalance.svg"] = line_chart_svg(
        title="Maximum task occupancy per true batch",
        x_label="optimizer step / true batch",
        y_label="max records from one task",
        x_values=batch_steps,
        series={
            "stratified": [
                max(int(value) for key, value in row.items() if key != "step")
                for row in batch_counts["stratified"]
            ],
            "global random": [
                max(int(value) for key, value in row.items() if key != "step")
                for row in batch_counts["global"]
            ],
        },
        y_bounds=(0.0, 12.0),
    )

    for name, content in charts.items():
        (args.out / name).write_text(content, encoding="utf-8")
    manifest = {
        "status": "valid",
        "batch_effect": decision["batch_effect"],
        "source_files": {
            str(decision_path): sha256(decision_path),
            str(gradient_path): sha256(gradient_path),
            str(verification_path): sha256(verification_path),
        },
        "charts": {
            name: {"bytes": (args.out / name).stat().st_size, "sha256": sha256(args.out / name)}
            for name in sorted(charts)
        },
        "final_half_scored": False,
    }
    (args.out / "CHARTS.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
