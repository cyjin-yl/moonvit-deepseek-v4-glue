#!/usr/bin/env python3
"""为固定预算 matched replay 生成确定性 SVG 图表。"""

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
    parser.add_argument("--ordinary-run", required=True, type=Path)
    parser.add_argument("--fixed-run", required=True, type=Path)
    parser.add_argument("--triggered-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite matched replay charts: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    decision_path = args.analysis / "DECISIONS.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    summaries = {
        "ordinary": json.loads((args.ordinary_run / "SUMMARY.json").read_text(encoding="utf-8")),
        "fixed": json.loads((args.fixed_run / "SUMMARY.json").read_text(encoding="utf-8")),
        "triggered": json.loads((args.triggered_run / "SUMMARY.json").read_text(encoding="utf-8")),
    }
    if decision.get("status") != "valid":
        raise ValueError("matched replay analysis is not valid")
    metrics = decision["state_metrics"]
    steps = [50, 75, 100]

    def policy_states(policy: str) -> list[str]:
        if policy == "ordinary":
            return ["exchange-step50", "ordinary-step75", "ordinary-step100"]
        if policy == "fixed":
            return ["exchange-step50", "fixed-step75", "fixed-step100"]
        return ["exchange-step50", "ordinary-step75", "triggered-step100"]

    charts = {
        "01-policy-summary-trajectories.svg": line_chart_svg(
            title="Fixed-budget replay: capability trajectories",
            x_label="global continuation step",
            y_label="macro paired score",
            x_values=steps,
            series={
                **{
                    f"{policy} preference": [metrics[state]["preference_macro"] for state in policy_states(policy)]
                    for policy in ("ordinary", "fixed", "triggered")
                },
                **{
                    f"{policy} generation": [metrics[state]["generation_macro"] for state in policy_states(policy)]
                    for policy in ("ordinary", "fixed", "triggered")
                },
            },
            y_bounds=(0.0, 0.65),
        ),
        "02-retention-task-trajectories.svg": line_chart_svg(
            title="Count and shape paired-preference retention",
            x_label="global continuation step",
            y_label="paired-preference accuracy",
            x_values=steps,
            series={
                f"{policy} {task}": [metrics[state]["preference"][task] for state in policy_states(policy)]
                for policy in ("ordinary", "fixed", "triggered")
                for task in ("count", "shape")
            },
            y_bounds=(0.0, 0.8),
        ),
    }
    tasks = ["color", "coordinate", "count", "ocr", "shape", "spatial"]
    charts["03-endpoint-task-deltas.svg"] = line_chart_svg(
        title="Endpoint paired-preference delta versus ordinary",
        x_label="task index: color, coordinate, count, OCR, shape, spatial",
        y_label="paired-preference gap",
        x_values=list(range(1, len(tasks) + 1)),
        series={
            "fixed": [
                metrics["fixed-step100"]["preference"][task]
                - metrics["ordinary-step100"]["preference"][task]
                for task in tasks
            ],
            "triggered": [
                metrics["triggered-step100"]["preference"][task]
                - metrics["ordinary-step100"]["preference"][task]
                for task in tasks
            ],
        },
        y_bounds=(-0.1, 0.45),
    )
    ordinary_counts = {task: 200 for task in tasks}
    fixed_counts = summaries["fixed"]["replay_policy"]["final_records_by_task"]
    triggered_late = summaries["triggered"]["replay_policy"]["final_records_by_task"]
    triggered_counts = {task: 100 + int(triggered_late[task]) for task in tasks}
    charts["04-fixed-budget-allocation.svg"] = line_chart_svg(
        title="Same 1,200-example budget: task allocation",
        x_label="task index: color, coordinate, count, OCR, shape, spatial",
        y_label="training examples in steps 51-100",
        x_values=list(range(1, len(tasks) + 1)),
        series={
            "ordinary": [ordinary_counts[task] for task in tasks],
            "fixed": [int(fixed_counts[task]) for task in tasks],
            "triggered": [triggered_counts[task] for task in tasks],
        },
        y_bounds=(160.0, 250.0),
    )

    for name, content in charts.items():
        (args.out / name).write_text(content, encoding="utf-8")
    manifest = {
        "status": "valid",
        "recommendation": decision["recommendation"],
        "source_files": {
            str(decision_path): sha256(decision_path),
            **{
                str(path / "SUMMARY.json"): sha256(path / "SUMMARY.json")
                for path in (args.ordinary_run, args.fixed_run, args.triggered_run)
            },
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
