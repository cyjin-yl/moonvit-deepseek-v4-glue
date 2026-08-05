#!/usr/bin/env python3
"""对顶部 LoRA 与 projector continuation 做 pair-unit bootstrap 比较。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from moonvit_glue.metrics import normalize_answer
from moonvit_glue.trajectory_analysis import paired_gap_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def pair_scores(rows: list[dict], metric: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    output = []
    for pair_id, pair in sorted(grouped.items()):
        if len(pair) != 2 or any(row.get("failure") is not None for row in pair):
            raise ValueError(f"adaptation analysis needs a valid complete pair: {pair_id}")
        if metric == "paired_preference":
            score = float(all(float(row["correct_margin"]) > 0 for row in pair))
        elif metric == "mean_margin":
            score = statistics.fmean(float(row["correct_margin"]) for row in pair)
        elif metric == "generation_paired":
            score = float(all(bool(row["correct"]) for row in pair))
        elif metric == "generation_sample":
            score = statistics.fmean(float(bool(row["correct"])) for row in pair)
        elif metric == "generation_flip":
            predictions = [normalize_answer(str(row["prediction"])) for row in pair]
            score = float(predictions[0] != predictions[1])
        else:
            raise ValueError(f"unknown adaptation pair metric: {metric}")
        output.append({"id": pair_id, "score": score})
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite adaptation analysis: {args.out}")
    summary = json.loads((args.run / "SUMMARY.json").read_text(encoding="utf-8"))
    if summary.get("status") != "valid" or summary.get("final_half_scored"):
        raise ValueError("adaptation evaluation input is not valid selection-only output")
    for name, entry in summary["files"].items():
        if sha256(args.run / name) != entry["sha256"]:
            raise ValueError(f"adaptation evaluation hash mismatch: {name}")
    preference = read_jsonl(args.run / "preference_records.jsonl")
    generation = read_jsonl(args.run / "generation_records.jsonl")
    states = [str(row["id"]) for row in json.loads((args.run / "CONFIG.json").read_text(encoding="utf-8"))["evaluation_states"]]
    frozen = "frozen-step1500"
    contrasts = []

    def append(
        *, family: str, metric: str, state_a: str, condition_a: str,
        state_b: str, condition_b: str, source: list[dict]
    ) -> None:
        rows_a = [
            row for row in source
            if row["state"] == state_a and row["condition"] == condition_a
        ]
        rows_b = [
            row for row in source
            if row["state"] == state_b and row["condition"] == condition_b
        ]
        interval = paired_gap_stats(
            pair_scores(rows_a, metric),
            pair_scores(rows_b, metric),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + len(contrasts),
        )
        contrasts.append(
            {
                "family": family,
                "metric": metric,
                "state_a": state_a,
                "condition_a": condition_a,
                "state_b": state_b,
                "condition_b": condition_b,
                "pairs": interval["denominator"],
                "mean_a": interval["mean_a"],
                "mean_b": interval["mean_b"],
                "mean_gap": interval["mean_gap"],
                "ci95_low": interval["ci95_low"],
                "ci95_high": interval["ci95_high"],
                "bootstrap_samples": interval["bootstrap_samples"],
                "bootstrap_seed": interval["seed"],
            }
        )

    for state in states:
        for metric in ("paired_preference", "mean_margin"):
            if state != frozen:
                append(
                    family="adaptation_minus_frozen",
                    metric=metric,
                    state_a=state,
                    condition_a="vision",
                    state_b=frozen,
                    condition_b="vision",
                    source=preference,
                )
            append(
                family="vision_minus_shuffle",
                metric=metric,
                state_a=state,
                condition_a="vision",
                state_b=state,
                condition_b="shuffled_image",
                source=preference,
            )
        append(
            family="vision_minus_paired_counterfactual",
            metric="mean_margin",
            state_a=state,
            condition_a="vision",
            state_b=state,
            condition_b="paired_counterfactual_image",
            source=preference,
        )
        if state != frozen:
            for metric in ("generation_paired", "generation_sample", "generation_flip"):
                append(
                    family="adaptation_minus_frozen",
                    metric=metric,
                    state_a=state,
                    condition_a="vision",
                    state_b=frozen,
                    condition_b="vision",
                    source=generation,
                )

    lora_states = [state for state in states if state.startswith("lora-step")]
    projector_states = [state for state in states if state.startswith("projector-step")]
    common_steps = sorted(
        set(int(state.removeprefix("lora-step")) for state in lora_states)
        & set(int(state.removeprefix("projector-step")) for state in projector_states)
    )
    for step in common_steps:
        lora = f"lora-step{step}"
        projector = f"projector-step{step}"
        for metric, source in (
            ("paired_preference", preference),
            ("mean_margin", preference),
            ("generation_paired", generation),
            ("generation_sample", generation),
        ):
            append(
                family="lora_minus_projector_equal_examples",
                metric=metric,
                state_a=lora,
                condition_a="vision",
                state_b=projector,
                condition_b="vision",
                source=source,
            )

    args.out.mkdir(parents=True)
    contrast_path = args.out / "adaptation_contrasts.csv"
    write_csv(contrast_path, contrasts)
    preference_curve = list(csv.DictReader((args.run / "preference_curve.csv").open(encoding="utf-8")))
    generation_curve = list(csv.DictReader((args.run / "generation_curve.csv").open(encoding="utf-8")))
    best_lora = max(
        (row for row in preference_curve if row["state"] in lora_states and row["condition"] == "vision"),
        key=lambda row: (float(row["paired_preference_accuracy"]), float(row["mean_correct_margin"])),
    )
    best_projector = (
        max(
            (row for row in preference_curve if row["state"] in projector_states and row["condition"] == "vision"),
            key=lambda row: (float(row["paired_preference_accuracy"]), float(row["mean_correct_margin"])),
        )
        if projector_states
        else None
    )
    best_lora_generation = max(
        (row for row in generation_curve if row["state"] in lora_states and row["condition"] == "vision"),
        key=lambda row: (float(row["paired_accuracy"]), float(row["accuracy"])),
    )
    lora_preference_gap = next(
        row for row in contrasts
        if row["family"] == "adaptation_minus_frozen"
        and row["metric"] == "paired_preference"
        and row["state_a"] == best_lora["state"]
    )
    lora_generation_gap = next(
        row for row in contrasts
        if row["family"] == "adaptation_minus_frozen"
        and row["metric"] == "generation_paired"
        and row["state_a"] == best_lora_generation["state"]
    )
    decisions = {
        "status": "valid",
        "best_lora_teacher_forced": best_lora,
        "best_projector_teacher_forced": best_projector,
        "best_lora_generation": best_lora_generation,
        "best_lora_teacher_forced_vs_frozen": lora_preference_gap,
        "best_lora_generation_vs_frozen": lora_generation_gap,
        "upper_layer_adaptation_teacher_forced_supported": float(lora_preference_gap["ci95_low"]) > 0,
        "upper_layer_adaptation_generation_supported": float(lora_generation_gap["ci95_low"]) > 0,
        "interpretation_limits": [
            "all confidence intervals resample complete a/b pairs",
            "the diagnostic trains only shape and cannot establish transfer to other tasks",
            "equal-examples LoRA/projector contrasts share the same frozen data order",
            "final odd halves remain unscored",
        ],
    }
    decisions_path = args.out / "DECISIONS.json"
    decisions_path.write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_summary = {
        "status": "valid",
        "format_version": "shape-adaptation-analysis-v1",
        "source_summary_sha256": sha256(args.run / "SUMMARY.json"),
        "preference_rows": len(preference),
        "generation_rows": len(generation),
        "contrast_rows": len(contrasts),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "files": {
            contrast_path.name: {"bytes": contrast_path.stat().st_size, "sha256": sha256(contrast_path)},
            decisions_path.name: {"bytes": decisions_path.stat().st_size, "sha256": sha256(decisions_path)},
        },
        "final_half_scored": False,
    }
    (args.out / "SUMMARY.json").write_text(
        json.dumps(output_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
