#!/usr/bin/env python3
"""对 activation patching 原始记录做 donor-specific 配对因果对照。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from moonvit_glue.mechanism_probe import aligned_effect_delta, pair_bootstrap_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patching", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite contrast analysis: {args.out}")
    summary_path = args.patching / "SUMMARY.json"
    raw_path = args.patching / "patching_records.jsonl"
    patch_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if patch_summary.get("status") != "valid":
        raise ValueError("activation patching input is not valid")
    if sha256(raw_path) != patch_summary["files"][raw_path.name]["sha256"]:
        raise ValueError("activation patching raw hash mismatch")
    rows = read_jsonl(raw_path)
    if len(rows) != int(patch_summary["raw_rows"]):
        raise ValueError("activation patching row count mismatch")

    by_cell: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[
            (str(row["checkpoint"]), str(row["intervention"]), int(row["layer_index"]))
        ].append(row)
    checkpoints = [str(value) for value in patch_summary["checkpoints"]]
    scan_layers = [int(value) for value in patch_summary["scan_layers"]]
    negative_layers = [int(value) for value in patch_summary["negative_control_layers"]]
    contrasts: list[dict] = []

    def append_contrast(
        *,
        family: str,
        checkpoint_a: str,
        intervention_a: str,
        checkpoint_b: str,
        intervention_b: str,
        layer_index: int,
    ) -> None:
        aligned = aligned_effect_delta(
            by_cell[(checkpoint_a, intervention_a, layer_index)],
            by_cell[(checkpoint_b, intervention_b, layer_index)],
        )
        index = len(contrasts)
        interval = pair_bootstrap_mean(
            values=[float(row["effect_delta"]) for row in aligned],
            pair_ids=[str(row["pair_id"]) for row in aligned],
            seed=args.seed + index,
            samples=args.bootstrap_samples,
        )
        contrasts.append(
            {
                "family": family,
                "checkpoint_a": checkpoint_a,
                "intervention_a": intervention_a,
                "checkpoint_b": checkpoint_b,
                "intervention_b": intervention_b,
                "layer_index": layer_index,
                "records": interval["records"],
                "pairs": interval["pairs"],
                "mean_delta": interval["mean"],
                "ci95_low": interval["ci95_low"],
                "ci95_high": interval["ci95_high"],
                "bootstrap_samples": interval["bootstrap_samples"],
                "bootstrap_seed": interval["seed"],
            }
        )

    for checkpoint in checkpoints:
        for layer_index in negative_layers:
            append_contrast(
                family="correct_image_minus_wrong_label_donor",
                checkpoint_a=checkpoint,
                intervention_a="correct_image_span",
                checkpoint_b=checkpoint,
                intervention_b="wrong_label_donor_image_span",
                layer_index=layer_index,
            )
            append_contrast(
                family="correct_image_minus_zero_donor",
                checkpoint_a=checkpoint,
                intervention_a="correct_image_span",
                checkpoint_b=checkpoint,
                intervention_b="zero_image_span",
                layer_index=layer_index,
            )
        append_contrast(
            family="outer_input_minus_center_input",
            checkpoint_a=checkpoint,
            intervention_a="input_clean_outer",
            checkpoint_b=checkpoint,
            intervention_b="input_clean_center",
            layer_index=-1,
        )
        append_contrast(
            family="full_input_minus_outer_input",
            checkpoint_a=checkpoint,
            intervention_a="input_clean_full",
            checkpoint_b=checkpoint,
            intervention_b="input_clean_outer",
            layer_index=-1,
        )
        for layer_index in scan_layers:
            append_contrast(
                family="correct_image_minus_assistant",
                checkpoint_a=checkpoint,
                intervention_a="correct_image_span",
                checkpoint_b=checkpoint,
                intervention_b="correct_assistant",
                layer_index=layer_index,
            )

    trained = [checkpoint for checkpoint in checkpoints if checkpoint != "step-000000"]
    comparisons = []
    if len(trained) >= 2:
        comparisons.append((trained[0], trained[-1]))
    for trained_checkpoint in trained:
        if "step-000000" in checkpoints:
            comparisons.append((trained_checkpoint, "step-000000"))
    for checkpoint_a, checkpoint_b in comparisons:
        for intervention in ("correct_image_span", "correct_assistant"):
            for layer_index in scan_layers:
                append_contrast(
                    family="checkpoint_effect_delta",
                    checkpoint_a=checkpoint_a,
                    intervention_a=intervention,
                    checkpoint_b=checkpoint_b,
                    intervention_b=intervention,
                    layer_index=layer_index,
                )

    args.out.mkdir(parents=True)
    csv_path = args.out / "patching_contrasts.csv"
    write_csv(csv_path, contrasts)
    decisions = {"status": "valid", "checkpoints": {}, "checkpoint_comparisons": {}}
    for checkpoint in checkpoints:
        donor_rows = [
            row
            for row in contrasts
            if row["family"] == "correct_image_minus_wrong_label_donor"
            and row["checkpoint_a"] == checkpoint
        ]
        region_rows = [
            row
            for row in contrasts
            if row["checkpoint_a"] == checkpoint
            and row["family"] in {"outer_input_minus_center_input", "full_input_minus_outer_input"}
        ]
        decisions["checkpoints"][checkpoint] = {
            "best_content_specific_image_layer": max(
                donor_rows, key=lambda row: float(row["mean_delta"])
            ),
            "input_region_contrasts": region_rows,
        }
    for checkpoint_a, checkpoint_b in comparisons:
        key = f"{checkpoint_a}_minus_{checkpoint_b}"
        candidate_rows = [
            row
            for row in contrasts
            if row["family"] == "checkpoint_effect_delta"
            and row["checkpoint_a"] == checkpoint_a
            and row["checkpoint_b"] == checkpoint_b
        ]
        decisions["checkpoint_comparisons"][key] = {
            "largest_absolute_delta": max(
                candidate_rows, key=lambda row: abs(float(row["mean_delta"]))
            )
        }
    decisions["interpretation_limits"] = [
        "all contrasts align the same sample IDs and resample complete a/b pairs",
        "correct-minus-wrong isolates donor content specificity from generic activation replacement",
        "projector token positions are globally contextualized and do not localize source pixels",
    ]
    decisions_path = args.out / "DECISIONS.json"
    decisions_path.write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "valid",
        "format_version": "activation-patching-contrasts-v1",
        "source_summary_sha256": sha256(summary_path),
        "source_raw_sha256": sha256(raw_path),
        "source_rows": len(rows),
        "contrast_rows": len(contrasts),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "files": {
            csv_path.name: {"bytes": csv_path.stat().st_size, "sha256": sha256(csv_path)},
            decisions_path.name: {
                "bytes": decisions_path.stat().st_size,
                "sha256": sha256(decisions_path),
            },
        },
    }
    (args.out / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
