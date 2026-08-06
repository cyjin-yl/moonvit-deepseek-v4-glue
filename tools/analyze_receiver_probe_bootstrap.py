#!/usr/bin/env python3
"""汇总冻结 receiver-prior probe，并计算固定 seed 的 paired bootstrap CI。"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take percentile of empty values")
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean(values: list[float], *, seed: int, samples: int) -> dict[str, float | int]:
    rng = random.Random(seed)
    boot = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    ]
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "ci95_lower": percentile(boot, 0.025),
        "ci95_upper": percentile(boot, 0.975),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def bootstrap_delta(
    left: list[float], right: list[float], *, seed: int, samples: int,
) -> dict[str, float | int]:
    if len(left) != len(right):
        raise ValueError("paired arrays have different lengths")
    deltas = [a - b for a, b in zip(left, right)]
    return bootstrap_mean(deltas, seed=seed, samples=samples)


def load_run(path: Path) -> dict:
    summary = json.loads((path / "SUMMARY.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (path / "probe_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != int(summary["sample_count"]):
        raise ValueError(f"row count mismatch in {path}")
    return {"summary": summary, "rows": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260805)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("runs", nargs="+", help="name=directory pairs; first pair is the reference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 2000:
        raise ValueError("the evaluation contract requires at least 2,000 bootstrap samples")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_ids = [str(record["id"]) for record in manifest["records"]]
    runs: dict[str, dict] = {}
    for item in args.runs:
        if "=" not in item:
            raise ValueError(f"run must be name=directory: {item}")
        name, raw_path = item.split("=", 1)
        run = load_run(Path(raw_path))
        ids = [str(row["sample_id"]) for row in run["rows"]]
        if ids != expected_ids:
            raise ValueError(f"sample order mismatch for {name}")
        runs[name] = run

    metrics: dict[str, dict] = {}
    for name, run in runs.items():
        rows = run["rows"]
        shuffle = [float(row["vision_minus_shuffle"]) for row in rows]
        blind = [float(row["vision_minus_blind"]) for row in rows]
        random_projector = [
            float(row["vision_answer_logp"]) - float(row["random_projector_answer_logp"])
            for row in rows
        ]
        metrics[name] = {
            "token_selection": run["summary"].get("token_selection"),
            "max_visual_tokens": run["summary"].get("max_visual_tokens"),
            "sample_count": len(rows),
            "vision_minus_shuffle": bootstrap_mean(shuffle, seed=args.bootstrap_seed, samples=args.bootstrap_samples),
            "vision_minus_blind": bootstrap_mean(blind, seed=args.bootstrap_seed + 1, samples=args.bootstrap_samples),
            "vision_minus_random_projector": bootstrap_mean(random_projector, seed=args.bootstrap_seed + 2, samples=args.bootstrap_samples),
            "positive_vision_minus_shuffle_count": sum(value > 0 for value in shuffle),
            "rows": rows,
        }

    reference_name = next(iter(runs))
    reference_shuffle = [float(row["vision_minus_shuffle"]) for row in runs[reference_name]["rows"]]
    deltas = {}
    for name, run in runs.items():
        if name == reference_name:
            continue
        candidate = [float(row["vision_minus_shuffle"]) for row in run["rows"]]
        deltas[f"{name}_minus_{reference_name}"] = bootstrap_delta(
            candidate, reference_shuffle, seed=args.bootstrap_seed + 10, samples=args.bootstrap_samples,
        )

    result = {
        "schema_version": "receiver-probe-bootstrap-v1",
        "status": "diagnostic_only",
        "manifest": str(args.manifest),
        "manifest_schema_version": manifest.get("schema_version"),
        "sample_count": len(expected_ids),
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "reference_run": reference_name,
        "runs": metrics,
        "paired_token_condition_deltas": deltas,
        "capability_claim_allowed": False,
        "deepseek_transfer_label": "transferable_with_runtime_validation",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("schema_version", "sample_count", "reference_run", "runs", "paired_token_condition_deltas")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
