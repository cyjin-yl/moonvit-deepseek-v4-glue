#!/usr/bin/env python3
"""独立验证 projector 插值、端点评测复现和分析哈希。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


IDENTITY_FIELDS = ("condition", "id", "pair_id", "pair_variant", "task")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def endpoint_equivalence(
    interpolation_rows: list[dict],
    reference_rows: list[dict],
    *,
    interpolation_state: str,
    reference_state: str,
    value_fields: tuple[str, ...],
) -> int:
    current = {
        (str(row["condition"]), str(row["id"])): row
        for row in interpolation_rows
        if str(row["state"]) == interpolation_state
    }
    reference = {
        (str(row["condition"]), str(row["id"])): row
        for row in reference_rows
        if str(row["state"]) == reference_state
    }
    if set(current) != set(reference) or not current:
        raise ValueError("endpoint evaluation key sets differ")
    for key in sorted(current):
        left = current[key]
        right = reference[key]
        for field in IDENTITY_FIELDS:
            if left.get(field) != right.get(field):
                raise ValueError(f"endpoint identity differs: {key}, {field}")
        for field in value_fields:
            if left.get(field) != right.get(field):
                raise ValueError(f"endpoint value differs: {key}, {field}")
    return len(current)


def verify_summary_files(directory: Path, summary: dict) -> int:
    count = 0
    for name, entry in summary["files"].items():
        path = directory / name
        if path.stat().st_size != int(entry["bytes"]) or sha256(path) != entry["sha256"]:
            raise ValueError(f"summary-bound file mismatch: {path}")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpolation-run", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--reference-evaluation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite interpolation verification: {args.out}")

    interpolation = json.loads(
        (args.interpolation_run / "SUMMARY.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (args.evaluation / "SUMMARY.json").read_text(encoding="utf-8")
    )
    analysis = json.loads((args.analysis / "SUMMARY.json").read_text(encoding="utf-8"))
    for name, summary in (
        ("interpolation", interpolation),
        ("evaluation", evaluation),
        ("analysis", analysis),
    ):
        if summary.get("status") != "valid" or summary.get("final_half_scored") is not False:
            raise ValueError(f"invalid or unsafe {name} summary")
    for alpha in ("0.0", "1.0"):
        reproduction = interpolation["endpoint_reproduction"][alpha]
        if not reproduction["exact_tensor_equality"]:
            raise ValueError(f"interpolation tensor endpoint did not reproduce: {alpha}")
        if reproduction["source_tensor_sha256"] != reproduction["output_tensor_sha256"]:
            raise ValueError(f"interpolation endpoint tensor hash differs: {alpha}")

    preference = read_jsonl(args.evaluation / "preference_records.jsonl")
    generation = read_jsonl(args.evaluation / "generation_records.jsonl")
    reference_preference = read_jsonl(
        args.reference_evaluation / "preference_records.jsonl"
    )
    reference_generation = read_jsonl(
        args.reference_evaluation / "generation_records.jsonl"
    )
    preference_fields = (
        "visual_source_id",
        "correct_answer",
        "counterfactual_answer",
        "correct_logp",
        "counterfactual_logp",
        "correct_margin",
        "correct_token_nll",
        "counterfactual_token_nll",
        "failure",
    )
    generation_fields = (
        "visual_source_id",
        "answers",
        "prediction",
        "normalized_prediction",
        "correct",
        "failure",
    )
    endpoint_pairs = (
        ("projector-interp000", "projector-step50"),
        ("projector-interp100", "projector-step100"),
    )
    endpoint_results = {}
    for current, reference in endpoint_pairs:
        endpoint_results[current] = {
            "reference_state": reference,
            "preference_rows_exact": endpoint_equivalence(
                preference,
                reference_preference,
                interpolation_state=current,
                reference_state=reference,
                value_fields=preference_fields,
            ),
            "generation_rows_exact": endpoint_equivalence(
                generation,
                reference_generation,
                interpolation_state=current,
                reference_state=reference,
                value_fields=generation_fields,
            ),
        }

    evaluation_files = verify_summary_files(args.evaluation, evaluation)
    analysis_files = verify_summary_files(args.analysis, analysis)
    if analysis["evaluation_summary_sha256"] != sha256(args.evaluation / "SUMMARY.json"):
        raise ValueError("analysis is not bound to the interpolation evaluation")
    output = {
        "status": "valid",
        "interpolation_checkpoints": len(interpolation["checkpoints"]),
        "endpoint_tensor_reproduction": interpolation["endpoint_reproduction"],
        "endpoint_evaluation_reproduction": endpoint_results,
        "evaluation": {
            "states": evaluation["states"],
            "preference_rows": evaluation["preference_rows"],
            "generation_rows": evaluation["generation_rows"],
            "files_verified": evaluation_files,
        },
        "analysis": {
            "metric_rows": analysis["metric_rows"],
            "contrasts": analysis["contrasts"],
            "files_verified": analysis_files,
        },
        "final_half_scored": False,
    }
    args.out.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
