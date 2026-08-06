#!/usr/bin/env python3
"""独立重算并校验 Qwen3B representation-retention 产物。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from moonvit_glue.representation_retention import (
    compare_geometry,
    decide_representation_action,
    pairwise_geometry,
    summarize_representation,
)

from train_qwen3b_proxy import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    config = json.loads((args.run / "RUN_CONFIG.json").read_text(encoding="utf-8"))
    summary = json.loads((args.run / "SUMMARY.json").read_text(encoding="utf-8"))
    tensors_path = args.run / "POOLED_REPRESENTATIONS.safetensors"
    if sha256_file(tensors_path) != summary["pooled_representations"]["sha256"]:
        raise ValueError("pooled representation SHA-256 differs")
    tensors = load_file(str(tensors_path), device="cpu")
    declared_tensors = summary["pooled_representations"]["tensors"]
    if set(tensors) != set(declared_tensors):
        raise ValueError("pooled representation tensor set differs")
    for name, tensor in tensors.items():
        if list(tensor.shape) != declared_tensors[name]:
            raise ValueError(f"pooled representation shape differs: {name}")

    recomputed = {name: summarize_representation(tensor) for name, tensor in tensors.items()}
    stage_keys = {
        "moonvit_flattened": {"shared": "moonvit_flattened"},
        "projector_4096": {
            "step0": "step0_projector_4096",
            "current_candidate": "current_candidate_projector_4096",
        },
        "fixed_receiver_2048": {
            "step0": "step0_fixed_receiver_2048",
            "current_candidate": "current_candidate_fixed_receiver_2048",
        },
    }
    for stage, conditions in stage_keys.items():
        for condition, key in conditions.items():
            declared = summary["stage_summaries"][stage][condition]["representation"]
            if canonical(declared) != canonical(recomputed[key]):
                raise ValueError(f"representation summary differs: {stage}/{condition}")

    geometry = {
        stage: compare_geometry(
            tensors[f"step0_{stage}"], tensors[f"current_candidate_{stage}"]
        )
        for stage in ("projector_4096", "fixed_receiver_2048")
    }
    if canonical(geometry) != canonical(summary["geometry_comparisons"]):
        raise ValueError("geometry comparisons differ")
    decisions = {
        stage: decide_representation_action(
            recomputed[f"step0_{stage}"],
            recomputed[f"current_candidate_{stage}"],
            config["analysis_contract"],
        )
        for stage in ("projector_4096", "fixed_receiver_2048")
    }
    if canonical(decisions) != canonical(summary["decisions"]):
        raise ValueError("registered representation decisions differ")

    pairwise_path = args.run / "PAIRWISE_GEOMETRY.jsonl"
    if sha256_file(pairwise_path) != summary["pairwise_geometry"]["sha256"]:
        raise ValueError("pairwise geometry SHA-256 differs")
    pairwise_rows = [
        json.loads(line) for line in pairwise_path.read_text(encoding="utf-8").splitlines()
    ]
    per_sample_path = args.run / "PER_SAMPLE.jsonl"
    if sha256_file(per_sample_path) != summary["per_sample"]["sha256"]:
        raise ValueError("per-sample SHA-256 differs")
    per_sample_rows = [
        json.loads(line) for line in per_sample_path.read_text(encoding="utf-8").splitlines()
    ]
    if len(per_sample_rows) != int(summary["sample_count"]):
        raise ValueError("per-sample row count differs")
    if [row["index"] for row in per_sample_rows] != list(range(len(per_sample_rows))):
        raise ValueError("per-sample index order differs")
    sample_ids = [str(row["sample_id"]) for row in per_sample_rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("per-sample IDs are not unique")

    expected_pairwise_rows = []
    for name, tensor in tensors.items():
        for row in pairwise_geometry(tensor):
            left = int(row.pop("left_index"))
            right = int(row.pop("right_index"))
            expected_pairwise_rows.append(
                {
                    "representation": name,
                    "left_index": left,
                    "right_index": right,
                    "left_id": sample_ids[left],
                    "right_id": sample_ids[right],
                    **row,
                }
            )
    if canonical(pairwise_rows) != canonical(expected_pairwise_rows):
        raise ValueError("pairwise geometry rows differ from pooled tensors")

    representation_names = set(tensors)
    for index, row in enumerate(per_sample_rows):
        observed = row["representations"]
        if set(observed) != representation_names:
            raise ValueError(f"per-sample representation set differs at row {index}")
        for name, tensor in tensors.items():
            values = observed[name]
            expected_rms = float(torch.sqrt(torch.mean(tensor[index].square())))
            if not math.isclose(
                float(values["pooled_rms"]), expected_rms, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"per-sample pooled RMS differs: {index}/{name}")
            if int(values["token_count"]) <= 0 or not math.isfinite(
                float(values["within_image_rms"])
            ):
                raise ValueError(f"per-sample token statistic is invalid: {index}/{name}")

    verification = {
        "format_version": "qwen3b-representation-retention-verification-v1",
        "status": "verified",
        "runner_git_sha": config["runner_git_sha"],
        "pooled_representations_sha256": sha256_file(tensors_path),
        "pooled_tensor_count": len(tensors),
        "pairwise_rows": len(expected_pairwise_rows),
        "pairwise_sha256": sha256_file(pairwise_path),
        "per_sample_rows": len(per_sample_rows),
        "per_sample_sha256": sha256_file(per_sample_path),
        "decisions": decisions,
        "registered_action": decisions[config["analysis_contract"]["decision_stage"]][
            "action"
        ],
        "paid_resources_used": False,
        "final_half_scored": False,
    }
    args.out.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
