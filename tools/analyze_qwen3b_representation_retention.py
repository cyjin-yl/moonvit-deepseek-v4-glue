#!/usr/bin/env python3
"""按预注册规则分析 Qwen3B ScreenSpot projector/receiver 表示保留。"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

import moonvit_glue.representation_retention as retention_module
from moonvit_glue import FeatureCache
from moonvit_glue.projector import PatchMergerProjector
from moonvit_glue.proxy_receiver import FixedPairwiseReceiverAdapter
from moonvit_glue.representation_retention import (
    compare_geometry,
    decide_representation_action,
    pairwise_geometry,
    summarize_token_sequences,
)
from moonvit_glue.screenspot_contract import verify_manifest
from moonvit_glue.screenspot_runtime import validate_screenspot_feature_cache

from eval_qwen3b_screenspot import (
    _Tee,
    canonical_sha256,
    git_sha,
    git_tracked_worktree_clean,
    set_stage,
    verify_projector,
    write_json,
)
from train_qwen3b_proxy import sha256_file
from verify_feature_cache import verify_feature_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--analysis-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--expected-cache-runner-git-sha", required=True)
    parser.add_argument("--expected-training-runner-git-sha", required=True)
    parser.add_argument("--receiver-dir", type=Path, required=True)
    parser.add_argument("--step0-projector", type=Path, required=True)
    parser.add_argument("--current-projector", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-dirty-development-run", action="store_true")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def runtime_source_files() -> list[dict[str, Any]]:
    paths = (
        Path(__file__).resolve(),
        Path(retention_module.__file__).resolve(),
        Path(__file__).with_name("eval_qwen3b_screenspot.py").resolve(),
        Path(__file__).with_name("verify_feature_cache.py").resolve(),
    )
    return [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ]


def _stage_payload(summary: Any) -> dict[str, Any]:
    return {
        "representation": summary.representation,
        "mean_within_image_rms": summary.mean_within_image_rms,
        "token_count": {
            "minimum": min(summary.token_counts),
            "maximum": max(summary.token_counts),
            "mean": sum(summary.token_counts) / len(summary.token_counts),
        },
    }


@torch.inference_mode()
def _run(args: argparse.Namespace, stage: dict[str, str]) -> dict[str, Any]:
    started = time.perf_counter()
    tracked_clean = git_tracked_worktree_clean()
    if not tracked_clean and not args.allow_dirty_development_run:
        raise RuntimeError("tracked Git worktree is dirty; formal analysis is refused")
    formal_run = tracked_clean and not args.allow_dirty_development_run
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    analysis_contract = json.loads(
        args.analysis_contract.read_text(encoding="utf-8")
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not verify_manifest(manifest):
        raise ValueError("ScreenSpot manifest self-hash verification failed")
    if (
        analysis_contract["dataset_name"] != manifest["name"]
        or analysis_contract["dataset_manifest_sha256"]
        != manifest["manifest_sha256"]
        or int(analysis_contract["sample_count"]) != len(manifest["samples"])
    ):
        raise ValueError("analysis contract does not bind the frozen ScreenSpot manifest")
    if analysis_contract["conditions"] != ["step0", "current_candidate"]:
        raise ValueError("analysis conditions differ from the preregistration")
    if analysis_contract["decision_stage"] != "fixed_receiver_2048":
        raise ValueError("analysis decision stage differs from the preregistration")

    set_stage(stage, "cache_and_artifact_verification")
    cache_verification = verify_feature_cache(
        args.feature_cache,
        expected_count=len(manifest["samples"]),
        expected_git_sha=args.expected_cache_runner_git_sha,
    )
    cache_manifest_path = args.feature_cache / "MANIFEST.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    preprocessing = contract["image_preprocessing"]
    cache_binding = validate_screenspot_feature_cache(
        manifest,
        cache_manifest,
        dataset_manifest_file_sha256=sha256_file(args.manifest),
        max_image_side=int(preprocessing["eval_max_image_side"]),
        max_visual_tokens=int(preprocessing["eval_max_visual_tokens"]),
        moonvit_weights_sha256=contract["vision_tower"]["extracted_weights_sha256"],
    )
    projector_sources = {
        role: verify_projector(
            directory,
            role=role,
            contract=contract,
            expected_training_runner_git_sha=args.expected_training_runner_git_sha,
        )
        for role, directory in (
            ("step0", args.step0_projector),
            ("current_candidate", args.current_projector),
        )
    }
    receiver_path = args.receiver_dir / "proxy_receiver.safetensors"
    if sha256_file(receiver_path) != contract["qwen_proxy_receiver"]["buffer_sha256"]:
        raise ValueError("proxy receiver weights differ from the contract")

    binding = {
        "format_version": "qwen3b-representation-retention-run-v1",
        "runner_git_sha": git_sha(),
        "git_tracked_worktree_clean": tracked_clean,
        "formal_run": formal_run,
        "contract_file_sha256": sha256_file(args.contract),
        "analysis_contract_file_sha256": sha256_file(args.analysis_contract),
        "analysis_contract": analysis_contract,
        "dataset_name": manifest["name"],
        "dataset_manifest_file_sha256": sha256_file(args.manifest),
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "feature_cache_manifest_file_sha256": sha256_file(cache_manifest_path),
        "cache_verification": cache_verification,
        "cache_binding": cache_binding,
        "projector_sources": projector_sources,
        "receiver_sha256": sha256_file(receiver_path),
        "runtime_source_files": runtime_source_files(),
        "paid_resources_used": False,
        "final_half_scored": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    write_json(args.out / "RUN_CONFIG.json", binding)

    set_stage(stage, "representation_extraction")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal representation analysis requires the local V100")
    torch.manual_seed(20260805)
    torch.cuda.manual_seed_all(20260805)
    torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    projectors = {
        "step0": PatchMergerProjector.from_pretrained(
            args.step0_projector, device=device, dtype=torch.float32
        ).eval(),
        "current_candidate": PatchMergerProjector.from_pretrained(
            args.current_projector, device=device, dtype=torch.float32
        ).eval(),
    }
    receiver = FixedPairwiseReceiverAdapter.from_pretrained(
        args.receiver_dir, device=device
    ).eval()
    cache = FeatureCache(args.feature_cache)
    sequences: dict[str, list[torch.Tensor]] = {
        "moonvit_flattened": [],
        "step0_projector_4096": [],
        "current_candidate_projector_4096": [],
        "step0_fixed_receiver_2048": [],
        "current_candidate_fixed_receiver_2048": [],
    }
    samples = list(manifest["samples"])
    for index, sample in enumerate(samples, start=1):
        feature_groups = cache.get(
            str(sample["sample_id"]), device=device, dtype=torch.float32
        )
        feature = feature_groups[0]
        sequences["moonvit_flattened"].append(
            feature.reshape(feature.shape[0], -1).detach().cpu()
        )
        for condition, projector in projectors.items():
            canonical = projector(feature_groups)[0]
            received = receiver(canonical)
            sequences[f"{condition}_projector_4096"].append(
                canonical.detach().cpu()
            )
            sequences[f"{condition}_fixed_receiver_2048"].append(
                received.detach().cpu()
            )
        if index % 10 == 0:
            print(f"representations [{index}/{len(samples)}]", flush=True)
    torch.cuda.synchronize(device)

    set_stage(stage, "statistics_and_action")
    summarized = {
        name: summarize_token_sequences(items) for name, items in sequences.items()
    }
    pooled = {name: value.pooled.contiguous() for name, value in summarized.items()}
    pooled_path = args.out / "POOLED_REPRESENTATIONS.safetensors"
    save_file(
        pooled,
        str(pooled_path),
        metadata={"format": "qwen3b-representation-retention-pooled-v1"},
    )

    stage_summaries = {
        "moonvit_flattened": {"shared": _stage_payload(summarized["moonvit_flattened"])},
        "projector_4096": {
            condition: _stage_payload(summarized[f"{condition}_projector_4096"])
            for condition in ("step0", "current_candidate")
        },
        "fixed_receiver_2048": {
            condition: _stage_payload(
                summarized[f"{condition}_fixed_receiver_2048"]
            )
            for condition in ("step0", "current_candidate")
        },
    }
    geometry_comparisons = {
        stage_name: compare_geometry(
            pooled[f"step0_{stage_name}"],
            pooled[f"current_candidate_{stage_name}"],
        )
        for stage_name in ("projector_4096", "fixed_receiver_2048")
    }
    decisions = {
        stage_name: decide_representation_action(
            stage_summaries[stage_name]["step0"]["representation"],
            stage_summaries[stage_name]["current_candidate"]["representation"],
            analysis_contract,
        )
        for stage_name in ("projector_4096", "fixed_receiver_2048")
    }

    pairwise_rows: list[dict[str, Any]] = []
    sample_ids = [str(row["sample_id"]) for row in samples]
    for key, matrix in pooled.items():
        for row in pairwise_geometry(matrix):
            left = int(row.pop("left_index"))
            right = int(row.pop("right_index"))
            pairwise_rows.append(
                {
                    "representation": key,
                    "left_index": left,
                    "right_index": right,
                    "left_id": sample_ids[left],
                    "right_id": sample_ids[right],
                    **row,
                }
            )
    pairwise_path = args.out / "PAIRWISE_GEOMETRY.jsonl"
    write_jsonl(pairwise_path, pairwise_rows)

    per_sample_rows = []
    for index, sample in enumerate(samples):
        row: dict[str, Any] = {
            "index": index,
            "sample_id": str(sample["sample_id"]),
            "platform": str(sample["platform"]),
            "target_type": str(sample["target_type"]),
            "representations": {},
        }
        for name, value in summarized.items():
            pooled_row = value.pooled[index]
            row["representations"][name] = {
                "token_count": value.token_counts[index],
                "within_image_rms": value.per_image_within_rms[index],
                "pooled_rms": float(torch.sqrt(torch.mean(pooled_row.square()))),
            }
        per_sample_rows.append(row)
    per_sample_path = args.out / "PER_SAMPLE.jsonl"
    write_jsonl(per_sample_path, per_sample_rows)

    total_wall = time.perf_counter() - started
    summary = {
        "format_version": "qwen3b-representation-retention-summary-v1",
        "status": "valid",
        "formal_analysis_complete": formal_run,
        "dataset_name": manifest["name"],
        "sample_count": len(samples),
        "stage_summaries": stage_summaries,
        "geometry_comparisons": geometry_comparisons,
        "decisions": decisions,
        "registered_decision_stage": analysis_contract["decision_stage"],
        "registered_action": decisions[analysis_contract["decision_stage"]]["action"],
        "pooled_representations": {
            "path": str(pooled_path),
            "bytes": pooled_path.stat().st_size,
            "sha256": sha256_file(pooled_path),
            "tensors": {name: list(value.shape) for name, value in pooled.items()},
        },
        "pairwise_geometry": {
            "path": str(pairwise_path),
            "rows": len(pairwise_rows),
            "bytes": pairwise_path.stat().st_size,
            "sha256": sha256_file(pairwise_path),
        },
        "per_sample": {
            "path": str(per_sample_path),
            "rows": len(per_sample_rows),
            "bytes": per_sample_path.stat().st_size,
            "sha256": sha256_file(per_sample_path),
        },
        "gpu_name": torch.cuda.get_device_name(device),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "total_wall_seconds": total_wall,
        "visual_ability_established": False,
        "capability_claim_allowed": False,
        "transfer_label": "directly_transferable",
        "paid_resources_used": False,
        "final_half_scored": False,
    }
    write_json(args.out / "SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "run.log"
    stage = {"name": "initialization"}
    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        with log_path.open("w", encoding="utf-8") as log:
            sys.stdout = _Tee(original_stdout, log)
            sys.stderr = _Tee(original_stderr, log)
            _run(args, stage)
    except Exception as exc:
        failure = {
            "format_version": "qwen3b-representation-retention-failure-v1",
            "status": "failed",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage["name"],
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "paid_resources_used": False,
            "final_half_scored": False,
        }
        write_json(args.out / "FAILURE.json", failure)
        raise
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    main()
