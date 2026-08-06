#!/usr/bin/env python3
"""按冻结 4k 合同训练 Qwen2.5-3B 的 DeepSeek-shaped projector。"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import torch

import moonvit_glue.chat_contract as chat_contract_module
import moonvit_glue.checkpointing as checkpointing_module
import moonvit_glue.feature_cache as feature_cache_module
import moonvit_glue.fixed_budget as fixed_budget_module
import moonvit_glue.geometry_regularization as geometry_regularization_module
import moonvit_glue.merge as merge_module
import moonvit_glue.model as model_module
import moonvit_glue.projector as projector_module
import moonvit_glue.proxy_receiver as proxy_receiver_module
import moonvit_glue.projector_binding as projector_binding_module
import moonvit_glue.training_health as training_health_module
import moonvit_glue.training_order as training_order_module
from moonvit_glue import FeatureCache
from moonvit_glue.chat_contract import build_chat_supervision
from moonvit_glue.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from moonvit_glue.fixed_budget import (
    fixed_batch_record_indices,
    route_training_example,
    validate_fixed_budget_contract,
    validate_resume_history,
)
from moonvit_glue.geometry_regularization import (
    geometry_payload,
    geometry_regularization_loss,
    global_gradient_norm,
    pool_projector_batch,
)
from moonvit_glue.grounding_preference import build_counterfactual_targets
from moonvit_glue.merge import expand_image_placeholders
from moonvit_glue.paired_preference import answer_logprob_stats
from moonvit_glue.training_health import (
    append_jsonl,
    evaluate_guards,
    jsonable_probe,
    probe_due,
    summarize_batch_embeddings,
    summarize_probe,
    tensor_sha256,
    validate_health_contract,
)
from moonvit_glue.model import VisionCausalLM
from moonvit_glue.projector import PatchMergerProjector
from moonvit_glue.projector_binding import canonical_binding, validate_variant_binding
from moonvit_glue.proxy_receiver import FixedPairwiseReceiverAdapter
from moonvit_glue.training_order import (
    load_ordered_records,
    verify_training_order_manifest,
)
from verify_feature_cache import verify_feature_cache


class _Tee:
    """日志文件始终保留；交互 stdout 断开时不杀训练。"""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = list(streams)

    def write(self, text: str) -> int:
        alive = []
        for stream in self.streams:
            try:
                stream.write(text)
                stream.flush()
                alive.append(stream)
            except (BrokenPipeError, OSError):
                continue
        self.streams = alive
        return len(text)

    def flush(self) -> None:
        for stream in list(self.streams):
            try:
                stream.flush()
            except (BrokenPipeError, OSError):
                self.streams.remove(stream)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def set_stage(stage: dict[str, str], name: str) -> None:
    stage["name"] = name
    print(f"stage: {name}", flush=True)


def git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def git_tracked_worktree_clean() -> bool:
    unstaged = subprocess.run(["git", "diff", "--quiet", "--"], check=False)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet", "--"], check=False)
    return unstaged.returncode == 0 and staged.returncode == 0


def verify_frozen_files(root: Path, expected: list[dict], *, label: str) -> list[dict]:
    verified = []
    for row in expected:
        relative = str(row["path"])
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"{label} file is absent: {path}")
        size = path.stat().st_size
        if size != int(row["bytes"]):
            raise ValueError(f"{label} byte count differs: {relative}")
        digest = sha256_file(path)
        if digest != str(row["sha256"]):
            raise ValueError(f"{label} SHA-256 differs: {relative}")
        verified.append({"path": relative, "bytes": size, "sha256": digest})
    return verified


def runtime_source_files(*, include_health: bool = False) -> list[dict[str, Any]]:
    paths = (
        Path(__file__).resolve(),
        Path(chat_contract_module.__file__).resolve(),
        Path(checkpointing_module.__file__).resolve(),
        Path(feature_cache_module.__file__).resolve(),
        Path(fixed_budget_module.__file__).resolve(),
        Path(geometry_regularization_module.__file__).resolve(),
        Path(merge_module.__file__).resolve(),
        Path(model_module.__file__).resolve(),
        Path(projector_module.__file__).resolve(),
        Path(projector_binding_module.__file__).resolve(),
        Path(proxy_receiver_module.__file__).resolve(),
        Path(training_order_module.__file__).resolve(),
        Path(__file__).with_name("verify_feature_cache.py").resolve(),
    )
    if include_health:
        paths = (*paths, Path(training_health_module.__file__).resolve())
    return [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ]


def checkpoint_files(directory: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "CHECKPOINT_MANIFEST.json":
            files.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return files


def save_bound_checkpoint(
    *,
    directory: Path,
    projector: PatchMergerProjector,
    optimizer: torch.optim.Optimizer,
    step: int,
    history: list[dict[str, Any]],
    rng: random.Random,
    binding: dict[str, Any],
) -> dict[str, Any]:
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {directory}")
    save_training_checkpoint(
        directory=directory,
        projector=projector,
        optimizer=optimizer,
        step=step,
        history=history,
        rng=rng,
    )
    files = checkpoint_files(directory)
    manifest = {
        "format_version": "qwen3b-fixed-budget-checkpoint-v1",
        **binding,
        "step": int(step),
        "progress": (
            {
                key: history[-1][key]
                for key in (
                    "optimizer_steps",
                    "examples_seen",
                    "answer_tokens_seen",
                    "effective_epochs",
                    "subset_passes",
                )
            }
            if history
            else {
                "optimizer_steps": 0,
                "examples_seen": 0,
                "answer_tokens_seen": 0,
                "effective_epochs": 0.0,
                "subset_passes": 0.0,
            }
        ),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
    }
    write_json(directory / "CHECKPOINT_MANIFEST.json", manifest)
    return manifest


def verify_bound_checkpoint(
    directory: Path, *, expected_binding: dict[str, Any]
) -> dict[str, Any]:
    manifest = json.loads(
        (directory / "CHECKPOINT_MANIFEST.json").read_text(encoding="utf-8")
    )
    if manifest.get("format_version") != "qwen3b-fixed-budget-checkpoint-v1":
        raise ValueError("resume checkpoint format differs")
    for key, expected in expected_binding.items():
        if manifest.get(key) != expected:
            raise ValueError(f"resume checkpoint binding differs: {key}")
    manifest_files = sorted(manifest["files"], key=lambda row: str(row["path"]))
    actual_files = checkpoint_files(directory)
    if manifest_files != actual_files:
        raise ValueError("resume checkpoint file inventory differs")
    if int(manifest.get("file_count", -1)) != len(actual_files):
        raise ValueError("resume checkpoint file count differs")
    if int(manifest.get("total_bytes", -1)) != sum(
        int(row["bytes"]) for row in actual_files
    ):
        raise ValueError("resume checkpoint byte count differs")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--architecture-control",
        type=Path,
        help="Optional V1/V2 architecture-control sidecar contract",
    )
    parser.add_argument(
        "--architecture-arm",
        help="Arm name in --architecture-control",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--training-order-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--expected-cache-runner-git-sha", required=True)
    parser.add_argument("--projector-dir", type=Path, required=True)
    parser.add_argument(
        "--projector-variant-contract",
        type=Path,
        help="可选的结构变体合同；提供后必须同时给 --projector-variant-arm 和 --projector-base-dir",
    )
    parser.add_argument("--projector-variant-arm")
    parser.add_argument(
        "--projector-base-dir",
        type=Path,
        help="结构变体对应的冻结 step0 projector 目录",
    )
    parser.add_argument("--receiver-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--development-max-optimizer-steps", type=int)
    parser.add_argument(
        "--projector-learning-rate",
        type=float,
        help="探索用 projector 学习率覆盖；examples/steps/order 仍绑定主合同",
    )
    parser.add_argument(
        "--causal-shuffle-margin-lambda",
        type=float,
        default=0.0,
        help="探索用 paired image-vs-shuffle hinge loss 权重；默认关闭",
    )
    parser.add_argument(
        "--causal-shuffle-margin",
        type=float,
        default=0.10,
        help="paired hinge 要求的 shuffled-loss 减 correct-loss 最小间隔",
    )
    parser.add_argument("--geometry-screen-contract", type=Path)
    parser.add_argument("--geometry-calibration", type=Path)
    parser.add_argument("--geometry-reference-projector", type=Path)
    parser.add_argument("--geometry-arm")
    parser.add_argument("--health-contract", type=Path)
    parser.add_argument("--health-probe-manifest", type=Path)
    parser.add_argument("--health-probe-feature-cache", type=Path)
    parser.add_argument(
        "--health-auto-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="health probe 触发 critical guard 时自动保存并停止",
    )
    parser.add_argument("--allow-dirty-development-run", action="store_true")
    return parser.parse_args()


def load_architecture_overlay(
    *,
    core_contract_path: Path,
    core_contract: dict[str, Any],
    architecture_control_path: Path | None,
    architecture_arm: str | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Overlay one sidecar V1/V2 arm onto the shared Qwen contract.

    The sidecar changes only the frozen vision/cache interface and projector
    initialization.  Qwen, receiver, prompts, data order and budget remain
    inherited from the core contract.  Omitting the sidecar returns the exact
    historical contract object and metadata ``None``.
    """

    if architecture_control_path is None:
        if architecture_arm is not None:
            raise ValueError("--architecture-arm requires --architecture-control")
        return core_contract, None
    if architecture_arm is None:
        raise ValueError("--architecture-arm is required with --architecture-control")

    root = Path(__file__).resolve().parents[1]
    sidecar = json.loads(architecture_control_path.read_text(encoding="utf-8"))
    base = sidecar.get("base_contract")
    if isinstance(base, dict):
        raw_base_path = Path(str(base.get("path", "")))
        base_path = raw_base_path if raw_base_path.is_absolute() else root / raw_base_path
        if not base_path.is_file():
            candidate = architecture_control_path.parent / raw_base_path
            if candidate.is_file():
                base_path = candidate
        if not base_path.is_file():
            raise FileNotFoundError(f"architecture base contract is absent: {base_path}")
        expected_sha = base.get("sha256")
        if expected_sha and sha256_file(base_path) != str(expected_sha):
            raise ValueError("architecture base contract SHA-256 differs")
        if base_path.resolve() != core_contract_path.resolve():
            raise ValueError("architecture sidecar is bound to a different core contract")
    arms = sidecar.get("arms")
    if not isinstance(arms, dict) or architecture_arm not in arms:
        raise ValueError(f"architecture arm is absent: {architecture_arm}")
    arm = arms[architecture_arm]
    vision = arm.get("vision_tower")
    projector = arm.get("projector")
    if not isinstance(vision, dict) or not isinstance(projector, dict):
        raise ValueError("architecture arm requires vision_tower and projector objects")
    for key in ("vision_width", "merge_factor"):
        if key not in vision:
            raise ValueError(f"architecture arm vision_tower is missing {key}")
    for key in ("config_path", "config_sha256", "output_width", "parameter_count"):
        if key not in projector:
            raise ValueError(f"architecture arm projector is missing {key}")
    raw_projector_config = Path(str(projector["config_path"]))
    projector_config_path = (
        raw_projector_config
        if raw_projector_config.is_absolute()
        else root / raw_projector_config
    )
    if not projector_config_path.is_file():
        candidate = architecture_control_path.parent / raw_projector_config
        if candidate.is_file():
            projector_config_path = candidate
    if not projector_config_path.is_file():
        raise FileNotFoundError(
            f"architecture projector source config is absent: {projector_config_path}"
        )
    source_config_sha = projector.get("source_config_sha256")
    if source_config_sha is not None and sha256_file(projector_config_path) != str(
        source_config_sha
    ):
        raise ValueError("architecture projector source config SHA-256 differs")

    effective = copy.deepcopy(core_contract)
    effective_vision = effective["vision_tower"]
    effective_vision.update(
        {
            "name": vision.get("name", effective_vision.get("name")),
            "source_repo": vision.get("model", effective_vision.get("source_repo")),
            "source_resolved_revision": vision.get(
                "revision",
                vision.get("resolved_revision", effective_vision.get("source_resolved_revision")),
            ),
            "vision_width": int(vision["vision_width"]),
            "merge_factor": int(vision["merge_factor"]),
            "frozen": True,
        }
    )
    if vision.get("weights_sha256") is not None:
        effective_vision["extracted_weights_sha256"] = str(vision["weights_sha256"])
    else:
        effective_vision.pop("extracted_weights_sha256", None)
    effective["feature_cache_binding"] = {
        "vision_tower": vision.get("cache_tower_id", vision.get("name")),
        "moonvit_model": vision.get("model"),
        "moonvit_revision": vision.get("revision", vision.get("resolved_revision")),
        "moonvit_weights_sha256": vision.get("weights_sha256"),
        "vision_width": int(vision["vision_width"]),
        "merge_factor": int(vision["merge_factor"]),
        "require_tower_identity": bool(vision.get("require_tower_identity", True)),
    }
    effective_projector = effective["canonical_projector"]
    effective_projector.update(
        {
            "config": str(projector["config_path"]),
            "config_sha256": str(projector["config_sha256"]),
            "source_config_sha256": (
                str(source_config_sha) if source_config_sha is not None else None
            ),
            "projector_variant": projector.get(
                "variant", effective_projector.get("projector_variant", "legacy_pre_norm")
            ),
            "output_width": int(projector["output_width"]),
            "parameter_count": int(projector["parameter_count"]),
        }
    )
    initialization = projector.get("initialization")
    if not isinstance(initialization, dict) or "step0" not in initialization:
        raise ValueError("architecture arm projector initialization is incomplete")
    effective_projector["initialization_contract"] = copy.deepcopy(initialization)
    effective_projector["initialization_seed"] = int(
        initialization["step0"]["seed"]
    )
    metadata = {
        "path": str(architecture_control_path.resolve()),
        "sha256": sha256_file(architecture_control_path),
        "arm": str(architecture_arm),
        "effective_contract_sha256": canonical_sha256(effective),
        "vision_tower": copy.deepcopy(vision),
        "projector": copy.deepcopy(projector),
    }
    return effective, metadata


def load_geometry_setup(
    args: argparse.Namespace, *, core_contract_path: Path, core_contract: dict[str, Any]
) -> dict[str, Any] | None:
    """校验并加载 Package 15P 的固定几何辅助项配置。"""

    fields = (
        args.geometry_screen_contract,
        args.geometry_calibration,
        args.geometry_reference_projector,
        args.geometry_arm,
    )
    if all(value is None for value in fields):
        return None
    if any(value is None for value in fields):
        raise ValueError("geometry screen arguments must be supplied together")
    screen_path = args.geometry_screen_contract
    calibration_path = args.geometry_calibration
    reference_dir = args.geometry_reference_projector
    assert screen_path is not None and calibration_path is not None
    assert reference_dir is not None and args.geometry_arm is not None
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if sha256_file(core_contract_path) != screen["core_contract_file_sha256"]:
        raise ValueError("geometry screen core contract hash differs")
    if calibration.get("status") != "valid" or calibration.get(
        "formal_calibration_complete"
    ) is not True:
        raise ValueError("geometry calibration is not a formal valid result")
    if calibration.get("screen_contract_file_sha256") != sha256_file(screen_path):
        raise ValueError("geometry calibration is bound to a different screen contract")
    reference_sha = sha256_file(reference_dir / "projector.safetensors")
    if (
        reference_sha != screen["reference_projector_sha256"]
        or reference_sha
        != core_contract["canonical_projector"]["initialization_contract"]["step0"][
            "weights_sha256"
        ]
    ):
        raise ValueError("geometry reference projector differs from exact step0")
    arm_name = str(args.geometry_arm)
    derived = calibration.get("derived_arms", {})
    if arm_name not in derived:
        raise ValueError(f"geometry arm is absent from calibration: {arm_name}")
    arm = derived[arm_name]
    target_ratio = float(arm["target_gradient_ratio"])
    geometry_lambda = float(arm["lambda"])
    if target_ratio < 0.0 or geometry_lambda < 0.0:
        raise ValueError("geometry target ratio and lambda must be non-negative")
    objective = screen["geometry_objective"]
    weights = objective["component_weights"]
    return {
        "screen_contract": screen,
        "screen_contract_file_sha256": sha256_file(screen_path),
        "calibration_summary": calibration,
        "calibration_summary_file_sha256": sha256_file(calibration_path),
        "reference_projector_dir": str(reference_dir.resolve()),
        "reference_projector_sha256": reference_sha,
        "arm": arm_name,
        "target_gradient_ratio": target_ratio,
        "geometry_lambda": geometry_lambda,
        "geometry_kwargs": {
            "scale_weight": float(weights["scale"]),
            "relative_spread_weight": float(weights["relative_spread"]),
            "centered_gram_weight": float(weights["centered_gram"]),
            "epsilon": float(objective["epsilon"]),
        },
    }


def compute_geometry_auxiliary_gradients(
    *,
    projector: PatchMergerProjector,
    reference_projector: PatchMergerProjector,
    feature_batches: list[list[torch.Tensor]],
    geometry_kwargs: dict[str, float],
) -> tuple[Any, tuple[torch.Tensor, ...], float]:
    """对一个真实 global batch 计算无权重 geometry loss 及其 projector 梯度。"""

    current_outputs = [projector(groups) for groups in feature_batches]
    current_pooled = pool_projector_batch(current_outputs)
    with torch.no_grad():
        reference_outputs = [reference_projector(groups) for groups in feature_batches]
        reference_pooled = pool_projector_batch(reference_outputs)
    result = geometry_regularization_loss(
        current_pooled, reference_pooled, **geometry_kwargs
    )
    gradients = torch.autograd.grad(result.total, tuple(projector.parameters()))
    gradient_norm = float(global_gradient_norm(gradients))
    return result, gradients, gradient_norm


def prepare_supervision(
    *,
    tokenizer: Any,
    contract: dict[str, Any],
    order_manifest: dict[str, Any],
    records: list[dict[str, Any]],
    cache_manifest: dict[str, Any],
    placeholder_token_id: int,
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared = []
    audit_rows = []
    prompt_lengths = []
    answer_tokens = []
    expanded_lengths = []
    route_counts: dict[str, int] = {}
    model_limit = int(contract["proxy_model"]["max_position_embeddings"])
    for index, (entry, record, cache_row) in enumerate(
        zip(order_manifest["records"], records, cache_manifest["records"], strict=True)
    ):
        routed = route_training_example(contract, entry, record)
        supervision = build_chat_supervision(
            tokenizer,
            system_prompt=routed.system_prompt,
            user_prompt=routed.user_prompt,
            answer=routed.target_answer,
            placeholder_token_id=placeholder_token_id,
            include_image=True,
        )
        if supervision.prompt.placeholder_count != 1:
            raise ValueError(f"supervision placeholder count differs: {routed.record_id}")
        nonmasked = sum(label != -100 for label in supervision.labels)
        if nonmasked != supervision.answer_tokens:
            raise ValueError(f"supervision answer-token count differs: {routed.record_id}")
        visual_tokens = int(cache_row["feature_shape"][0])
        expanded_length = len(supervision.input_ids) - 1 + visual_tokens
        if expanded_length > model_limit:
            raise ValueError(f"expanded supervision exceeds model context: {routed.record_id}")
        row = {
            "index": index,
            "id": routed.record_id,
            "prompt_route": routed.prompt_route,
            "target_answer": routed.target_answer,
            "input_ids": supervision.input_ids,
            "labels": supervision.labels,
            "answer_tokens": supervision.answer_tokens,
            "visual_tokens": visual_tokens,
        }
        prepared.append(row)
        audit_rows.append(
            {
                "index": index,
                "id": routed.record_id,
                "prompt_route": routed.prompt_route,
                "target_answer": routed.target_answer,
                "target_answer_sha256": hashlib.sha256(
                    routed.target_answer.encode("utf-8")
                ).hexdigest(),
                "template_text_sha256": hashlib.sha256(
                    supervision.prompt.template_text_for_audit.encode("utf-8")
                ).hexdigest(),
                "input_ids_sha256": canonical_sha256(supervision.input_ids),
                "labels_sha256": canonical_sha256(supervision.labels),
                "prompt_length": supervision.prompt_length,
                "answer_tokens": supervision.answer_tokens,
                "visual_tokens": visual_tokens,
                "expanded_sequence_length": expanded_length,
            }
        )
        prompt_lengths.append(supervision.prompt_length)
        answer_tokens.append(supervision.answer_tokens)
        expanded_lengths.append(expanded_length)
        route_counts[routed.prompt_route] = route_counts.get(routed.prompt_route, 0) + 1

    audit_path = out / "SUPERVISION_RECORDS.jsonl"
    with audit_path.open("w", encoding="utf-8") as stream:
        for row in audit_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "records": len(prepared),
        "route_counts": dict(sorted(route_counts.items())),
        "answer_tokens_total": sum(answer_tokens),
        "answer_tokens_min": min(answer_tokens),
        "answer_tokens_max": max(answer_tokens),
        "prompt_length_min": min(prompt_lengths),
        "prompt_length_max": max(prompt_lengths),
        "expanded_sequence_length_min": min(expanded_lengths),
        "expanded_sequence_length_max": max(expanded_lengths),
        "records_file": audit_path.name,
        "records_file_sha256": sha256_file(audit_path),
    }
    write_json(out / "SUPERVISION_SUMMARY.json", summary)
    return prepared, summary


def _verify_self_hash(payload: dict[str, Any], *, field: str) -> bool:
    expected = payload.get(field)
    if not isinstance(expected, str):
        return False
    copy = dict(payload)
    copy.pop(field, None)
    return canonical_sha256(copy) == expected


def load_health_contract(contract_path: Path) -> dict[str, Any]:
    """加载共享阈值合同，并允许 architecture arm 只覆盖 probe 身份。"""

    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    extends = raw.get("extends")
    if not extends:
        return raw
    base_path = Path(str(extends))
    if not base_path.is_absolute():
        base_path = contract_path.parent / base_path
    if not base_path.is_file():
        raise FileNotFoundError(f"health contract base is absent: {base_path}")
    base = load_health_contract(base_path)

    def merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(left)
        for key, value in right.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    return merge(base, raw)


def load_health_setup(
    args: argparse.Namespace,
    *,
    core_contract_path: Path,
    core_contract: dict[str, Any],
) -> dict[str, Any] | None:
    """在加载模型前冻结并校验通用 projector health 合同。"""

    fields = (
        args.health_contract,
        args.health_probe_manifest,
        args.health_probe_feature_cache,
    )
    if all(value is None for value in fields):
        return None
    if any(value is None for value in fields):
        raise ValueError("health arguments must be supplied together")
    contract_path = args.health_contract
    probe_path = args.health_probe_manifest
    cache_path = args.health_probe_feature_cache
    assert contract_path is not None and probe_path is not None and cache_path is not None
    health_contract = load_health_contract(contract_path)
    validate_health_contract(health_contract)
    if health_contract.get("core_qwen_contract_file_sha256") != sha256_file(
        core_contract_path
    ):
        raise ValueError("health contract is bound to a different core contract")
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if not _verify_self_hash(probe, field="manifest_sha256"):
        raise ValueError("health probe manifest self-hash verification failed")
    probe_contract = health_contract["probe_manifest"]
    if sha256_file(probe_path) != probe_contract["file_sha256"]:
        raise ValueError("health probe manifest file SHA-256 differs")
    if probe.get("manifest_sha256") != probe_contract["manifest_sha256"]:
        raise ValueError("health probe manifest canonical SHA-256 differs")
    if int(probe.get("count", -1)) != int(probe_contract["count"]):
        raise ValueError("health probe sample count differs")
    cache_manifest_path = cache_path / "MANIFEST.json"
    if sha256_file(cache_manifest_path) != probe_contract[
        "feature_cache_manifest_file_sha256"
    ]:
        raise ValueError("health probe feature-cache manifest SHA-256 differs")
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    cache = FeatureCache(cache_path)
    cache_by_id = {str(row["id"]): row for row in cache_manifest["records"]}
    feature_ids = []
    for row in probe["samples"]:
        sample_id = str(row["sample_id"])
        cache_row = cache_by_id.get(sample_id)
        if cache_row is None:
            raise ValueError(f"health probe cache is missing sample: {sample_id}")
        groups = cache.get(sample_id, device="cpu", dtype=torch.float32)
        if list(groups[0].shape) != [int(value) for value in row["feature_shape"]]:
            raise ValueError(f"health probe feature shape differs: {sample_id}")
        if tensor_sha256(groups[0]) != str(row["feature_sha256"]):
            raise ValueError(f"health probe feature SHA-256 differs: {sample_id}")
        if str(cache_row["image_sha256"]) != str(row["image_sha256"]):
            raise ValueError(f"health probe image SHA-256 differs: {sample_id}")
        feature_ids.append(sample_id)
    if feature_ids != [str(row["sample_id"]) for row in probe["samples"]]:
        raise ValueError("health probe sample order is not stable")
    screen_path = Path(__file__).resolve().parents[1] / health_contract["screen_contract"]["manifest_file"]
    if sha256_file(screen_path) != health_contract["screen_contract"]["manifest_file_sha256"]:
        raise ValueError("health probe ScreenSpot manifest file SHA-256 differs")
    screen_manifest = json.loads(screen_path.read_text(encoding="utf-8"))
    if screen_manifest.get("manifest_sha256") != health_contract["screen_contract"]["manifest_sha256"]:
        raise ValueError("health probe ScreenSpot manifest SHA-256 differs")
    if [str(row["sample_id"]) for row in screen_manifest["samples"]] != feature_ids:
        raise ValueError("health probe IDs differ from frozen ScreenSpot order")
    return {
        "contract": health_contract,
        "contract_file_sha256": sha256_file(contract_path),
        "probe": probe,
        "probe_file_sha256": sha256_file(probe_path),
        "probe_cache": cache,
        "probe_cache_root": str(cache_path.resolve()),
        "probe_cache_manifest_file_sha256": sha256_file(cache_manifest_path),
        "screen_manifest": screen_manifest,
        "screen_manifest_file": str(screen_path.resolve()),
        "screen_manifest_file_sha256": sha256_file(screen_path),
        "teacher_ids": [
            str(value) for value in probe["teacher_forced_sample_ids"]
        ],
    }


def build_health_supervisions(
    *,
    tokenizer: Any,
    prompt_contract: dict[str, Any],
    screen_manifest: dict[str, Any],
    teacher_ids: list[str],
    placeholder_id: int,
) -> dict[str, dict[str, Any]]:
    """构造固定 teacher-forced health probe 的三种图像条件。"""

    targets = build_counterfactual_targets(screen_manifest)
    samples = {str(row["sample_id"]): row for row in screen_manifest["samples"]}
    output: dict[str, dict[str, Any]] = {}
    for sample_id in teacher_ids:
        sample = samples.get(sample_id)
        if sample is None:
            raise ValueError(f"health teacher probe sample is absent: {sample_id}")
        target = targets[sample_id]
        user_prompt = prompt_contract["user_prompt"].format(
            instruction=sample["instruction"]
        )
        answers = (target["correct_answer"], target["counterfactual_answer"])
        output[sample_id] = {
            "sample": sample,
            "target": target,
            "vision": [
                build_chat_supervision(
                    tokenizer,
                    system_prompt=prompt_contract["system_prompt"],
                    user_prompt=user_prompt,
                    answer=answer,
                    placeholder_token_id=placeholder_id,
                    include_image=True,
                )
                for answer in answers
            ],
            "blind": [
                build_chat_supervision(
                    tokenizer,
                    system_prompt=prompt_contract["system_prompt"],
                    user_prompt=user_prompt,
                    answer=answer,
                    placeholder_token_id=placeholder_id,
                    include_image=False,
                )
                for answer in answers
            ],
        }
    return output


def _health_supervision_batch(
    supervisions: list[Any], *, pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(len(row.input_ids) for row in supervisions)
    input_ids = torch.full(
        (len(supervisions), max_length), int(pad_token_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for index, row in enumerate(supervisions):
        length = len(row.input_ids)
        input_ids[index, :length] = torch.tensor(row.input_ids, dtype=torch.long, device=device)
        attention_mask[index, :length] = 1
        labels[index, :length] = torch.tensor(row.labels, dtype=torch.long, device=device)
    return input_ids, attention_mask, labels


@torch.inference_mode()
def _score_health_condition(
    *,
    condition: str,
    rows: list[dict[str, Any]],
    language_model: torch.nn.Module,
    projector: PatchMergerProjector,
    receiver: FixedPairwiseReceiverAdapter,
    feature_cache: FeatureCache,
    placeholder_id: int,
    pad_token_id: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    supervision_condition = "vision" if condition == "shuffled" else condition
    all_supervisions = [item[supervision_condition] for item in rows]
    flat_supervisions = [supervision for pair in all_supervisions for supervision in pair]
    input_ids, attention_mask, labels = _health_supervision_batch(
        flat_supervisions, pad_token_id=pad_token_id, device=device
    )
    if condition == "blind":
        outputs = language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    else:
        image_embeddings: list[torch.Tensor] = []
        for item in rows:
            image_id = str(item["sample"]["sample_id"])
            if condition == "shuffled":
                image_id = str(item["target"]["counterfactual_sample_id"])
            projected = projector(
                feature_cache.get(image_id, device=device, dtype=torch.float32)
            )[0]
            received = receiver(projected)
            image_embeddings.extend((received, received))
        text_embeddings = language_model.get_input_embeddings()(input_ids)
        merged = expand_image_placeholders(
            input_ids=input_ids,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            placeholder_token_id=placeholder_id,
            attention_mask=attention_mask,
            labels=labels,
            pad_token_id=pad_token_id,
        )
        outputs = language_model(
            inputs_embeds=merged.inputs_embeds,
            attention_mask=merged.attention_mask,
            position_ids=merged.position_ids,
            use_cache=False,
        )
        labels = merged.labels
        if labels is None:
            raise AssertionError("health visual labels were not expanded")
    stats = answer_logprob_stats(outputs.logits, labels)
    result = []
    for index, item in enumerate(rows):
        correct = stats[index * 2]
        counterfactual = stats[index * 2 + 1]
        margin = float(correct["logp_mean"]) - float(counterfactual["logp_mean"])
        result.append(
            {
                "sample_id": str(item["sample"]["sample_id"]),
                "condition": condition,
                "correct_logp_mean": float(correct["logp_mean"]),
                "counterfactual_logp_mean": float(counterfactual["logp_mean"]),
                "correct_margin": margin,
                "correct_preferred": margin > 0.0,
            }
        )
    return result


def run_health_causal_probe(
    *,
    rows: list[dict[str, Any]],
    language_model: torch.nn.Module,
    projector: PatchMergerProjector,
    receiver: FixedPairwiseReceiverAdapter,
    feature_cache: FeatureCache,
    placeholder_id: int,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    condition_rows = {
        condition: _score_health_condition(
            condition=condition,
            rows=rows,
            language_model=language_model,
            projector=projector,
            receiver=receiver,
            feature_cache=feature_cache,
            placeholder_id=placeholder_id,
            pad_token_id=pad_token_id,
            device=device,
        )
        for condition in ("vision", "shuffled", "blind")
    }

    def summary(condition: str) -> dict[str, Any]:
        values = condition_rows[condition]
        margins = [float(row["correct_margin"]) for row in values]
        return {
            "records": len(values),
            "preference_count": sum(float(value) > 0.0 for value in margins),
            "preference": sum(float(value) > 0.0 for value in margins) / len(values),
            "mean_correct_margin": sum(margins) / len(margins),
            "mean_correct_logp": sum(
                float(row["correct_logp_mean"]) for row in values
            )
            / len(values),
            "rows": values,
        }

    summaries = {condition: summary(condition) for condition in condition_rows}
    return {
        "vision_preference": summaries["vision"]["preference"],
        "shuffled_preference": summaries["shuffled"]["preference"],
        "blind_preference": summaries["blind"]["preference"],
        "vision_minus_shuffle_correct_logp": (
            summaries["vision"]["mean_correct_logp"]
            - summaries["shuffled"]["mean_correct_logp"]
        ),
        "vision_minus_blind_correct_logp": (
            summaries["vision"]["mean_correct_logp"]
            - summaries["blind"]["mean_correct_logp"]
        ),
        "rows": summaries,
    }


@torch.inference_mode()
def collect_health_representation(
    *,
    projector: PatchMergerProjector,
    receiver: FixedPairwiseReceiverAdapter,
    feature_cache: FeatureCache,
    sample_ids: list[str],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """逐图前向并立即转回 CPU，限制 V100 probe 峰值显存。"""

    projector_sequences: list[torch.Tensor] = []
    receiver_sequences: list[torch.Tensor] = []
    for sample_id in sample_ids:
        projected = projector(
            feature_cache.get(sample_id, device=device, dtype=torch.float32)
        )[0]
        received = receiver(projected)
        projector_sequences.append(projected.detach().cpu())
        receiver_sequences.append(received.detach().cpu())
        del projected, received
    return projector_sequences, receiver_sequences


def run_health_probe(
    *,
    step: int,
    projector: PatchMergerProjector,
    receiver: FixedPairwiseReceiverAdapter,
    language_model: torch.nn.Module,
    feature_cache: FeatureCache,
    sample_ids: list[str],
    step0_projector_sequences: list[torch.Tensor],
    step0_receiver_sequences: list[torch.Tensor],
    teacher_rows: list[dict[str, Any]],
    placeholder_id: int,
    pad_token_id: int,
    device: torch.device,
    training_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    """运行一次固定 representation 与 teacher-forced 小探针。"""

    started = time.perf_counter()
    projector_sequences, receiver_sequences = collect_health_representation(
        projector=projector,
        receiver=receiver,
        feature_cache=feature_cache,
        sample_ids=sample_ids,
        device=device,
    )
    payload = summarize_probe(
        projector_sequences,
        receiver_sequences,
        step0_projector_sequences=step0_projector_sequences,
        step0_receiver_sequences=step0_receiver_sequences,
        step=step,
    )
    payload["causal"] = run_health_causal_probe(
        rows=teacher_rows,
        language_model=language_model,
        projector=projector,
        receiver=receiver,
        feature_cache=feature_cache,
        placeholder_id=placeholder_id,
        pad_token_id=pad_token_id,
        device=device,
    )
    payload["training"] = training_metrics or {
        "gradient_norm_before_clip": None,
        "gradient_norm_after_clip": None,
    }
    payload["has_nan_or_inf"] = False
    payload["probe_wall_seconds"] = time.perf_counter() - started
    payload["sample_count"] = len(sample_ids)
    payload["teacher_forced_sample_count"] = len(teacher_rows)
    return jsonable_probe(payload)


def save_health_stop_and_rollback(
    *,
    out: Path,
    step: int,
    projector: PatchMergerProjector,
    optimizer: torch.optim.Optimizer,
    history: list[dict[str, Any]],
    rng: random.Random,
    checkpoint_binding: dict[str, Any],
    batch_ids: list[str],
    health_row: dict[str, Any] | None,
    probe: dict[str, Any],
    guard: dict[str, Any],
    previous_probe: dict[str, Any] | None,
    last_healthy_checkpoint: Path | None,
    device: torch.device,
) -> dict[str, Any]:
    """保存 critical 轨迹、failure checkpoint，并把内存状态回滚到健康点。"""

    failure_checkpoint = (
        out
        / "health_snapshots"
        / "checkpoints"
        / f"failure-step-{int(step):06d}"
    )
    failure_binding = {**checkpoint_binding, "health_checkpoint_role": "failure"}
    failure_manifest = save_bound_checkpoint(
        directory=failure_checkpoint,
        projector=projector,
        optimizer=optimizer,
        step=step,
        history=history,
        rng=rng,
        binding=failure_binding,
    )
    onset_left = (
        int(previous_probe["step"])
        if previous_probe is not None
        else max(0, int(step) - 1)
    )
    failure = {
        "status": "auto_stopped_by_projector_health_guard",
        "optimizer_step": int(step),
        "collapse_onset_interval": [onset_left, int(step)],
        "critical_reasons": list(guard["critical"]),
        "warnings": list(guard["warnings"]),
        "current_batch_ids": list(batch_ids),
        "current_batch_ids_sha256": canonical_sha256(batch_ids),
        "health_metrics": health_row,
        "probe_metrics": probe,
        "failure_checkpoint": str(failure_checkpoint),
        "failure_checkpoint_file_count": failure_manifest["file_count"],
        "failure_checkpoint_total_bytes": failure_manifest["total_bytes"],
        "last_healthy_checkpoint": (
            str(last_healthy_checkpoint) if last_healthy_checkpoint else None
        ),
        "resume_from_failure_forbidden": True,
        "capability_claim_allowed": False,
        "final_half_scored": False,
        "paid_resources_used": False,
    }
    write_json(out / "FAILURE.json", failure)
    rollback = {
        "status": "no_healthy_checkpoint_available",
        "source": None,
        "restored_step": None,
    }
    if last_healthy_checkpoint is not None:
        restored_step, _restored_history, _restored_rng, restored_dir = (
            load_training_checkpoint(
                source=last_healthy_checkpoint,
                projector=projector,
                optimizer=optimizer,
                device=device,
            )
        )
        rollback = {
            "status": "rolled_back_to_last_healthy_checkpoint",
            "source": str(restored_dir),
            "restored_step": int(restored_step),
            "failed_step": int(step),
            "critical_checkpoint_must_not_resume": str(failure_checkpoint),
        }
    write_json(out / "ROLLBACK.json", rollback)
    failure["rollback"] = rollback
    return failure


def write_health_artifact_manifest(out: Path) -> dict[str, Any]:
    """绑定训练结束时已封闭的 health/probe/checkpoint 产物。"""

    candidates = [out / "train_health.jsonl", out / "probe_metrics.jsonl"]
    health_root = out / "health_snapshots"
    if health_root.exists():
        candidates.extend(path for path in health_root.rglob("*") if path.is_file())
    candidates.extend(path for path in (out / "FAILURE.json", out / "ROLLBACK.json") if path.is_file())
    files = []
    for path in sorted(set(candidates)):
        files.append(
            {
                "path": path.relative_to(out).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "format_version": "projector-health-artifact-manifest-v1",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "run_log": "run.log (closed after the runner returns; independently verify later)",
        "paid_resources_used": False,
    }
    write_json(out / "HEALTH_ARTIFACT_MANIFEST.json", manifest)
    return manifest


def _run(args: argparse.Namespace, stage: dict[str, str]) -> dict[str, Any]:
    started = time.perf_counter()
    tracked_clean = git_tracked_worktree_clean()
    current_git_sha = git_sha()
    if not tracked_clean and not args.allow_dirty_development_run:
        raise RuntimeError("tracked Git worktree is dirty; formal training is refused")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")

    set_stage(stage, "contract_order_cache_verification")
    core_contract = json.loads(args.contract.read_text(encoding="utf-8"))
    contract, architecture_metadata = load_architecture_overlay(
        core_contract_path=args.contract,
        core_contract=core_contract,
        architecture_control_path=args.architecture_control,
        architecture_arm=args.architecture_arm,
    )
    order_manifest = json.loads(
        args.training_order_manifest.read_text(encoding="utf-8")
    )
    cache_manifest_path = args.feature_cache / "MANIFEST.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if not verify_training_order_manifest(order_manifest):
        raise ValueError("training-order manifest self-verification failed")
    binding_summary = validate_fixed_budget_contract(
        contract, order_manifest, cache_manifest
    )
    records = load_ordered_records(data_path=args.data, manifest=order_manifest)
    cache_verification = verify_feature_cache(
        args.feature_cache,
        expected_count=binding_summary["examples_seen"],
        training_order_manifest_path=args.training_order_manifest,
        expected_git_sha=args.expected_cache_runner_git_sha,
    )
    write_json(args.out / "CACHE_VERIFICATION.json", cache_verification)
    gc.collect()

    set_stage(stage, "frozen_file_verification")
    model_files = verify_frozen_files(
        args.model_dir, contract["proxy_model"]["files"], label="Qwen contract"
    )
    receiver_path = args.receiver_dir / "proxy_receiver.safetensors"
    projector_path = args.projector_dir / "projector.safetensors"
    projector_config_path = args.projector_dir / "projector_config.json"
    canonical_projector = contract["canonical_projector"]
    canonical_projector_sha = canonical_projector["initialization_contract"][
        "step0"
    ]["weights_sha256"]
    if args.projector_variant_contract is None:
        if args.projector_variant_arm is not None or args.projector_base_dir is not None:
            raise ValueError(
                "projector variant arm/base-dir require --projector-variant-contract"
            )
        if sha256_file(projector_path) != canonical_projector_sha:
            raise ValueError("step0 projector SHA-256 differs from the frozen contract")
        if sha256_file(projector_config_path) != str(
            canonical_projector["config_sha256"]
        ):
            raise ValueError("step0 projector config SHA-256 differs from the contract")
        projector_binding = canonical_binding(
            weights_sha256=canonical_projector_sha,
            parameter_count=int(canonical_projector["parameter_count"]),
        ).as_dict()
    else:
        if args.projector_variant_arm is None or args.projector_base_dir is None:
            raise ValueError(
                "projector variant contract requires --projector-variant-arm and --projector-base-dir"
            )
        variant_contract_path = args.projector_variant_contract.resolve()
        projector_binding = validate_variant_binding(
            root=Path(__file__).resolve().parents[1],
            contract_path=variant_contract_path,
            arm_name=str(args.projector_variant_arm),
            projector_dir=args.projector_dir.resolve(),
            base_dir=args.projector_base_dir.resolve(),
        ).as_dict()
    expected_projector_sha = str(projector_binding["weights_sha256"])
    expected_projector_parameter_count = int(projector_binding["parameter_count"])
    expected_receiver_sha = contract["qwen_proxy_receiver"]["buffer_sha256"]
    if sha256_file(receiver_path) != expected_receiver_sha:
        raise ValueError("proxy receiver SHA-256 differs from the frozen contract")

    set_stage(stage, "tokenizer_config_supervision")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import transformers

    model_config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False
    )
    expected_model = contract["proxy_model"]
    if (
        model_config.architectures != [expected_model["architecture"]]
        or model_config.model_type != expected_model["model_type"]
        or hasattr(model_config, "vision_config")
        or int(model_config.hidden_size) != int(expected_model["hidden_size"])
    ):
        raise ValueError("runtime backbone differs from the frozen pure-text Qwen model")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False
    )
    prompt_contract = contract["prompt_and_generation"]
    placeholder_id = int(prompt_contract["image_placeholder_token_id"])
    if tokenizer.convert_tokens_to_ids(prompt_contract["image_placeholder_token"]) != placeholder_id:
        raise ValueError("Qwen image placeholder ID differs from the contract")
    if int(tokenizer.eos_token_id) != int(prompt_contract["eos_token_id"]):
        raise ValueError("Qwen EOS ID differs from the contract")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    chat_template_sha = hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()
    if chat_template_sha != expected_model["chat_template_sha256"]:
        raise ValueError("Qwen chat template SHA-256 differs from the contract")
    prepared, supervision_summary = prepare_supervision(
        tokenizer=tokenizer,
        contract=contract,
        order_manifest=order_manifest,
        records=records,
        cache_manifest=cache_manifest,
        placeholder_token_id=placeholder_id,
        out=args.out,
    )

    total_steps = int(binding_summary["optimizer_steps"])
    target_steps = (
        int(args.development_max_optimizer_steps)
        if args.development_max_optimizer_steps is not None
        else total_steps
    )
    if target_steps <= 0 or target_steps > total_steps:
        raise ValueError("development max optimizer steps falls outside the 4k budget")
    geometry_setup = load_geometry_setup(
        args, core_contract_path=args.contract, core_contract=contract
    )
    if geometry_setup is not None:
        expected_screen_steps = int(
            geometry_setup["screen_contract"]["screen"]["optimizer_steps"]
        )
        if target_steps != expected_screen_steps:
            raise ValueError("geometry screen optimizer-step budget differs")
    health_setup = load_health_setup(
        args, core_contract_path=args.contract, core_contract=contract
    )
    formal_run = (
        tracked_clean
        and not args.allow_dirty_development_run
        and target_steps == total_steps
    )
    source_files = runtime_source_files(include_health=health_setup is not None)
    contract_file_sha = sha256_file(args.contract)
    order_file_sha = sha256_file(args.training_order_manifest)
    cache_manifest_file_sha = sha256_file(cache_manifest_path)
    checkpoint_binding = {
        "runner_git_sha": current_git_sha,
        "contract_file_sha256": contract_file_sha,
        "training_order_manifest_file_sha256": order_file_sha,
        "training_order_manifest_sha256": order_manifest["manifest_sha256"],
        "training_order_records_sha256": order_manifest["records_sha256"],
        "feature_cache_manifest_file_sha256": cache_manifest_file_sha,
        "feature_cache_records_sha256": cache_manifest["records_sha256"],
        "feature_cache_runner_git_sha": cache_manifest["git_sha"],
        "initial_projector_sha256": expected_projector_sha,
        "projector_binding": projector_binding,
        "proxy_receiver_sha256": expected_receiver_sha,
    }
    if architecture_metadata is not None:
        checkpoint_binding.update(
            {
                "architecture_control_path": architecture_metadata["path"],
                "architecture_control_sha256": architecture_metadata["sha256"],
                "architecture_arm": architecture_metadata["arm"],
                "architecture_effective_contract_sha256": architecture_metadata[
                    "effective_contract_sha256"
                ],
            }
        )
    if geometry_setup is not None:
        checkpoint_binding.update(
            {
                "geometry_screen_contract_file_sha256": geometry_setup[
                    "screen_contract_file_sha256"
                ],
                "geometry_calibration_summary_file_sha256": geometry_setup[
                    "calibration_summary_file_sha256"
                ],
                "geometry_reference_projector_sha256": geometry_setup[
                    "reference_projector_sha256"
                ],
                "geometry_arm": geometry_setup["arm"],
                "geometry_target_gradient_ratio": geometry_setup[
                    "target_gradient_ratio"
                ],
                "geometry_lambda": geometry_setup["geometry_lambda"],
            }
        )
    if health_setup is not None:
        checkpoint_binding.update(
            {
                "health_contract_file_sha256": health_setup[
                    "contract_file_sha256"
                ],
                "health_probe_manifest_file_sha256": health_setup[
                    "probe_file_sha256"
                ],
                "health_probe_cache_manifest_file_sha256": health_setup[
                    "probe_cache_manifest_file_sha256"
                ],
                "health_auto_stop": bool(args.health_auto_stop),
            }
        )
    run_config = {
        "format_version": "qwen3b-fixed-budget-training-run-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_run": formal_run,
        "capability_claim_allowed_before_evaluation": False,
        "final_half_scored": False,
        "paid_resources_used": False,
        "git_sha": current_git_sha,
        "git_tracked_worktree_clean": tracked_clean,
        "runtime_source_files": source_files,
        "transformers_version": transformers.__version__,
        "contract": str(args.contract.resolve()),
        "contract_file_sha256": contract_file_sha,
        "architecture_control": architecture_metadata,
        "model_dir": str(args.model_dir.resolve()),
        "model_files": model_files,
        "data": str(args.data.resolve()),
        "training_order_manifest": str(args.training_order_manifest.resolve()),
        "training_order_manifest_file_sha256": order_file_sha,
        "feature_cache": str(args.feature_cache.resolve()),
        "feature_cache_manifest_file_sha256": cache_manifest_file_sha,
        "feature_cache_verification": cache_verification,
        "projector_dir": str(args.projector_dir.resolve()),
        "projector_binding": projector_binding,
        "receiver_dir": str(args.receiver_dir.resolve()),
        "binding": binding_summary,
        "target_optimizer_steps": target_steps,
        "formal_optimizer_steps": total_steps,
        "projector_learning_rate_contract": float(
            contract["training_budget"]["learning_rate"]
        ),
        "projector_learning_rate_override": (
            float(args.projector_learning_rate)
            if args.projector_learning_rate is not None
            else None
        ),
        "causal_shuffle_margin_lambda": float(args.causal_shuffle_margin_lambda),
        "causal_shuffle_margin": float(args.causal_shuffle_margin),
        "checkpoint_every": args.checkpoint_every,
        "resume": str(args.resume.resolve()) if args.resume else None,
        "supervision": supervision_summary,
        "geometry_setup": geometry_setup,
        "health_setup": (
            {
                key: value
                for key, value in health_setup.items()
                if key not in {"probe_cache", "screen_manifest", "probe"}
            }
            if health_setup is not None
            else None
        ),
        "health_auto_stop": bool(args.health_auto_stop),
    }
    write_json(args.out / "RUN_CONFIG.json", run_config)

    set_stage(stage, "cuda_and_model_load")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Qwen3B training requires the existing CUDA V100")
    seed = int(contract["canonical_projector"]["initialization_seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    language_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    language_model.requires_grad_(False).eval()
    language_model.config.use_cache = False
    language_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if not language_model.is_gradient_checkpointing:
        raise ValueError("Qwen activation checkpointing is not enabled")
    qwen_parameters = sum(parameter.numel() for parameter in language_model.parameters())
    qwen_dtypes = sorted({str(parameter.dtype) for parameter in language_model.parameters()})
    if qwen_parameters != int(expected_model["parameter_count_bf16"]):
        raise ValueError("loaded Qwen parameter count differs from the contract")
    if qwen_dtypes != ["torch.float16"]:
        raise ValueError(f"loaded Qwen runtime dtype differs: {qwen_dtypes}")

    set_stage(stage, "projector_receiver_optimizer_load")
    projector = PatchMergerProjector.from_pretrained(
        args.projector_dir, device=device, dtype=torch.float32
    )
    if sum(parameter.numel() for parameter in projector.parameters()) != expected_projector_parameter_count:
        raise ValueError("projector parameter count differs from the contract")
    receiver = FixedPairwiseReceiverAdapter.from_pretrained(
        args.receiver_dir, device=device
    )
    if sum(parameter.numel() for parameter in receiver.parameters()) != 0:
        raise ValueError("proxy receiver unexpectedly has trainable parameters")
    reference_projector = None
    if geometry_setup is not None:
        reference_projector = PatchMergerProjector.from_pretrained(
            geometry_setup["reference_projector_dir"],
            device=device,
            dtype=torch.float32,
        ).requires_grad_(False).eval()
    health_reference_projector = None
    health_probe_rows: list[dict[str, Any]] = []
    if health_setup is not None:
        health_reference_projector = (
            reference_projector
            if reference_projector is not None
            else PatchMergerProjector.from_pretrained(
                args.projector_dir, device=device, dtype=torch.float32
            ).requires_grad_(False).eval()
        )
        health_supervisions = build_health_supervisions(
            tokenizer=tokenizer,
            prompt_contract=prompt_contract,
            screen_manifest=health_setup["screen_manifest"],
            teacher_ids=health_setup["teacher_ids"],
            placeholder_id=placeholder_id,
        )
        health_probe_rows = [
            health_supervisions[sample_id]
            for sample_id in health_setup["teacher_ids"]
        ]
    model = VisionCausalLM(
        language_model=language_model,
        projector=projector,
        receiver_adapter=receiver,
        placeholder_token_id=placeholder_id,
        backbone_kind="generic",
        freeze_language_model=True,
        pad_token_id=int(tokenizer.pad_token_id),
    ).to(device)
    model.train()
    budget = contract["training_budget"]
    learning_rate = (
        float(args.projector_learning_rate)
        if args.projector_learning_rate is not None
        else float(budget["learning_rate"])
    )
    if learning_rate <= 0.0:
        raise ValueError("projector learning rate must be positive")
    causal_shuffle_lambda = float(args.causal_shuffle_margin_lambda)
    causal_shuffle_margin = float(args.causal_shuffle_margin)
    if causal_shuffle_lambda < 0.0 or causal_shuffle_margin < 0.0:
        raise ValueError("causal shuffle lambda and margin must be non-negative")
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=learning_rate,
        betas=tuple(float(value) for value in budget["betas"]),
        eps=float(budget["epsilon"]),
        weight_decay=float(budget["weight_decay"]),
    )

    rng = random.Random(seed)
    if args.resume:
        set_stage(stage, "checkpoint_resume_verification")
        resume_manifest = verify_bound_checkpoint(
            args.resume, expected_binding=checkpoint_binding
        )
        if resume_manifest.get("health_checkpoint_role") == "failure":
            raise ValueError("resume from a critical-collapse checkpoint is forbidden")
        start_step, history, rng, restored_dir = load_training_checkpoint(
            source=args.resume,
            projector=projector,
            optimizer=optimizer,
            device=device,
        )
        if int(resume_manifest["step"]) != int(start_step):
            raise ValueError("resume manifest and training state steps differ")
        print(f"resumed from {restored_dir} at optimizer step {start_step}", flush=True)
    else:
        start_step = 0
        history = []
    resume_progress = validate_resume_history(
        start_step=start_step,
        history=history,
        total_examples=len(prepared),
        gradient_accumulation=binding_summary["gradient_accumulation"],
    )
    if start_step >= target_steps:
        raise ValueError("resume checkpoint is at or beyond the requested target step")
    examples_seen = int(resume_progress["examples_seen"])
    answer_tokens_seen = int(resume_progress["answer_tokens_seen"])
    expected_resumed_tokens = sum(
        int(row["answer_tokens"]) for row in prepared[:examples_seen]
    )
    if answer_tokens_seen != expected_resumed_tokens:
        raise ValueError("resume answer-token count differs from frozen supervision")

    health_probe_path = args.out / "probe_metrics.jsonl"
    health_log_path = args.out / "train_health.jsonl"
    health_state: dict[str, int] = {}
    health_previous_probe: dict[str, Any] | None = None
    health_last_healthy_checkpoint: Path | None = None
    health_stop_record: dict[str, Any] | None = None
    health_step0_projector_sequences: list[torch.Tensor] = []
    health_step0_receiver_sequences: list[torch.Tensor] = []
    if health_setup is not None:
        set_stage(stage, "health_probe_step0_reference")
        probe_cache = health_setup["probe_cache"]
        probe_ids = [str(row["sample_id"]) for row in health_setup["probe"]["samples"]]
        health_step0_projector_sequences, health_step0_receiver_sequences = (
            collect_health_representation(
                projector=health_reference_projector,
                receiver=receiver,
                feature_cache=probe_cache,
                sample_ids=probe_ids,
                device=device,
            )
        )
        # step0 只记录一次；恢复运行从已有 step 之后继续，避免伪造新的 step0。
        if start_step == 0:
            initial_probe = run_health_probe(
                step=0,
                projector=projector,
                receiver=receiver,
                language_model=language_model,
                feature_cache=probe_cache,
                sample_ids=probe_ids,
                step0_projector_sequences=health_step0_projector_sequences,
                step0_receiver_sequences=health_step0_receiver_sequences,
                teacher_rows=health_probe_rows,
                placeholder_id=placeholder_id,
                pad_token_id=int(tokenizer.pad_token_id),
                device=device,
                training_metrics=None,
            )
            guard = evaluate_guards(
                initial_probe,
                previous=None,
                state=health_state,
                contract=health_setup["contract"],
            )
            initial_probe["guards"] = guard
            append_jsonl(health_probe_path, initial_probe)
            health_previous_probe = initial_probe
            if guard["stop"] and args.health_auto_stop:
                health_stop_record = {
                    "step": 0,
                    "guard": guard,
                    "probe": initial_probe,
                }
            else:
                healthy_dir = (
                    args.out / "health_snapshots" / "checkpoints" / "healthy-step-000000"
                )
                healthy_binding = {**checkpoint_binding, "health_checkpoint_role": "healthy"}
                save_bound_checkpoint(
                    directory=healthy_dir,
                    projector=projector,
                    optimizer=optimizer,
                    step=0,
                    history=history,
                    rng=rng,
                    binding=healthy_binding,
                )
                health_last_healthy_checkpoint = healthy_dir
        else:
            health_previous_probe = None

    if health_setup is not None:
        append_jsonl(
            health_log_path,
            {
                "schema_version": "projector-health-step-v1",
                "event": "run_start",
                "optimizer_step": int(start_step),
                "examples_seen": int(examples_seen),
            },
        )

    history_path = args.out / "TRAINING_HISTORY.jsonl"
    with history_path.open("w", encoding="utf-8") as history_stream:
        for row in history:
            history_stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        history_stream.flush()

        set_stage(stage, "projector_training")
        cache = FeatureCache(args.feature_cache)
        training_started = time.perf_counter()
        first_gradient_report = None
        last_gradient_report = None
        for zero_based_step in range(start_step, target_steps):
            if health_stop_record is not None:
                break
            one_based_step = zero_based_step + 1
            indices = fixed_batch_record_indices(
                optimizer_step=zero_based_step,
                total_examples=len(prepared),
                gradient_accumulation=binding_summary["gradient_accumulation"],
            )
            torch.cuda.synchronize(device)
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            micro_losses = []
            causal_shuffle_losses = []
            batch_answer_tokens = 0
            batch_feature_groups = None
            auxiliary_result = None
            auxiliary_gradients: tuple[torch.Tensor, ...] | None = None
            auxiliary_gradient_norm = 0.0
            batch_projector_sequences: list[torch.Tensor] = []
            batch_receiver_sequences: list[torch.Tensor] = []
            if geometry_setup is not None or causal_shuffle_lambda > 0.0:
                batch_feature_groups = [
                    cache.get(
                        prepared[index]["id"], device=device, dtype=torch.float32
                    )
                    for index in indices
                ]
            if geometry_setup is not None:
                assert reference_projector is not None
                (
                    auxiliary_result,
                    auxiliary_gradients,
                    auxiliary_gradient_norm,
                ) = compute_geometry_auxiliary_gradients(
                    projector=projector,
                    reference_projector=reference_projector,
                    feature_batches=batch_feature_groups,
                    geometry_kwargs=geometry_setup["geometry_kwargs"],
                )
                geometry_lambda = float(geometry_setup["geometry_lambda"])
                if geometry_lambda > 0.0:
                    for parameter, gradient in zip(
                        projector.parameters(), auxiliary_gradients, strict=True
                    ):
                        parameter.grad = gradient.detach().mul(geometry_lambda)

            for micro_index, index in enumerate(indices):
                item = prepared[index]
                feature_groups = (
                    batch_feature_groups[micro_index]
                    if batch_feature_groups is not None
                    else cache.get(item["id"], device=device, dtype=torch.float32)
                )
                input_ids = torch.tensor(
                    [item["input_ids"]], dtype=torch.long, device=device
                )
                labels = torch.tensor([item["labels"]], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids)
                if health_setup is not None or causal_shuffle_lambda > 0.0:
                    projected_embeddings = projector(feature_groups)
                    if health_setup is not None:
                        batch_projector_sequences.extend(
                            value.detach().cpu() for value in projected_embeddings
                        )
                        batch_receiver_sequences.extend(
                            receiver(value).detach().cpu()
                            for value in projected_embeddings
                        )
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        image_embeddings=projected_embeddings,
                    )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        image_feature_groups=feature_groups,
                    )
                loss = outputs.loss
                if not bool(torch.isfinite(loss)):
                    raise ValueError(
                        f"non-finite Qwen3B training loss: {item['id']}"
                    )
                causal_penalty = torch.zeros((), device=device, dtype=loss.dtype)
                if causal_shuffle_lambda > 0.0:
                    assert batch_feature_groups is not None
                    shuffled_groups = batch_feature_groups[
                        (micro_index + 1) % len(batch_feature_groups)
                    ]
                    shuffled_embeddings = projector(shuffled_groups)
                    shuffled_outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        image_embeddings=shuffled_embeddings,
                    )
                    shuffled_loss = shuffled_outputs.loss
                    if not bool(torch.isfinite(shuffled_loss)):
                        raise ValueError(
                            f"non-finite shuffled Qwen3B loss: {item['id']}"
                        )
                    causal_penalty = torch.relu(
                        torch.as_tensor(causal_shuffle_margin, device=device, dtype=loss.dtype)
                        - (shuffled_loss - loss)
                    )
                    causal_shuffle_losses.append(float(causal_penalty.detach().item()))
                    del shuffled_outputs, shuffled_embeddings, shuffled_loss, shuffled_groups
                loss_for_backward = loss + causal_shuffle_lambda * causal_penalty
                (loss_for_backward / binding_summary["gradient_accumulation"]).backward()
                micro_losses.append(float(loss.detach().item()))
                batch_answer_tokens += int(item["answer_tokens"])
                del outputs, loss, loss_for_backward, input_ids, labels, attention_mask, feature_groups
                if health_setup is not None or causal_shuffle_lambda > 0.0:
                    del projected_embeddings

            geometry_history = None
            if geometry_setup is not None:
                assert auxiliary_result is not None and auxiliary_gradients is not None
                geometry_lambda = float(geometry_setup["geometry_lambda"])
                ce_squared = torch.zeros((), device=device, dtype=torch.float64)
                auxiliary_squared = torch.zeros((), device=device, dtype=torch.float64)
                dot = torch.zeros((), device=device, dtype=torch.float64)
                for parameter, auxiliary_gradient in zip(
                    projector.parameters(), auxiliary_gradients, strict=True
                ):
                    if parameter.grad is None:
                        raise ValueError("projector CE gradient is absent")
                    weighted = auxiliary_gradient.detach().mul(geometry_lambda)
                    ce_gradient = parameter.grad.detach() - weighted
                    ce_norm = torch.linalg.vector_norm(ce_gradient.to(torch.float32))
                    auxiliary_norm = torch.linalg.vector_norm(weighted.to(torch.float32))
                    ce_squared += ce_norm.to(torch.float64).square()
                    auxiliary_squared += auxiliary_norm.to(torch.float64).square()
                    dot += torch.sum(
                        ce_gradient.to(torch.float32) * weighted.to(torch.float32)
                    ).to(torch.float64)
                ce_gradient_norm = float(torch.sqrt(ce_squared))
                weighted_auxiliary_gradient_norm = float(torch.sqrt(auxiliary_squared))
                denominator = ce_gradient_norm * weighted_auxiliary_gradient_norm
                geometry_history = {
                    **geometry_payload(auxiliary_result),
                    "arm": geometry_setup["arm"],
                    "target_gradient_ratio": geometry_setup[
                        "target_gradient_ratio"
                    ],
                    "lambda": geometry_lambda,
                    "unweighted_auxiliary_gradient_norm": auxiliary_gradient_norm,
                    "weighted_auxiliary_gradient_norm": weighted_auxiliary_gradient_norm,
                    "ce_gradient_norm_before_clip": ce_gradient_norm,
                    "weighted_auxiliary_over_ce_gradient_norm": (
                        weighted_auxiliary_gradient_norm / ce_gradient_norm
                        if ce_gradient_norm > 0.0
                        else None
                    ),
                    "ce_auxiliary_gradient_cosine": (
                        float(dot) / denominator if denominator > 0.0 else None
                    ),
                }

            parameter_gradients = []
            for name, parameter in projector.named_parameters():
                if parameter.grad is None:
                    raise ValueError(f"projector gradient is absent: {name}")
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise ValueError(f"projector gradient is non-finite: {name}")
                nonzero = int(torch.count_nonzero(parameter.grad).item())
                if nonzero == 0:
                    raise ValueError(f"projector gradient is exactly zero: {name}")
                parameter_gradients.append(
                    {"name": name, "nonzero": nonzero, "numel": parameter.grad.numel()}
                )
            language_gradient_tensors = sum(
                parameter.grad is not None for parameter in language_model.parameters()
            )
            if language_gradient_tensors:
                raise ValueError("frozen Qwen accumulated parameter gradients")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                projector.parameters(), float(budget["gradient_clip"])
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise ValueError("projector gradient norm is non-finite")
            gradient_norm_after_clip = float(
                global_gradient_norm(
                    [parameter.grad for parameter in projector.parameters()]
                )
            )
            if not math.isfinite(gradient_norm_after_clip):
                raise ValueError("projector gradient norm after clip is non-finite")
            optimizer.step()
            if not all(
                bool(torch.isfinite(parameter).all())
                for parameter in projector.parameters()
            ):
                raise ValueError("projector parameter became non-finite")
            torch.cuda.synchronize(device)

            examples_seen += len(indices)
            answer_tokens_seen += batch_answer_tokens
            step_wall = time.perf_counter() - step_started
            gradient_report = {
                "step": one_based_step,
                "gradient_norm_before_clip": float(gradient_norm.detach().item()),
                "gradient_norm_after_clip": gradient_norm_after_clip,
                "parameter_gradients": parameter_gradients,
                "language_parameter_gradient_tensors": language_gradient_tensors,
            }
            if first_gradient_report is None:
                first_gradient_report = gradient_report
            last_gradient_report = gradient_report
            causal_shuffle_loss_value = (
                sum(causal_shuffle_losses) / len(causal_shuffle_losses)
                if causal_shuffle_losses
                else 0.0
            )
            history_row = {
                "step": one_based_step,
                "optimizer_steps": one_based_step,
                "examples_seen": examples_seen,
                "answer_tokens_seen": answer_tokens_seen,
                "effective_epochs": examples_seen
                / int(budget["effective_epochs_denominator"]),
                "subset_passes": examples_seen / len(prepared),
                "batch_start_index": indices[0],
                "batch_end_index": indices[-1],
                "batch_record_ids_sha256": canonical_sha256(
                    [prepared[index]["id"] for index in indices]
                ),
                "loss": sum(micro_losses) / len(micro_losses),
                "micro_loss_min": min(micro_losses),
                "micro_loss_max": max(micro_losses),
                "gradient_norm_before_clip": float(gradient_norm.detach().item()),
                "gradient_norm_after_clip": gradient_norm_after_clip,
                "step_wall_seconds": step_wall,
                "examples_per_second": len(indices) / step_wall,
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "causal_shuffle_hinge_loss": causal_shuffle_loss_value,
                "causal_shuffle_margin_lambda": causal_shuffle_lambda,
                "causal_shuffle_margin": causal_shuffle_margin,
            }
            if geometry_history is not None:
                history_row["geometry"] = geometry_history
            history.append(history_row)
            history_stream.write(
                json.dumps(history_row, ensure_ascii=False, sort_keys=True) + "\n"
            )
            history_stream.flush()
            health_row: dict[str, Any] | None = None
            if health_setup is not None:
                if len(batch_projector_sequences) < 2:
                    raise ValueError("health batch probe needs at least two images")
                batch_health = summarize_batch_embeddings(
                    batch_projector_sequences, batch_receiver_sequences
                )
                geometry_loss_value = (
                    float(auxiliary_result.total.detach().item())
                    if auxiliary_result is not None
                    else 0.0
                )
                geometry_lambda_value = (
                    float(geometry_setup["geometry_lambda"])
                    if geometry_setup is not None
                    else 0.0
                )
                ce_loss_value = float(history_row["loss"])
                causal_lambda_value = causal_shuffle_lambda
                health_row = {
                    "schema_version": "projector-health-step-v1",
                    "optimizer_step": one_based_step,
                    "step": one_based_step,
                    "examples_seen": examples_seen,
                    "answer_tokens_seen": answer_tokens_seen,
                    "projector_output_rms": batch_health["projector_output_rms"],
                    "receiver_output_rms": batch_health["receiver_output_rms"],
                    "between_image_rms": batch_health["between_image_rms"],
                    "within_image_token_rms": batch_health["within_image_token_rms"],
                    "relative_spread": batch_health["relative_spread"],
                    "projector_relative_spread": batch_health[
                        "projector_relative_spread"
                    ],
                    "receiver_relative_spread": batch_health[
                        "receiver_relative_spread"
                    ],
                    "mean_direction_fraction": batch_health[
                        "mean_direction_fraction"
                    ],
                    "projector_gradient_norm_before_clip": float(
                        gradient_norm.detach().item()
                    ),
                    "projector_gradient_norm_after_clip": gradient_norm_after_clip,
                    "ce_loss": ce_loss_value,
                    "geometry_loss": geometry_loss_value,
                    "causal_shuffle_hinge_loss": causal_shuffle_loss_value,
                    "causal_shuffle_margin_lambda": causal_lambda_value,
                    "causal_shuffle_margin": causal_shuffle_margin,
                    "total_loss": (
                        ce_loss_value
                        + geometry_lambda_value * geometry_loss_value
                        + causal_lambda_value * causal_shuffle_loss_value
                    ),
                    "has_nan_or_inf": False,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "optimizer_steps": one_based_step,
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                }
                append_jsonl(health_log_path, health_row)

                if probe_due(
                    one_based_step,
                    max_step=target_steps,
                    every_after=int(
                        health_setup["contract"]["probe_schedule"]["every_after_step"]
                    ),
                ):
                    set_stage(stage, f"health_probe_step_{one_based_step}")
                    probe = run_health_probe(
                        step=one_based_step,
                        projector=projector,
                        receiver=receiver,
                        language_model=language_model,
                        feature_cache=health_setup["probe_cache"],
                        sample_ids=probe_ids,
                        step0_projector_sequences=health_step0_projector_sequences,
                        step0_receiver_sequences=health_step0_receiver_sequences,
                        teacher_rows=health_probe_rows,
                        placeholder_id=placeholder_id,
                        pad_token_id=int(tokenizer.pad_token_id),
                        device=device,
                        training_metrics={
                            "gradient_norm_before_clip": health_row[
                                "projector_gradient_norm_before_clip"
                            ],
                            "gradient_norm_after_clip": health_row[
                                "projector_gradient_norm_after_clip"
                            ],
                            "ce_loss": health_row["ce_loss"],
                            "geometry_loss": health_row["geometry_loss"],
                            "total_loss": health_row["total_loss"],
                            "learning_rate": health_row["learning_rate"],
                            "examples_seen": examples_seen,
                        },
                    )
                    guard = evaluate_guards(
                        probe,
                        previous=health_previous_probe,
                        state=health_state,
                        contract=health_setup["contract"],
                    )
                    probe["guards"] = guard
                    append_jsonl(health_probe_path, probe)
                    if guard["stop"] and args.health_auto_stop:
                        health_stop_record = save_health_stop_and_rollback(
                            out=args.out,
                            step=one_based_step,
                            projector=projector,
                            optimizer=optimizer,
                            history=history,
                            rng=rng,
                            checkpoint_binding=checkpoint_binding,
                            batch_ids=[prepared[index]["id"] for index in indices],
                            health_row=health_row,
                            probe=probe,
                            guard=guard,
                            previous_probe=health_previous_probe,
                            last_healthy_checkpoint=health_last_healthy_checkpoint,
                            device=device,
                        )
                        print(
                            f"health auto-stop at optimizer_step {one_based_step}: "
                            f"{guard['critical']}",
                            flush=True,
                        )
                    else:
                        healthy_dir = (
                            args.out
                            / "health_snapshots"
                            / "checkpoints"
                            / f"healthy-step-{one_based_step:06d}"
                        )
                        healthy_binding = {
                            **checkpoint_binding,
                            "health_checkpoint_role": "healthy",
                        }
                        save_bound_checkpoint(
                            directory=healthy_dir,
                            projector=projector,
                            optimizer=optimizer,
                            step=one_based_step,
                            history=history,
                            rng=rng,
                            binding=healthy_binding,
                        )
                        health_last_healthy_checkpoint = healthy_dir
                    health_previous_probe = probe
                    projector.train()
                    model.train()
            if one_based_step == 1 or one_based_step % 10 == 0:
                print(
                    f"optimizer_step {one_based_step}/{target_steps} "
                    f"loss={history_row['loss']:.6f} examples_seen={examples_seen} "
                    f"answer_tokens_seen={answer_tokens_seen} "
                    f"step_wall={step_wall:.3f}s",
                    flush=True,
                )
            if health_stop_record is not None:
                break
            if (
                one_based_step % args.checkpoint_every == 0
                or one_based_step == target_steps
            ):
                checkpoint_dir = (
                    args.out / "checkpoints" / f"step-{one_based_step:06d}"
                )
                manifest = save_bound_checkpoint(
                    directory=checkpoint_dir,
                    projector=projector,
                    optimizer=optimizer,
                    step=one_based_step,
                    history=history,
                    rng=rng,
                    binding=checkpoint_binding,
                )
                print(
                    f"checkpoint saved: {checkpoint_dir} "
                    f"({manifest['total_bytes']} bytes)",
                    flush=True,
                )
        training_wall = time.perf_counter() - training_started

    expected_tokens_seen = sum(
        int(row["answer_tokens"]) for row in prepared[:examples_seen]
    )
    if answer_tokens_seen != expected_tokens_seen:
        raise ValueError("final answer-token count differs from frozen supervision")
    actual_steps = int(history[-1]["optimizer_steps"]) if history else int(start_step)
    if health_stop_record is not None:
        if health_stop_record.get("status") != "auto_stopped_by_projector_health_guard":
            health_stop_record = save_health_stop_and_rollback(
                out=args.out,
                step=int(health_stop_record["step"]),
                projector=projector,
                optimizer=optimizer,
                history=history,
                rng=rng,
                checkpoint_binding=checkpoint_binding,
                batch_ids=[],
                health_row=None,
                probe=health_stop_record["probe"],
                guard=health_stop_record["guard"],
                previous_probe=None,
                last_healthy_checkpoint=health_last_healthy_checkpoint,
                device=device,
            )
        health_artifacts = write_health_artifact_manifest(args.out)
        stopped_losses = [float(row["loss"]) for row in history]
        summary = {
            "status": "auto_stopped_by_projector_health_guard",
            "formal_training_complete": False,
            "capability_claim_allowed": False,
            "visual_ability_established": False,
            "previous_best": "step0",
            "final_half_scored": False,
            "paid_resources_used": False,
            "runner_git_sha": current_git_sha,
            "architecture_control": architecture_metadata,
            "projector_binding": projector_binding,
            "git_tracked_worktree_clean": tracked_clean,
            "target_optimizer_steps": target_steps,
            "optimizer_steps_completed": actual_steps,
            "examples_seen": examples_seen,
            "answer_tokens_seen": answer_tokens_seen,
            "loss_first": stopped_losses[0] if stopped_losses else None,
            "loss_last": stopped_losses[-1] if stopped_losses else None,
            "training_wall_seconds": training_wall,
            "total_wall_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "critical_reasons": health_stop_record["critical_reasons"],
            "collapse_onset_interval": health_stop_record[
                "collapse_onset_interval"
            ],
            "failure_checkpoint": health_stop_record["failure_checkpoint"],
            "last_healthy_checkpoint": health_stop_record[
                "last_healthy_checkpoint"
            ],
            "rollback": health_stop_record["rollback"],
            "health_artifact_file_count": health_artifacts["file_count"],
            "health_artifact_total_bytes": health_artifacts["total_bytes"],
            "transfer_label": "directly_transferable",
        }
        set_stage(stage, "health_auto_stopped")
        write_json(args.out / "SUMMARY.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    final_checkpoint = args.out / "checkpoints" / f"step-{target_steps:06d}"
    final_checkpoint_manifest = json.loads(
        (final_checkpoint / "CHECKPOINT_MANIFEST.json").read_text(encoding="utf-8")
    )
    set_stage(stage, "complete")
    losses = [float(row["loss"]) for row in history]
    geometry_rows = [row["geometry"] for row in history if "geometry" in row]
    summary = {
        "status": "valid" if formal_run else "development_only",
        "formal_training_complete": formal_run,
        "capability_claim_allowed": False,
        "evaluation_required_for_capability_claim": True,
        "visual_ability_established": False,
        "previous_best": "step0",
        "final_half_scored": False,
        "paid_resources_used": False,
        "runner_git_sha": current_git_sha,
        "architecture_control": architecture_metadata,
        "projector_binding": projector_binding,
        "git_tracked_worktree_clean": tracked_clean,
        "optimizer_steps": target_steps,
        "examples_seen": examples_seen,
        "answer_tokens_seen": answer_tokens_seen,
        "effective_epochs": examples_seen
        / int(budget["effective_epochs_denominator"]),
        "subset_passes": examples_seen / len(prepared),
        "micro_batch_size": binding_summary["micro_batch_size"],
        "gradient_accumulation": binding_summary["gradient_accumulation"],
        "real_global_batch": binding_summary["real_global_batch"],
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "loss_mean": sum(losses) / len(losses),
        "training_wall_seconds": training_wall,
        "total_wall_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "gpu_name": torch.cuda.get_device_name(device),
        "qwen_parameter_count": qwen_parameters,
        "qwen_runtime_dtypes": qwen_dtypes,
        "qwen_trainable_parameter_count": sum(
            parameter.numel()
            for parameter in language_model.parameters()
            if parameter.requires_grad
        ),
        "projector_parameter_count": sum(
            parameter.numel() for parameter in projector.parameters()
        ),
        "receiver_trainable_parameter_count": sum(
            parameter.numel() for parameter in receiver.parameters()
        ),
        "activation_checkpointing": language_model.is_gradient_checkpointing,
        "first_gradient": first_gradient_report,
        "last_gradient": last_gradient_report,
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_file_count": final_checkpoint_manifest["file_count"],
        "final_checkpoint_total_bytes": final_checkpoint_manifest["total_bytes"],
        "supervision_records_sha256": supervision_summary["records_file_sha256"],
        "transfer_label": "transferable_with_runtime_validation",
    }
    if health_setup is not None:
        health_artifacts = write_health_artifact_manifest(args.out)
        summary.update(
            {
                "health_contract_file_sha256": health_setup[
                    "contract_file_sha256"
                ],
                "health_probe_manifest_file_sha256": health_setup[
                    "probe_file_sha256"
                ],
                "health_probe_count": len(health_setup["probe"]["samples"]),
                "health_probe_schedule": health_setup["contract"][
                    "probe_schedule"
                ],
                "health_artifact_file_count": health_artifacts["file_count"],
                "health_artifact_total_bytes": health_artifacts["total_bytes"],
                "health_auto_stop": bool(args.health_auto_stop),
                "health_completed_without_critical": True,
            }
        )
    if geometry_setup is not None:
        if len(geometry_rows) != target_steps:
            raise ValueError("geometry history is missing optimizer steps")
        summary["geometry_arm"] = geometry_setup["arm"]
        summary["geometry_target_gradient_ratio"] = geometry_setup[
            "target_gradient_ratio"
        ]
        summary["geometry_lambda"] = geometry_setup["geometry_lambda"]
        summary["geometry_first"] = geometry_rows[0]
        summary["geometry_last"] = geometry_rows[-1]
        summary["geometry_auxiliary_total_mean"] = sum(
            float(row["total"]) for row in geometry_rows
        ) / len(geometry_rows)
        summary["geometry_weighted_auxiliary_over_ce_mean"] = sum(
            float(row["weighted_auxiliary_over_ce_gradient_norm"]) for row in geometry_rows
        ) / len(geometry_rows)
    write_json(args.out / "SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite training run: {args.out}")
    args.out.mkdir(parents=True)
    log_handle = (args.out / "run.log").open("w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _Tee(original_stdout, log_handle)
    sys.stderr = _Tee(original_stderr, log_handle)
    stage = {"name": "initialization"}
    write_json(
        args.out / "ATTEMPT.json",
        {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv,
            "git_sha": git_sha(),
            "git_tracked_worktree_clean": git_tracked_worktree_clean(),
            "formal_result_allowed": not args.allow_dirty_development_run
            and args.development_max_optimizer_steps is None,
            "capability_claim_allowed": False,
            "final_half_scored": False,
            "paid_resources_used": False,
        },
    )
    try:
        _run(args, stage)
    except BaseException as exc:
        failure = {
            "status": "failed",
            "stage": stage["name"],
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "capability_claim_allowed": False,
            "final_half_scored": False,
            "paid_resources_used": False,
        }
        write_json(args.out / "FAILURE.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    main()
