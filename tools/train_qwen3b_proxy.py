#!/usr/bin/env python3
"""按冻结 4k 合同训练 Qwen2.5-3B 的 DeepSeek-shaped projector。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
import moonvit_glue.merge as merge_module
import moonvit_glue.model as model_module
import moonvit_glue.projector as projector_module
import moonvit_glue.proxy_receiver as proxy_receiver_module
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
from moonvit_glue.model import VisionCausalLM
from moonvit_glue.projector import PatchMergerProjector
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


def runtime_source_files() -> list[dict[str, Any]]:
    paths = (
        Path(__file__).resolve(),
        Path(chat_contract_module.__file__).resolve(),
        Path(checkpointing_module.__file__).resolve(),
        Path(feature_cache_module.__file__).resolve(),
        Path(fixed_budget_module.__file__).resolve(),
        Path(merge_module.__file__).resolve(),
        Path(model_module.__file__).resolve(),
        Path(projector_module.__file__).resolve(),
        Path(proxy_receiver_module.__file__).resolve(),
        Path(training_order_module.__file__).resolve(),
        Path(__file__).with_name("verify_feature_cache.py").resolve(),
    )
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
        "progress": {
            key: history[-1][key]
            for key in (
                "optimizer_steps",
                "examples_seen",
                "answer_tokens_seen",
                "effective_epochs",
                "subset_passes",
            )
        },
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
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--training-order-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--expected-cache-runner-git-sha", required=True)
    parser.add_argument("--projector-dir", type=Path, required=True)
    parser.add_argument("--receiver-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--development-max-optimizer-steps", type=int)
    parser.add_argument("--allow-dirty-development-run", action="store_true")
    return parser.parse_args()


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


def _run(args: argparse.Namespace, stage: dict[str, str]) -> dict[str, Any]:
    started = time.perf_counter()
    tracked_clean = git_tracked_worktree_clean()
    current_git_sha = git_sha()
    if not tracked_clean and not args.allow_dirty_development_run:
        raise RuntimeError("tracked Git worktree is dirty; formal training is refused")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")

    set_stage(stage, "contract_order_cache_verification")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
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
    projector_path = args.projector_dir / "projector.safetensors"
    receiver_path = args.receiver_dir / "proxy_receiver.safetensors"
    expected_projector_sha = contract["canonical_projector"][
        "initialization_contract"
    ]["step0"]["weights_sha256"]
    if sha256_file(projector_path) != expected_projector_sha:
        raise ValueError("step0 projector SHA-256 differs from the frozen contract")
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
    formal_run = (
        tracked_clean
        and not args.allow_dirty_development_run
        and target_steps == total_steps
    )
    source_files = runtime_source_files()
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
        "proxy_receiver_sha256": expected_receiver_sha,
    }
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
        "model_dir": str(args.model_dir.resolve()),
        "model_files": model_files,
        "data": str(args.data.resolve()),
        "training_order_manifest": str(args.training_order_manifest.resolve()),
        "training_order_manifest_file_sha256": order_file_sha,
        "feature_cache": str(args.feature_cache.resolve()),
        "feature_cache_manifest_file_sha256": cache_manifest_file_sha,
        "feature_cache_verification": cache_verification,
        "projector_dir": str(args.projector_dir.resolve()),
        "receiver_dir": str(args.receiver_dir.resolve()),
        "binding": binding_summary,
        "target_optimizer_steps": target_steps,
        "formal_optimizer_steps": total_steps,
        "checkpoint_every": args.checkpoint_every,
        "resume": str(args.resume.resolve()) if args.resume else None,
        "supervision": supervision_summary,
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
    if sum(parameter.numel() for parameter in projector.parameters()) != int(
        contract["canonical_projector"]["parameter_count"]
    ):
        raise ValueError("projector parameter count differs from the contract")
    receiver = FixedPairwiseReceiverAdapter.from_pretrained(
        args.receiver_dir, device=device
    )
    if sum(parameter.numel() for parameter in receiver.parameters()) != 0:
        raise ValueError("proxy receiver unexpectedly has trainable parameters")
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
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=float(budget["learning_rate"]),
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
            batch_answer_tokens = 0
            for index in indices:
                item = prepared[index]
                feature_groups = cache.get(
                    item["id"], device=device, dtype=torch.float32
                )
                input_ids = torch.tensor(
                    [item["input_ids"]], dtype=torch.long, device=device
                )
                labels = torch.tensor([item["labels"]], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids)
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
                (loss / binding_summary["gradient_accumulation"]).backward()
                micro_losses.append(float(loss.detach().item()))
                batch_answer_tokens += int(item["answer_tokens"])
                del outputs, loss, input_ids, labels, attention_mask, feature_groups

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
                "parameter_gradients": parameter_gradients,
                "language_parameter_gradient_tensors": language_gradient_tensors,
            }
            if first_gradient_report is None:
                first_gradient_report = gradient_report
            last_gradient_report = gradient_report
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
                "step_wall_seconds": step_wall,
                "examples_per_second": len(indices) / step_wall,
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(history_row)
            history_stream.write(
                json.dumps(history_row, ensure_ascii=False, sort_keys=True) + "\n"
            )
            history_stream.flush()
            if one_based_step == 1 or one_based_step % 10 == 0:
                print(
                    f"optimizer_step {one_based_step}/{target_steps} "
                    f"loss={history_row['loss']:.6f} examples_seen={examples_seen} "
                    f"answer_tokens_seen={answer_tokens_seen} "
                    f"step_wall={step_wall:.3f}s",
                    flush=True,
                )
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
    final_checkpoint = args.out / "checkpoints" / f"step-{target_steps:06d}"
    final_checkpoint_manifest = json.loads(
        (final_checkpoint / "CHECKPOINT_MANIFEST.json").read_text(encoding="utf-8")
    )
    set_stage(stage, "complete")
    losses = [float(row["loss"]) for row in history]
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
