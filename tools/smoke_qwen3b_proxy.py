"""在 V100 上验证 Qwen2.5-3B 真实图像 glue、梯度与 checkpoint 闭环。"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from moonvit_glue.chat_contract import build_chat_prompt, build_chat_supervision
from moonvit_glue.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from moonvit_glue.grounding_contract import format_click_action, parse_click_action
from moonvit_glue.model import VisionCausalLM
from moonvit_glue.moonvit_v2 import load_moonvit_v2_encoder
from moonvit_glue.projector import PatchMergerProjector
from moonvit_glue.proxy_receiver import FixedPairwiseReceiverAdapter
from moonvit_glue.screenspot_contract import verify_manifest


class _Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen_files(
    root: Path, entries: list[dict[str, Any]], *, label: str
) -> list[dict[str, Any]]:
    """逐文件验证冻结合同中的相对路径、字节数与 SHA-256。"""

    verified = []
    for entry in entries:
        relative = str(entry["path"])
        path = root / relative
        expected_bytes = int(entry["bytes"])
        expected_sha256 = str(entry["sha256"])
        if not path.is_file():
            raise ValueError(f"{label} missing frozen file: {relative}")
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
            raise ValueError(
                f"{label} mismatch for {relative}: "
                f"bytes={actual_bytes}/{expected_bytes}, "
                f"sha256={actual_sha256}/{expected_sha256}"
            )
        verified.append(
            {"path": relative, "bytes": actual_bytes, "sha256": actual_sha256}
        )
    return verified


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def _read_screenspot_image(
    source_dir: Path, sample: dict[str, Any]
) -> tuple[bytes, Image.Image]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for the ScreenSpot smoke") from error

    parquet_path = source_dir / sample["source_parquet"]
    target_index = int(sample["source_row_index"])
    parquet = pq.ParquetFile(parquet_path)
    offset = 0
    encoded: bytes | None = None
    for batch in parquet.iter_batches(batch_size=32, columns=["image"]):
        if offset <= target_index < offset + batch.num_rows:
            row = batch.slice(target_index - offset, 1).to_pylist()[0]
            encoded = row["image"]["bytes"]
            break
        offset += batch.num_rows
    if not isinstance(encoded, bytes):
        raise ValueError(f"unable to load ScreenSpot row {target_index} from {parquet_path}")
    if hashlib.sha256(encoded).hexdigest() != sample["image_sha256"]:
        raise ValueError("ScreenSpot smoke image SHA-256 differs from the frozen manifest")
    image = Image.open(io.BytesIO(encoded)).convert("RGB")
    if list(image.size) != [sample["image_width"], sample["image_height"]]:
        raise ValueError("ScreenSpot smoke image dimensions differ from the manifest")
    return encoded, image


def _decode_continuation(
    tokenizer: Any, generated: torch.Tensor, prefix_length: int
) -> tuple[list[int], str]:
    if generated.ndim != 2 or generated.shape[0] != 1:
        raise ValueError(f"unexpected generated shape: {tuple(generated.shape)}")
    if generated.shape[1] < prefix_length:
        raise ValueError("generation is shorter than its expanded prompt")
    token_ids = [int(value) for value in generated[0, prefix_length:].tolist()]
    return token_ids, tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def _tensor_finite_and_nonzero(tensor: torch.Tensor) -> tuple[bool, int, int]:
    detached = tensor.detach()
    return (
        bool(torch.isfinite(detached).all()),
        int(torch.count_nonzero(detached).item()),
        detached.numel(),
    )


def _optimizer_states_equal(
    first: torch.optim.Optimizer, second: torch.optim.Optimizer
) -> bool:
    left = first.state_dict()
    right = second.state_dict()
    if left["param_groups"] != right["param_groups"] or left["state"].keys() != right[
        "state"
    ].keys():
        return False
    for key in left["state"]:
        if left["state"][key].keys() != right["state"][key].keys():
            return False
        for name, value in left["state"][key].items():
            other = right["state"][key][name]
            if torch.is_tensor(value):
                if value.shape != other.shape or value.dtype != other.dtype:
                    return False
                comparable = other.to(device=value.device)
                if not torch.equal(value, comparable):
                    return False
            elif value != other:
                return False
    return True


def _checkpoint_files(directory: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def _artifact_manifest(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    payload = {
        "schema_version": "qwen3b-proxy-smoke-artifacts-v1",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "final_half_scored": False,
    }
    _write_json(root / "ARTIFACT_MANIFEST.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--projector-dir", type=Path, required=True)
    parser.add_argument("--receiver-dir", type=Path, required=True)
    parser.add_argument("--moonvit-v2-weights", type=Path, required=True)
    parser.add_argument("--screenspot-manifest", type=Path, required=True)
    parser.add_argument("--screenspot-source-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-image-side", type=int, default=448)
    return parser.parse_args()


def _run(args: argparse.Namespace, stage: dict[str, str]) -> dict[str, Any]:
    started = time.perf_counter()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    stage["name"] = "frozen_input_verification"
    model_file_verification = _verify_frozen_files(
        args.model_dir, contract["proxy_model"]["files"], label="Qwen contract"
    )
    moonvit_weights_sha256 = sha256_file(args.moonvit_v2_weights)
    if moonvit_weights_sha256 != contract["vision_tower"][
        "extracted_weights_sha256"
    ]:
        raise ValueError("MoonViT-V2 weights SHA-256 differs from the frozen contract")
    expected_max_side = int(contract["image_preprocessing"]["train_max_image_side"])
    if args.max_image_side != expected_max_side:
        raise ValueError(
            f"smoke max image side must match the frozen contract: {expected_max_side}"
        )
    manifest = json.loads(args.screenspot_manifest.read_text(encoding="utf-8"))
    if not verify_manifest(manifest):
        raise ValueError("ScreenSpot manifest self-hash verification failed")
    expected_dataset = contract["datasets"]["screenspot_glm50"]
    if (
        manifest["name"] != expected_dataset["name"]
        or manifest["manifest_sha256"] != expected_dataset["manifest_sha256"]
    ):
        raise ValueError("ScreenSpot smoke manifest differs from the frozen contract")
    sample = manifest["samples"][0]
    encoded_image, image = _read_screenspot_image(args.screenspot_source_dir, sample)
    (args.out / "input_image.bin").write_bytes(encoded_image)

    expected_projector_sha = contract["canonical_projector"][
        "initialization_contract"
    ]["step0"]["weights_sha256"]
    projector_weights = args.projector_dir / "projector.safetensors"
    if sha256_file(projector_weights) != expected_projector_sha:
        raise ValueError("step0 projector SHA-256 differs from the frozen contract")
    expected_receiver_sha = contract["qwen_proxy_receiver"]["buffer_sha256"]
    receiver_weights = args.receiver_dir / "proxy_receiver.safetensors"
    if sha256_file(receiver_weights) != expected_receiver_sha:
        raise ValueError("proxy receiver SHA-256 differs from the frozen contract")

    stage["name"] = "runtime_initialization"
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Qwen3B smoke requires the existing CUDA V100")
    torch.manual_seed(20260805)
    torch.cuda.manual_seed_all(20260805)
    torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32 = False

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import transformers

    stage["name"] = "tokenizer_and_config_load"
    model_config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False
    )
    if model_config.architectures != ["Qwen2ForCausalLM"] or hasattr(
        model_config, "vision_config"
    ):
        raise ValueError("smoke backbone must remain the pinned pure-text Qwen2 model")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False
    )
    placeholder_id = int(contract["prompt_and_generation"]["image_placeholder_token_id"])
    if tokenizer.convert_tokens_to_ids("<|image_pad|>") != placeholder_id:
        raise ValueError("Qwen image placeholder ID differs from the contract")
    if tokenizer.eos_token_id != contract["prompt_and_generation"]["eos_token_id"]:
        raise ValueError("Qwen EOS ID differs from the contract")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    user_prompt = contract["prompt_and_generation"]["user_prompt"].format(
        instruction=sample["instruction"]
    )
    system_prompt = contract["prompt_and_generation"]["system_prompt"]
    visual_prompt = build_chat_prompt(
        tokenizer,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        placeholder_token_id=placeholder_id,
        include_image=True,
    )
    blind_prompt = build_chat_prompt(
        tokenizer,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        placeholder_token_id=placeholder_id,
        include_image=False,
    )
    x1, y1, x2, y2 = (float(value) for value in sample["bbox_999_xyxy"])
    target_answer = format_click_action(
        (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0)))
    )
    supervision = build_chat_supervision(
        tokenizer,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        answer=target_answer,
        placeholder_token_id=placeholder_id,
        include_image=True,
    )
    _write_json(
        args.out / "PROMPTS.json",
        {
            "sample_id": sample["sample_id"],
            "instruction": sample["instruction"],
            "target_answer": target_answer,
            "vision": {
                "template_text": visual_prompt.template_text_for_audit,
                "input_ids": visual_prompt.input_ids,
            },
            "blind": {
                "template_text": blind_prompt.template_text_for_audit,
                "input_ids": blind_prompt.input_ids,
            },
            "supervision": {
                "input_ids": supervision.input_ids,
                "labels": supervision.labels,
                "prompt_length": supervision.prompt_length,
                "answer_tokens": supervision.answer_tokens,
            },
        },
    )

    stage["name"] = "qwen_fp16_load"
    load_started = time.perf_counter()
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
    qwen_load_seconds = time.perf_counter() - load_started
    qwen_parameter_count = sum(parameter.numel() for parameter in language_model.parameters())
    qwen_dtypes = sorted({str(parameter.dtype) for parameter in language_model.parameters()})
    if qwen_parameter_count != int(contract["proxy_model"]["parameter_count_bf16"]):
        raise ValueError("loaded Qwen parameter count differs from the frozen contract")
    if qwen_dtypes != ["torch.float16"]:
        raise ValueError(f"loaded Qwen runtime dtype is not exact FP16: {qwen_dtypes}")

    stage["name"] = "projector_and_receiver_load"
    projector = PatchMergerProjector.from_pretrained(
        args.projector_dir, device=device, dtype=torch.float32
    )
    receiver = FixedPairwiseReceiverAdapter.from_pretrained(
        args.receiver_dir, device=device
    )
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

    stage["name"] = "moonvit_real_image_forward"
    moonvit_started = time.perf_counter()
    moonvit = load_moonvit_v2_encoder(
        args.moonvit_v2_weights,
        attn_implementation="sdpa",
        torch_dtype=torch.float32,
        device=device,
        freeze=True,
    )
    resized = image.copy()
    resized.thumbnail((args.max_image_side, args.max_image_side), Image.Resampling.LANCZOS)
    image_inputs = moonvit.preprocess(resized, device=device)
    feature_group = moonvit(**image_inputs)[0].detach().to(dtype=torch.float32)
    moonvit_seconds = time.perf_counter() - moonvit_started
    if feature_group.shape[0] > contract["image_preprocessing"]["train_max_visual_tokens"]:
        raise ValueError("MoonViT smoke exceeds the frozen training visual-token limit")
    feature_finite, feature_nonzero, feature_numel = _tensor_finite_and_nonzero(
        feature_group
    )
    if not feature_finite or feature_nonzero == 0:
        raise ValueError("MoonViT smoke features are non-finite or exactly zero")
    del moonvit, image_inputs
    torch.cuda.empty_cache()

    generation_kwargs = {
        "do_sample": False,
        "max_new_tokens": int(contract["prompt_and_generation"]["max_new_tokens"]),
        "eos_token_id": int(tokenizer.eos_token_id),
        "pad_token_id": int(tokenizer.pad_token_id),
    }
    stage["name"] = "vision_generation"
    model.eval()
    visual_ids = torch.tensor([visual_prompt.input_ids], dtype=torch.long, device=device)
    visual_mask = torch.ones_like(visual_ids)
    generation_started = time.perf_counter()
    generated_visual = model.generate(
        input_ids=visual_ids,
        attention_mask=visual_mask,
        image_feature_groups=[feature_group],
        **generation_kwargs,
    )
    torch.cuda.synchronize(device)
    vision_generation_seconds = time.perf_counter() - generation_started
    expanded_visual_prefix = len(visual_prompt.input_ids) - 1 + feature_group.shape[0]
    visual_tokens, visual_text = _decode_continuation(
        tokenizer, generated_visual, expanded_visual_prefix
    )

    stage["name"] = "blind_generation"
    blind_ids = torch.tensor([blind_prompt.input_ids], dtype=torch.long, device=device)
    blind_mask = torch.ones_like(blind_ids)
    generation_started = time.perf_counter()
    generated_blind = language_model.generate(
        input_ids=blind_ids,
        attention_mask=blind_mask,
        **generation_kwargs,
    )
    torch.cuda.synchronize(device)
    blind_generation_seconds = time.perf_counter() - generation_started
    blind_tokens, blind_text = _decode_continuation(
        tokenizer, generated_blind, len(blind_prompt.input_ids)
    )
    generation_rows = [
        {
            "condition": "vision_step0",
            "sample_id": sample["sample_id"],
            "prediction": visual_text,
            "continuation_token_ids": visual_tokens,
            "parse_result": parse_click_action(visual_text),
            "capability_claim_allowed": False,
        },
        {
            "condition": "blind_step0",
            "sample_id": sample["sample_id"],
            "prediction": blind_text,
            "continuation_token_ids": blind_tokens,
            "parse_result": parse_click_action(blind_text),
            "capability_claim_allowed": False,
        },
    ]
    with (args.out / "GENERATIONS.jsonl").open("w", encoding="utf-8") as handle:
        for row in generation_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    stage["name"] = "real_image_backward"
    model.train()
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=float(contract["training_budget"]["learning_rate"]),
        betas=tuple(float(value) for value in contract["training_budget"]["betas"]),
        eps=float(contract["training_budget"]["epsilon"]),
        weight_decay=float(contract["training_budget"]["weight_decay"]),
    )
    input_ids = torch.tensor([supervision.input_ids], dtype=torch.long, device=device)
    labels = torch.tensor([supervision.labels], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    optimizer.zero_grad(set_to_none=True)
    canonical_embeddings = projector([feature_group])
    canonical_embeddings[0].retain_grad()
    backward_started = time.perf_counter()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        image_embeddings=canonical_embeddings,
    )
    loss = outputs.loss
    if not bool(torch.isfinite(loss)):
        raise ValueError("Qwen3B real-image smoke loss is non-finite")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        projector.parameters(), float(contract["training_budget"]["gradient_clip"])
    )
    if not bool(torch.isfinite(grad_norm)):
        raise ValueError("Qwen3B projector gradient norm is non-finite")
    torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - backward_started

    parameter_gradients = []
    for name, parameter in projector.named_parameters():
        if parameter.grad is None:
            parameter_gradients.append(
                {"name": name, "present": False, "finite": False, "nonzero": 0}
            )
            continue
        finite, nonzero, numel = _tensor_finite_and_nonzero(parameter.grad)
        parameter_gradients.append(
            {
                "name": name,
                "present": True,
                "finite": finite,
                "nonzero": nonzero,
                "numel": numel,
            }
        )
    if not all(item["present"] and item["finite"] for item in parameter_gradients):
        raise ValueError("one or more projector parameter gradients are missing/non-finite")
    if not all(item["nonzero"] > 0 for item in parameter_gradients):
        raise ValueError("one or more projector parameter gradients are exactly zero")
    canonical_finite, canonical_nonzero, canonical_numel = _tensor_finite_and_nonzero(
        canonical_embeddings[0].grad
    )
    if not canonical_finite or canonical_nonzero == 0:
        raise ValueError("canonical visual embedding gradient is non-finite or zero")
    language_gradient_tensors = sum(
        parameter.grad is not None for parameter in language_model.parameters()
    )
    if language_gradient_tensors:
        raise ValueError("frozen Qwen unexpectedly accumulated parameter gradients")
    optimizer.step()
    if not all(
        bool(torch.isfinite(parameter).all()) for parameter in projector.parameters()
    ):
        raise ValueError("projector parameters became non-finite after the smoke step")

    gradient_report = {
        "loss": float(loss.detach()),
        "gradient_norm_before_clip": float(grad_norm),
        "projector_parameter_gradients": parameter_gradients,
        "canonical_embedding_gradient": {
            "shape": list(canonical_embeddings[0].grad.shape),
            "finite": canonical_finite,
            "nonzero": canonical_nonzero,
            "numel": canonical_numel,
        },
        "language_parameter_gradient_tensors": language_gradient_tensors,
        "receiver_trainable_parameter_count": sum(
            parameter.numel() for parameter in receiver.parameters()
        ),
        "backward_seconds": backward_seconds,
    }
    _write_json(args.out / "GRADIENT.json", gradient_report)

    stage["name"] = "checkpoint_save_restore"
    checkpoint_dir = args.out / "large_checkpoint" / "step-000001"
    history = [
        {
            "step": 1,
            "examples_seen": 1,
            "answer_tokens_seen": supervision.answer_tokens,
            "loss": float(loss.detach()),
        }
    ]
    python_rng = random.Random(20260805)
    checkpoint_started = time.perf_counter()
    save_training_checkpoint(
        directory=checkpoint_dir,
        projector=projector,
        optimizer=optimizer,
        step=1,
        history=history,
        rng=python_rng,
    )
    restored_projector = PatchMergerProjector(projector.config).to(
        device=device, dtype=torch.float32
    )
    restored_optimizer = torch.optim.AdamW(
        restored_projector.parameters(),
        lr=float(contract["training_budget"]["learning_rate"]),
        betas=tuple(float(value) for value in contract["training_budget"]["betas"]),
        eps=float(contract["training_budget"]["epsilon"]),
        weight_decay=float(contract["training_budget"]["weight_decay"]),
    )
    restored_step, restored_history, restored_rng, restored_dir = load_training_checkpoint(
        source=checkpoint_dir,
        projector=restored_projector,
        optimizer=restored_optimizer,
        device=device,
    )
    projector_exact = all(
        torch.equal(value, restored_projector.state_dict()[name])
        for name, value in projector.state_dict().items()
    )
    optimizer_exact = _optimizer_states_equal(optimizer, restored_optimizer)
    rng_exact = restored_rng.getstate() == python_rng.getstate()
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    checkpoint_files = _checkpoint_files(checkpoint_dir)
    checkpoint_report = {
        "directory": str(checkpoint_dir),
        "restored_directory": str(restored_dir),
        "restored_step": restored_step,
        "history_exact": restored_history == history,
        "projector_exact": projector_exact,
        "optimizer_exact": optimizer_exact,
        "python_rng_exact": rng_exact,
        "checkpoint_seconds": checkpoint_seconds,
        "files": checkpoint_files,
        "file_count": len(checkpoint_files),
        "total_bytes": sum(item["bytes"] for item in checkpoint_files),
    }
    if not all(
        (
            restored_step == 1,
            checkpoint_report["history_exact"],
            projector_exact,
            optimizer_exact,
            rng_exact,
        )
    ):
        raise ValueError("Qwen3B smoke checkpoint round-trip is not exact")
    _write_json(args.out / "CHECKPOINT.json", checkpoint_report)

    torch.cuda.synchronize(device)
    summary = {
        "schema_version": "qwen3b-proxy-smoke-v1",
        "status": "valid",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "contract_path": str(args.contract),
        "contract_sha256": sha256_file(args.contract),
        "model": {
            "path": str(args.model_dir),
            "repo": contract["proxy_model"]["repo"],
            "resolved_revision": contract["proxy_model"]["resolved_revision"],
            "architecture": model_config.architectures[0],
            "parameter_count": qwen_parameter_count,
            "dtypes": qwen_dtypes,
            "frozen": True,
            "load_seconds": qwen_load_seconds,
            "verified_file_count": len(model_file_verification),
            "verified_files": model_file_verification,
        },
        "input": {
            "dataset": manifest["name"],
            "manifest_sha256": manifest["manifest_sha256"],
            "sample_id": sample["sample_id"],
            "image_sha256": sample["image_sha256"],
            "original_size": [sample["image_width"], sample["image_height"]],
            "resized_size": list(resized.size),
            "max_image_side": args.max_image_side,
        },
        "moonvit": {
            "weights": str(args.moonvit_v2_weights),
            "weights_sha256": moonvit_weights_sha256,
            "feature_shape": list(feature_group.shape),
            "feature_finite": feature_finite,
            "feature_nonzero": feature_nonzero,
            "feature_numel": feature_numel,
            "forward_seconds_including_load": moonvit_seconds,
        },
        "generation": {
            "registered_temperature": contract["prompt_and_generation"]["temperature"],
            "runtime_temperature_argument": "omitted because do_sample=false makes it inert",
            "do_sample": False,
            "max_new_tokens": generation_kwargs["max_new_tokens"],
            "vision_prediction": visual_text,
            "vision_parse": parse_click_action(visual_text),
            "blind_prediction": blind_text,
            "blind_parse": parse_click_action(blind_text),
            "vision_seconds": vision_generation_seconds,
            "blind_seconds": blind_generation_seconds,
        },
        "gradient": gradient_report,
        "checkpoint": checkpoint_report,
        "runtime": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "wall_seconds": time.perf_counter() - started,
            "pid": os.getpid(),
        },
        "claims": {
            "real_qwen3b_load": True,
            "real_moonvit_image_forward": True,
            "real_image_gradient_reaches_projector": True,
            "qwen_parameter_gradients_absent": True,
            "checkpoint_round_trip_exact": True,
            "visual_ability_established": False,
            "reason": "step0 generation is an engineering smoke, not a trained or paired benchmark",
            "deepseek_transfer": "transferable_with_runtime_validation",
        },
        "paid_resources_used": False,
        "final_half_scored": False,
    }
    _write_json(args.out / "SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite smoke output: {args.out}")
    args.out.mkdir(parents=True)
    _write_json(
        args.out / "RUN_CONFIG.json",
        {
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_sha(),
        },
    )
    log = (args.out / "run.log").open("w", encoding="utf-8", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(original_stdout, log)
    sys.stderr = _Tee(original_stderr, log)
    stage = {"name": "startup"}
    failure: BaseException | None = None
    try:
        _run(args, stage)
    except BaseException as error:
        failure = error
        traceback.print_exc()
        _write_json(
            args.out / "FAILURE.json",
            {
                "schema_version": "qwen3b-proxy-smoke-failure-v1",
                "status": "invalid",
                "failed_at_stage": stage["name"],
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated())
                    if torch.cuda.is_available()
                    else 0
                ),
                "paid_resources_used": False,
                "capability_claim_allowed": False,
            },
        )
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log.close()
        artifact_manifest = _artifact_manifest(args.out)
        print(json.dumps(artifact_manifest, indent=2, sort_keys=True))
    if failure is not None:
        raise RuntimeError(
            f"Qwen3B smoke failed at {stage['name']}; see {args.out / 'FAILURE.json'}"
        ) from failure


if __name__ == "__main__":
    main()
