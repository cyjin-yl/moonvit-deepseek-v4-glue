#!/usr/bin/env python3
"""在冻结 shape 协议上训练顶部 LoRA 或等量 projector continuation。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from extract_layerwise_representations import visual_prompt_batch
from moonvit_glue import FeatureCache, PatchMergerProjector, ProjectorConfig, VisionCausalLM
from moonvit_glue.lora import freeze_non_lora, inject_lora, lora_state_dict
from tools_common import load_records, validate_text_only_backbone_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def resolve_projector_config_source(config: dict) -> Path:
    """允许权重断点与不可变 projector 结构定义分开存放。"""
    return Path(config.get("projector_config_source", config["base_projector"]))


def maybe_resume_projector_optimizer(
    *,
    optimizer,
    arm: dict,
    base_projector: Path,
    device: torch.device,
) -> dict:
    """仅给真实 projector continuation 恢复 AdamW 动量。"""
    provenance = {"restored": False, "source": None, "source_step": None}
    if arm["kind"] != "projector" or not bool(arm.get("resume_optimizer", False)):
        return provenance
    state_path = base_projector / "training_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"projector optimizer state is missing: {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if "optimizer" not in state or "step" not in state:
        raise ValueError("projector training state is incomplete")
    optimizer.load_state_dict(state["optimizer"])
    # PyTorch 通常会自动迁移状态；显式迁移避免旧版本把 AdamW 动量留在 CPU。
    for parameter_state in optimizer.state.values():
        for key, value in parameter_state.items():
            if torch.is_tensor(value):
                parameter_state[key] = value.to(device=device)
    return {
        "restored": True,
        "source": str(state_path),
        "source_step": int(state["step"]),
        "source_sha256": sha256(state_path),
    }


def read_frozen_records(data_path: Path, ids_path: Path) -> list[dict]:
    all_records = {str(row["id"]): row for row in load_records(data_path)}
    requested = [
        str(row["id"])
        for row in json.loads(ids_path.read_text(encoding="utf-8"))["records"]
    ]
    missing = [sample_id for sample_id in requested if sample_id not in all_records]
    if missing:
        raise ValueError(f"frozen adaptation IDs are missing: {missing[:3]}")
    records = [all_records[sample_id] for sample_id in requested]
    if len(records) != 400 or len({str(row["pair_id"]) for row in records}) != 200:
        raise ValueError("shape adaptation requires exactly 400 records / 200 pairs")
    return records


def read_adaptation_records(dataset: dict) -> list[dict]:
    if dataset.get("train_ids"):
        return read_frozen_records(
            Path(dataset["train_data"]), Path(dataset["train_ids"])
        )
    tasks = [str(task) for task in dataset["tasks"]]
    records = [
        row
        for row in load_records(Path(dataset["train_data"]))
        if str(row.get("task")) in tasks
    ]
    expected = int(dataset["expected_train_records"])
    if len(records) != expected:
        raise ValueError(f"adaptation train denominator mismatch: {len(records)} != {expected}")
    counts = {
        task: sum(str(row.get("task")) == task for row in records) for task in tasks
    }
    if len(set(counts.values())) != 1:
        raise ValueError(f"adaptation tasks are not balanced: {counts}")
    pair_counts = {task: set() for task in tasks}
    for record in records:
        pair_counts[str(record["task"])].add(str(record["pair_id"]))
    if any(len(pair_counts[task]) * 2 != counts[task] for task in tasks):
        raise ValueError("adaptation task contains incomplete counterfactual pairs")
    return records


def balanced_epoch_indices(
    records: list[dict],
    *,
    tasks: list[str],
    batch_size: int,
    generator: torch.Generator,
) -> list[int]:
    if batch_size % len(tasks) != 0:
        raise ValueError("balanced adaptation batch must be divisible by task count")
    quota = batch_size // len(tasks)
    grouped = {
        task: [index for index, row in enumerate(records) if str(row["task"]) == task]
        for task in tasks
    }
    sizes = {task: len(indices) for task, indices in grouped.items()}
    if not sizes or 0 in sizes.values() or len(set(sizes.values())) != 1:
        raise ValueError(f"balanced adaptation task sizes drifted: {sizes}")
    if next(iter(sizes.values())) % quota != 0:
        raise ValueError("per-task record count must be divisible by per-batch quota")
    shuffled = {
        task: [indices[index] for index in torch.randperm(len(indices), generator=generator)]
        for task, indices in grouped.items()
    }
    batches = next(iter(sizes.values())) // quota
    order: list[int] = []
    for batch_index in range(batches):
        start = batch_index * quota
        for task in tasks:
            order.extend(shuffled[task][start : start + quota])
    return order


def teacher_forced_batch(
    tokenizer,
    records: list[dict],
    *,
    prompt_template: str,
    placeholder: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    prompt_ids, prompt_mask = visual_prompt_batch(
        tokenizer,
        prompt_template,
        [str(record["question"]) for record in records],
        placeholder,
        device,
    )
    answers = [
        tokenizer.encode(" " + str(record["answers"][0]), add_special_tokens=False)
        + [int(tokenizer.eos_token_id)]
        for record in records
    ]
    if any(len(answer) < 2 for answer in answers):
        raise ValueError("teacher-forced answer must contain a token plus EOS")
    answer_width = max(map(len, answers))
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    ids = torch.full(
        (len(records), prompt_ids.shape[1] + answer_width),
        pad,
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    ids[:, : prompt_ids.shape[1]] = prompt_ids
    mask[:, : prompt_ids.shape[1]] = prompt_mask
    for index, answer in enumerate(answers):
        value = torch.tensor(answer, dtype=torch.long, device=device)
        start = prompt_ids.shape[1]
        ids[index, start : start + value.numel()] = value
        mask[index, start : start + value.numel()] = 1
        labels[index, start : start + value.numel()] = value
    return ids, mask, labels, sum(len(answer) for answer in answers)


def tensor_state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        tensor = state[name].detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(
    output: Path,
    *,
    step: int,
    arm: dict,
    arm_name: str,
    projector: PatchMergerProjector,
    language_model,
    optimizer,
    resolved_modules: list[str],
    base_projector_sha256: str,
    batch_size: int,
) -> dict:
    directory = output / "checkpoints" / f"step-{step:06d}"
    directory.mkdir(parents=True)
    if arm["kind"] == "lora":
        state = lora_state_dict(language_model)
        weights_path = directory / "lora.safetensors"
        save_file(state, str(weights_path))
        adapter_config = {
            "format_version": "explicit-top-lora-v1",
            "arm": arm_name,
            "step": step,
            "layer_indices": arm["layer_indices"],
            "target_modules": arm["target_modules"],
            "rank": arm["rank"],
            "alpha": arm["alpha"],
            "resolved_modules": resolved_modules,
            "base_projector_sha256": base_projector_sha256,
        }
        (directory / "adapter_config.json").write_text(
            json.dumps(adapter_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        projector.save_pretrained(directory)
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in projector.state_dict().items()
        }
        weights_path = directory / projector.weights_filename
    training_state_path = directory / "training_state.pt"
    torch.save({"step": step, "optimizer": optimizer.state_dict()}, training_state_path)
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(directory.iterdir())
        if path.is_file()
    }
    manifest = {
        "status": "valid",
        "arm": arm_name,
        "kind": arm["kind"],
        "step": step,
        "examples_seen": step * batch_size,
        "weights_tensor_sha256": tensor_state_hash(state),
        "files": files,
    }
    (directory / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def run(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.arm not in config["arms"]:
        raise ValueError(f"unknown adaptation arm: {args.arm}")
    arm = config["arms"][args.arm]
    training = config["training"]
    steps = int(args.steps if args.steps is not None else training["steps"])
    batch_size = int(
        args.batch_size if args.batch_size is not None else training["batch_size"]
    )
    if steps <= 0 or batch_size <= 0:
        raise ValueError("adaptation steps and batch size must be positive")
    config["runtime_override"] = {"steps": args.steps, "batch_size": args.batch_size}
    config["resolved_arm"] = args.arm
    args.out.mkdir(parents=True)
    (args.out / "CONFIG.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    started = time.time()
    device = torch.device(config["device"])
    language_dtype = getattr(torch, config["language_dtype"])
    projector_dtype = getattr(torch, config["projector_dtype"])
    torch.manual_seed(int(config["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["seed"]))
        torch.cuda.reset_peak_memory_stats(device)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_config = AutoConfig.from_pretrained(config["text_model"], local_files_only=True)
    validate_text_only_backbone_config(model_config)
    tokenizer = AutoTokenizer.from_pretrained(config["text_model"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    language_model = AutoModelForCausalLM.from_pretrained(
        config["text_model"], dtype=language_dtype, local_files_only=True
    ).to(device)
    language_model.requires_grad_(False)
    projector_source = Path(config["base_projector"])
    projector_config_source = resolve_projector_config_source(config)
    projector_config = ProjectorConfig(
        **json.loads(
            (projector_config_source / "projector_config.json").read_text(
                encoding="utf-8"
            )
        )
    )
    projector = PatchMergerProjector(projector_config).to(
        device=device, dtype=projector_dtype
    )
    base_state_path = projector_source / "projector.safetensors"
    projector.load_state_dict(load_file(str(base_state_path), device="cpu"), strict=True)
    base_projector_sha256 = sha256(base_state_path)
    resolved_modules: list[str] = []
    if arm["kind"] == "lora":
        projector.requires_grad_(False).eval()
        resolved_modules = inject_lora(
            language_model,
            layer_indices=arm["layer_indices"],
            target_modules=arm["target_modules"],
            rank=int(arm["rank"]),
            alpha=float(arm["alpha"]),
            seed=int(config["seed"]),
        )
        trainable_parameters = freeze_non_lora(language_model)
        language_model.train()
        parameters = [parameter for parameter in language_model.parameters() if parameter.requires_grad]
        freeze_language_model = False
    elif arm["kind"] == "projector":
        language_model.eval()
        projector.requires_grad_(True).train()
        trainable_parameters = sum(parameter.numel() for parameter in projector.parameters())
        parameters = list(projector.parameters())
        freeze_language_model = True
    else:
        raise ValueError(f"unsupported adaptation kind: {arm['kind']}")
    model = VisionCausalLM(
        language_model=language_model,
        projector=projector,
        placeholder_token_id=int(config["placeholder_token_id"]),
        backbone_kind="generic",
        freeze_language_model=freeze_language_model,
        pad_token_id=int(tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    optimizer_resume = maybe_resume_projector_optimizer(
        optimizer=optimizer,
        arm=arm,
        base_projector=projector_source,
        device=device,
    )
    dataset = config["dataset"]
    train_records = read_adaptation_records(dataset)
    configured_tasks = dataset.get("tasks")
    if configured_tasks is None:
        configured_tasks = [dataset["task"]]
    tasks = [str(task) for task in configured_tasks]
    cache = FeatureCache(dataset["train_feature_cache"])
    if int(cache.manifest["max_image_side"]) != int(dataset["max_image_side"]):
        raise ValueError("shape adaptation cache resolution mismatch")
    checkpoint_steps = sorted(
        {0, steps}
        | {int(value) for value in training["checkpoint_steps"] if int(value) <= steps}
    )
    checkpoint_manifests = {}
    checkpoint_manifests["step-000000"] = save_checkpoint(
        args.out,
        step=0,
        arm=arm,
        arm_name=args.arm,
        projector=projector,
        language_model=language_model,
        optimizer=optimizer,
        resolved_modules=resolved_modules,
        base_projector_sha256=base_projector_sha256,
        batch_size=batch_size,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
    order: list[int] = []
    cursor = 0
    answer_tokens_seen = 0
    history_path = args.out / "train_history.jsonl"
    order_path = args.out / "training_order.jsonl"
    with history_path.open("w", encoding="utf-8", newline="\n") as history_stream, order_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as order_stream:
        for step in range(1, steps + 1):
            if cursor + batch_size > len(order):
                order = balanced_epoch_indices(
                    train_records,
                    tasks=tasks,
                    batch_size=batch_size,
                    generator=generator,
                )
                cursor = 0
            indices = order[cursor : cursor + batch_size]
            cursor += batch_size
            batch = [train_records[index] for index in indices]
            groups = [
                cache.get(str(record["id"]), device=device, dtype=projector_dtype)[0]
                for record in batch
            ]
            ids, mask, labels, answer_tokens = teacher_forced_batch(
                tokenizer,
                batch,
                prompt_template=str(config["prompt_template"]),
                placeholder=int(config["placeholder_token_id"]),
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            step_started = time.time()
            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                labels=labels,
                image_feature_groups=groups,
            )
            if not bool(torch.isfinite(outputs.loss)):
                raise ValueError(f"non-finite adaptation loss at step {step}")
            outputs.loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(training["gradient_clip"])
            )
            if not bool(torch.isfinite(grad_norm)):
                raise ValueError(f"non-finite adaptation gradient at step {step}")
            optimizer.step()
            answer_tokens_seen += answer_tokens
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row = {
                "step": step,
                "loss": float(outputs.loss.detach()),
                "gradient_norm_before_clip": float(grad_norm),
                "batch_size": batch_size,
                "examples_seen": step * batch_size,
                "answer_tokens_seen": answer_tokens_seen,
                "task_counts": {
                    task: sum(str(record["task"]) == task for record in batch)
                    for task in tasks
                },
                "step_wall_seconds": time.time() - step_started,
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                ),
            }
            history_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            history_stream.flush()
            order_stream.write(
                json.dumps(
                    {"step": step, "ids": [str(record["id"]) for record in batch]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            order_stream.flush()
            if step in checkpoint_steps:
                checkpoint_manifests[f"step-{step:06d}"] = save_checkpoint(
                    args.out,
                    step=step,
                    arm=arm,
                    arm_name=args.arm,
                    projector=projector,
                    language_model=language_model,
                    optimizer=optimizer,
                    resolved_modules=resolved_modules,
                    base_projector_sha256=base_projector_sha256,
                    batch_size=batch_size,
                )
            if step == 1 or step % 25 == 0:
                print(json.dumps(row), flush=True)
    summary = {
        "status": "valid",
        "format_version": config.get(
            "training_format_version", "shape-adaptation-training-v1"
        ),
        "arm": args.arm,
        "kind": arm["kind"],
        "metadata": {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha(),
            "host": platform.node(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "wall_seconds": time.time() - started,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "final_half_scored": False,
        },
        "steps": steps,
        "batch_size": batch_size,
        "examples_seen": steps * batch_size,
        "train_records": len(train_records),
        "train_pairs": len({str(record["pair_id"]) for record in train_records}),
        "tasks": tasks,
        "train_records_by_task": {
            task: sum(str(record["task"]) == task for record in train_records)
            for task in tasks
        },
        "trainable_parameters": trainable_parameters,
        "resolved_lora_modules": resolved_modules,
        "base_projector_sha256": base_projector_sha256,
        "projector_config_source": str(projector_config_source),
        "projector_config_sha256": sha256(
            projector_config_source / "projector_config.json"
        ),
        "optimizer_resume": optimizer_resume,
        "selection_manifest_sha256": sha256(Path(dataset["selection_manifest"])),
        "checkpoints": checkpoint_manifests,
        "files": {
            history_path.name: {
                "bytes": history_path.stat().st_size,
                "sha256": sha256(history_path),
            },
            order_path.name: {
                "bytes": order_path.stat().st_size,
                "sha256": sha256(order_path),
            },
        },
        "final_half_scored": False,
    }
    (args.out / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite adaptation run: {args.out}")
    try:
        run(args)
    except Exception as error:
        args.out.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "invalid",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "final_half_scored": False,
        }
        (args.out / "FAILURE.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
