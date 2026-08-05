#!/usr/bin/env python3
"""提取 projector 与 Qwen 全层的预注册池化表示，供独立 probe 重放。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from moonvit_glue import FeatureCache, PatchMergerProjector, ProjectorConfig
from moonvit_glue.lora import inject_lora, load_lora_state_dict
from moonvit_glue.mechanism_probe import (
    last_active_indices,
    masked_token_mean,
    pool_token_grid,
    select_complete_task_pairs,
)
from moonvit_glue.merge import expand_image_placeholders
from moonvit_glue.trajectory_metrics import resolve_control_features
from tools_common import load_records, validate_text_only_backbone_config


IMAGE_SENTINEL = "<mechanism-image-sentinel>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--checkpoint-ids", nargs="*", default=None)
    parser.add_argument("--checkpoint-label", default=None)
    parser.add_argument(
        "--projector-checkpoint",
        type=Path,
        default=None,
        help="用显式 projector checkpoint 替换筛选后的唯一 checkpoint",
    )
    parser.add_argument(
        "--language-adapter",
        type=Path,
        default=None,
        help="含 adapter_config.json 与 lora.safetensors 的显式顶部 LoRA checkpoint",
    )
    return parser.parse_args()


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
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


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def override_projector_checkpoint(
    config: dict[str, Any], checkpoint_dir: Path, checkpoint_label: str
) -> None:
    if len(config["checkpoints"]) != 1:
        raise ValueError("projector checkpoint override requires exactly one source checkpoint")
    weights_path = checkpoint_dir / "projector.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    source = dict(config["checkpoints"][0])
    config["checkpoints"] = [
        {
            "id": checkpoint_label,
            "source_id": source["id"],
            "kind": "trained",
            "path": str(checkpoint_dir),
        }
    ]
    config["projector_checkpoint_override"] = {
        "directory": str(checkpoint_dir),
        "weights_sha256": file_sha256(weights_path),
    }


def matched_random_state(config: ProjectorConfig, seed: int) -> dict[str, torch.Tensor]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        projector = PatchMergerProjector(config)
    return {key: value.detach().clone() for key, value in projector.state_dict().items()}


def load_checkpoint_state(
    projector: PatchMergerProjector,
    checkpoint: dict,
    random_state: dict[str, torch.Tensor],
) -> str:
    if checkpoint["kind"] == "random":
        state = random_state
    else:
        state = load_file(str(Path(checkpoint["path"]) / "projector.safetensors"), device="cpu")
    projector.load_state_dict(state, strict=True)
    projector.eval()
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def visual_prompt_batch(tokenizer, template: str, questions: list[str], placeholder: int, device):
    encoded_rows: list[list[int]] = []
    for question in questions:
        rendered = template.replace("{image}", IMAGE_SENTINEL).format(question=question)
        before, after = rendered.split(IMAGE_SENTINEL)
        encoded_rows.append(
            tokenizer.encode(before, add_special_tokens=False)
            + [placeholder]
            + tokenizer.encode(after, add_special_tokens=False)
        )
    maximum = max(map(len, encoded_rows))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if pad_id == placeholder:
        pad_id = 0 if placeholder != 0 else 1
    ids = torch.full((len(encoded_rows), maximum), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(encoded_rows):
        start = maximum - len(row)
        ids[index, start:] = torch.tensor(row, dtype=torch.long, device=device)
        mask[index, start:] = 1
    return ids, mask


def pair_index(records: list[dict]) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for record in records:
        grouped.setdefault(str(record["pair_id"]), []).append(str(record["id"]))
    result: dict[str, str] = {}
    for pair_id, ids in grouped.items():
        if len(ids) != 2:
            raise ValueError(f"pair {pair_id!r} is incomplete")
        result[ids[0]], result[ids[1]] = ids[1], ids[0]
    return result


def take_pair_limit(records: list[dict], limit_pairs: int | None) -> list[dict]:
    if limit_pairs is None:
        return records
    pairs = sorted({str(record["pair_id"]) for record in records})[:limit_pairs]
    keep = set(pairs)
    return [record for record in records if str(record["pair_id"]) in keep]


def resolve_batch_features(
    records: list[dict],
    condition: str,
    *,
    cache: FeatureCache,
    controls: dict[str, dict],
    paired: dict[str, str],
    device,
    dtype,
) -> tuple[list[torch.Tensor], list[str]]:
    groups: list[torch.Tensor] = []
    source_ids: list[str] = []

    def cache_get(sample_id: str):
        return cache.get(sample_id, device=device, dtype=dtype)

    for record in records:
        sample_id = str(record["id"])
        if condition == "paired_counterfactual_image":
            source_id = paired[sample_id]
            resolved = cache_get(source_id)
        else:
            control = controls.get(sample_id, {})
            if condition == "shuffled_image":
                source_id = str(control["shuffled_image_id"])
            else:
                source_id = sample_id
            resolved = resolve_control_features(
                condition, sample_id, str(record["split"]), control, cache_get
            )
        if resolved is None or len(resolved) != 1:
            raise ValueError("mechanism extraction expects one visual group per sample")
        groups.append(resolved[0])
        source_ids.append(source_id)
    return groups, source_ids


def answer_token_ids(tokenizer, classes: list[str], device) -> torch.Tensor:
    rows = [tokenizer.encode(" " + answer, add_special_tokens=False) for answer in classes]
    if any(len(row) != 1 for row in rows):
        raise ValueError(f"shape logit lens requires one token per class: {rows}")
    return torch.tensor([row[0] for row in rows], dtype=torch.long, device=device)


def collect_condition(
    *,
    language_model,
    projector,
    tokenizer,
    records: list[dict],
    condition: str,
    cache: FeatureCache,
    controls: dict[str, dict],
    paired: dict[str, str],
    labels_by_id: dict[str, int],
    classes: list[str],
    poolings: list[str],
    batch_size: int,
    placeholder: int,
    prompt_template: str,
    save_dtype: torch.dtype,
    device,
    dtype,
) -> tuple[dict[str, torch.Tensor], list[dict]]:
    tensors: dict[str, list[torch.Tensor]] = {}
    metadata: list[dict] = []
    class_tokens = answer_token_ids(tokenizer, classes, device)
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        groups, source_ids = resolve_batch_features(
            batch,
            condition,
            cache=cache,
            controls=controls,
            paired=paired,
            device=device,
            dtype=dtype,
        )
        raw = torch.stack(groups)
        projected_list = projector(groups)
        projected = torch.stack(projected_list)
        input_ids, input_mask = visual_prompt_batch(
            tokenizer,
            prompt_template,
            [str(record["question"]) for record in batch],
            placeholder,
            device,
        )
        text = language_model.get_input_embeddings()(input_ids)
        merged = expand_image_placeholders(
            input_ids=input_ids,
            text_embeddings=text,
            image_embeddings=projected_list,
            placeholder_token_id=placeholder,
            attention_mask=input_mask,
            pad_token_id=int(tokenizer.pad_token_id),
        )
        outputs = language_model(
            inputs_embeds=merged.inputs_embeds,
            attention_mask=merged.attention_mask,
            position_ids=merged.position_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError("language model did not return hidden states")
        image_mask = merged.routing_input_ids.eq(placeholder) & merged.attention_mask.bool()
        assistant = last_active_indices(merged.attention_mask)
        batch_indices = torch.arange(len(batch), device=device)
        for mode in poolings:
            tensors.setdefault(f"tower_{mode}", []).append(
                pool_token_grid(raw, mode).to(save_dtype).cpu()
            )
            tensors.setdefault(f"projector_{mode}", []).append(
                pool_token_grid(projected, mode).to(save_dtype).cpu()
            )
        for layer_index, hidden in enumerate(hidden_states):
            tensors.setdefault(f"layer_{layer_index:02d}_assistant", []).append(
                hidden[batch_indices, assistant].to(save_dtype).cpu()
            )
            tensors.setdefault(f"layer_{layer_index:02d}_image_mean", []).append(
                masked_token_mean(hidden, image_mask).to(save_dtype).cpu()
            )
        tensors.setdefault("shape_logits", []).append(
            outputs.logits[batch_indices, assistant][:, class_tokens].float().cpu()
        )
        tensors.setdefault("labels", []).append(
            torch.tensor([labels_by_id[str(row["id"])] for row in batch], dtype=torch.long)
        )
        tensors.setdefault("source_labels", []).append(
            torch.tensor([labels_by_id[source_id] for source_id in source_ids], dtype=torch.long)
        )
        for record, source_id in zip(batch, source_ids, strict=True):
            metadata.append(
                {
                    "id": str(record["id"]),
                    "pair_id": str(record["pair_id"]),
                    "pair_variant": str(record["pair_variant"]),
                    "condition": condition,
                    "target_answer": str(record["answers"][0]),
                    "source_id": source_id,
                    "source_answer": classes[labels_by_id[source_id]],
                }
            )
    return {key: torch.cat(values) for key, values in tensors.items()}, metadata


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite representation run: {args.out}")
    args.out.mkdir(parents=True)
    config = deepcopy(json.loads(args.config.read_text(encoding="utf-8")))
    if args.checkpoint_ids:
        requested = set(args.checkpoint_ids)
        config["checkpoints"] = [row for row in config["checkpoints"] if row["id"] in requested]
        if len(config["checkpoints"]) != len(requested):
            raise ValueError("unknown checkpoint in screening override")
    if args.checkpoint_label is not None and args.projector_checkpoint is None:
        if len(config["checkpoints"]) != 1:
            raise ValueError("checkpoint label override requires exactly one source checkpoint")
        checkpoint = dict(config["checkpoints"][0])
        checkpoint["source_id"] = checkpoint["id"]
        checkpoint["id"] = str(args.checkpoint_label)
        config["checkpoints"] = [checkpoint]
    if args.projector_checkpoint is not None:
        if args.checkpoint_label is None:
            raise ValueError("--projector-checkpoint requires --checkpoint-label")
        override_projector_checkpoint(
            config, args.projector_checkpoint, str(args.checkpoint_label)
        )
    adapter_config = None
    if args.language_adapter is not None:
        adapter_config_path = args.language_adapter / "adapter_config.json"
        adapter_weights_path = args.language_adapter / "lora.safetensors"
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        config["language_adapter"] = {
            "directory": str(args.language_adapter),
            "config_sha256": file_sha256(adapter_config_path),
            "weights_sha256": file_sha256(adapter_weights_path),
            "format_version": adapter_config["format_version"],
            "layer_indices": adapter_config["layer_indices"],
            "target_modules": adapter_config["target_modules"],
            "rank": adapter_config["rank"],
            "alpha": adapter_config["alpha"],
        }
    config["screening_override"] = {
        "limit_pairs": args.limit_pairs,
        "checkpoint_ids": args.checkpoint_ids,
        "checkpoint_label": args.checkpoint_label,
        "projector_checkpoint": (
            str(args.projector_checkpoint) if args.projector_checkpoint else None
        ),
        "language_adapter": str(args.language_adapter) if args.language_adapter else None,
    }
    write_json(args.out / "CONFIG.json", config)
    started = time.time()
    device = torch.device(config["device"])
    dtype = getattr(torch, config["dtype"])
    save_dtype = getattr(torch, config["extraction"]["save_dtype"])
    torch.manual_seed(int(config["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["seed"]))
        torch.cuda.reset_peak_memory_stats(device)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_path = str(config["text_model"])
    text_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    validate_text_only_backbone_config(text_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    language_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, local_files_only=True
    ).to(device).eval()
    language_model.requires_grad_(False)
    if args.language_adapter is not None:
        assert adapter_config is not None
        resolved_modules = inject_lora(
            language_model,
            layer_indices=adapter_config["layer_indices"],
            target_modules=adapter_config["target_modules"],
            rank=int(adapter_config["rank"]),
            alpha=float(adapter_config["alpha"]),
            seed=int(config["seed"]),
        )
        if resolved_modules != adapter_config["resolved_modules"]:
            raise ValueError("representation LoRA module resolution drifted from training")
        load_lora_state_dict(
            language_model,
            load_file(str(args.language_adapter / "lora.safetensors"), device="cpu"),
        )
        language_model.eval()
    source = Path(config["projector_config_source"])
    projector_config = ProjectorConfig(
        **json.loads((source / "projector_config.json").read_text(encoding="utf-8"))
    )
    projector = PatchMergerProjector(projector_config).to(device=device, dtype=dtype).eval()
    random_checkpoint = next(
        (row for row in config["checkpoints"] if row["kind"] == "random"), None
    )
    random_state = matched_random_state(
        projector_config,
        int(random_checkpoint["random_seed"]) if random_checkpoint else 0,
    )

    dataset = config["dataset"]
    task = str(dataset["task"])
    all_train = select_complete_task_pairs(load_records(Path(dataset["train_data"])), task)
    all_selection = select_complete_task_pairs(
        load_records(Path(dataset["selection_data"])), task
    )
    train = take_pair_limit(all_train, args.limit_pairs)
    selection = take_pair_limit(all_selection, args.limit_pairs)
    expected_train = args.limit_pairs * 2 if args.limit_pairs else int(dataset["expected_train_records"])
    expected_selection = args.limit_pairs * 2 if args.limit_pairs else int(dataset["expected_selection_records"])
    if len(train) != expected_train or len(selection) != expected_selection:
        raise ValueError("mechanism extraction denominator mismatch")
    controls = {str(row["id"]): row for row in load_records(Path(dataset["controls"]))}
    paired = pair_index(selection)
    classes = [str(value) for value in dataset["classes"]]
    class_index = {answer: index for index, answer in enumerate(classes)}
    labels_by_id = {
        str(record["id"]): class_index[str(record["answers"][0])]
        for record in all_selection
    }
    labels_by_id.update(
        {
            str(record["id"]): class_index[str(record["answers"][0])]
            for record in all_train
        }
    )
    # shuffled source 始终来自同任务的 selection；显式拒绝跨任务或未知标签。
    for record in selection:
        source_id = str(controls[str(record["id"])]["shuffled_image_id"])
        if source_id not in labels_by_id:
            raise ValueError(f"shuffled source leaves selected task: {source_id}")
    train_cache = FeatureCache(dataset["train_feature_cache"])
    selection_cache = FeatureCache(dataset["selection_feature_cache"])

    summary: dict[str, Any] = {
        "status": "running",
        "format_version": "layerwise-representations-v1",
        "metadata": {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha(),
            "host": platform.node(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "dtype": str(dtype).removeprefix("torch."),
            "save_dtype": str(save_dtype).removeprefix("torch."),
            "final_half_scored": False,
        },
        "selection_manifest_sha256": file_sha256(args.selection / "MANIFEST.json"),
        "train_records": len(train),
        "selection_records": len(selection),
        "classes": classes,
        "checkpoints": {},
        "files": {},
    }
    conditions = [str(value) for value in config["extraction"]["conditions"]]
    with torch.inference_mode():
        for checkpoint in config["checkpoints"]:
            state_hash = load_checkpoint_state(projector, checkpoint, random_state)
            checkpoint_summary = {"state_sha256": state_hash, "cells": {}}
            cells = [("train", "vision", train, train_cache)] + [
                ("selection", condition, selection, selection_cache) for condition in conditions
            ]
            for split, condition, records, cache in cells:
                tensors, metadata = collect_condition(
                    language_model=language_model,
                    projector=projector,
                    tokenizer=tokenizer,
                    records=records,
                    condition=condition,
                    cache=cache,
                    controls=controls,
                    paired=paired,
                    labels_by_id=labels_by_id,
                    classes=classes,
                    poolings=list(config["extraction"]["projector_poolings"]),
                    batch_size=int(config["extraction"]["batch_size"]),
                    placeholder=int(config["placeholder_token_id"]),
                    prompt_template=str(config["prompt_template"]),
                    save_dtype=save_dtype,
                    device=device,
                    dtype=dtype,
                )
                stem = f"{checkpoint['id']}__{split}__{condition}"
                tensor_path = args.out / f"{stem}.safetensors"
                metadata_path = args.out / f"{stem}.jsonl"
                save_file(tensors, str(tensor_path), metadata={"format": "layerwise-representations-v1"})
                with metadata_path.open("w", encoding="utf-8", newline="\n") as stream:
                    for row in metadata:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                entry = {
                    "records": len(records),
                    "tensor_keys": len(tensors),
                    "tensor_file": tensor_path.name,
                    "tensor_bytes": tensor_path.stat().st_size,
                    "tensor_sha256": file_sha256(tensor_path),
                    "metadata_file": metadata_path.name,
                    "metadata_bytes": metadata_path.stat().st_size,
                    "metadata_sha256": file_sha256(metadata_path),
                }
                checkpoint_summary["cells"][f"{split}/{condition}"] = entry
                summary["files"][tensor_path.name] = entry["tensor_sha256"]
                summary["files"][metadata_path.name] = entry["metadata_sha256"]
                summary["checkpoints"][checkpoint["id"]] = checkpoint_summary
                write_json(args.out / "SUMMARY.partial.json", summary)
            summary["checkpoints"][checkpoint["id"]] = checkpoint_summary
    summary["status"] = "valid"
    summary["metadata"]["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["metadata"]["wall_seconds"] = time.time() - started
    summary["metadata"]["peak_gpu_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    write_json(args.out / "SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
