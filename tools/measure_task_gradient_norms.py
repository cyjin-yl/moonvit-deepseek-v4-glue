#!/usr/bin/env python3
"""在固定小批次上测量各任务对 projector 的梯度范数与夹角。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from moonvit_glue import FeatureCache, PatchMergerProjector, ProjectorConfig, VisionCausalLM
from tools_common import load_records, validate_text_only_backbone_config
from train_shape_adaptation import teacher_forced_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--projector-index", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--records-per-task", type=int, default=8)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() or None


def gradient_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    """计算两个梯度向量的余弦；零范数会被明确拒绝。"""
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm) == 0.0 or float(right_norm) == 0.0:
        raise ValueError("gradient cosine is undefined for a zero vector")
    return float(torch.dot(left, right) / (left_norm * right_norm))


def select_complete_pairs(records: list[dict], *, task: str, count: int) -> list[dict]:
    if count <= 0 or count % 2:
        raise ValueError("records per task must be a positive even number")
    task_rows = [row for row in records if str(row["task"]) == task]
    pair_ids = []
    for row in task_rows:
        pair_id = str(row["pair_id"])
        if pair_id not in pair_ids:
            pair_ids.append(pair_id)
        if len(pair_ids) == count // 2:
            break
    selected_ids = set(pair_ids)
    selected = [row for row in task_rows if str(row["pair_id"]) in selected_ids]
    if len(selected) != count:
        raise ValueError(f"task {task} does not expose {count} complete-pair records")
    return selected


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite gradient diagnostics: {args.out}")
    started = time.time()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    index_summary_path = args.projector_index / "SUMMARY.json"
    index_summary = json.loads(index_summary_path.read_text(encoding="utf-8"))
    if index_summary.get("status") != "valid":
        raise ValueError("projector comparison index is invalid")
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
    language_model.requires_grad_(False).eval()
    projector_config_source = Path(config.get("projector_config_source", config["base_projector"]))
    projector_config = ProjectorConfig(
        **json.loads((projector_config_source / "projector_config.json").read_text(encoding="utf-8"))
    )
    projector = PatchMergerProjector(projector_config).to(device=device, dtype=projector_dtype)
    projector.requires_grad_(True).train()
    model = VisionCausalLM(
        language_model=language_model,
        projector=projector,
        placeholder_token_id=int(config["placeholder_token_id"]),
        backbone_kind="generic",
        freeze_language_model=True,
        pad_token_id=int(tokenizer.pad_token_id),
    )
    parameters = [parameter for parameter in projector.parameters() if parameter.requires_grad]
    dataset = config["dataset"]
    tasks = [str(task) for task in dataset["tasks"]]
    all_records = [
        row for row in load_records(Path(dataset["train_data"]))
        if str(row.get("task")) in tasks
    ]
    selected = {
        task: select_complete_pairs(all_records, task=task, count=args.records_per_task)
        for task in tasks
    }
    cache = FeatureCache(dataset["train_feature_cache"])

    states = [
        {
            "id": "frozen-base",
            "path": Path(config["base_projector"]) / "projector.safetensors",
            "step": 0,
        }
    ]
    for state_id, manifest in index_summary["checkpoints"].items():
        relative = Path(manifest["relative_path"])
        checkpoint = relative if relative.is_absolute() else args.projector_index / relative
        states.append({
            "id": state_id,
            "path": checkpoint / "projector.safetensors",
            "step": int(manifest["step"]),
        })

    norm_rows = []
    cosine_rows = []
    state_summaries = {}
    for state in states:
        projector.load_state_dict(load_file(str(state["path"]), device="cpu"), strict=True)
        task_gradients = {}
        state_started = time.time()
        for task in tasks:
            batch = selected[task]
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
            projector.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                labels=labels,
                image_feature_groups=groups,
            )
            if not bool(torch.isfinite(outputs.loss)):
                raise ValueError(f"non-finite gradient diagnostic loss: {state['id']} {task}")
            outputs.loss.backward()
            missing = [name for (name, parameter) in projector.named_parameters() if parameter.grad is None]
            if missing:
                raise ValueError(f"projector gradients are missing: {missing[:3]}")
            gradient = torch.cat([
                parameter.grad.detach().float().cpu().reshape(-1)
                for parameter in parameters
            ])
            norm = float(torch.linalg.vector_norm(gradient))
            if not math.isfinite(norm) or norm == 0.0:
                raise ValueError(f"invalid gradient norm: {state['id']} {task}")
            task_gradients[task] = gradient
            norm_rows.append({
                "state": state["id"],
                "step": state["step"],
                "task": task,
                "records": len(batch),
                "answer_tokens": answer_tokens,
                "loss": float(outputs.loss.detach()),
                "gradient_norm": norm,
            })
        state_cosines = []
        for left_index, left in enumerate(tasks):
            for right in tasks[left_index + 1 :]:
                cosine = gradient_cosine(task_gradients[left], task_gradients[right])
                state_cosines.append(cosine)
                cosine_rows.append({
                    "state": state["id"],
                    "step": state["step"],
                    "task_a": left,
                    "task_b": right,
                    "cosine": cosine,
                    "conflict": cosine < 0,
                })
        state_summaries[state["id"]] = {
            "step": state["step"],
            "weights": str(state["path"]),
            "weights_sha256": sha256(state["path"]),
            "mean_gradient_norm": sum(
                row["gradient_norm"] for row in norm_rows if row["state"] == state["id"]
            ) / len(tasks),
            "mean_task_cosine": sum(state_cosines) / len(state_cosines),
            "negative_cosine_pairs": sum(value < 0 for value in state_cosines),
            "task_pairs": len(state_cosines),
            "wall_seconds": time.time() - state_started,
        }
        del task_gradients

    args.out.mkdir(parents=True)
    write_csv(args.out / "gradient_norms.csv", norm_rows)
    write_csv(args.out / "gradient_cosines.csv", cosine_rows)
    selected_path = args.out / "SELECTED_RECORDS.json"
    selected_path.write_text(
        json.dumps(
            {
                "records_per_task": args.records_per_task,
                "tasks": {
                    task: [str(row["id"]) for row in selected[task]] for task in tasks
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": "valid",
        "format_version": "fixed-task-projector-gradient-diagnostic-v1",
        "metadata": {
            "git_sha": git_sha(),
            "host": platform.node(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "wall_seconds": time.time() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "final_half_scored": False,
        },
        "records_per_task": args.records_per_task,
        "state_summaries": state_summaries,
        "sources": {
            "config_sha256": sha256(args.config),
            "projector_index_sha256": sha256(index_summary_path),
            "selected_records_sha256": sha256(selected_path),
        },
        "files": {},
        "final_half_scored": False,
    }
    for name in ("gradient_norms.csv", "gradient_cosines.csv", "SELECTED_RECORDS.json"):
        path = args.out / name
        summary["files"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (args.out / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run(parse_args())
