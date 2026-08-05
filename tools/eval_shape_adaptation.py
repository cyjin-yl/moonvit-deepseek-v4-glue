#!/usr/bin/env python3
"""在冻结 shape selection 上比较 frozen、顶部 LoRA 与 projector continuation。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file

from eval_checkpoint_trajectory import generate_visual_batch
from eval_paired_preference import score_answers
from extract_layerwise_representations import pair_index
from moonvit_glue import FeatureCache, PatchMergerProjector, ProjectorConfig, VisionCausalLM
from moonvit_glue.lora import inject_lora, load_lora_state_dict
from moonvit_glue.metrics import normalize_answer
from moonvit_glue.paired_preference import build_pair_index, summarize_preference_rows
from moonvit_glue.trajectory_metrics import summarize_synthetic_rows
from tools_common import load_records, validate_text_only_backbone_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--lora-run", type=Path)
    parser.add_argument("--projector-run", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--teacher-batch-size", type=int, default=32)
    parser.add_argument("--generation-batch-size", type=int, default=16)
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


def read_frozen_records(data_path: Path, ids_path: Path) -> list[dict]:
    by_id = {str(row["id"]): row for row in load_records(data_path)}
    ids = [
        str(row["id"])
        for row in json.loads(ids_path.read_text(encoding="utf-8"))["records"]
    ]
    missing = [sample_id for sample_id in ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"adaptation evaluation IDs are missing: {missing[:3]}")
    return [by_id[sample_id] for sample_id in ids]


def take_complete_pair_limit(records: list[dict], limit: int | None) -> list[dict]:
    if limit is None:
        return records
    if limit <= 0 or limit % 2:
        raise ValueError("adaptation evaluation limit must retain complete pairs")
    pair_ids = []
    for record in records:
        pair_id = str(record["pair_id"])
        if pair_id not in pair_ids:
            pair_ids.append(pair_id)
        if len(pair_ids) * 2 >= limit:
            break
    keep = set(pair_ids)
    selected = [row for row in records if str(row["pair_id"]) in keep]
    if len(selected) != limit:
        raise ValueError("adaptation evaluation pair limit mismatch")
    return selected


def checkpoint_states(
    *, config: dict, lora_run: Path | None, projector_run: Path | None
) -> tuple[list[dict], dict]:
    if lora_run is None:
        raise ValueError("a LoRA run is required to define the explicit adapter structure")
    lora_summary = json.loads((lora_run / "SUMMARY.json").read_text(encoding="utf-8"))
    if lora_summary.get("status") != "valid":
        raise ValueError("LoRA adaptation run is not valid")
    states = [
        {
            "id": "frozen-step1500",
            "kind": "frozen",
            "adaptation_step": 0,
            "adaptation_examples_seen": 0,
            "lora": lora_run / "checkpoints" / "step-000000" / "lora.safetensors",
            "projector": Path(config["base_projector"]) / "projector.safetensors",
        }
    ]
    for key, manifest in sorted(lora_summary["checkpoints"].items()):
        step = int(manifest["step"])
        if step == 0:
            continue
        states.append(
            {
                "id": f"lora-step{step}",
                "kind": "lora",
                "adaptation_step": step,
                "adaptation_examples_seen": int(manifest["examples_seen"]),
                "lora": lora_run / "checkpoints" / f"step-{step:06d}" / "lora.safetensors",
                "projector": Path(config["base_projector"]) / "projector.safetensors",
            }
        )
    projector_summary = None
    if projector_run is not None:
        projector_summary = json.loads(
            (projector_run / "SUMMARY.json").read_text(encoding="utf-8")
        )
        if projector_summary.get("status") != "valid":
            raise ValueError("projector continuation run is not valid")
        for key, manifest in sorted(projector_summary["checkpoints"].items()):
            step = int(manifest["step"])
            if step == 0:
                continue
            states.append(
                {
                    "id": f"projector-step{step}",
                    "kind": "projector",
                    "adaptation_step": step,
                    "adaptation_examples_seen": int(manifest["examples_seen"]),
                    "lora": lora_run / "checkpoints" / "step-000000" / "lora.safetensors",
                    "projector": projector_run
                    / "checkpoints"
                    / f"step-{step:06d}"
                    / "projector.safetensors",
                }
            )
    sources = {
        "lora_summary_sha256": sha256(lora_run / "SUMMARY.json"),
        "projector_summary_sha256": (
            sha256(projector_run / "SUMMARY.json") if projector_run else None
        ),
    }
    return states, sources


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    states, sources = checkpoint_states(
        config=config, lora_run=args.lora_run, projector_run=args.projector_run
    )
    config["evaluation_override"] = {
        "limit": args.limit,
        "teacher_batch_size": args.teacher_batch_size,
        "generation_batch_size": args.generation_batch_size,
    }
    config["evaluation_states"] = [
        {
            **{key: value for key, value in state.items() if key not in {"lora", "projector"}},
            "lora": str(state["lora"]),
            "lora_sha256": sha256(state["lora"]),
            "projector": str(state["projector"]),
            "projector_sha256": sha256(state["projector"]),
        }
        for state in states
    ]
    args.out.mkdir(parents=True)
    (args.out / "CONFIG.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    started = time.time()
    device = torch.device(config["device"])
    language_dtype = getattr(torch, config["language_dtype"])
    projector_dtype = getattr(torch, config["projector_dtype"])
    if device.type == "cuda":
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
    adapter_config = json.loads(
        (
            args.lora_run
            / "checkpoints"
            / "step-000000"
            / "adapter_config.json"
        ).read_text(encoding="utf-8")
    )
    resolved = inject_lora(
        language_model,
        layer_indices=adapter_config["layer_indices"],
        target_modules=adapter_config["target_modules"],
        rank=int(adapter_config["rank"]),
        alpha=float(adapter_config["alpha"]),
        seed=int(config["seed"]),
    )
    if resolved != adapter_config["resolved_modules"]:
        raise ValueError("resolved LoRA modules drifted from training")
    projector_source = Path(config["base_projector"])
    projector_config = ProjectorConfig(
        **json.loads((projector_source / "projector_config.json").read_text(encoding="utf-8"))
    )
    projector = PatchMergerProjector(projector_config).to(
        device=device, dtype=projector_dtype
    ).eval()
    projector.requires_grad_(False)
    model = VisionCausalLM(
        language_model=language_model,
        projector=projector,
        placeholder_token_id=int(config["placeholder_token_id"]),
        backbone_kind="generic",
        freeze_language_model=False,
        pad_token_id=int(tokenizer.pad_token_id),
    ).eval()
    dataset = config["dataset"]
    selection = take_complete_pair_limit(
        read_frozen_records(Path(dataset["selection_data"]), Path(dataset["selection_ids"])),
        args.limit,
    )
    generation = take_complete_pair_limit(
        read_frozen_records(Path(dataset["selection_data"]), Path(dataset["patching_ids"])),
        args.limit,
    )
    pair_details = build_pair_index(selection)
    generation_pair_details = build_pair_index(generation)
    mates = pair_index(selection)
    generation_mates = pair_index(generation)
    controls = {
        str(row["id"]): row for row in load_records(Path(dataset["controls"]))
    }
    cache = FeatureCache(dataset["selection_feature_cache"])
    raw_preference_path = args.out / "preference_records.jsonl"
    raw_generation_path = args.out / "generation_records.jsonl"
    preference_curves = []
    generation_curves = []
    with raw_preference_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as preference_stream, raw_generation_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as generation_stream, torch.inference_mode():
        for state in states:
            load_lora_state_dict(
                language_model, load_file(str(state["lora"]), device="cpu")
            )
            projector.load_state_dict(
                load_file(str(state["projector"]), device="cpu"), strict=True
            )
            state_preference_rows = []
            for condition in ("vision", "paired_counterfactual_image", "shuffled_image"):
                condition_rows = []
                for start in range(0, len(selection), args.teacher_batch_size):
                    batch = selection[start : start + args.teacher_batch_size]
                    source_ids = []
                    groups = []
                    for record in batch:
                        sample_id = str(record["id"])
                        if condition == "vision":
                            source_id = sample_id
                        elif condition == "paired_counterfactual_image":
                            source_id = mates[sample_id]
                        else:
                            source_id = str(controls[sample_id]["shuffled_image_id"])
                        source_ids.append(source_id)
                        groups.append(
                            cache.get(source_id, device=device, dtype=projector_dtype)[0]
                        )
                    correct_answers = [
                        pair_details[str(record["id"])]["correct_answer"] for record in batch
                    ]
                    counter_answers = [
                        pair_details[str(record["id"])]["counterfactual_answer"]
                        for record in batch
                    ]
                    correct = score_answers(
                        model=model,
                        tokenizer=tokenizer,
                        records=batch,
                        answers=correct_answers,
                        image_feature_groups=groups,
                        placeholder_token_id=int(config["placeholder_token_id"]),
                        prompt_template=str(config["prompt_template"]),
                        device=device,
                    )
                    counter = score_answers(
                        model=model,
                        tokenizer=tokenizer,
                        records=batch,
                        answers=counter_answers,
                        image_feature_groups=groups,
                        placeholder_token_id=int(config["placeholder_token_id"]),
                        prompt_template=str(config["prompt_template"]),
                        device=device,
                    )
                    for record, source_id, correct_row, counter_row in zip(
                        batch, source_ids, correct, counter, strict=True
                    ):
                        row = {
                            "state": state["id"],
                            "kind": state["kind"],
                            "adaptation_step": state["adaptation_step"],
                            "adaptation_examples_seen": state["adaptation_examples_seen"],
                            "condition": condition,
                            "id": str(record["id"]),
                            "pair_id": str(record["pair_id"]),
                            "pair_variant": str(record["pair_variant"]),
                            "task": "shape",
                            "visual_source_id": source_id,
                            "correct_answer": pair_details[str(record["id"])]["correct_answer"],
                            "counterfactual_answer": pair_details[str(record["id"])][
                                "counterfactual_answer"
                            ],
                            "correct_logp": correct_row["logp_mean"],
                            "counterfactual_logp": counter_row["logp_mean"],
                            "correct_margin": correct_row["logp_mean"]
                            - counter_row["logp_mean"],
                            "correct_token_nll": correct_row["token_normalized_nll"],
                            "counterfactual_token_nll": counter_row[
                                "token_normalized_nll"
                            ],
                            "failure": None,
                        }
                        preference_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        condition_rows.append(row)
                preference_stream.flush()
                state_preference_rows.extend(condition_rows)
                summary = summarize_preference_rows(condition_rows)
                preference_curves.append(
                    {
                        "state": state["id"],
                        "kind": state["kind"],
                        "adaptation_step": state["adaptation_step"],
                        "adaptation_examples_seen": state["adaptation_examples_seen"],
                        "condition": condition,
                        "records": summary["samples"],
                        "pairs": summary["pairs"],
                        "sample_preference_accuracy": summary[
                            "sample_preference_accuracy"
                        ]["value"],
                        "paired_preference_accuracy": summary[
                            "paired_preference_accuracy"
                        ]["value"],
                        "mean_correct_margin": summary["mean_correct_margin"],
                    }
                )
            for condition in ("vision", "paired_counterfactual_image"):
                condition_rows = []
                for start in range(0, len(generation), args.generation_batch_size):
                    batch = generation[start : start + args.generation_batch_size]
                    source_ids = [
                        str(record["id"])
                        if condition == "vision"
                        else generation_mates[str(record["id"])]
                        for record in batch
                    ]
                    groups = [
                        cache.get(source_id, device=device, dtype=projector_dtype)[0]
                        for source_id in source_ids
                    ]
                    predictions = generate_visual_batch(
                        model=model,
                        tokenizer=tokenizer,
                        records=batch,
                        feature_groups=groups,
                        placeholder_token_id=int(config["placeholder_token_id"]),
                        prompt_template=str(config["prompt_template"]),
                        max_new_tokens=int(config["evaluation"]["generation_max_new_tokens"]),
                        device=device,
                    )
                    for record, source_id, prediction in zip(
                        batch, source_ids, predictions, strict=True
                    ):
                        details = generation_pair_details[str(record["id"])]
                        expected = (
                            details["correct_answer"]
                            if condition == "vision"
                            else details["counterfactual_answer"]
                        )
                        row = {
                            "state": state["id"],
                            "kind": state["kind"],
                            "adaptation_step": state["adaptation_step"],
                            "adaptation_examples_seen": state["adaptation_examples_seen"],
                            "condition": condition,
                            "id": str(record["id"]),
                            "pair_id": str(record["pair_id"]),
                            "pair_variant": str(record["pair_variant"]),
                            "task": "shape",
                            "visual_source_id": source_id,
                            "answers": [expected],
                            "prediction": prediction,
                            "normalized_prediction": normalize_answer(prediction),
                            "correct": normalize_answer(prediction)
                            == normalize_answer(expected),
                            "failure": None,
                        }
                        generation_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        condition_rows.append(row)
                generation_stream.flush()
                summary = summarize_synthetic_rows(condition_rows)
                generation_curves.append(
                    {
                        "state": state["id"],
                        "kind": state["kind"],
                        "adaptation_step": state["adaptation_step"],
                        "adaptation_examples_seen": state["adaptation_examples_seen"],
                        "condition": condition,
                        "records": summary["samples"],
                        "pairs": summary["pairs"],
                        "accuracy": summary["accuracy"]["value"],
                        "paired_accuracy": summary["paired_accuracy"]["value"],
                        "answer_flip_accuracy": summary["answer_flip_accuracy"]["value"],
                        "prediction_flip_rate": summary["prediction_flip_rate"]["value"],
                    }
                )
            print(f"completed {state['id']}", flush=True)
    preference_curve_path = args.out / "preference_curve.csv"
    generation_curve_path = args.out / "generation_curve.csv"
    write_csv(preference_curve_path, preference_curves)
    write_csv(generation_curve_path, generation_curves)
    decisions = {
        "status": "valid",
        "best_vision_paired_preference": max(
            (row for row in preference_curves if row["condition"] == "vision"),
            key=lambda row: (float(row["paired_preference_accuracy"]), float(row["mean_correct_margin"])),
        ),
        "best_vision_paired_generation": max(
            (row for row in generation_curves if row["condition"] == "vision"),
            key=lambda row: (float(row["paired_accuracy"]), float(row["accuracy"])),
        ),
        "interpretation_limits": [
            "all selection records are disjoint from the 400-record shape adaptation train split",
            "LoRA and projector continuation see equal true-batch examples from the same frozen order",
            "this shape-only diagnostic does not estimate general benchmark transfer",
            "final odd halves remain unscored",
        ],
    }
    decisions_path = args.out / "DECISIONS.json"
    decisions_path.write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "valid",
        "format_version": "shape-adaptation-eval-v1",
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
        "states": len(states),
        "teacher_forced_records_per_cell": len(selection),
        "generation_records_per_cell": len(generation),
        "preference_rows": len(states) * len(selection) * 3,
        "generation_rows": len(states) * len(generation) * 2,
        "sources": sources,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (
                raw_preference_path,
                raw_generation_path,
                preference_curve_path,
                generation_curve_path,
                decisions_path,
            )
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
        raise FileExistsError(f"refusing to overwrite adaptation evaluation: {args.out}")
    try:
        run(args)
    except Exception as error:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "FAILURE.json").write_text(
            json.dumps(
                {
                    "status": "invalid",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "final_half_scored": False,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
