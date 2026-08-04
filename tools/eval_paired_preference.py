#!/usr/bin/env python3
"""跨 projector checkpoint 评测 teacher-forced 答案偏好。"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from moonvit_glue import FeatureCache, PatchMergerProjector, ProjectorConfig, VisionCausalLM
from moonvit_glue.merge import expand_image_placeholders
from moonvit_glue.paired_preference import (
    answer_logprob_stats,
    build_pair_index,
    summarize_preference_rows,
)
from moonvit_glue.trajectory_metrics import resolve_control_features
from tools_common import load_records, validate_text_only_backbone_config

from eval_checkpoint_trajectory import (
    _cache_getter,
    file_sha256,
    load_checkpoint_state,
    load_controls,
    matched_random_state,
    visual_prompt_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint-id", action="append", dest="checkpoint_ids")
    return parser.parse_args()


def git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def git_status_porcelain() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in result.stdout.splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def apply_screening_overrides(
    config: dict,
    *,
    limit: int | None,
    checkpoint_ids: list[str] | None,
) -> dict:
    """评测前把筛选选择固化进 run 内部配置。"""

    screened = deepcopy(config)
    selected = [str(value) for value in checkpoint_ids] if checkpoint_ids else None
    if selected is not None:
        known = {str(row["id"]) for row in screened["checkpoints"]}
        unknown = set(selected) - known
        if unknown:
            raise ValueError(f"unknown screening checkpoints: {sorted(unknown)}")
        screened["checkpoints"] = [
            row for row in screened["checkpoints"] if str(row["id"]) in selected
        ]
        selected_set = set(selected)
        screened["aliases"] = [
            row
            for row in screened.get("aliases", [])
            if str(row["source"]) in selected_set
        ]
    if limit is not None:
        if limit < 2 or limit % 2:
            raise ValueError("preference screening limit must be a positive pair multiple")
        screened["synthetic"]["limit"] = int(limit)
        screened["synthetic"]["expected_records"] = int(limit)
    if limit is not None or selected is not None:
        screened["screening_override"] = {
            "record_limit": limit,
            "checkpoint_ids": selected,
        }
    return screened


def condition_source_id(
    condition: str,
    sample_id: str,
    split: str,
    control: dict,
    pair_index: dict[str, dict[str, str]],
) -> str | None:
    """返回一次干预实际读取的缓存特征 ID。"""

    if condition == "blind":
        return None
    if condition in {"vision", "patch_permutation", "background_matched_aux"}:
        return sample_id
    if condition == "blank":
        return str(control.get("blank_image_id", f"control:{split}:blank"))
    if condition == "same_image":
        return f"control:{split}:same"
    if condition == "shuffled_image":
        return str(control["shuffled_image_id"])
    if condition == "paired_counterfactual_image":
        return pair_index[sample_id]["paired_image_id"]
    raise ValueError(f"unknown preference condition: {condition}")


def _safe_pad_id(tokenizer, placeholder_token_id: int | None) -> int:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if placeholder_token_id is not None and pad_id == placeholder_token_id:
        return 0 if placeholder_token_id != 0 else 1
    return int(pad_id)


def answer_batch(
    tokenizer,
    prompt_template: str,
    records: list[dict],
    answers: list[str],
    *,
    placeholder_token_id: int,
    visual: bool,
    device,
):
    if visual:
        prompt_ids, prompt_mask = visual_prompt_batch(
            tokenizer,
            prompt_template,
            [str(record["question"]) for record in records],
            placeholder_token_id,
            device,
        )
    else:
        prompts = [
            prompt_template.format(image="", question=str(record["question"]))
            for record in records
        ]
        old_padding = tokenizer.padding_side
        tokenizer.padding_side = "left"
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        tokenizer.padding_side = old_padding
        prompt_ids = encoded["input_ids"].to(device)
        prompt_mask = encoded["attention_mask"].to(device)

    answer_ids = [
        tokenizer.encode(" " + answer, add_special_tokens=False)
        for answer in answers
    ]
    if any(not row for row in answer_ids):
        raise ValueError("every preference candidate needs at least one answer token")
    answer_length = max(len(row) for row in answer_ids)
    total_length = prompt_ids.shape[1] + answer_length
    input_ids = torch.full(
        (len(records), total_length),
        _safe_pad_id(tokenizer, placeholder_token_id if visual else None),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    input_ids[:, : prompt_ids.shape[1]] = prompt_ids
    attention_mask[:, : prompt_ids.shape[1]] = prompt_mask
    for index, tokens in enumerate(answer_ids):
        end = prompt_ids.shape[1] + len(tokens)
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
        input_ids[index, prompt_ids.shape[1] : end] = token_tensor
        attention_mask[index, prompt_ids.shape[1] : end] = 1
        labels[index, prompt_ids.shape[1] : end] = token_tensor
    return input_ids, attention_mask, labels


def score_answers(
    *,
    model,
    tokenizer,
    records: list[dict],
    answers: list[str],
    image_feature_groups,
    placeholder_token_id: int,
    prompt_template: str,
    device,
) -> list[dict]:
    visual = image_feature_groups is not None
    input_ids, attention_mask, labels = answer_batch(
        tokenizer,
        prompt_template,
        records,
        answers,
        placeholder_token_id=placeholder_token_id,
        visual=visual,
        device=device,
    )
    if not visual:
        outputs = model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return answer_logprob_stats(outputs.logits, labels)

    image_embeddings = model.projector(image_feature_groups)
    text_embeddings = model.language_model.get_input_embeddings()(input_ids)
    merged = expand_image_placeholders(
        input_ids=input_ids,
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        placeholder_token_id=placeholder_token_id,
        attention_mask=attention_mask,
        labels=labels,
        pad_token_id=model.pad_token_id,
    )
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_embeddings=image_embeddings,
    )
    assert merged.labels is not None
    return answer_logprob_stats(outputs.logits, merged.labels)


def condition_groups(
    condition: str,
    records: list[dict],
    *,
    pair_index: dict[str, dict[str, str]],
    controls: dict[str, dict],
    cache_get,
    auxiliary_get,
):
    if condition == "blind":
        return None, [None] * len(records)
    groups = []
    source_ids = []
    for record in records:
        sample_id = str(record["id"])
        split = str(record.get("split", "selection"))
        control = controls[sample_id]
        source_id = condition_source_id(
            condition, sample_id, split, control, pair_index
        )
        if condition == "paired_counterfactual_image":
            assert source_id is not None
            resolved = cache_get(source_id)
        elif condition == "background_matched_aux":
            resolved = auxiliary_get(sample_id)
        else:
            resolved = resolve_control_features(
                condition,
                sample_id,
                split,
                control,
                cache_get,
            )
        if resolved is None or len(resolved) != 1:
            raise ValueError("preference runner expects one visual group per sample")
        groups.extend(resolved)
        source_ids.append(source_id)
    return groups, source_ids


def preference_row(
    record: dict,
    pair: dict[str, str],
    correct: dict | None,
    counterfactual: dict | None,
    *,
    checkpoint: dict,
    condition: str,
    visual_source_id: str | None,
    wall_seconds: float,
    failure: dict | None = None,
) -> dict:
    margin = None
    if correct is not None and counterfactual is not None:
        margin = float(correct["logp_mean"]) - float(counterfactual["logp_mean"])
        if not math.isfinite(margin):
            raise FloatingPointError(f"non-finite correct margin: {margin}")
    return {
        "checkpoint": str(checkpoint["id"]),
        "optimizer_steps": int(checkpoint["optimizer_steps"]),
        "examples_seen": int(checkpoint["examples_seen"]),
        "effective_epochs": float(checkpoint["effective_epochs"]),
        "condition": condition,
        "id": str(record["id"]),
        "pair_id": str(record["pair_id"]),
        "pair_variant": str(record["pair_variant"]),
        "task": str(record["task"]),
        "question": str(record["question"]),
        "correct_answer": pair["correct_answer"],
        "counterfactual_answer": pair["counterfactual_answer"],
        "visual_source_id": visual_source_id,
        "correct_answer_tokens": correct["answer_tokens"] if correct else None,
        "correct_logp_sum": correct["logp_sum"] if correct else None,
        "correct_logp_mean": correct["logp_mean"] if correct else None,
        "correct_token_nll": correct["token_normalized_nll"] if correct else None,
        "counterfactual_answer_tokens": counterfactual["answer_tokens"] if counterfactual else None,
        "counterfactual_logp_sum": counterfactual["logp_sum"] if counterfactual else None,
        "counterfactual_logp_mean": counterfactual["logp_mean"] if counterfactual else None,
        "counterfactual_token_nll": counterfactual["token_normalized_nll"] if counterfactual else None,
        "correct_margin": margin,
        "preference_correct": bool(margin is not None and margin > 0),
        "wall_seconds": wall_seconds,
        "failure": failure,
    }


def evaluate_condition(
    *,
    model,
    tokenizer,
    records: list[dict],
    pair_index: dict[str, dict[str, str]],
    controls: dict[str, dict],
    cache_get,
    auxiliary_get,
    checkpoint: dict,
    condition: str,
    batch_size: int,
    placeholder_token_id: int,
    prompt_template: str,
    device,
    raw_stream,
    failure_stream,
) -> list[dict]:
    output = []

    def run_batch(batch: list[dict]):
        groups, source_ids = condition_groups(
            condition,
            batch,
            pair_index=pair_index,
            controls=controls,
            cache_get=cache_get,
            auxiliary_get=auxiliary_get,
        )
        correct_answers = [pair_index[str(row["id"])]["correct_answer"] for row in batch]
        counterfactual_answers = [
            pair_index[str(row["id"])]["counterfactual_answer"] for row in batch
        ]
        correct = score_answers(
            model=model,
            tokenizer=tokenizer,
            records=batch,
            answers=correct_answers,
            image_feature_groups=groups,
            placeholder_token_id=placeholder_token_id,
            prompt_template=prompt_template,
            device=device,
        )
        counterfactual = score_answers(
            model=model,
            tokenizer=tokenizer,
            records=batch,
            answers=counterfactual_answers,
            image_feature_groups=groups,
            placeholder_token_id=placeholder_token_id,
            prompt_template=prompt_template,
            device=device,
        )
        return correct, counterfactual, source_ids

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        started = time.time()
        try:
            correct, counterfactual, source_ids = run_batch(batch)
            per_sample = (time.time() - started) / len(batch)
            batch_rows = [
                preference_row(
                    record,
                    pair_index[str(record["id"])],
                    correct[index],
                    counterfactual[index],
                    checkpoint=checkpoint,
                    condition=condition,
                    visual_source_id=source_ids[index],
                    wall_seconds=per_sample,
                )
                for index, record in enumerate(batch)
            ]
        except Exception as batch_error:
            event = {
                "scope": "batch",
                "checkpoint": checkpoint["id"],
                "condition": condition,
                "start": start,
                "count": len(batch),
                "error_type": type(batch_error).__name__,
                "error": str(batch_error),
                "traceback": traceback.format_exc(),
                "fallback": "single-sample scoring",
            }
            failure_stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            failure_stream.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch_rows = []
            for record in batch:
                sample_started = time.time()
                try:
                    correct, counterfactual, source_ids = run_batch([record])
                    row = preference_row(
                        record,
                        pair_index[str(record["id"])],
                        correct[0],
                        counterfactual[0],
                        checkpoint=checkpoint,
                        condition=condition,
                        visual_source_id=source_ids[0],
                        wall_seconds=time.time() - sample_started,
                    )
                except Exception as sample_error:
                    failure = {
                        "scope": "sample",
                        "checkpoint": checkpoint["id"],
                        "condition": condition,
                        "id": str(record["id"]),
                        "error_type": type(sample_error).__name__,
                        "error": str(sample_error),
                        "traceback": traceback.format_exc(),
                    }
                    failure_stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    failure_stream.flush()
                    row = preference_row(
                        record,
                        pair_index[str(record["id"])],
                        None,
                        None,
                        checkpoint=checkpoint,
                        condition=condition,
                        visual_source_id=None,
                        wall_seconds=time.time() - sample_started,
                        failure=failure,
                    )
                batch_rows.append(row)
        for row in batch_rows:
            raw_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        raw_stream.flush()
        output.extend(batch_rows)
        print(
            f"[{checkpoint['id']}] preference/{condition} "
            f"{min(start + len(batch), len(records))}/{len(records)}",
            flush=True,
        )
    return output


def paired_condition_delta(rows_a: list[dict], rows_b: list[dict], field: str) -> dict:
    by_a = {str(row["id"]): row for row in rows_a}
    by_b = {str(row["id"]): row for row in rows_b}
    if set(by_a) != set(by_b):
        raise ValueError("condition delta requires identical sample IDs")
    values = [
        float(by_a[sample_id][field]) - float(by_b[sample_id][field])
        for sample_id in sorted(by_a)
        if by_a[sample_id].get("failure") is None
        and by_b[sample_id].get("failure") is None
    ]
    return {
        "field": field,
        "denominator": len(values),
        "sum": sum(values),
        "mean": sum(values) / len(values) if values else None,
    }


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite preference run: {args.out}")
    args.out.mkdir(parents=True)
    config = apply_screening_overrides(
        json.loads(args.config.read_text(encoding="utf-8")),
        limit=args.limit,
        checkpoint_ids=args.checkpoint_ids,
    )
    write_json(args.out / "CONFIG.json", config)
    started = time.time()
    device = torch.device(config.get("device", "cuda"))
    dtype = getattr(torch, config.get("dtype", "bfloat16"))
    seed = int(config["seed"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    text_model = str(config["text_model"])
    text_config = AutoConfig.from_pretrained(text_model, local_files_only=True)
    validate_text_only_backbone_config(text_config)
    tokenizer = AutoTokenizer.from_pretrained(text_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    language_model = AutoModelForCausalLM.from_pretrained(
        text_model, dtype=dtype, local_files_only=True
    ).to(device)

    projector_config_source = Path(config["projector_config_source"])
    projector_config = ProjectorConfig(
        **json.loads(
            (projector_config_source / "projector_config.json").read_text(encoding="utf-8")
        )
    )
    projector = PatchMergerProjector(projector_config).to(device=device, dtype=dtype)
    model = VisionCausalLM(
        language_model=language_model,
        projector=projector,
        placeholder_token_id=int(config["placeholder_token_id"]),
        backbone_kind="auto",
        freeze_language_model=True,
        pad_token_id=int(tokenizer.pad_token_id),
    ).eval()
    random_checkpoint = next(
        (
            checkpoint
            for checkpoint in config["checkpoints"]
            if checkpoint["kind"] == "random"
        ),
        None,
    )
    random_state = (
        matched_random_state(projector_config, int(random_checkpoint["random_seed"]))
        if random_checkpoint is not None
        else {}
    )

    dataset = config["synthetic"]
    records = load_records(Path(dataset["data"]))
    if dataset.get("limit") is not None:
        records = records[: int(dataset["limit"])]
    if len(records) != int(dataset["expected_records"]):
        raise ValueError(
            f"synthetic preference expected {dataset['expected_records']} records, got {len(records)}"
        )
    pair_index = build_pair_index(records)
    controls = load_controls(Path(dataset["controls"]))
    primary_cache = FeatureCache(dataset["feature_cache"])
    control_cache = FeatureCache(dataset["control_cache"])
    auxiliary_cache = FeatureCache(dataset["background_aux_feature_cache"])
    cache_get = _cache_getter(primary_cache, control_cache, device=device, dtype=dtype)
    auxiliary_get = _cache_getter(auxiliary_cache, None, device=device, dtype=dtype)

    summary: dict[str, Any] = {
        "status": "running",
        "format_version": "paired-preference-trajectory-v1",
        "metadata": {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha(),
            "git_status_porcelain": git_status_porcelain(),
            "host": platform.node(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "dtype": str(dtype).removeprefix("torch."),
            "seed": seed,
            "text_model": text_model,
            "prompt_template": str(config["prompt_template"]),
            "scorer_version": "token-normalized-answer-logprob-v1-no-eos",
            "batch_size": int(dataset["batch_size"]),
            "actual_batched_forward": True,
            "paired_preference_rule": "both pair variants have correct_margin > 0",
            "final_half_scored": False,
            "config_sha256": file_sha256(args.out / "CONFIG.json"),
        },
        "dataset": {
            "records": len(records),
            "pairs": len(records) // 2,
            "logical_dataset_sha256": str(dataset["logical_dataset_sha256"]),
            "data_sha256": file_sha256(Path(dataset["data"])),
            "selection_manifest_sha256": file_sha256(
                Path(dataset["data"]).parent / "MANIFEST.json"
            ),
            "controls_sha256": file_sha256(Path(dataset["controls"])),
            "feature_cache_manifest_sha256": file_sha256(
                Path(dataset["feature_cache"]) / "MANIFEST.json"
            ),
            "control_cache_manifest_sha256": file_sha256(
                Path(dataset["control_cache"]) / "MANIFEST.json"
            ),
            "background_aux_manifest_sha256": file_sha256(
                Path(dataset["background_aux_data"]).parent / "MANIFEST.json"
            ),
            "background_aux_cache_manifest_sha256": file_sha256(
                Path(dataset["background_aux_feature_cache"]) / "MANIFEST.json"
            ),
            "max_image_side": int(dataset["max_image_side"]),
        },
        "checkpoints": {},
        "aliases": config.get("aliases", []),
        "failures": 0,
    }
    raw_path = args.out / "preference_records.jsonl"
    failures_path = args.out / "failures.jsonl"
    with raw_path.open("w", encoding="utf-8", newline="\n") as raw_stream, failures_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as failure_stream, torch.inference_mode():
        for checkpoint_source in config["checkpoints"]:
            checkpoint = deepcopy(checkpoint_source)
            checkpoint["state_sha256"] = load_checkpoint_state(
                model, checkpoint, random_state, device
            )
            checkpoint_rows: dict[str, list[dict]] = {}
            checkpoint_summary = {
                key: checkpoint[key]
                for key in (
                    "id",
                    "kind",
                    "optimizer_steps",
                    "examples_seen",
                    "effective_epochs",
                    "state_sha256",
                )
            }
            checkpoint_summary["initialization_claim"] = checkpoint.get(
                "initialization_claim"
            )
            checkpoint_summary["source_path"] = checkpoint.get("path")
            checkpoint_summary["projector_config_sha256"] = file_sha256(
                projector_config_source / "projector_config.json"
            )
            checkpoint_summary["conditions"] = {}
            for condition in dataset["conditions"]:
                rows = evaluate_condition(
                    model=model,
                    tokenizer=tokenizer,
                    records=records,
                    pair_index=pair_index,
                    controls=controls,
                    cache_get=cache_get,
                    auxiliary_get=auxiliary_get,
                    checkpoint=checkpoint,
                    condition=str(condition),
                    batch_size=int(dataset["batch_size"]),
                    placeholder_token_id=int(config["placeholder_token_id"]),
                    prompt_template=str(config["prompt_template"]),
                    device=device,
                    raw_stream=raw_stream,
                    failure_stream=failure_stream,
                )
                checkpoint_rows[str(condition)] = rows
                checkpoint_summary["conditions"][str(condition)] = summarize_preference_rows(rows)
                summary["failures"] += sum(bool(row.get("failure")) for row in rows)
                summary["checkpoints"][checkpoint["id"]] = checkpoint_summary
                write_json(args.out / "SUMMARY.partial.json", summary)
            checkpoint_summary["image_swap_drop"] = paired_condition_delta(
                checkpoint_rows["vision"],
                checkpoint_rows["paired_counterfactual_image"],
                "correct_logp_mean",
            )
            checkpoint_summary["background_shift"] = paired_condition_delta(
                checkpoint_rows["background_matched_aux"],
                checkpoint_rows["vision"],
                "correct_logp_mean",
            )
            summary["checkpoints"][checkpoint["id"]] = checkpoint_summary

    for alias in config.get("aliases", []):
        source = str(alias["source"])
        alias_id = str(alias["id"])
        aliased = deepcopy(summary["checkpoints"][source])
        aliased["id"] = alias_id
        aliased["alias_of"] = source
        summary["checkpoints"][alias_id] = aliased
    summary["failure_events"] = sum(
        bool(line) for line in failures_path.read_text(encoding="utf-8").splitlines()
    )
    if summary["failures"]:
        summary["status"] = "completed_with_sample_failures"
    elif summary["failure_events"]:
        summary["status"] = "valid_with_recovered_batch_failures"
    else:
        summary["status"] = "valid"
    summary["metadata"]["wall_seconds"] = time.time() - started
    summary["metadata"]["peak_gpu_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    summary["metadata"]["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["raw_files"] = {
        raw_path.name: {"bytes": raw_path.stat().st_size, "sha256": file_sha256(raw_path)},
        failures_path.name: {
            "bytes": failures_path.stat().st_size,
            "sha256": file_sha256(failures_path),
        },
    }
    write_json(args.out / "SUMMARY.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "wall_seconds": summary["metadata"]["wall_seconds"],
                "failures": summary["failures"],
                "summary": str(args.out / "SUMMARY.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
