#!/usr/bin/env python3
"""用固定 synthetic 控制和 selection 数据评测 projector checkpoint。

语言模型只加载一次，projector state 原位替换；所有含图条件都读取已审计的
冻结特征缓存。结果持续写入 JSONL，单次失败不会抹掉已完成样本。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
import traceback
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from moonvit_glue import FeatureCache, PatchMergerProjector, ProjectorConfig, VisionCausalLM
from moonvit_glue.metrics import normalize_answer, score_record, summarize
from moonvit_glue.paired_preference import build_pair_index
from moonvit_glue.trajectory_data import configured_conditions
from moonvit_glue.trajectory_metrics import (
    derangement_indices,
    resolve_control_features,
    summarize_synthetic_rows,
)
from tools_common import load_records, validate_text_only_backbone_config
from training_protocol import records_manifest_sha256, select_supervision


IMAGE_SENTINEL = "\x00image\x00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint-id", action="append", dest="checkpoint_ids")
    return parser.parse_args()


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() or None


def git_status_porcelain() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return [line for line in result.stdout.splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def apply_screening_overrides(
    config: dict,
    *,
    limit: int | None,
    checkpoint_ids: list[str] | None,
) -> dict:
    """在不修改源配置的前提下固化小规模筛选配置。"""

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
            raise ValueError("trajectory screening limit must be a positive pair multiple")
        for dataset in screened["datasets"]:
            fixed_count = int(dataset.get("expected_records", limit))
            effective_limit = min(int(limit), fixed_count)
            dataset["limit"] = effective_limit
            dataset["expected_records"] = effective_limit
        heldout = screened["heldout_shuffle_loss"]
        fixed_count = int(heldout.get("expected_records", limit))
        effective_limit = min(int(limit), fixed_count)
        heldout["limit"] = effective_limit
        heldout["expected_records"] = effective_limit
    if limit is not None or selected is not None:
        screened["screening_override"] = {
            "record_limit": limit,
            "checkpoint_ids": selected,
        }
    return screened


def matched_random_state(config: ProjectorConfig, seed: int) -> dict[str, torch.Tensor]:
    """生成不受加载顺序影响、可复现的 matched random projector。"""

    # 使用独立 CPU RNG，避免语言模型加载顺序改变随机对照。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        projector = PatchMergerProjector(config)
    return {key: value.detach().clone() for key, value in projector.state_dict().items()}


def load_controls(path: Path) -> dict[str, dict]:
    controls = {
        str(row["id"]): row
        for row in load_records(path)
    }
    if len(controls) == 0:
        raise ValueError("control manifest is empty")
    return controls


def visual_prompt_batch(tokenizer, template: str, questions: list[str], placeholder_token_id: int, device):
    encoded_rows: list[list[int]] = []
    for question in questions:
        rendered = template.replace("{image}", IMAGE_SENTINEL).format(question=question)
        before, after = rendered.split(IMAGE_SENTINEL)
        encoded_rows.append(
            tokenizer.encode(before, add_special_tokens=False)
            + [placeholder_token_id]
            + tokenizer.encode(after, add_special_tokens=False)
        )
    max_length = max(len(row) for row in encoded_rows)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if pad_id == placeholder_token_id:
        # Qwen2.5 同时把 end-of-text 用作 padding，Gate B 又把它用作图像
        # placeholder。被 mask 的 padding 仍会进入 placeholder 展开，因此
        # 这里改用词表内另一个 ID。
        pad_id = 0 if placeholder_token_id != 0 else 1
    input_ids = torch.full((len(encoded_rows), max_length), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(encoded_rows):
        start = max_length - len(row)
        input_ids[index, start:] = torch.tensor(row, dtype=torch.long, device=device)
        attention_mask[index, start:] = 1
    return input_ids, attention_mask


def generate_visual_batch(
    *, model, tokenizer, records: list[dict], feature_groups: list[torch.Tensor],
    placeholder_token_id: int, prompt_template: str, max_new_tokens: int, device,
) -> list[str]:
    input_ids, attention_mask = visual_prompt_batch(
        tokenizer,
        prompt_template,
        [str(record["question"]) for record in records],
        placeholder_token_id,
        device,
    )
    raw_length = input_ids.shape[1]
    expanded_prefix = max(raw_length - 1 + int(group.shape[0]) for group in feature_groups)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_feature_groups=feature_groups,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=model.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return [
        tokenizer.decode(row[expanded_prefix:], skip_special_tokens=True).strip()
        for row in output
    ]


def generate_blind_batch(
    *, model, tokenizer, records: list[dict], prompt_template: str,
    max_new_tokens: int, device,
) -> list[str]:
    prompts = [prompt_template.format(image="", question=record["question"]) for record in records]
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    tokenizer.padding_side = old_padding
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    output = model.language_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=model.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prefix = input_ids.shape[1]
    return [tokenizer.decode(row[prefix:], skip_special_tokens=True).strip() for row in output]


def primary_score(scored: dict) -> tuple[str, float]:
    if "exact_match" in scored:
        return "exact_match", float(scored["exact_match"])
    if "soft_vqa" in scored:
        return "soft_vqa", float(scored["soft_vqa"])
    if "anls" in scored:
        return "anls", float(scored["anls"])
    if "token_f1" in scored:
        return "token_f1", float(scored["token_f1"])
    if "grounding" in scored:
        return "accuracy", float(scored["grounding"]["correct"])
    raise ValueError(f"score payload has no primary metric: {scored}")


def scoring_record_for_condition(
    record: dict,
    condition: str,
    pair_index: dict[str, dict[str, str]] | None,
) -> dict:
    """返回视觉干预对应的预注册期望答案。"""

    if condition != "paired_counterfactual_image":
        return record
    if pair_index is None:
        raise ValueError("paired counterfactual scoring needs a pair index")
    scored = dict(record)
    scored["authoritative_answers"] = record.get("answers")
    scored["answers"] = [pair_index[str(record["id"])]["counterfactual_answer"]]
    return scored


def scored_row(
    record: dict,
    prediction: str | None,
    *, checkpoint: dict,
    dataset: str,
    condition: str,
    visual_source_id: str | None,
    wall_seconds: float,
    failure: dict | None = None,
) -> dict:
    actual_prediction = prediction or ""
    scores = score_record(actual_prediction, record)
    metric_name, value = primary_score(scores)
    row = {
        "checkpoint": checkpoint["id"],
        "optimizer_steps": checkpoint["optimizer_steps"],
        "examples_seen": checkpoint["examples_seen"],
        "dataset": dataset,
        "benchmark": record.get("benchmark"),
        "condition": condition,
        "visual_source_id": visual_source_id,
        "id": str(record["id"]),
        "pair_id": record.get("pair_id"),
        "task": record.get("task"),
        "question": record["question"],
        "image": record.get("image"),
        "image_sha256": record.get("image_sha256"),
        "answers": record.get("answers"),
        "authoritative_answers": record.get("authoritative_answers", record.get("answers")),
        "prediction": prediction,
        "normalized_prediction": normalize_answer(actual_prediction),
        "metric": metric_name,
        "correct": bool(value == 1.0),
        "score": value,
        "scores": scores,
        "wall_seconds": wall_seconds,
        "failure": failure,
    }
    for key in ("gt_box", "gt_point"):
        if record.get(key) is not None:
            row[key] = record[key]
    return row


def _cache_getter(primary: FeatureCache, control: FeatureCache | None, *, device, dtype):
    def get(sample_id: str):
        try:
            return primary.get(sample_id, device=device, dtype=dtype)
        except KeyError:
            if control is None:
                raise
            return control.get(sample_id, device=device, dtype=dtype)
    return get


def generate_condition(
    *,
    model,
    tokenizer,
    records: list[dict],
    condition: str,
    checkpoint: dict,
    dataset_name: str,
    cache_get,
    controls: dict[str, dict] | None,
    auxiliary_get,
    pair_index: dict[str, dict[str, str]] | None,
    split: str,
    batch_size: int,
    placeholder_token_id: int,
    prompt_template: str,
    max_new_tokens: int,
    device,
    raw_stream,
    failure_stream,
) -> list[dict]:
    output_rows: list[dict] = []

    def run_batch(batch: list[dict]) -> tuple[list[str], list[str | None]]:
        if condition == "blind":
            predictions = generate_blind_batch(
                model=model,
                tokenizer=tokenizer,
                records=batch,
                prompt_template=prompt_template,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            return predictions, [None] * len(batch)
        groups: list[torch.Tensor] = []
        source_ids: list[str] = []
        for record in batch:
            sample_id = str(record["id"])
            control = controls[str(record["id"])] if controls is not None else {}
            if condition == "paired_counterfactual_image":
                if pair_index is None:
                    raise ValueError("paired counterfactual condition needs a pair index")
                source_id = pair_index[sample_id]["paired_image_id"]
                resolved = cache_get(source_id)
            elif condition == "background_matched_aux":
                if auxiliary_get is None:
                    raise ValueError("background condition needs an auxiliary feature cache")
                source_id = sample_id
                resolved = auxiliary_get(sample_id)
            else:
                if condition == "blank":
                    source_id = f"control:{split}:blank"
                elif condition == "same_image":
                    source_id = f"control:{split}:same"
                elif condition == "shuffled_image":
                    source_id = str(control["shuffled_image_id"])
                else:
                    source_id = sample_id
                resolved = resolve_control_features(
                    condition,
                    sample_id,
                    split,
                    control,
                    cache_get,
                )
            if resolved is None or len(resolved) != 1:
                raise ValueError("trajectory runner expects exactly one image group per record")
            groups.extend(resolved)
            source_ids.append(source_id)
        predictions = generate_visual_batch(
            model=model,
            tokenizer=tokenizer,
            records=batch,
            feature_groups=groups,
            placeholder_token_id=placeholder_token_id,
            prompt_template=prompt_template,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        return predictions, source_ids

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        batch_started = time.time()
        try:
            predictions, source_ids = run_batch(batch)
            if len(predictions) != len(batch):
                raise ValueError("generation output batch cardinality mismatch")
            elapsed = (time.time() - batch_started) / len(batch)
            batch_rows = [
                scored_row(
                    scoring_record_for_condition(record, condition, pair_index),
                    prediction,
                    checkpoint=checkpoint,
                    dataset=dataset_name,
                    condition=condition,
                    visual_source_id=source_id,
                    wall_seconds=elapsed,
                )
                for record, prediction, source_id in zip(batch, predictions, source_ids)
            ]
        except Exception as batch_error:
            batch_failure = {
                "scope": "batch",
                "checkpoint": checkpoint["id"],
                "dataset": dataset_name,
                "condition": condition,
                "start": start,
                "count": len(batch),
                "error_type": type(batch_error).__name__,
                "error": str(batch_error),
                "traceback": traceback.format_exc(),
                "fallback": "single-sample generation",
            }
            failure_stream.write(json.dumps(batch_failure, ensure_ascii=False) + "\n")
            failure_stream.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch_rows = []
            for record in batch:
                sample_started = time.time()
                try:
                    predictions, source_ids = run_batch([record])
                    prediction = predictions[0]
                    row = scored_row(
                        scoring_record_for_condition(record, condition, pair_index),
                        prediction,
                        checkpoint=checkpoint,
                        dataset=dataset_name,
                        condition=condition,
                        visual_source_id=source_ids[0],
                        wall_seconds=time.time() - sample_started,
                    )
                except Exception as sample_error:
                    failure = {
                        "scope": "sample",
                        "checkpoint": checkpoint["id"],
                        "dataset": dataset_name,
                        "condition": condition,
                        "id": str(record["id"]),
                        "error_type": type(sample_error).__name__,
                        "error": str(sample_error),
                        "traceback": traceback.format_exc(),
                    }
                    failure_stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    failure_stream.flush()
                    row = scored_row(
                        scoring_record_for_condition(record, condition, pair_index),
                        None,
                        checkpoint=checkpoint,
                        dataset=dataset_name,
                        condition=condition,
                        visual_source_id=None,
                        wall_seconds=time.time() - sample_started,
                        failure=failure,
                    )
                batch_rows.append(row)
        for row in batch_rows:
            raw_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        raw_stream.flush()
        output_rows.extend(batch_rows)
        print(
            f"[{checkpoint['id']}] {dataset_name}/{condition} "
            f"{min(start + len(batch), len(records))}/{len(records)}",
            flush=True,
        )
    return output_rows


def benchmark_summary(rows: list[dict]) -> dict:
    score_payloads = [row["scores"] for row in rows]
    result = summarize(score_payloads)
    failures = sum(bool(row.get("failure")) for row in rows)
    result["failures"] = {"numerator": failures, "denominator": len(rows), "value": failures / len(rows) if rows else None}
    if rows:
        result["primary_metric"] = rows[0]["metric"]
        result["primary_numerator"] = sum(float(row["score"]) for row in rows)
        result["primary_denominator"] = len(rows)
        result["primary_value"] = result["primary_numerator"] / len(rows)
    grounding = [
        row["scores"]["grounding"]
        for row in rows
        if "grounding" in row["scores"]
    ]
    if grounding:
        parsed = sum(bool(row["parse_ok"]) for row in grounding)
        coordinates = Counter(
            tuple(float(value) for value in row["prediction_point"])
            for row in grounding
            if row.get("prediction_point") is not None
        )
        common_coordinate, common_count = (
            coordinates.most_common(1)[0] if coordinates else (None, 0)
        )
        result["coordinate_parse"] = {
            "numerator": parsed,
            "denominator": len(grounding),
            "value": parsed / len(grounding),
        }
        result["most_common_coordinate"] = (
            list(common_coordinate) if common_coordinate is not None else None
        )
        result["most_common_coordinate_ratio"] = {
            "numerator": common_count,
            "denominator": len(grounding),
            "value": common_count / len(grounding),
        }
        result["most_common_coordinate_among_parsed_ratio"] = {
            "numerator": common_count,
            "denominator": parsed,
            "value": common_count / parsed if parsed else None,
        }
    return result


def summarize_dataset(rows: list[dict], dataset_config: dict) -> dict:
    if dataset_config["kind"] == "synthetic":
        return summarize_synthetic_rows(rows)
    group_key = dataset_config.get("group_by", "benchmark")
    groups = sorted({str(row[group_key]) for row in rows})
    return {
        "count": len(rows),
        "by_group": {
            group: benchmark_summary([row for row in rows if str(row[group_key]) == group])
            for group in groups
        },
    }


def add_vision_gaps(dataset_summary: dict, dataset_config: dict) -> None:
    conditions = dataset_summary["conditions"]
    if "vision" not in conditions or "blind" not in conditions:
        return
    if dataset_config["kind"] == "synthetic":
        vision = conditions["vision"]
        blind = conditions["blind"]
        dataset_summary["vision_minus_blind"] = {
            "overall": vision["accuracy"]["value"] - blind["accuracy"]["value"],
            "by_task": {
                task: vision["by_task"][task]["accuracy"]["value"]
                - blind["by_task"][task]["accuracy"]["value"]
                for task in vision["by_task"]
            },
        }
    else:
        dataset_summary["vision_minus_blind"] = {
            group: conditions["vision"]["by_group"][group]["primary_value"]
            - conditions["blind"]["by_group"][group]["primary_value"]
            for group in conditions["vision"]["by_group"]
        }


def checkpoint_training_metrics(checkpoint: dict) -> dict:
    if checkpoint["kind"] == "random":
        return {"last_train_loss": None, "mean_last_50_train_loss": None}
    history_path = Path(checkpoint["path"]) / "history.json"
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    history = payload["history"]
    trailing = [float(row["loss"]) for row in history[-50:]]
    return {
        "last_train_loss": float(history[-1]["loss"]),
        "mean_last_50_train_loss": sum(trailing) / len(trailing),
        "history_sha256": file_sha256(history_path),
    }


def teacher_forced_loss(model, tokenizer, prompt_template: str, placeholder_token_id: int, record: dict, answer: str, groups, device) -> float:
    input_ids, attention_mask = visual_prompt_batch(
        tokenizer, prompt_template, [str(record["question"])], placeholder_token_id, device
    )
    answer_ids = tokenizer.encode(
        " " + answer, return_tensors="pt", add_special_tokens=False
    ).to(device)
    combined = torch.cat([input_ids, answer_ids], dim=1)
    combined_mask = torch.cat([attention_mask, torch.ones_like(answer_ids)], dim=1)
    labels = combined.clone()
    labels[:, : input_ids.shape[1]] = -100
    outputs = model(
        input_ids=combined,
        attention_mask=combined_mask,
        image_feature_groups=groups,
        labels=labels,
    )
    value = float(outputs.loss.detach())
    if not math.isfinite(value):
        raise FloatingPointError(f"non-finite teacher-forced loss: {value}")
    return value


def run_shuffle_loss(
    *, model, tokenizer, records: list[dict], cache_get, checkpoint: dict,
    repeats: int, seed: int, prompt_template: str, placeholder_token_id: int,
    device, raw_stream,
) -> dict:
    mappings = [derangement_indices(len(records), seed=seed + repeat) for repeat in range(repeats)]
    rows = []
    consecutive_nonfinite = 0
    for index, record in enumerate(records):
        started = time.time()
        supervision = select_supervision(record["answers"], rule="canonical")
        try:
            true_loss = teacher_forced_loss(
                model, tokenizer, prompt_template, placeholder_token_id,
                record, supervision.selected_answer, cache_get(str(record["id"])), device,
            )
            shuffled_losses = [
                teacher_forced_loss(
                    model,
                    tokenizer,
                    prompt_template,
                    placeholder_token_id,
                    record,
                    supervision.selected_answer,
                    cache_get(str(records[mapping[index]]["id"])),
                    device,
                )
                for mapping in mappings
            ]
            mean_shuffled = sum(shuffled_losses) / len(shuffled_losses)
            row = {
                "checkpoint": checkpoint["id"],
                "optimizer_steps": checkpoint["optimizer_steps"],
                "examples_seen": checkpoint["examples_seen"],
                "id": str(record["id"]),
                "source": record.get("source"),
                "raw_answers": supervision.raw_answers,
                "canonical_answer": supervision.canonical_answer,
                "normalization_rule": supervision.normalization_rule,
                "true_loss": true_loss,
                "shuffled_losses": shuffled_losses,
                "mean_shuffled_loss": mean_shuffled,
                "delta": mean_shuffled - true_loss,
                "shuffle_repeats": repeats,
                "wall_seconds": time.time() - started,
                "failure": None,
            }
            consecutive_nonfinite = 0
        except FloatingPointError as error:
            consecutive_nonfinite += 1
            row = {
                "checkpoint": checkpoint["id"],
                "optimizer_steps": checkpoint["optimizer_steps"],
                "examples_seen": checkpoint["examples_seen"],
                "id": str(record["id"]),
                "source": record.get("source"),
                "raw_answers": supervision.raw_answers,
                "canonical_answer": supervision.canonical_answer,
                "normalization_rule": supervision.normalization_rule,
                "true_loss": None,
                "shuffled_losses": [],
                "mean_shuffled_loss": None,
                "delta": None,
                "shuffle_repeats": repeats,
                "wall_seconds": time.time() - started,
                "failure": {"error_type": type(error).__name__, "error": str(error)},
            }
            if consecutive_nonfinite >= 2:
                raw_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                raw_stream.flush()
                raise RuntimeError("stop condition: consecutive non-finite losses") from error
        raw_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        raw_stream.flush()
        rows.append(row)
        print(f"[{checkpoint['id']}] shuffle-loss {index + 1}/{len(records)}", flush=True)

    valid = [row for row in rows if row["failure"] is None]
    def group_summary(group_rows: list[dict]) -> dict:
        denominator = len(group_rows)
        valid_rows = [row for row in group_rows if row["failure"] is None]
        repeat_means = [
            statistics.fmean(float(row["shuffled_losses"][repeat]) for row in valid_rows)
            for repeat in range(repeats)
        ] if valid_rows else []
        mean_true = (
            statistics.fmean(float(row["true_loss"]) for row in valid_rows)
            if valid_rows
            else None
        )
        repeat_deltas = [value - mean_true for value in repeat_means] if mean_true is not None else []
        return {
            "records": denominator,
            "failures": sum(row["failure"] is not None for row in group_rows),
            "sum_true_loss": sum(row["true_loss"] for row in group_rows if row["true_loss"] is not None),
            "sum_shuffled_loss": sum(row["mean_shuffled_loss"] for row in group_rows if row["mean_shuffled_loss"] is not None),
            "sum_delta": sum(row["delta"] for row in group_rows if row["delta"] is not None),
            "mean_true_loss": (sum(row["true_loss"] for row in group_rows if row["true_loss"] is not None) / denominator if denominator else None),
            "mean_shuffled_loss": (sum(row["mean_shuffled_loss"] for row in group_rows if row["mean_shuffled_loss"] is not None) / denominator if denominator else None),
            "mean_delta": (sum(row["delta"] for row in group_rows if row["delta"] is not None) / denominator if denominator else None),
            "repeat_mean_shuffled_losses": repeat_means,
            "repeat_shuffle_deltas": repeat_deltas,
            "shuffle_delta_std": statistics.pstdev(repeat_deltas) if repeat_deltas else None,
        }
    sources = sorted({str(row.get("source") or "unknown") for row in rows})
    return {
        **group_summary(rows),
        "shuffle_repeats": repeats,
        "valid_records": len(valid),
        "by_source": {
            source: group_summary([row for row in rows if str(row.get("source") or "unknown") == source])
            for source in sources
        },
    }


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_checkpoint_state(model, checkpoint: dict, random_state: dict[str, torch.Tensor], device) -> str:
    if checkpoint["kind"] == "random":
        model.projector.load_state_dict(random_state, strict=True)
        return state_sha256(random_state)
    weights = Path(checkpoint["path"]) / "projector.safetensors"
    state = load_file(str(weights), device="cpu")
    model.projector.load_state_dict(state, strict=True)
    model.projector.to(device=device, dtype=next(model.language_model.parameters()).dtype)
    return file_sha256(weights)


def write_charts(summary: dict, output: Path) -> None:
    with (output / "synthetic_curve.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["checkpoint", "optimizer_steps", "examples_seen", "condition", "task", "accuracy_numerator", "accuracy_denominator", "accuracy", "paired_numerator", "paired_denominator", "paired_accuracy", "answer_flip_numerator", "answer_flip_denominator", "answer_flip_accuracy", "prediction_flip_rate"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for checkpoint_id, checkpoint in summary["checkpoints"].items():
            dataset = checkpoint["datasets"].get("synthetic")
            if not dataset:
                continue
            for condition, condition_summary in dataset["conditions"].items():
                for task, task_summary in {"overall": condition_summary, **condition_summary["by_task"]}.items():
                    writer.writerow({
                        "checkpoint": checkpoint_id,
                        "optimizer_steps": checkpoint["optimizer_steps"],
                        "examples_seen": checkpoint["examples_seen"],
                        "condition": condition,
                        "task": task,
                        "accuracy_numerator": task_summary["accuracy"]["numerator"],
                        "accuracy_denominator": task_summary["accuracy"]["denominator"],
                        "accuracy": task_summary["accuracy"]["value"],
                        "paired_numerator": task_summary["paired_accuracy"]["numerator"],
                        "paired_denominator": task_summary["paired_accuracy"]["denominator"],
                        "paired_accuracy": task_summary["paired_accuracy"]["value"],
                        "answer_flip_numerator": task_summary["answer_flip_accuracy"]["numerator"],
                        "answer_flip_denominator": task_summary["answer_flip_accuracy"]["denominator"],
                        "answer_flip_accuracy": task_summary["answer_flip_accuracy"]["value"],
                        "prediction_flip_rate": task_summary["prediction_flip_rate"]["value"],
                    })
    with (output / "benchmark_curve.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "checkpoint", "optimizer_steps", "examples_seen", "condition", "benchmark",
            "metric", "raw_score_sum", "denominator", "score", "parse_numerator",
            "parse_denominator", "parse_rate", "most_common_coordinate",
            "collapse_numerator", "collapse_denominator", "collapse_rate",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for checkpoint_id, checkpoint in summary["checkpoints"].items():
            dataset = checkpoint["datasets"].get("benchmarks")
            if not dataset:
                continue
            for condition, condition_summary in dataset["conditions"].items():
                for benchmark, values in condition_summary["by_group"].items():
                    writer.writerow({
                        "checkpoint": checkpoint_id,
                        "optimizer_steps": checkpoint["optimizer_steps"],
                        "examples_seen": checkpoint["examples_seen"],
                        "condition": condition,
                        "benchmark": benchmark,
                        "metric": values["primary_metric"],
                        "raw_score_sum": values["primary_numerator"],
                        "denominator": values["primary_denominator"],
                        "score": values["primary_value"],
                        "parse_numerator": values.get("coordinate_parse", {}).get("numerator"),
                        "parse_denominator": values.get("coordinate_parse", {}).get("denominator"),
                        "parse_rate": values.get("coordinate_parse", {}).get("value"),
                        "most_common_coordinate": values.get("most_common_coordinate"),
                        "collapse_numerator": values.get("most_common_coordinate_ratio", {}).get("numerator"),
                        "collapse_denominator": values.get("most_common_coordinate_ratio", {}).get("denominator"),
                        "collapse_rate": values.get("most_common_coordinate_ratio", {}).get("value"),
                    })
    with (output / "shuffle_loss_curve.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "checkpoint", "optimizer_steps", "examples_seen", "source", "records",
            "shuffle_repeats", "mean_true_loss", "mean_shuffled_loss", "mean_delta",
            "shuffle_delta_std",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for checkpoint_id, checkpoint in summary["checkpoints"].items():
            values = checkpoint.get("shuffle_loss")
            if values:
                for source, source_values in {
                    "overall": values,
                    **values["by_source"],
                }.items():
                    writer.writerow({
                        "checkpoint": checkpoint_id,
                        "optimizer_steps": checkpoint["optimizer_steps"],
                        "examples_seen": checkpoint["examples_seen"],
                        "source": source,
                        "records": source_values["records"],
                        "shuffle_repeats": values["shuffle_repeats"],
                        "mean_true_loss": source_values["mean_true_loss"],
                        "mean_shuffled_loss": source_values["mean_shuffled_loss"],
                        "mean_delta": source_values["mean_delta"],
                        "shuffle_delta_std": source_values["shuffle_delta_std"],
                    })


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite trajectory run: {args.out}")
    args.out.mkdir(parents=True)
    config = apply_screening_overrides(
        json.loads(args.config.read_text(encoding="utf-8")),
        limit=args.limit,
        checkpoint_ids=args.checkpoint_ids,
    )
    write_json(args.out / "CONFIG.json", config)
    started = time.time()
    seed = int(config["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(config.get("device", "cuda"))
    dtype = getattr(torch, config.get("dtype", "bfloat16"))
    if device.type == "cuda":
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

    projector_config_source = config.get("projector_config_source")
    if projector_config_source is None:
        projector_config_source = next(
            checkpoint["path"] for checkpoint in config["checkpoints"] if checkpoint["kind"] == "trained"
        )
    projector_config_source = Path(projector_config_source)
    projector_config = ProjectorConfig(
        **json.loads((projector_config_source / "projector_config.json").read_text(encoding="utf-8"))
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
        checkpoint for checkpoint in config["checkpoints"] if checkpoint["kind"] == "random"
    )
    random_state = matched_random_state(
        projector_config, int(random_checkpoint["random_seed"])
    )

    datasets = []
    dataset_provenance: dict[str, dict] = {}
    for dataset_config in config["datasets"]:
        records = load_records(Path(dataset_config["data"]))
        if dataset_config.get("limit") is not None:
            records = records[: int(dataset_config["limit"])]
        expected = dataset_config.get("expected_records")
        if expected is not None and len(records) != int(expected):
            raise ValueError(f"{dataset_config['name']} expected {expected} records, got {len(records)}")
        primary_cache = FeatureCache(dataset_config["feature_cache"])
        control_cache = FeatureCache(dataset_config["control_cache"]) if dataset_config.get("control_cache") else None
        controls = load_controls(Path(dataset_config["controls"])) if dataset_config.get("controls") else None
        if controls is not None:
            missing = [str(record["id"]) for record in records if str(record["id"]) not in controls]
            if missing:
                raise ValueError(f"control assignments missing ids: {missing[:3]}")
        auxiliary_cache = (
            FeatureCache(dataset_config["background_aux_feature_cache"])
            if dataset_config.get("background_aux_feature_cache")
            else None
        )
        all_conditions = {
            str(condition) for condition in dataset_config["conditions"]
        } | {
            str(condition)
            for extension in dataset_config.get("checkpoint_condition_extensions", [])
            for condition in extension["conditions"]
        }
        pair_index = (
            build_pair_index(records)
            if "paired_counterfactual_image" in all_conditions
            else None
        )
        datasets.append(
            (
                dataset_config,
                records,
                primary_cache,
                control_cache,
                controls,
                auxiliary_cache,
                pair_index,
            )
        )
        dataset_provenance[str(dataset_config["name"])] = {
            "data_sha256": file_sha256(Path(dataset_config["data"])),
            "logical_dataset_sha256": dataset_config.get("logical_dataset_sha256"),
            "authoritative_selection_sha256": (
                file_sha256(Path(dataset_config["authoritative_selection_data"]))
                if dataset_config.get("authoritative_selection_data")
                else None
            ),
            "generation_selection_manifest_sha256": (
                file_sha256(Path(dataset_config["generation_selection_manifest"]))
                if dataset_config.get("generation_selection_manifest")
                else None
            ),
            "records": len(records),
            "feature_cache_manifest_sha256": file_sha256(
                Path(dataset_config["feature_cache"]) / "MANIFEST.json"
            ),
            "controls_sha256": (
                file_sha256(Path(dataset_config["controls"]))
                if dataset_config.get("controls")
                else None
            ),
            "controls_manifest_sha256": (
                file_sha256(Path(dataset_config["controls"]).parent / "MANIFEST.json")
                if dataset_config.get("controls")
                and (Path(dataset_config["controls"]).parent / "MANIFEST.json").exists()
                else None
            ),
            "control_cache_manifest_sha256": (
                file_sha256(Path(dataset_config["control_cache"]) / "MANIFEST.json")
                if dataset_config.get("control_cache")
                else None
            ),
            "background_aux_manifest_sha256": (
                file_sha256(Path(dataset_config["background_aux_data"]).parent / "MANIFEST.json")
                if dataset_config.get("background_aux_data")
                else None
            ),
            "background_aux_cache_manifest_sha256": (
                file_sha256(
                    Path(dataset_config["background_aux_feature_cache"]) / "MANIFEST.json"
                )
                if dataset_config.get("background_aux_feature_cache")
                else None
            ),
            "max_image_side": dataset_config.get("max_image_side"),
            "max_new_tokens": int(dataset_config["max_new_tokens"]),
            "conditions_by_checkpoint": {
                str(checkpoint["id"]): configured_conditions(
                    dataset_config, str(checkpoint["id"])
                )
                for checkpoint in config["checkpoints"]
            },
        }

    heldout_config = config["heldout_shuffle_loss"]
    heldout_records = load_records(Path(heldout_config["data"]))
    if heldout_config.get("limit") is not None:
        heldout_records = heldout_records[: int(heldout_config["limit"])]
    heldout_cache = FeatureCache(heldout_config["feature_cache"])
    if len(heldout_records) != int(heldout_config["expected_records"]):
        raise ValueError("held-out record count drift")

    summary: dict[str, Any] = {
        "status": "running",
        "format_version": "checkpoint-trajectory-v1",
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
            "language_parameters": sum(parameter.numel() for parameter in language_model.parameters()),
            "projector_parameters": sum(parameter.numel() for parameter in projector.parameters()),
            "eval_batch_size": int(config["eval_batch_size"]),
            "actual_batched_forward": True,
            "prompt_template": str(config["prompt_template"]),
            "scorer_version": "repository-metrics-v1-fixed-short-answer",
            "final_half_scored": False,
            "config_sha256": file_sha256(args.out / "CONFIG.json"),
        },
        "checkpoints": {},
        "datasets": dataset_provenance,
        "heldout_shuffle_loss": {
            "data_sha256": file_sha256(Path(heldout_config["data"])),
            "feature_cache_manifest_sha256": file_sha256(
                Path(heldout_config["feature_cache"]) / "MANIFEST.json"
            ),
            "records": len(heldout_records),
            "shuffle_repeats": int(heldout_config["shuffle_repeats"]),
            "max_image_side": int(heldout_config["max_image_side"]),
        },
        "aliases": config.get("aliases", []),
        "failures": 0,
    }
    raw_path = args.out / "records.jsonl"
    shuffle_path = args.out / "shuffle_loss_records.jsonl"
    failures_path = args.out / "failures.jsonl"
    with raw_path.open("w", encoding="utf-8", newline="\n") as raw_stream, shuffle_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as shuffle_stream, failures_path.open("w", encoding="utf-8", newline="\n") as failure_stream:
        with torch.inference_mode():
            for checkpoint_source in config["checkpoints"]:
                checkpoint = deepcopy(checkpoint_source)
                checkpoint["state_sha256"] = load_checkpoint_state(
                    model, checkpoint, random_state, device
                )
                checkpoint.update(checkpoint_training_metrics(checkpoint))
                checkpoint_summary = {
                    key: checkpoint[key]
                    for key in (
                        "id", "kind", "optimizer_steps", "examples_seen", "state_sha256",
                        "last_train_loss", "mean_last_50_train_loss",
                    )
                }
                checkpoint_summary["effective_epochs"] = float(
                    checkpoint.get("effective_epochs", 0.0)
                )
                checkpoint_summary["initialization_claim"] = checkpoint.get(
                    "initialization_claim"
                )
                checkpoint_summary["source_path"] = checkpoint.get("path")
                checkpoint_summary["projector_config_sha256"] = file_sha256(
                    projector_config_source / "projector_config.json"
                )
                checkpoint_summary["datasets"] = {}
                for (
                    dataset_config,
                    records,
                    primary_cache,
                    control_cache,
                    controls,
                    auxiliary_cache,
                    pair_index,
                ) in datasets:
                    dataset_name = dataset_config["name"]
                    dataset_summary = {
                        "kind": dataset_config["kind"],
                        "records": len(records),
                        "records_manifest_sha256": records_manifest_sha256(records),
                        "conditions": {},
                    }
                    getter = _cache_getter(
                        primary_cache,
                        control_cache,
                        device=device,
                        dtype=next(model.projector.parameters()).dtype,
                    )
                    auxiliary_getter = (
                        _cache_getter(
                            auxiliary_cache,
                            None,
                            device=device,
                            dtype=next(model.projector.parameters()).dtype,
                        )
                        if auxiliary_cache is not None
                        else None
                    )
                    for condition in configured_conditions(
                        dataset_config, str(checkpoint["id"])
                    ):
                        rows = generate_condition(
                            model=model,
                            tokenizer=tokenizer,
                            records=records,
                            condition=condition,
                            checkpoint=checkpoint,
                            dataset_name=dataset_name,
                            cache_get=getter,
                            controls=controls,
                            auxiliary_get=auxiliary_getter,
                            pair_index=pair_index,
                            split=dataset_config.get("split", "selection"),
                            batch_size=int(dataset_config.get("batch_size", config["eval_batch_size"])),
                            placeholder_token_id=int(config["placeholder_token_id"]),
                            prompt_template=str(config["prompt_template"]),
                            max_new_tokens=int(dataset_config["max_new_tokens"]),
                            device=device,
                            raw_stream=raw_stream,
                            failure_stream=failure_stream,
                        )
                        dataset_summary["conditions"][condition] = summarize_dataset(rows, dataset_config)
                        summary["failures"] += sum(bool(row.get("failure")) for row in rows)
                        checkpoint_summary["datasets"][dataset_name] = dataset_summary
                        summary["checkpoints"][checkpoint["id"]] = checkpoint_summary
                        write_json(args.out / "SUMMARY.partial.json", summary)
                    add_vision_gaps(dataset_summary, dataset_config)
                    checkpoint_summary["datasets"][dataset_name] = dataset_summary

                heldout_get = _cache_getter(
                    heldout_cache,
                    None,
                    device=device,
                    dtype=next(model.projector.parameters()).dtype,
                )
                checkpoint_summary["shuffle_loss"] = run_shuffle_loss(
                    model=model,
                    tokenizer=tokenizer,
                    records=heldout_records,
                    cache_get=heldout_get,
                    checkpoint=checkpoint,
                    repeats=int(heldout_config["shuffle_repeats"]),
                    seed=int(heldout_config["seed"]),
                    prompt_template=str(config["prompt_template"]),
                    placeholder_token_id=int(config["placeholder_token_id"]),
                    device=device,
                    raw_stream=shuffle_stream,
                )
                summary["checkpoints"][checkpoint["id"]] = checkpoint_summary
                write_json(args.out / "SUMMARY.partial.json", summary)

    for alias in config.get("aliases", []):
        source = str(alias["source"])
        alias_id = str(alias["id"])
        if source not in summary["checkpoints"]:
            raise ValueError(f"alias source is absent: {source}")
        aliased = deepcopy(summary["checkpoints"][source])
        aliased["id"] = alias_id
        aliased["alias_of"] = source
        summary["checkpoints"][alias_id] = aliased

    failure_events = sum(1 for line in failures_path.read_text(encoding="utf-8").splitlines() if line)
    summary["failure_events"] = failure_events
    if summary["failures"]:
        summary["status"] = "completed_with_sample_failures"
    elif failure_events:
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
        shuffle_path.name: {"bytes": shuffle_path.stat().st_size, "sha256": file_sha256(shuffle_path)},
        failures_path.name: {"bytes": failures_path.stat().st_size, "sha256": file_sha256(failures_path)},
    }
    write_charts(summary, args.out)
    for filename in ("synthetic_curve.csv", "benchmark_curve.csv", "shuffle_loss_curve.csv"):
        path = args.out / filename
        summary["raw_files"][filename] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    write_json(args.out / "SUMMARY.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "wall_seconds": summary["metadata"]["wall_seconds"],
        "peak_gpu_memory_bytes": summary["metadata"]["peak_gpu_memory_bytes"],
        "failures": summary["failures"],
        "summary": str(args.out / "SUMMARY.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
