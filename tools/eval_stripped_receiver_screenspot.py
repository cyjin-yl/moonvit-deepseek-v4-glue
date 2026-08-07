#!/usr/bin/env python3
"""对 Qwen2.5-7B stripped receiver-prior 运行最小 GLM-format ScreenSpot。"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from moonvit_glue import FeatureCache
from moonvit_glue.grounding_contract import score_click_prediction, summarize_click_scores
from moonvit_glue.projector import PatchMergerProjector, seeded_projector
from moonvit_glue.token_selection import select_visual_tokens
from probe_stripped_receiver import FixedGroupedReceiverAdapter
from generate_stripped_receiver_probe import _prompt_inputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--feature-cache", type=Path, required=True)
    p.add_argument("--screenspot-manifest", type=Path, required=True)
    p.add_argument("--projector-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-visual-tokens", type=int, default=256)
    p.add_argument("--token-selection", choices=("prefix", "uniform", "mean_pool"), default="prefix")
    p.add_argument("--projector-scale", type=float, default=1.0)
    p.add_argument("--random-seed", type=int, default=20260806)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--sample-indices", default=None, help="只跑指定的冻结样本下标，逗号分隔；用于失败后的最小重试")
    p.add_argument("--bootstrap-seed", type=int, default=20260805)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    return p.parse_args()


def _bootstrap_delta(rows_a, rows_b, key, *, lower_is_better=False, seed=20260805, samples=2000):
    rng = random.Random(seed)
    deltas = []
    for a, b in zip(rows_a, rows_b):
        av = float(a[key])
        bv = float(b[key])
        deltas.append((bv - av) if lower_is_better else (bv - av))
    means = []
    for _ in range(samples):
        draw = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        means.append(sum(draw) / len(draw))
    ordered = sorted(means)
    return {
        "mean": sum(deltas) / len(deltas),
        "ci95_lower": ordered[int(0.025 * (samples - 1))],
        "ci95_upper": ordered[int(0.975 * (samples - 1))],
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    manifest = json.loads(args.screenspot_manifest.read_text(encoding="utf-8"))
    samples = list(manifest["samples"])
    if args.sample_indices:
        indices = [int(part.strip()) for part in args.sample_indices.split(",") if part.strip()]
        if any(index < 0 or index >= len(samples) for index in indices):
            raise ValueError("sample-indices contains an out-of-range index")
        samples = [samples[index] for index in indices]
    mapping = {str(row["sample_id"]): str(row["shuffled_image_sample_id"]) for row in manifest["shuffled_image_control"]["mapping"]}
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    placeholder = int(config.get("image_token_id") or tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    model_class = AutoModelForCausalLM if config.get("model_type") == "qwen2" else AutoModelForImageTextToText
    model = model_class.from_pretrained(
        str(args.model_dir), dtype=dtype, local_files_only=True, low_cpu_mem_usage=True,
    ).to(device).eval()
    model.requires_grad_(False)
    projector = PatchMergerProjector.from_pretrained(args.projector_dir, device=device, dtype=dtype).eval()
    text_config = config.get("text_config") or config
    receiver_width = int(text_config.get("hidden_size") or model.config.hidden_size)
    receiver = FixedGroupedReceiverAdapter(4096, receiver_width, seed=20260806).to(device) if receiver_width != 4096 else None
    random_projector = seeded_projector(projector.config, seed=args.random_seed).to(device).eval()
    cache = FeatureCache(args.feature_cache)
    cache_records = {str(row["id"]): row for row in cache.manifest.get("records", [])}
    missing = [str(row["sample_id"]) for row in samples if str(row["sample_id"]) not in cache_records]
    if missing:
        raise ValueError(f"ScreenSpot samples missing from feature cache: {missing[:5]}")

    rows_by_condition = {name: [] for name in ("vision", "blind", "shuffled", "random_projector")}
    all_rows = []
    for index, sample in enumerate(samples):
        sample_id = str(sample["sample_id"])
        shuffled_id = mapping[sample_id]
        feature = select_visual_tokens(cache.get(sample_id, device=device, dtype=torch.float32)[0], args.max_visual_tokens, args.token_selection)
        shuffled = select_visual_tokens(cache.get(shuffled_id, device=device, dtype=torch.float32)[0], args.max_visual_tokens, args.token_selection)
        for condition, current_feature, current_projector in (
            ("vision", feature, projector),
            ("blind", None, projector),
            ("shuffled", shuffled, projector),
            ("random_projector", feature, random_projector),
        ):
            with torch.no_grad():
                embeds, attention, position_ids = _prompt_inputs(
                    model=model, tokenizer=tokenizer, sample={"question": sample["instruction"]},
                    placeholder=placeholder, feature=current_feature, projector=current_projector,
                    receiver=receiver, prompt_route="grounding", scale=args.projector_scale, device=device,
                )
                generated = model.generate(
                    inputs_embeds=embeds, attention_mask=attention, position_ids=position_ids,
                    do_sample=False, max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id, use_cache=True,
                )
            prediction = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            scored = score_click_prediction(
                sample_id=sample_id, prediction=prediction, target_box=sample["bbox_999_xyxy"]
            )
            scored.update({"sample_index": index, "condition": condition, "shuffled_sample_id": shuffled_id})
            rows_by_condition[condition].append(scored)
            all_rows.append(scored)

    summary = {
        "schema_version": "stripped-receiver-screenspot-v1",
        "status": "diagnostic_only",
        "dataset": manifest.get("name", "screenspot_glm50_v1"),
        "sample_count": len(samples),
        "conditions": list(rows_by_condition),
        "max_visual_tokens": args.max_visual_tokens,
        "token_selection": args.token_selection,
        "projector_scale": args.projector_scale,
        "community_contract": {
            "global_batch": 64,
            "learning_rate": 5e-4,
            "max_visual_tokens": 256,
            "token_selection": "prefix",
            "do_sample": False,
            "temperature": 0.0,
            "bootstrap_samples": args.bootstrap_samples,
        },
        "max_new_tokens": args.max_new_tokens,
        "generation": {"do_sample": False, "temperature": 0.0, "prompt_route": "grounding"},
        "condition_summaries": {k: summarize_click_scores(v) for k, v in rows_by_condition.items()},
        "paired": {
            "vision_minus_blind_click_in_box": _bootstrap_delta(rows_by_condition["blind"], rows_by_condition["vision"], "click_in_box", seed=args.bootstrap_seed, samples=args.bootstrap_samples),
            "vision_minus_shuffled_click_in_box": _bootstrap_delta(rows_by_condition["shuffled"], rows_by_condition["vision"], "click_in_box", seed=args.bootstrap_seed, samples=args.bootstrap_samples),
            "vision_minus_blind_center_l2": _bootstrap_delta(rows_by_condition["blind"], rows_by_condition["vision"], "center_l2_penalized", lower_is_better=True, seed=args.bootstrap_seed, samples=args.bootstrap_samples),
            "vision_minus_shuffled_center_l2": _bootstrap_delta(rows_by_condition["shuffled"], rows_by_condition["vision"], "center_l2_penalized", lower_is_better=True, seed=args.bootstrap_seed, samples=args.bootstrap_samples),
        },
        "rows": all_rows,
        "capability_claim_allowed": False,
        "deepseek_transfer_label": "transferable_with_runtime_validation",
        "wall_seconds": time.perf_counter() - started,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "generation_rows.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows), encoding="utf-8")
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
