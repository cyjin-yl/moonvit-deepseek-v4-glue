#!/usr/bin/env python3
"""多样样本的 stripped-native receiver-prior 小筛选。

只做冻结 step0 projector 的 teacher-forced forward，另外评估同结构随机
projector。原生 Qwen3.5 visual 路径始终不传 pixel_values；Qwen2.5 纯文本
模型也可复用这个工具。结果用于诊断 token 顺序、receiver prior 和随机基线，
不进入 ScreenSpot/Qwen 社区排行榜。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_stripped_qwen35 import (  # noqa: E402
    FixedGroupedReceiverAdapter,
    answer_logprob,
    build_inputs,
    expanded_forward,
)

from moonvit_glue import FeatureCache  # noqa: E402
from moonvit_glue.merge import expand_image_placeholders  # noqa: E402
from moonvit_glue.projector import PatchMergerProjector, seeded_projector  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--weight-manifest", type=Path, required=True)
    p.add_argument("--feature-cache", type=Path, required=True)
    p.add_argument("--projector-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sample-indices", default="0,1,2,3,4,5,6,7")
    p.add_argument("--max-visual-tokens", type=int, default=16)
    p.add_argument("--random-seed", type=int, default=20260806)
    p.add_argument("--qwen-mrope", action="store_true", help="Qwen3.5-only positional diagnostic; not transferable")
    return p.parse_args()


def _grid_for_token_count(token_count: int, device: torch.device) -> torch.Tensor:
    target = int(token_count) * 4
    candidates = []
    for height in range(2, target + 1, 2):
        if target % height:
            continue
        width = target // height
        if width % 2 == 0:
            candidates.append((abs(width - height), height, width))
    if not candidates:
        raise ValueError(f"cannot construct even Qwen3.5 grid for {token_count} tokens")
    _, height, width = min(candidates)
    return torch.tensor([[1, height, width]], dtype=torch.long, device=device)


def eval_case(*, model, projector, receiver, tokenizer, sample, feature, placeholder, device, qwen_mrope=False):
    if qwen_mrope and feature is not None and hasattr(getattr(model, "model", None), "compute_3d_position_ids"):
        supervision, input_ids, labels, attention = build_inputs(
            tokenizer=tokenizer, features=feature, sample=sample,
            placeholder_token_id=placeholder, device=device,
        )
        text_embeddings = model.get_input_embeddings()(input_ids)
        projected = projector([feature])[0]
        if receiver is not None:
            projected = receiver(projected)
        merged = expand_image_placeholders(
            input_ids=input_ids, text_embeddings=text_embeddings,
            image_embeddings=[projected], placeholder_token_id=placeholder,
            attention_mask=attention, labels=labels,
        )
        mm_token_type_ids = torch.zeros_like(merged.routing_input_ids, dtype=torch.int32)
        mm_token_type_ids = mm_token_type_ids.masked_fill(
            merged.routing_input_ids.eq(placeholder) & merged.attention_mask.bool(), 1
        )
        rope_result = model.model.compute_3d_position_ids(
            input_ids=merged.routing_input_ids, inputs_embeds=merged.inputs_embeds,
            image_grid_thw=_grid_for_token_count(feature.shape[0], device),
            mm_token_type_ids=mm_token_type_ids, attention_mask=merged.attention_mask,
        )
        position_ids = rope_result[0] if isinstance(rope_result, tuple) else rope_result
        with torch.no_grad():
            outputs = model(
                inputs_embeds=merged.inputs_embeds, attention_mask=merged.attention_mask,
                position_ids=position_ids, labels=merged.labels, use_cache=False,
            )
        return answer_logprob(outputs.logits, merged.labels)
    with torch.no_grad():
        _, outputs, labels = expanded_forward(
            model=model, projector=projector, receiver=receiver, features=feature,
            tokenizer=tokenizer, sample=sample, placeholder_token_id=placeholder,
            device=device,
        )
    return answer_logprob(outputs.logits, labels)


def main() -> None:
    args = parse_args()
    if args.max_visual_tokens <= 0:
        raise ValueError("max-visual-tokens must be positive")
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    weight_manifest = json.loads(args.weight_manifest.read_text(encoding="utf-8"))
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    placeholder = int(config.get("image_token_id") or tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    model_class = AutoModelForCausalLM if config.get("model_type") == "qwen2" else AutoModelForImageTextToText
    model = model_class.from_pretrained(
        str(args.model_dir), dtype=dtype, local_files_only=True, low_cpu_mem_usage=True,
    ).to(device)
    model.requires_grad_(False)
    model.eval()
    vision_calls = {"count": 0}
    visual = getattr(model, "visual", None)
    hook = visual.register_forward_pre_hook(lambda *_: vision_calls.__setitem__("count", vision_calls["count"] + 1)) if visual is not None else None

    projector = PatchMergerProjector.from_pretrained(args.projector_dir, device=device)
    projector.eval()
    text_config = config.get("text_config") or config
    receiver_width = int(text_config.get("hidden_size") or model.config.hidden_size)
    receiver = FixedGroupedReceiverAdapter(4096, receiver_width, seed=20260806).to(device) if receiver_width != 4096 else None
    random_projector = seeded_projector(projector.config, seed=args.random_seed).to(device)
    random_projector.eval()
    cache = FeatureCache(args.feature_cache)
    records = cache.manifest.get("records", [])
    indices = [int(item.strip()) for item in args.sample_indices.split(",") if item.strip()]
    if not indices or any(i < 0 or i >= len(records) for i in indices):
        raise ValueError(f"sample indices outside cache: {indices}")

    rows = []
    for index in indices:
        sample = records[index]
        shuffled = records[(index + 1) % len(records)]
        feature = cache.get(sample["id"], device=device, dtype=torch.float32)[0][: args.max_visual_tokens].contiguous()
        shuffled_feature = cache.get(shuffled["id"], device=device, dtype=torch.float32)[0][: args.max_visual_tokens].contiguous()
        correct = eval_case(model=model, projector=projector, receiver=receiver, tokenizer=tokenizer, sample=sample, feature=feature, placeholder=placeholder, device=device, qwen_mrope=args.qwen_mrope)
        shuffled_lp = eval_case(model=model, projector=projector, receiver=receiver, tokenizer=tokenizer, sample=sample, feature=shuffled_feature, placeholder=placeholder, device=device, qwen_mrope=args.qwen_mrope)
        _, input_ids, labels, attention = build_inputs(tokenizer=tokenizer, features=None, sample=sample, placeholder_token_id=placeholder, device=device)
        with torch.no_grad():
            text_embeddings = model.get_input_embeddings()(input_ids)
            blind_out = model(inputs_embeds=text_embeddings, attention_mask=attention, position_ids=(attention.long().cumsum(dim=-1) - 1).clamp_min(0), labels=labels, use_cache=False)
        blind_lp = answer_logprob(blind_out.logits, labels)
        random_lp = eval_case(model=model, projector=random_projector, receiver=receiver, tokenizer=tokenizer, sample=sample, feature=feature, placeholder=placeholder, device=device, qwen_mrope=args.qwen_mrope)
        rows.append({
            "sample_index": index, "sample_id": sample["id"], "shuffled_sample_id": shuffled["id"],
            "vision_answer_logp": correct, "shuffled_answer_logp": shuffled_lp, "blind_answer_logp": blind_lp,
            "random_projector_answer_logp": random_lp,
            "vision_minus_shuffle": correct - shuffled_lp, "vision_minus_blind": correct - blind_lp,
        })
        (args.out / "probe_metrics.jsonl").open("a", encoding="utf-8").write(json.dumps(rows[-1], ensure_ascii=False) + "\n")

    if hook is not None:
        hook.remove()
    margins = [row["vision_minus_shuffle"] for row in rows]
    blind_margins = [row["vision_minus_blind"] for row in rows]
    summary = {
        "schema_version": "stripped-receiver-probe-v1", "status": "diagnostic_only",
        "model_type": config.get("model_type"), "model_loader": model_class.__name__,
        "model_hf_revision": weight_manifest.get("resolved_revision"),
        "config_sha256": sha256_file(args.model_dir / "config.json"),
        "weight_manifest_sha256": sha256_file(args.weight_manifest),
        "dtype": args.dtype, "max_visual_tokens": args.max_visual_tokens,
        "qwen_mrope": args.qwen_mrope,
        "sample_indices": indices, "sample_count": len(rows), "random_projector_seed": args.random_seed,
        "native_vision_forward_calls": vision_calls["count"],
        "vision_minus_shuffle_mean": mean(margins), "vision_minus_shuffle_std": stdev(margins) if len(margins) > 1 else 0.0,
        "vision_minus_blind_mean": mean(blind_margins), "vision_minus_blind_std": stdev(blind_margins) if len(blind_margins) > 1 else 0.0,
        "rows": rows, "capability_claim_allowed": False,
        "deepseek_transfer_label": "transferable_with_runtime_validation",
        "wall_seconds": time.perf_counter() - started,
    }
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
