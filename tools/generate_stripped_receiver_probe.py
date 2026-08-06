#!/usr/bin/env python3
"""在冻结 receiver-prior checkpoint 上做最小自由生成一致性检查。

该工具只使用外部 MoonViT/projector token，绕过原生视觉模块；输出是
teacher-forced probe 的 companion diagnostic，不进入正式 ScreenSpot 排名。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch

from moonvit_glue import FeatureCache
from moonvit_glue.chat_contract import build_chat_prompt
from moonvit_glue.grounding_contract import parse_click_action
from moonvit_glue.merge import expand_image_placeholders
from moonvit_glue.probe_samples import load_receiver_probe_records
from moonvit_glue.projector import PatchMergerProjector, seeded_projector
from moonvit_glue.token_selection import select_visual_tokens
from probe_stripped_receiver import FixedGroupedReceiverAdapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--feature-cache", type=Path, required=True)
    p.add_argument("--sample-manifest", type=Path, required=True)
    p.add_argument("--projector-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sample-indices", default="16,17,18,19,20,21,22,23")
    p.add_argument("--max-visual-tokens", type=int, default=16)
    p.add_argument("--token-selection", choices=("prefix", "uniform", "mean_pool"), default="mean_pool")
    p.add_argument("--projector-scale", type=float, default=0.1)
    p.add_argument("--random-seed", type=int, default=20260806)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--prompt-route", choices=("grounding", "generic"), default="grounding")
    return p.parse_args()


def _target_point(sample: dict) -> tuple[int, int] | None:
    answers = sample.get("answers") or []
    for answer in answers:
        match = re.search(r"start_box=\[\s*(\d+)\s*,\s*(\d+)\s*\]", str(answer))
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _prompt_inputs(*, model, tokenizer, sample, placeholder, feature, projector, receiver, prompt_route, scale, device):
    question = str(sample.get("question") or sample.get("instruction") or "").strip()
    if not question:
        raise ValueError(f"sample {sample.get('id')} has no question/instruction")
    if prompt_route == "grounding":
        system_prompt = (
            "You are a GUI grounding model. Return exactly one click action and no other text. "
            "Use integer coordinates from 0 to 999 with the top-left origin. "
            "Required format: click(start_box=[x, y])"
        )
        user_prompt = f"Locate the UI element described below and click its center.\nTarget: {question}"
    else:
        system_prompt = "Use the image to answer the user's question. Return only the answer with no explanation."
        user_prompt = f"Question: {question}"
    prompt = build_chat_prompt(
        tokenizer,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        placeholder_token_id=placeholder,
        include_image=feature is not None,
        enable_thinking=False if "enable_thinking" in str(getattr(tokenizer, "chat_template", "")) else None,
    )
    input_ids = torch.tensor([prompt.input_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(input_ids)
    text_embeddings = model.get_input_embeddings()(input_ids)
    if feature is None:
        position_ids = (attention.long().cumsum(dim=-1) - 1).clamp_min(0)
        return text_embeddings, attention, position_ids
    projected = projector([feature])[0] * float(scale)
    if receiver is not None:
        projected = receiver(projected)
    merged = expand_image_placeholders(
        input_ids=input_ids,
        text_embeddings=text_embeddings,
        image_embeddings=[projected],
        placeholder_token_id=placeholder,
        attention_mask=attention,
    )
    return merged.inputs_embeds, merged.attention_mask, merged.position_ids


def main() -> None:
    args = parse_args()
    if args.max_visual_tokens <= 0 or args.max_new_tokens <= 0:
        raise ValueError("token limits must be positive")
    if not math.isfinite(args.projector_scale) or args.projector_scale <= 0:
        raise ValueError("projector-scale must be finite and positive")
    started = time.perf_counter()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    placeholder = int(config.get("image_token_id") or tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), dtype=dtype, local_files_only=True, low_cpu_mem_usage=True,
    ).to(device).eval()
    model.requires_grad_(False)
    projector = PatchMergerProjector.from_pretrained(args.projector_dir, device=device).eval()
    text_config = config.get("text_config") or config
    receiver_width = int(text_config.get("hidden_size") or model.config.hidden_size)
    receiver = FixedGroupedReceiverAdapter(4096, receiver_width, seed=20260806).to(device) if receiver_width != 4096 else None
    random_projector = seeded_projector(projector.config, seed=args.random_seed).to(device).eval()
    cache = FeatureCache(args.feature_cache)
    _, records = load_receiver_probe_records(args.sample_manifest, cache.manifest.get("records", []))
    indices = [int(value) for value in args.sample_indices.split(",") if value.strip()]
    selected_ids = [str(records[index]["id"]) for index in indices]
    rows = []
    for index, sample_id in zip(indices, selected_ids):
        sample = records[index]
        shuffled_sample = records[(index + 1) % len(records)]
        shuffled_id = str(shuffled_sample["id"])
        feature = cache.get(sample_id, device=device, dtype=torch.float32)[0]
        shuffled = cache.get(shuffled_id, device=device, dtype=torch.float32)[0]
        feature = select_visual_tokens(feature, args.max_visual_tokens, args.token_selection)
        shuffled = select_visual_tokens(shuffled, args.max_visual_tokens, args.token_selection)
        target = _target_point(sample)
        for condition, current_feature, current_projector in (
            ("vision", feature, projector),
            ("blind", None, projector),
            ("shuffled", shuffled, projector),
            ("random_projector", feature, random_projector),
        ):
            with torch.no_grad():
                embeds, attention, position_ids = _prompt_inputs(
                    model=model, tokenizer=tokenizer, sample=sample, placeholder=placeholder,
                    feature=current_feature, projector=current_projector, receiver=receiver,
                    prompt_route=args.prompt_route, scale=args.projector_scale, device=device,
                )
                generated = model.generate(
                    inputs_embeds=embeds,
                    attention_mask=attention,
                    position_ids=position_ids,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            point = parse_click_action(text)
            distance = None
            if point is not None and target is not None:
                distance = math.hypot(point[0] - target[0], point[1] - target[1])
            rows.append({
                "sample_index": index,
                "sample_id": sample_id,
                "shuffled_sample_id": shuffled_id,
                "condition": condition,
                "target_point_999": list(target) if target else None,
                "prediction": text,
                "parsed": point is not None,
                "prediction_point_999": list(point) if point else None,
                "distance_to_target_point": distance,
            })
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "generation_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    summary = {
        "schema_version": "stripped-receiver-free-generation-v1",
        "status": "diagnostic_only",
        "sample_count": len(indices),
        "conditions": ["vision", "blind", "shuffled", "random_projector"],
        "max_visual_tokens": args.max_visual_tokens,
        "token_selection": args.token_selection,
        "projector_scale": args.projector_scale,
        "max_new_tokens": args.max_new_tokens,
        "prompt_route": args.prompt_route,
        "rows": rows,
        "capability_claim_allowed": False,
        "deepseek_transfer_label": "transferable_with_runtime_validation",
        "wall_seconds": time.perf_counter() - started,
    }
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
