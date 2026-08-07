#!/usr/bin/env python3
"""Evaluate the matched text-only no-vision ScreenSpot control.

This deliberately injects no image token. It reuses the grounding prompt and
greedy decoding path, so its rows can sit beside external-MoonViT conditions
without pretending that a text-only model has a random visual projector.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from moonvit_glue.grounding_contract import parse_click_action, score_click_prediction, summarize_click_scores
from generate_stripped_receiver_probe import _prompt_inputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--screenspot-manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    manifest = json.loads(args.screenspot_manifest.read_text(encoding="utf-8"))
    samples = list(manifest["samples"])
    if args.limit is not None:
        samples = samples[: args.limit]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), dtype=dtype, local_files_only=True, low_cpu_mem_usage=True,
    ).to(device).eval()
    model.requires_grad_(False)
    placeholder = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    rows = []
    for index, sample in enumerate(samples):
        embeds, attention, position_ids = _prompt_inputs(
            model=model, tokenizer=tokenizer, sample={"instruction": sample["instruction"]},
            placeholder=placeholder, feature=None, projector=None, receiver=None,
            prompt_route="grounding", scale=1.0, device=device,
        )
        with torch.no_grad():
            generated = model.generate(
                inputs_embeds=embeds, attention_mask=attention, position_ids=position_ids,
                do_sample=False, max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id, use_cache=True,
            )
        prediction = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        row = score_click_prediction(
            sample_id=str(sample["sample_id"]), prediction=prediction,
            target_box=sample["bbox_999_xyxy"],
        )
        row.update({
            "sample_index": index, "condition": "no_vision",
            "platform": sample.get("platform"), "target_type": sample.get("target_type"),
            "prediction_text": prediction,
        })
        rows.append(row)
        print(f"[{index + 1}/{len(samples)}] {sample['sample_id']} -> {prediction!r}", flush=True)
    summary = summarize_click_scores(rows)
    report = {
        "schema_version": "no-vision-screenspot-v1",
        "status": "control_only",
        "model_dir": str(args.model_dir),
        "screenspot_manifest": str(args.screenspot_manifest),
        "sample_count": len(rows),
        "condition_summaries": {"no_vision": summary},
        "rows": rows,
        "generation": {"do_sample": False, "temperature": 0.0, "max_new_tokens": args.max_new_tokens},
        "image_tokens_injected": False,
        "capability_claim_allowed": False,
        "wall_seconds": time.perf_counter() - started,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "generation_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (args.out / "SUMMARY.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
