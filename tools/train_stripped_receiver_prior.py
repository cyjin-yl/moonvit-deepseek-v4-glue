#!/usr/bin/env python3
"""在视觉预训练 receiver 上做最小 projector-only 训练筛选。

使用 8 个固定样本和固定 derangement，冻结 Qwen3.5 语言接收器及 MoonViT
特征，只更新 projector。每步同时记录 CE、paired image margin、RMS、跨图
spread、梯度和 NaN/Inf；这是 receiver-prior 诊断，不是正式能力训练。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from moonvit_glue import FeatureCache
from moonvit_glue.probe_samples import load_receiver_probe_records
from moonvit_glue.projector import PatchMergerProjector
from moonvit_glue.token_selection import select_visual_tokens
from probe_stripped_receiver import FixedGroupedReceiverAdapter, build_inputs, expanded_forward


def answer_logprob_tensor(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = labels.ne(-100)
    logp = F.log_softmax(logits.float(), dim=-1)
    selected = torch.gather(logp, -1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return selected.masked_select(mask).mean()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--weight-manifest", type=Path, required=True)
    p.add_argument("--feature-cache", type=Path, required=True)
    p.add_argument("--sample-manifest", type=Path, required=True)
    p.add_argument("--projector-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sample-indices", default="0,1,2,3,4,5,6,7")
    p.add_argument("--max-visual-tokens", type=int, default=240)
    p.add_argument(
        "--token-selection", choices=("prefix", "uniform", "mean_pool"), default="prefix",
        help="固定 token 预算下的序列选择/压缩方式",
    )
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--shuffle-margin-lambda", type=float, default=0.0)
    p.add_argument("--shuffle-margin", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.max_visual_tokens <= 0:
        raise ValueError("steps and max-visual-tokens must be positive")
    started = time.perf_counter()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.out}")
    args.out.mkdir(parents=True)
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    placeholder = int(config.get("image_token_id") or tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    model_class = AutoModelForCausalLM if config.get("model_type") == "qwen2" else AutoModelForImageTextToText
    model = model_class.from_pretrained(str(args.model_dir), dtype=dtype, local_files_only=True, low_cpu_mem_usage=True).to(device)
    model.requires_grad_(False)
    model.eval()
    vision_calls = {"count": 0}
    visual = getattr(model, "visual", None)
    hook = visual.register_forward_pre_hook(lambda *_: vision_calls.__setitem__("count", vision_calls["count"] + 1)) if visual is not None else None

    projector = PatchMergerProjector.from_pretrained(args.projector_dir, device=device)
    projector.train()
    text_config = config.get("text_config") or config
    receiver_width = int(text_config.get("hidden_size") or model.config.hidden_size)
    receiver = FixedGroupedReceiverAdapter(4096, receiver_width, seed=20260806).to(device) if receiver_width != 4096 else None
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.learning_rate)
    cache = FeatureCache(args.feature_cache)
    cache_cap = cache.manifest.get("max_visual_tokens")
    if cache_cap is not None and args.max_visual_tokens > int(cache_cap):
        raise ValueError(
            f"requested max-visual-tokens {args.max_visual_tokens} exceeds cache contract {cache_cap}"
        )
    sample_manifest, records = load_receiver_probe_records(
        args.sample_manifest, cache.manifest.get("records", [])
    )
    indices = [int(item.strip()) for item in args.sample_indices.split(",") if item.strip()]
    if not indices or any(i < 0 or i >= len(records) for i in indices):
        raise ValueError(f"sample indices outside cache: {indices}")
    feature_rows = []
    feature_selection_meta = []
    for index in indices:
        sample = records[index]
        shuffled = records[(index + 1) % len(records)]
        source_feature = cache.get(sample["id"], device=device, dtype=torch.float32)[0]
        source_shuffled_feature = cache.get(shuffled["id"], device=device, dtype=torch.float32)[0]
        selected_feature = select_visual_tokens(source_feature, args.max_visual_tokens, args.token_selection)
        selected_shuffled_feature = select_visual_tokens(source_shuffled_feature, args.max_visual_tokens, args.token_selection)
        feature_rows.append((
            sample,
            selected_feature,
            selected_shuffled_feature,
        ))
        feature_selection_meta.append({
            "sample_index": index, "sample_id": sample["id"],
            "shuffled_sample_id": shuffled["id"],
            "source_feature_shape": list(source_feature.shape),
            "selected_feature_shape": list(selected_feature.shape),
            "shuffled_source_feature_shape": list(source_shuffled_feature.shape),
            "shuffled_selected_feature_shape": list(selected_shuffled_feature.shape),
        })
    (args.out / "RUN_CONFIG.json").write_text(json.dumps({
        "schema_version": "stripped-receiver-prior-train-v1", "model_dir": str(args.model_dir),
        "model_type": config.get("model_type"), "model_loader": model_class.__name__,
        "weight_manifest": str(args.weight_manifest), "feature_cache": str(args.feature_cache),
        "sample_manifest": str(args.sample_manifest),
        "sample_manifest_sha256": sha256_file(args.sample_manifest),
        "sample_manifest_schema_version": sample_manifest.get("schema_version"),
        "projector_dir": str(args.projector_dir), "dtype": args.dtype,
        "max_visual_tokens": args.max_visual_tokens, "sample_indices": indices,
        "token_selection": {
            "schema_version": "visual-token-selection-v1",
            "mode": args.token_selection,
            "requested_max_tokens": args.max_visual_tokens,
            "axis": 0,
            "source_layout": "row_major_cache_tokens",
            "uniform_rule": "integer_nearest_endpoint_v1",
            "mean_pool_rule": "integer_floor_bins_v1",
            "feature_rows": feature_selection_meta,
        },
        "steps": args.steps, "learning_rate": args.learning_rate,
        "shuffle_margin_lambda": args.shuffle_margin_lambda, "shuffle_margin": args.shuffle_margin,
        "receiver_width": receiver_width, "native_vision_bypassed": True,
        "native_vision_forward_calls": 0, "capability_claim_allowed": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    health_path = args.out / "train_health.jsonl"
    rows = []
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        ce_values = []
        margin_values = []
        projected_values = []
        for sample, feature, shuffled_feature in feature_rows:
            _, vision_out, vision_labels = expanded_forward(model=model, projector=projector, receiver=receiver, features=feature, tokenizer=tokenizer, sample=sample, placeholder_token_id=placeholder, device=device)
            _, shuffle_out, shuffle_labels = expanded_forward(model=model, projector=projector, receiver=receiver, features=shuffled_feature, tokenizer=tokenizer, sample=sample, placeholder_token_id=placeholder, device=device)
            vision_lp = answer_logprob_tensor(vision_out.logits, vision_labels)
            shuffle_lp = answer_logprob_tensor(shuffle_out.logits, shuffle_labels)
            ce_values.append(vision_out.loss.detach().float())
            margin_values.append((vision_lp - shuffle_lp).detach())
            projected_values.append(projector([feature])[0].detach())
            if step < args.steps:
                loss = vision_out.loss + args.shuffle_margin_lambda * F.relu(args.shuffle_margin - (vision_lp - shuffle_lp))
                loss.backward()
        if step < args.steps:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(projector.parameters(), 1e9).item())
            optimizer.step()
        else:
            grad_norm = None
        with torch.no_grad():
            pcat = torch.cat([p.reshape(-1, p.shape[-1]) for p in projected_values], dim=0).float()
            means = torch.stack([p.float().mean(dim=0) for p in projected_values])
            ce_mean = float(torch.stack(ce_values).mean().item())
            margin_mean = float(torch.stack(margin_values).mean().item())
            row = {
                "optimizer_step": step, "ce_loss": ce_mean, "vision_minus_shuffle": margin_mean,
                "projector_output_rms": float(pcat.pow(2).mean().sqrt().item()),
                "between_image_rms": float(means.std(dim=0).pow(2).mean().sqrt().item()),
                "projector_gradient_norm_before_clip": grad_norm,
                "native_vision_forward_calls": vision_calls["count"],
                "nan_or_inf": not all(torch.isfinite(v).all().item() for v in projector.parameters()),
            }
        rows.append(row)
        with health_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    projector.save_pretrained(args.out / "projector_final")
    torch.save(optimizer.state_dict(), args.out / "optimizer_final.pt")
    if hook is not None:
        hook.remove()
    summary = {
        "schema_version": "stripped-receiver-prior-train-v1", "status": "diagnostic_only",
        "trajectory": rows, "native_vision_forward_calls": vision_calls["count"],
        "capability_claim_allowed": False, "deepseek_transfer_label": "transferable_with_runtime_validation",
        "wall_seconds": time.perf_counter() - started,
    }
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
