#!/usr/bin/env python3
"""外部 MoonViT projector 接入 Qwen3.5 原生 VLM 语言接收器的最小闸门。

这个脚本保留 Qwen3.5 的 ``language_model`` 和 ``lm_head`` 权重，完全不传
``pixel_values``，只把本仓库 projector 产生的 4096 维视觉 token 插入
``inputs_embeds``。它用于区分“视觉预训练过的 receiver 是否更容易读外部
MoonViT token”和“原生 vision tower 在起作用”。结果是诊断证据，不进入
Qwen2.5-3B 社区排行榜，也不能替代 DeepSeek Gate D。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from moonvit_glue import FeatureCache
from moonvit_glue.chat_contract import build_chat_supervision
from moonvit_glue.merge import expand_image_placeholders
from moonvit_glue.projector import PatchMergerProjector


class FixedGroupedReceiverAdapter(torch.nn.Module):
    """4096->arbitrary-width parameter-free signed grouping for diagnostics."""

    def __init__(self, canonical_width: int, receiver_width: int, seed: int) -> None:
        super().__init__()
        if canonical_width < receiver_width or receiver_width <= 0:
            raise ValueError("receiver width must be in (0, canonical width]")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        permutation = torch.randperm(canonical_width, generator=generator)
        signs = torch.randint(0, 2, (canonical_width,), generator=generator, dtype=torch.int8)
        signs = signs.mul_(2).sub_(1)
        group_sizes = [2] * (canonical_width - receiver_width) + [1] * (2 * receiver_width - canonical_width)
        if sum(group_sizes) != canonical_width or len(group_sizes) != receiver_width:
            raise AssertionError("invalid grouped receiver layout")
        self.register_buffer("permutation", permutation, persistent=True)
        self.register_buffer("signs", signs, persistent=True)
        self.register_buffer("group_sizes", torch.tensor(group_sizes, dtype=torch.long), persistent=True)
        self.canonical_width = int(canonical_width)
        self.receiver_width = int(receiver_width)
        self.seed = int(seed)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.canonical_width:
            raise ValueError(f"expected {self.canonical_width} dims, got {inputs.shape[-1]}")
        values = inputs.index_select(-1, self.permutation) * self.signs.to(inputs.dtype)
        outputs: list[torch.Tensor] = []
        offset = 0
        for size in self.group_sizes.tolist():
            outputs.append(values[..., offset : offset + size].sum(dim=-1) / math.sqrt(float(size)))
            offset += size
        return torch.stack(outputs, dim=-1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weight-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--projector-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-visual-tokens", type=int, default=None,
                        help="Diagnostic-only token cap; omit for the fixed cache contract")
    return parser.parse_args()


def answer_logprob(logits: torch.Tensor, labels: torch.Tensor) -> float:
    mask = labels.ne(-100)
    if not bool(mask.any()):
        raise ValueError("supervision contains no answer labels")
    logp = F.log_softmax(logits.float(), dim=-1)
    selected = torch.gather(logp, -1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return float(selected.masked_select(mask).mean().item())


def build_inputs(
    *, tokenizer: Any, features: torch.Tensor | None, sample: dict[str, Any],
    placeholder_token_id: int, device: torch.device,
):
    bbox = sample.get("bbox_999_xyxy") or [0.0, 0.0, 999.0, 999.0]
    center_x = round((float(bbox[0]) + float(bbox[2])) / 2.0)
    center_y = round((float(bbox[1]) + float(bbox[3])) / 2.0)
    answer = f"click(start_box=[{center_x}, {center_y}])"
    supervision = build_chat_supervision(
        tokenizer,
        system_prompt=(
            "You are a GUI grounding model. Return exactly one click action and no other text. "
            "Use integer coordinates from 0 to 999 with the top-left origin. "
            "Required format: click(start_box=[x, y])"
        ),
        user_prompt=(
            "Locate the UI element described below and click its center.\n"
            f"Target: {sample.get('instruction', 'the target UI element')}"
        ),
        answer=answer,
        placeholder_token_id=placeholder_token_id,
        include_image=features is not None,
    )
    input_ids = torch.tensor([supervision.input_ids], dtype=torch.long, device=device)
    labels = torch.tensor([supervision.labels], dtype=torch.long, device=device)
    attention = torch.ones_like(input_ids)
    return supervision, input_ids, labels, attention


def expanded_forward(
    *, model: torch.nn.Module, projector: PatchMergerProjector | None,
    receiver: FixedGroupedReceiverAdapter | None, features: torch.Tensor | None, tokenizer: Any, sample: dict[str, Any],
    placeholder_token_id: int, device: torch.device,
):
    supervision, input_ids, labels, attention = build_inputs(
        tokenizer=tokenizer, features=features, sample=sample,
        placeholder_token_id=placeholder_token_id, device=device,
    )
    text_embeddings = model.get_input_embeddings()(input_ids)
    if features is None:
        outputs = model(
            inputs_embeds=text_embeddings, attention_mask=attention,
            position_ids=(attention.long().cumsum(dim=-1) - 1).clamp_min(0),
            labels=labels, use_cache=False,
        )
        return supervision, outputs, labels
    assert projector is not None
    projected = projector([features])[0]
    if receiver is not None:
        projected = receiver(projected)
    merged = expand_image_placeholders(
        input_ids=input_ids, text_embeddings=text_embeddings,
        image_embeddings=[projected], placeholder_token_id=placeholder_token_id,
        attention_mask=attention, labels=labels,
    )
    outputs = model(
        inputs_embeds=merged.inputs_embeds, attention_mask=merged.attention_mask,
        position_ids=merged.position_ids, labels=merged.labels, use_cache=False,
    )
    return supervision, outputs, merged.labels


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    started = time.perf_counter()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    config_path = args.model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    weight_manifest = json.loads(args.weight_manifest.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    placeholder_token_id = int(config.get("image_token_id") or tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    # Qwen3.5 走多模态类但不传 pixel_values；纯文本 Qwen2.5 走 CausalLM 类，
    # 两者最终共享 inputs_embeds/labels 接口，避免把原生 VLM 层混入代理合同。
    model_class = AutoModelForCausalLM if config.get("model_type") == "qwen2" else AutoModelForImageTextToText
    model = model_class.from_pretrained(
        str(args.model_dir), dtype=dtype, local_files_only=True, low_cpu_mem_usage=True,
    ).to(device)
    model.requires_grad_(False)
    model.eval()

    vision_calls = {"count": 0}
    visual = getattr(model, "visual", None)
    hook = None
    if visual is not None:
        hook = visual.register_forward_pre_hook(lambda *_args: vision_calls.__setitem__("count", vision_calls["count"] + 1))

    projector = PatchMergerProjector.from_pretrained(args.projector_dir, device=device)
    projector.train()
    text_config = config.get("text_config") or config
    receiver_width = int(text_config.get("hidden_size") or model.config.hidden_size)
    receiver = (
        FixedGroupedReceiverAdapter(4096, receiver_width, seed=20260806).to(device)
        if receiver_width != 4096 else None
    )
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.learning_rate)
    cache = FeatureCache(args.feature_cache)
    records = cache.manifest.get("records", [])
    if not records:
        raise ValueError("feature cache has no records")
    index = int(args.sample_index)
    if index < 0 or index >= len(records):
        raise IndexError(f"sample-index {index} outside cache with {len(records)} records")
    sample = records[index]
    shuffled_sample = records[(index + 1) % len(records)]
    feature = cache.get(sample["id"], device=device, dtype=torch.float32)[0]
    shuffled_feature = cache.get(shuffled_sample["id"], device=device, dtype=torch.float32)[0]
    if args.max_visual_tokens is not None:
        if args.max_visual_tokens <= 0:
            raise ValueError("--max-visual-tokens must be positive")
        feature = feature[: args.max_visual_tokens].contiguous()
        shuffled_feature = shuffled_feature[: args.max_visual_tokens].contiguous()

    (out / "RUN_CONFIG.json").write_text(json.dumps({
        "schema_version": "stripped-native-qwen35-receiver-smoke-v1",
        "model_dir": str(args.model_dir), "config_sha256": sha256_file(config_path),
        "weight_manifest": str(args.weight_manifest),
        "weight_manifest_sha256": sha256_file(args.weight_manifest),
        "weights": weight_manifest, "model_type": config.get("model_type"),
        "model_loader": model_class.__name__,
        "architectures": config.get("architectures"),
        "text_hidden_size": text_config.get("hidden_size"),
        "receiver_adapter": {
            "kind": "fixed_grouped_signed_projection" if receiver is not None else "identity",
            "canonical_width": 4096, "receiver_width": receiver_width,
            "seed": 20260806, "trainable_parameter_count": 0,
        },
        "native_vision_bypassed": True, "native_vision_forward_calls": 0,
        "placeholder_token_id": placeholder_token_id, "dtype": args.dtype,
        "max_visual_tokens": args.max_visual_tokens,
        "device": str(device), "steps": args.steps, "learning_rate": args.learning_rate,
        "feature_cache": str(args.feature_cache),
        "feature_cache_manifest_sha256": sha256_file(args.feature_cache / "MANIFEST.json"),
        "projector_dir": str(args.projector_dir), "sample_id": sample["id"],
        "shuffled_sample_id": shuffled_sample["id"], "started_utc": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    health_rows = []
    step0_projector = {key: value.detach().clone() for key, value in projector.state_dict().items()}
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        _, vision_outputs, vision_labels = expanded_forward(
            model=model, projector=projector, features=feature, tokenizer=tokenizer,
            receiver=receiver, sample=sample, placeholder_token_id=placeholder_token_id, device=device,
        )
        loss = vision_outputs.loss
        if step < args.steps:
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(projector.parameters(), 1e9).item())
            optimizer.step()
        else:
            grad_norm = None
        with torch.no_grad():
            _, correct_eval, correct_labels = expanded_forward(
                model=model, projector=projector, features=feature, tokenizer=tokenizer,
                receiver=receiver, sample=sample, placeholder_token_id=placeholder_token_id, device=device,
            )
            _, shuffle_eval, shuffle_labels = expanded_forward(
                model=model, projector=projector, features=shuffled_feature, tokenizer=tokenizer,
                receiver=receiver, sample=sample, placeholder_token_id=placeholder_token_id, device=device,
            )
            _, blind_eval, blind_labels = expanded_forward(
                model=model, projector=None, features=None, tokenizer=tokenizer,
                receiver=None, sample=sample, placeholder_token_id=placeholder_token_id, device=device,
            )
        health_rows.append({
            "optimizer_step": step, "ce_loss": float(loss.detach().float().item()),
            "correct_answer_logp": answer_logprob(correct_eval.logits, correct_labels),
            "shuffled_answer_logp": answer_logprob(shuffle_eval.logits, shuffle_labels),
            "blind_answer_logp": answer_logprob(blind_eval.logits, blind_labels),
            "vision_minus_shuffle": answer_logprob(correct_eval.logits, correct_labels) - answer_logprob(shuffle_eval.logits, shuffle_labels),
            "projector_output_rms": float(projector([feature])[0].detach().float().pow(2).mean().sqrt().item()),
            "projector_gradient_norm_before_clip": grad_norm,
            "vision_forward_calls": vision_calls["count"],
            "nan_or_inf": not all(torch.isfinite(value).all().item() for value in projector.parameters()),
        })
        with (out / "train_health.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(health_rows[-1], ensure_ascii=False) + "\n")

    if hook is not None:
        hook.remove()
    finite_gradient = any(
        row["projector_gradient_norm_before_clip"] is not None
        and math.isfinite(float(row["projector_gradient_norm_before_clip"]))
        and float(row["projector_gradient_norm_before_clip"]) > 0.0
        for row in health_rows[:-1]
    )
    finite_outputs = all(
        math.isfinite(float(row["ce_loss"]))
        and math.isfinite(float(row["correct_answer_logp"]))
        and math.isfinite(float(row["shuffled_answer_logp"]))
        and not row["nan_or_inf"]
        for row in health_rows
    )
    summary = {
        "schema_version": "stripped-native-qwen35-receiver-smoke-v1",
        "status": "passed_input_gradient_gate" if finite_gradient and finite_outputs and vision_calls["count"] == 0 else "failed_input_gradient_gate",
        "failure_reasons": ([
            reason for reason, condition in (
                ("nonfinite_projector_or_loss", not finite_outputs),
                ("zero_or_nonfinite_projector_gradient", not finite_gradient),
                ("native_vision_module_called", vision_calls["count"] != 0),
            ) if condition
        ]),
        "native_vision_forward_calls": vision_calls["count"],
        "trajectory": health_rows,
        "projector_step0_state_keys": sorted(step0_projector),
        "projector_trainable_parameter_count": sum(parameter.numel() for parameter in projector.parameters()),
        "wall_seconds": time.perf_counter() - started,
        "deepseek_transfer_label": "transferable_with_runtime_validation",
        "capability_claim_allowed": False,
        "qwen_proxy_conclusion": "This is a receiver-prior and input-gradient smoke only; it is not a community benchmark result.",
    }
    (out / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
