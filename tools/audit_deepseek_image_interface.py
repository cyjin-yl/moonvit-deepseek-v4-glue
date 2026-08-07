#!/usr/bin/env python3
"""Run the local, software-only DeepSeek image-token interface screen.

This screen freezes the seam that can be checked without the full
``DeepSeek-V4-Flash-0731`` weights: image placeholder expansion, contiguous
position IDs, repeated routing IDs, masked image labels, projector gradients,
and the fact that routing IDs and position IDs are actually consumed by the
tiny real Transformers DeepSeek-V4 implementation.  A pass here is an
interface result, not evidence that the full 0731 checkpoint or FP4/FP8
backward path is ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

from moonvit_glue.merge import MultimodalInputs, expand_image_placeholders
from moonvit_glue.model import VisionCausalLM
from moonvit_glue.projector import PatchMergerProjector, ProjectorConfig


def tiny_config(*, vocab_size: int) -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=vocab_size,
        hidden_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        q_lora_rank=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        o_groups=2,
        o_lora_rank=16,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        hc_mult=2,
        num_nextn_predict_layers=0,
        pad_token_id=0,
    )


def build_model(*, seed: int, device: torch.device, placeholder_token_id: int) -> VisionCausalLM:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    language_model = DeepseekV4ForCausalLM(
        tiny_config(vocab_size=max(64, placeholder_token_id + 1))
    ).to(device=device, dtype=torch.float32)
    # Random tiny configs initialise every tid2eid entry to zero.  That default
    # makes routing-ID plumbing observationally inert, so install a frozen,
    # non-degenerate table solely for this software interface screen.
    with torch.no_grad():
        token_ids = torch.arange(language_model.config.vocab_size, device=device)
        for layer in language_model.model.layers:
            if not layer.mlp.is_hash:
                continue
            expert_count = int(layer.mlp.gate.num_experts)
            top_k = int(layer.mlp.gate.top_k)
            columns = [((token_ids + offset) % expert_count) for offset in range(top_k)]
            layer.mlp.gate.tid2eid.copy_(torch.stack(columns, dim=-1))
    projector = PatchMergerProjector(
        ProjectorConfig(vision_width=3, language_width=32, merge_factor=1, projector_width=8)
    ).to(device=device, dtype=torch.float32)
    return VisionCausalLM(
        language_model=language_model,
        projector=projector,
        placeholder_token_id=placeholder_token_id,
        backbone_kind="deepseek_v4",
        freeze_language_model=True,
        pad_token_id=0,
    ).to(device)


def _tensor_sha256(value: torch.Tensor) -> str:
    data = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def merge_invariants(
    merged: MultimodalInputs,
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    placeholder_token_id: int,
    image_token_count: int,
) -> dict[str, object]:
    """Return frozen, JSON-friendly checks for the canonical merge contract."""

    ids = merged.routing_input_ids[0]
    mask = merged.attention_mask[0]
    positions = merged.position_ids[0]
    merged_labels = merged.labels[0] if merged.labels is not None else None
    image_start = int((ids == placeholder_token_id).nonzero(as_tuple=False)[0].item())
    image_slice = ids[image_start : image_start + image_token_count]
    image_label_slice = (
        merged_labels[image_start : image_start + image_token_count]
        if merged_labels is not None
        else torch.empty(0, dtype=torch.long)
    )
    expected_text_labels = torch.tensor(
        [int(labels[0, 0]), int(labels[0, 2]), int(labels[0, 3]), int(labels[0, 4])],
        device=labels.device,
        dtype=labels.dtype,
    )
    actual_text_labels = torch.cat(
        [merged_labels[:image_start], merged_labels[image_start + image_token_count :]]
    ) if merged_labels is not None else torch.empty(0, dtype=labels.dtype, device=labels.device)
    checks = {
        "raw_sequence_length": int(input_ids.shape[1]),
        "expanded_sequence_length": int(merged.inputs_embeds.shape[1]),
        "image_token_count": int(image_token_count),
        "routing_placeholder_repeated": bool(
            torch.equal(image_slice, torch.full_like(image_slice, placeholder_token_id))
        ),
        "routing_placeholder_count": int((ids == placeholder_token_id).sum().item()),
        "attention_mask_all_active": bool(torch.equal(mask, torch.ones_like(mask))),
        "position_ids_contiguous": bool(
            torch.equal(positions, torch.arange(merged.inputs_embeds.shape[1], device=positions.device))
        ),
        "image_labels_ignored": bool(
            torch.equal(image_label_slice, torch.full_like(image_label_slice, -100))
        ),
        "text_labels_preserved": bool(torch.equal(actual_text_labels, expected_text_labels)),
    }
    checks["pass"] = all(bool(value) for key, value in checks.items() if key.endswith(("repeated", "active", "contiguous", "ignored", "preserved")))
    return checks


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def run_screen(*, out: Path, device: torch.device, seed: int, placeholder_token_id: int) -> dict[str, object]:
    started = time.perf_counter()
    torch.manual_seed(seed + 1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 1)
    model = build_model(seed=seed, device=device, placeholder_token_id=placeholder_token_id)
    model.eval()
    input_ids = torch.tensor([[1, placeholder_token_id, 5, 7, 2]], dtype=torch.long, device=device)
    labels = torch.tensor([[1, -100, 5, 7, 2]], dtype=torch.long, device=device)
    features = [torch.randn(3, 1, 3, device=device, dtype=torch.float32)]
    with torch.no_grad():
        projected = model.projector(features)
        text_embeddings = model.language_model.get_input_embeddings()(input_ids)
        merged = expand_image_placeholders(
            input_ids=input_ids,
            text_embeddings=text_embeddings,
            image_embeddings=projected,
            placeholder_token_id=placeholder_token_id,
            labels=labels,
        )
        canonical = model._language_forward(merged)
        route_ablated_ids = merged.routing_input_ids.clone()
        route_ablated_ids[:, 1:4] = 1
        route_ablated = model._language_forward(
            MultimodalInputs(
                inputs_embeds=merged.inputs_embeds,
                routing_input_ids=route_ablated_ids,
                attention_mask=merged.attention_mask,
                position_ids=merged.position_ids,
                labels=merged.labels,
            )
        )
        position_ablated = model._language_forward(
            MultimodalInputs(
                inputs_embeds=merged.inputs_embeds,
                routing_input_ids=merged.routing_input_ids,
                attention_mask=merged.attention_mask,
                position_ids=merged.position_ids.clone().index_fill(1, torch.tensor([2], device=device), 1),
                labels=merged.labels,
            )
        )
    model.train()
    model.zero_grad(set_to_none=True)
    outputs = model(input_ids=input_ids, image_feature_groups=features, labels=labels)
    outputs.loss.backward()
    projector_gradients = [parameter.grad for parameter in model.projector.parameters()]
    projector_grad_finite = all(gradient is not None and torch.isfinite(gradient).all().item() for gradient in projector_gradients)
    projector_grad_norm = float(torch.stack([gradient.detach().float().norm() for gradient in projector_gradients if gradient is not None]).norm().cpu())
    checks = merge_invariants(
        merged,
        input_ids=input_ids,
        labels=labels,
        placeholder_token_id=placeholder_token_id,
        image_token_count=3,
    )
    routing_delta = _max_abs(canonical.logits, route_ablated.logits)
    position_delta = _max_abs(canonical.logits, position_ablated.logits)
    checks["projector_gradient_finite_nonzero"] = bool(projector_grad_finite and projector_grad_norm > 0)
    checks["language_gradient_all_none"] = all(parameter.grad is None for parameter in model.language_model.parameters())
    checks["routing_ids_are_consumed"] = bool(routing_delta > 0)
    checks["position_ids_are_consumed"] = bool(position_delta > 0)
    checks["pass"] = bool(
        checks["pass"]
        and checks["projector_gradient_finite_nonzero"]
        and checks["language_gradient_all_none"]
        and checks["routing_ids_are_consumed"]
        and checks["position_ids_are_consumed"]
    )
    summary: dict[str, object] = {
        "schema_version": "deepseek-v4-image-interface-screen-v2",
        "status": "software_interface_pass_hardware_pending" if checks["pass"] else "software_interface_fail",
        "gate_d_status": "NO-GO",
        "interpretation": "Canonical placeholder expansion, routing IDs, contiguous positions, label masking, projector input-DGRAD, and tiny real DeepSeek-V4 consumption were checked. Full 0731 weights, runtime quantization, memory, and FP4/FP8 backward remain unverified.",
        "contract": {
            "placeholder_token_id": placeholder_token_id,
            "position_policy": "expanded_contiguous",
            "routing_policy": "repeat_placeholder_id_over_image_span",
            "image_label_policy": "ignore_index_-100",
            "canonical_projector_width": 4096,
            "tiny_receiver_hidden_size": 32,
            "tiny_hash_route_table": "tid2eid[token, k] = (token + k) mod num_experts",
        },
        "checks": checks,
        "causal_interface_screen": {
            "routing_id_ablation_max_abs_logit_delta": routing_delta,
            "position_id_ablation_max_abs_logit_delta": position_delta,
            "routing_ids_are_consumed": bool(routing_delta > 0),
            "position_ids_are_consumed": bool(position_delta > 0),
        },
        "forward_backward": {
            "loss": float(outputs.loss.detach().cpu()),
            "projector_gradient_norm": projector_grad_norm,
            "projector_gradient_finite_nonzero": bool(projector_grad_finite and projector_grad_norm > 0),
            "language_gradient_all_none": checks["language_gradient_all_none"],
        },
        "fingerprints": {
            "projected_image_sha256": [_tensor_sha256(value) for value in projected],
            "expanded_embeddings_sha256": _tensor_sha256(merged.inputs_embeds),
            "routing_ids_sha256": _tensor_sha256(merged.routing_input_ids),
            "position_ids_sha256": _tensor_sha256(merged.position_ids),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "paid_hardware_required": True,
        "wall_seconds": time.perf_counter() - started,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    torch.save(merged, out / "MERGED_TENSORS.pt")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--placeholder-token-id", type=int, default=129279)
    args = parser.parse_args()
    summary = run_screen(
        out=args.out,
        device=torch.device(args.device),
        seed=args.seed,
        placeholder_token_id=args.placeholder_token_id,
    )
    print(json.dumps(summary, indent=2))
    if summary["status"] != "software_interface_pass_hardware_pending":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
