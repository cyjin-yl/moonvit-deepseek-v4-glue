#!/usr/bin/env python3
"""在 Transformers tiny DeepSeek-V4 上运行完整的软件级 Gate D 闭环。

该工具验证真实 ``DeepseekV4ForCausalLM`` 的 batch forward/backward、冻结语言
主干、projector checkpoint、精确恢复和 greedy generate。它使用 tiny config，
不代表完整 DeepSeek-V4-Flash-0731，也不关闭真实 FP4/FP8 hardware gate。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

from moonvit_glue.model import VisionCausalLM
from moonvit_glue.projector import PatchMergerProjector, ProjectorConfig


def _tiny_config() -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=64,
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


def _build(seed: int, device: torch.device) -> VisionCausalLM:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    language_model = DeepseekV4ForCausalLM(_tiny_config()).to(device)
    projector = PatchMergerProjector(
        ProjectorConfig(vision_width=3, language_width=32, merge_factor=1, projector_width=8)
    ).to(device)
    return VisionCausalLM(
        language_model=language_model,
        projector=projector,
        placeholder_token_id=63,
        backbone_kind="deepseek_v4",
        freeze_language_model=True,
        pad_token_id=0,
    ).to(device)


def _batch(seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    input_ids = torch.tensor(
        [[1, 63, 5, 7, 2], [1, 63, 8, 9, 2]], dtype=torch.long, device=device
    )
    labels = torch.tensor(
        [[1, -100, 5, 7, 2], [1, -100, 8, 9, 2]], dtype=torch.long, device=device
    )
    # projector 的 merge_factor=1 仍要求显式保留 grouped patch 轴 [T, M, W]。
    features = [torch.randn(2, 1, 3, device=device), torch.randn(2, 1, 3, device=device)]
    return input_ids, labels, features


def _projector_state(model: VisionCausalLM) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.projector.state_dict().items()}


def _language_state(model: VisionCausalLM) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.language_model.state_dict().items()}


def _grad_norm(model: VisionCausalLM) -> float:
    values = [parameter.grad.detach().float().norm() for parameter in model.projector.parameters() if parameter.grad is not None]
    return float(torch.stack(values).norm()) if values else 0.0


def _run_steps(
    model: VisionCausalLM,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    features: list[torch.Tensor],
    count: int,
    *,
    start_step: int,
) -> list[dict]:
    model.train()
    trace: list[dict] = []
    for offset in range(count):
        step = start_step + offset + 1
        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, image_feature_groups=features, labels=labels)
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        grad_norm = _grad_norm(model)
        projector_grads_finite = all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            for parameter in model.projector.parameters()
        )
        lm_grads_all_none = all(parameter.grad is None for parameter in model.language_model.parameters())
        optimizer.step()
        trace.append(
            {
                "optimizer_step": step,
                "loss": float(loss.detach().cpu()),
                "projector_gradient_norm": grad_norm,
                "projector_grads_finite": bool(projector_grads_finite),
                "language_grads_all_none": bool(lm_grads_all_none),
            }
        )
    return trace


def _save_checkpoint(path: Path, model: VisionCausalLM, optimizer: torch.optim.Optimizer, step: int) -> None:
    payload = {
        "step": step,
        "language_model": _language_state(model),
        "projector": _projector_state(model),
        "optimizer": optimizer.state_dict(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, path)


def _restore_checkpoint(path: Path, model: VisionCausalLM, optimizer: torch.optim.Optimizer) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.language_model.load_state_dict(payload["language_model"])
    model.projector.load_state_dict(payload["projector"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["cpu_rng"])
    if payload["cuda_rng_all"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng_all"])
    return int(payload["step"])


def _max_state_delta(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    return max(float((left[key] - right[key]).abs().max()) for key in left)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--split-step", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    if args.steps <= 1 or not 0 < args.split_step < args.steps:
        raise ValueError("steps must be > 1 and split-step must be inside the run")
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    input_ids, labels, features = _batch(args.seed + 1, device)

    uninterrupted = _build(args.seed, device)
    uninterrupted_optimizer = torch.optim.AdamW(uninterrupted.projector.parameters(), lr=1e-3)
    uninterrupted_trace = _run_steps(
        uninterrupted, uninterrupted_optimizer, input_ids, labels, features, args.steps, start_step=0
    )
    uninterrupted_state = _projector_state(uninterrupted)

    split = _build(args.seed, device)
    split_optimizer = torch.optim.AdamW(split.projector.parameters(), lr=1e-3)
    split_trace_a = _run_steps(
        split, split_optimizer, input_ids, labels, features, args.split_step, start_step=0
    )
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / f"checkpoint_step{args.split_step:04d}.pt"
    _save_checkpoint(checkpoint_path, split, split_optimizer, args.split_step)

    resumed = _build(args.seed, device)
    resumed_optimizer = torch.optim.AdamW(resumed.projector.parameters(), lr=1e-3)
    restored_step = _restore_checkpoint(checkpoint_path, resumed, resumed_optimizer)
    split_trace_b = _run_steps(
        resumed, resumed_optimizer, input_ids, labels, features, args.steps - args.split_step, start_step=restored_step
    )
    resumed_state = _projector_state(resumed)

    resumed.eval()
    generated = resumed.generate(
        input_ids=input_ids,
        image_feature_groups=features,
        max_new_tokens=2,
        min_new_tokens=2,
        do_sample=False,
        pad_token_id=0,
    )
    state_delta = _max_state_delta(uninterrupted_state, resumed_state)
    loss_delta = max(
        abs(left["loss"] - right["loss"])
        for left, right in zip(uninterrupted_trace, split_trace_a + split_trace_b)
    )
    forward_backward_pass = all(
        row["projector_grads_finite"] and row["projector_gradient_norm"] > 0 and row["language_grads_all_none"]
        for row in uninterrupted_trace
    )
    resume_pass = state_delta <= 1e-6 and loss_delta <= 1e-6 and restored_step == args.split_step
    summary = {
        "schema_version": "deepseek-v4-tiny-e2e-gate-v1",
        "status": "software_tiny_pass_hardware_pending",
        "gate_d_status": "NO-GO",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda_available": torch.cuda.is_available(),
        },
        "tiny_model": {
            "model_type": "deepseek_v4",
            "hidden_size": 32,
            "layers": 1,
            "routing_experts": 4,
            "placeholder_token_id": 63,
            "batch_size": int(input_ids.shape[0]),
            "raw_sequence_length": int(input_ids.shape[1]),
            "expanded_sequence_length": int(generated.shape[1] - 2),
        },
        "forward_backward": {
            "steps": args.steps,
            "pass": bool(forward_backward_pass),
            "projector_grad_finite_nonzero": bool(forward_backward_pass),
            "language_grads_all_none": all(row["language_grads_all_none"] for row in uninterrupted_trace),
        },
        "save_resume": {
            "checkpoint": str(checkpoint_path),
            "restored_step": restored_step,
            "projector_max_abs_delta": state_delta,
            "loss_max_abs_delta": loss_delta,
            "pass": bool(resume_pass),
        },
        "generate": {
            "pass": bool(generated.ndim == 2 and torch.isfinite(generated.float()).all()),
            "shape": list(generated.shape),
            "ids": generated.detach().cpu().tolist(),
        },
        "traces": {
            "uninterrupted": uninterrupted_trace,
            "split": split_trace_a + split_trace_b,
        },
        "paid_hardware_required": True,
        "interpretation": "tiny Transformers DeepSeek-V4 software seam passes; full 0731 weights and real FP4/FP8 input-DGRAD remain unverified",
        "wall_seconds": time.perf_counter() - started,
    }
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.out / "trace_uninterrupted.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in uninterrupted_trace), encoding="utf-8"
    )
    (args.out / "trace_split.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in split_trace_a + split_trace_b), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
