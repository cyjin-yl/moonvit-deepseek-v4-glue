#!/usr/bin/env python3
"""Gate D 三模式 input-gradient reproducer。

模式 ``reference`` 用普通冻结 Linear 验证测试 harness；模式 ``native`` 从
Transformers 实际加载的量化模型中分别探测 FP8Linear、expert gate/up 和 expert
down；模式 ``candidate`` 验证 input-only autograd 接口与 BF16 reference，并把尚未
接入的真实量化 DGRAD 明确标成 ``hardware_pending``。

reference 成功不代表量化路径通过。native 只有三个目标全部返回有限非零 input
gradient、且冻结权重无梯度时才通过。
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import platform
import sys
import traceback
from pathlib import Path
from typing import Callable

import torch


REQUIRED_TARGET_KINDS = (
    "fp8_linear",
    "fp8_expert_gate_up",
    "fp4_expert_down",
)


@dataclasses.dataclass(frozen=True)
class QuantizedTarget:
    """一个需要独立判定 input gradient 的实际量化计算。"""

    kind: str
    module_path: str
    module: torch.nn.Module


class InputOnlyDgradFunction(torch.autograd.Function):
    """只返回 grad_input 的候选 autograd 边界。

    ``forward_op`` 必须由调用方绑定到真实 forward kernel；``dgrad_op`` 必须只计算
    input gradient。weight、scale 与 bias 都不会获得梯度。V100 单元测试使用普通
    Linear 数学验证接口，不把它登记为量化 runtime 通过。
    """

    @staticmethod
    def forward(ctx, inputs, weight, scale, bias, forward_op, dgrad_op):
        ctx.save_for_backward(weight, scale)
        ctx.dgrad_op = dgrad_op
        with torch.no_grad():
            return forward_op(inputs, weight, scale, bias)

    @staticmethod
    def backward(ctx, grad_output):
        weight, scale = ctx.saved_tensors
        grad_input = ctx.dgrad_op(grad_output.contiguous(), weight, scale)
        return grad_input, None, None, None, None, None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dtype_from_name(name: str) -> torch.dtype:
    aliases = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
    canonical = aliases.get(name.lower(), name.lower())
    dtype = getattr(torch, canonical, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported dtype: {name}")
    return dtype


def discover_config(weights: Path) -> dict:
    path = weights / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    quant = config.get("quantization_config") or config.get("quantization")
    print("== config discovery ==")
    print(f"model_type={config.get('model_type')} torch_dtype={config.get('torch_dtype')}")
    print(f"quantization_config={json.dumps(quant, indent=2, default=str) if quant else None}")
    return config


def discover_shard_dtypes(weights: Path) -> dict:
    """只查看第一个现有 shard；最小 gate 可以因此验证实际存储 dtype。"""

    from safetensors import safe_open

    index_path = weights / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        candidates = sorted(set(index["weight_map"].values()))
        first_shard = next((name for name in candidates if (weights / name).exists()), None)
        if first_shard is None:
            raise FileNotFoundError("weight index exists but no referenced shard is present")
    else:
        first_shard = "model.safetensors"
    dtypes: dict[str, int] = {}
    with safe_open(str(weights / first_shard), framework="pt") as handle:
        for key in list(handle.keys())[:200]:
            dtype = str(handle.get_slice(key).get_dtype())
            dtypes[dtype] = dtypes.get(dtype, 0) + 1
    print(f"== first shard ({first_shard}) dtypes (first 200 tensors): {dtypes} ==")
    return {"shard": first_shard, "dtypes": dtypes, "sha256": sha256(weights / first_shard)}


def find_target_linear(root: torch.nn.Module) -> tuple[str, torch.nn.Module]:
    """兼容旧测试：找到 decoder layer 0 的首个直接二维权重模块。"""

    try:
        layer0 = root.model.layers[0]
    except AttributeError as exc:
        raise SystemExit(f"cannot locate decoder layer 0 on {type(root).__name__}: {exc}")
    for name, module in layer0.named_modules():
        direct = dict(module.named_parameters(recurse=False))
        buffers = dict(module.named_buffers(recurse=False))
        weight = direct.get("weight")
        if weight is None:
            weight = buffers.get("weight")
        if weight is not None and weight.ndim >= 2:
            return name, module
    raise SystemExit("no linear-like module with a direct 2D weight found in layer 0")


def find_quantized_targets(root: torch.nn.Module) -> list[QuantizedTarget]:
    """按固定顺序选出普通 FP8 与 expert 的两段计算。"""

    linear: tuple[str, torch.nn.Module] | None = None
    experts: tuple[str, torch.nn.Module] | None = None
    for name, module in root.named_modules():
        class_name = type(module).__name__
        if linear is None and class_name == "FP8Linear":
            linear = (name, module)
        if experts is None and class_name == "FP8Experts":
            required = ("gate_up_proj", "gate_up_proj_scale_inv", "down_proj", "down_proj_scale_inv")
            if all(hasattr(module, attribute) for attribute in required):
                experts = (name, module)
        if linear is not None and experts is not None:
            break
    missing = []
    if linear is None:
        missing.append("FP8Linear")
    if experts is None:
        missing.append("FP8Experts")
    if missing:
        raise LookupError(f"actual loaded model is missing required quantized modules: {', '.join(missing)}")
    linear_path, linear_module = linear
    expert_path, expert_module = experts
    return [
        QuantizedTarget("fp8_linear", linear_path, linear_module),
        QuantizedTarget("fp8_expert_gate_up", expert_path, expert_module),
        QuantizedTarget("fp4_expert_down", expert_path, expert_module),
    ]


def module_device(module: torch.nn.Module) -> torch.device:
    for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
        if tensor.device.type != "meta":
            return tensor.device
    return torch.device("cpu")


@contextlib.contextmanager
def capture_backend(module: torch.nn.Module):
    """包裹真实 dispatcher，记录 DeepGEMM 尝试与 Triton fallback。"""

    trace: list[str] = []
    if not type(module).__module__.startswith("transformers"):
        trace.append("reference")
        yield trace
        return
    from transformers.integrations import finegrained_fp8

    original_deepgemm = finegrained_fp8.deepgemm_fp8_fp4_linear
    original_triton = finegrained_fp8.finegrained_fp8_linear

    def tracked_deepgemm(*args, **kwargs):
        trace.append("deepgemm")
        return original_deepgemm(*args, **kwargs)

    def tracked_triton(*args, **kwargs):
        trace.append("triton")
        return original_triton(*args, **kwargs)

    finegrained_fp8.deepgemm_fp8_fp4_linear = tracked_deepgemm
    finegrained_fp8.finegrained_fp8_linear = tracked_triton
    try:
        yield trace
    finally:
        finegrained_fp8.deepgemm_fp8_fp4_linear = original_deepgemm
        finegrained_fp8.finegrained_fp8_linear = original_triton


def classify_exception(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "does not require grad" in text or "no grad_fn" in text:
        return "no_grad_fn"
    if "derivative" in text or "backward" in text and "not implemented" in text:
        return "derivative_not_implemented"
    if "unknown sf transformation" in text or "layout" in text:
        return "weight_or_scale_layout"
    if "out of memory" in text:
        return "oom"
    if "kernel" in text or "triton" in text or "deepgemm" in text:
        return "kernel_error"
    return "other_error"


def target_tensors(target: QuantizedTarget) -> tuple[int, Callable[[torch.Tensor], torch.Tensor], dict]:
    module = target.module
    if target.kind == "fp8_linear":
        weight = getattr(module, "weight")
        input_dim = int(weight.shape[-1])
        return input_dim, module, {
            "weight_dtype": str(weight.dtype),
            "weight_shape": list(weight.shape),
        }
    if target.kind == "fp8_expert_gate_up":
        weight = module.gate_up_proj[0]
        scale = module.gate_up_proj_scale_inv[0]

        def call(inputs: torch.Tensor) -> torch.Tensor:
            return module.linear(inputs, weight, scale, activation_scale=None)

        return int(module.hidden_dim), call, {
            "weight_dtype": str(weight.dtype),
            "weight_shape": list(weight.shape),
            "scale_dtype": str(scale.dtype),
            "scale_shape": list(scale.shape),
            "expert_index": 0,
        }
    if target.kind == "fp4_expert_down":
        weight = module.down_proj[0]
        scale = module.down_proj_scale_inv[0]

        def call(inputs: torch.Tensor) -> torch.Tensor:
            return module.linear(inputs, weight, scale, activation_scale=None)

        return int(module.intermediate_dim), call, {
            "weight_dtype": str(weight.dtype),
            "weight_shape": list(weight.shape),
            "scale_dtype": str(scale.dtype),
            "scale_shape": list(scale.shape),
            "expert_index": 0,
        }
    raise ValueError(f"unknown target kind: {target.kind}")


def run_target_probe(
    target: QuantizedTarget,
    *,
    dtype: torch.dtype,
    device: str | torch.device = "auto",
    seed: int = 0,
) -> dict:
    """对实际 target 调用 ``torch.autograd.grad`` 并保留完整失败分类。"""

    module = target.module
    module.requires_grad_(False)
    for parameter in module.parameters():
        parameter.grad = None
    selected_device = module_device(module) if str(device) == "auto" else torch.device(device)
    metadata = {
        "kind": target.kind,
        "module_path": target.module_path,
        "module_class": type(module).__name__,
        "module_python_path": type(module).__module__,
        "backend": None,
        "backend_attempts": [],
        "device": str(selected_device),
        "input_dtype": str(dtype),
    }
    try:
        input_dim, call, tensor_metadata = target_tensors(target)
        metadata.update(tensor_metadata)
        generator = torch.Generator(device=selected_device)
        generator.manual_seed(seed)
        inputs = torch.randn(
            4,
            input_dim,
            generator=generator,
            device=selected_device,
            dtype=dtype,
            requires_grad=True,
        )
        with capture_backend(module) as backend_attempts:
            output = call(inputs)
        metadata["backend_attempts"] = list(backend_attempts)
        metadata["backend"] = backend_attempts[-1] if backend_attempts else "unknown"
        if isinstance(output, tuple):
            output = output[0]
        metadata.update({
            "input_shape": list(inputs.shape),
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype),
            "output_grad_fn": type(output.grad_fn).__name__ if output.grad_fn is not None else None,
            "output_finite": bool(torch.isfinite(output).all()),
        })
        (gradient,) = torch.autograd.grad(output.float().sum(), (inputs,), allow_unused=True)
        weight_grads = [parameter.grad for parameter in module.parameters()]
        metadata.update({
            "input_grad_exists": gradient is not None,
            "input_grad_finite": bool(gradient is not None and torch.isfinite(gradient).all()),
            "input_grad_nonzero": bool(gradient is not None and float(gradient.detach().abs().max()) > 0),
            "input_grad_norm": float(torch.linalg.vector_norm(gradient.float())) if gradient is not None else None,
            "weight_grads_all_none": all(value is None for value in weight_grads),
            "exception_class": None,
            "exception_category": None,
            "exception_message": None,
            "traceback": None,
        })
        metadata["pass"] = bool(
            metadata["output_finite"]
            and metadata["output_grad_fn"]
            and metadata["input_grad_finite"]
            and metadata["input_grad_nonzero"]
            and metadata["weight_grads_all_none"]
        )
    except Exception as exc:  # 失败本身就是 Gate D 产物，必须结构化保存
        if "backend_attempts" in locals():
            metadata["backend_attempts"] = list(backend_attempts)
            metadata["backend"] = backend_attempts[-1] if backend_attempts else "unknown"
        metadata.update({
            "pass": False,
            "input_grad_exists": False,
            "input_grad_finite": False,
            "input_grad_nonzero": False,
            "input_grad_norm": None,
            "weight_grads_all_none": all(parameter.grad is None for parameter in module.parameters()),
            "output_grad_fn": None,
            "exception_class": type(exc).__name__,
            "exception_category": classify_exception(exc),
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        })
    return metadata


def dgrad_verdict(module: torch.nn.Module, hidden_size: int, dtype: torch.dtype, device: str) -> dict:
    """兼容旧调用的普通冻结 Linear reference 判定。"""

    module.requires_grad_(False)
    module.to(device=device, dtype=dtype)
    for parameter in module.parameters():
        parameter.grad = None
    inputs = torch.randn(2, 4, hidden_size, device=device, dtype=dtype, requires_grad=True)
    output = module(inputs)
    if isinstance(output, tuple):
        output = output[0]
    output_grad_fn = type(output.grad_fn).__name__ if output.grad_fn is not None else None
    (gradient,) = torch.autograd.grad(output.float().square().mean(), (inputs,), allow_unused=True)
    verdict = {
        "module_class": type(module).__name__,
        "input_shape": [2, 4, hidden_size],
        "input_grad_exists": gradient is not None,
        "input_grad_finite": bool(gradient is not None and torch.isfinite(gradient).all()),
        "input_grad_nonzero": bool(gradient is not None and float(gradient.detach().abs().max()) > 0),
        "input_grad_norm": float(torch.linalg.vector_norm(gradient.float())) if gradient is not None else None,
        "weight_grads_all_none": all(parameter.grad is None for parameter in module.parameters()),
        "output_dtype": str(output.dtype),
        "output_grad_fn": output_grad_fn,
    }
    verdict["pass"] = bool(
        verdict["input_grad_finite"]
        and verdict["input_grad_nonzero"]
        and verdict["weight_grads_all_none"]
        and verdict["output_grad_fn"]
    )
    return verdict


def run_reference_mode(
    *,
    dtype: torch.dtype,
    device: str,
    hidden_size: int = 64,
    out_size: int = 48,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    layer = torch.nn.Linear(hidden_size, out_size, bias=True)
    probe = dgrad_verdict(layer, hidden_size=hidden_size, dtype=dtype, device=device)
    return {
        "mode": "reference",
        "quantized_path": False,
        "verdict": "pass" if probe["pass"] else "fail",
        "seed": seed,
        "probe": probe,
        "interpretation": "harness-only; this result cannot pass a quantized Gate D",
    }


def candidate_error_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    if candidate.shape != reference.shape:
        raise ValueError(f"candidate/reference shape mismatch: {candidate.shape} vs {reference.shape}")
    candidate = candidate.detach().float().reshape(-1)
    reference = reference.detach().float().reshape(-1)
    absolute = (candidate - reference).abs()
    relative = absolute / reference.abs().clamp_min(torch.finfo(torch.float32).eps)
    cosine = torch.nn.functional.cosine_similarity(candidate, reference, dim=0)
    return {
        "max_abs_error": float(absolute.max()),
        "max_relative_error": float(relative.max()),
        "cosine_similarity": float(cosine),
    }


def run_candidate_reference_mode(
    *, dtype: torch.dtype, device: str, hidden_size: int = 32, out_size: int = 24, seed: int = 0
) -> dict:
    """验证自定义 Function 接口；真实量化 kernel 状态保持 hardware_pending。"""

    torch.manual_seed(seed)
    selected_device = torch.device(device)
    weight = torch.randn(out_size, hidden_size, device=selected_device, dtype=dtype)
    weight.requires_grad_(False)
    scale = torch.ones(1, device=selected_device, dtype=torch.float32)
    inputs = torch.randn(5, hidden_size, device=selected_device, dtype=dtype, requires_grad=True)

    def forward_op(x, w, _scale, _bias):
        return torch.nn.functional.linear(x, w)

    def dgrad_op(grad_output, w, _scale):
        return grad_output.to(w.dtype) @ w

    output = InputOnlyDgradFunction.apply(inputs, weight, scale, None, forward_op, dgrad_op)
    candidate_grad = torch.autograd.grad(output.float().square().mean(), inputs)[0]
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    reference_output = torch.nn.functional.linear(reference_inputs, weight)
    reference_grad = torch.autograd.grad(reference_output.float().square().mean(), reference_inputs)[0]
    metrics = candidate_error_metrics(candidate_grad, reference_grad)
    interface_pass = bool(
        torch.isfinite(candidate_grad).all()
        and float(candidate_grad.abs().max()) > 0
        and weight.grad is None
        and scale.grad is None
        and math.isfinite(metrics["cosine_similarity"])
    )
    return {
        "mode": "candidate",
        "quantized_forward_executed": False,
        "candidate_interface": "input-only-autograd-v1",
        "reference_interface_pass": interface_pass,
        "reference_error_metrics": metrics,
        "frozen_weight_grad_none": weight.grad is None,
        "frozen_scale_grad_none": scale.grad is None,
        "verdict": "hardware_pending" if interface_pass else "reference_fail",
        "interpretation": (
            "the input-only autograd boundary is validated with ordinary linear math; "
            "an actual FP8/FP4 forward and DGRAD adapter is still required on target hardware"
        ),
    }


def load_actual_quantized_model(weights: Path, *, dtype: torch.dtype, device: str):
    """通过 Transformers from_pretrained 触发真实 quantizer replacement。"""

    import transformers

    config = transformers.AutoConfig.from_pretrained(str(weights), local_files_only=True)
    placement = "auto" if device == "auto" else {"": device}
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(weights),
        config=config,
        dtype=dtype,
        device_map=placement,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.requires_grad_(False).eval()
    return model


def runtime_metadata() -> dict:
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        result.update({
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        })
    try:
        import transformers

        result["transformers"] = transformers.__version__
    except Exception as exc:
        result["transformers_error"] = f"{type(exc).__name__}: {exc}"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("reference", "native", "candidate"), default="native")
    parser.add_argument("--weights", type=Path, help="本地 DeepSeek-V4-Flash checkpoint 目录")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--targets", nargs="+", choices=REQUIRED_TARGET_KINDS, default=list(REQUIRED_TARGET_KINDS))
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def run(args: argparse.Namespace) -> tuple[dict, int]:
    dtype = dtype_from_name(args.dtype)
    metadata = runtime_metadata()
    if args.mode == "reference":
        result = run_reference_mode(dtype=dtype, device=args.device, seed=args.seed)
        report = {"status": result["verdict"], "runtime": metadata, "result": result}
        return report, 0 if result["verdict"] == "pass" else 1
    if args.mode == "candidate":
        result = run_candidate_reference_mode(dtype=dtype, device=args.device, seed=args.seed)
        report = {"status": result["verdict"], "runtime": metadata, "result": result}
        if args.weights:
            config_path = args.weights / "config.json"
            report["source"] = {
                "weights": str(args.weights),
                "config_sha256": sha256(config_path) if config_path.exists() else None,
            }
        return report, 2 if result["verdict"] == "hardware_pending" else 1
    if args.weights is None:
        raise ValueError("--weights is required for native mode")
    config = discover_config(args.weights)
    shard_inventory = discover_shard_dtypes(args.weights)
    model = load_actual_quantized_model(args.weights, dtype=dtype, device=args.device)
    all_targets = find_quantized_targets(model)
    wanted = set(args.targets)
    targets = [target for target in all_targets if target.kind in wanted]
    rows = [
        run_target_probe(target, dtype=dtype, device="auto", seed=args.seed + index)
        for index, target in enumerate(targets)
    ]
    complete = set(row["kind"] for row in rows) == wanted
    passed = complete and all(row["pass"] for row in rows)
    report = {
        "status": "pass" if passed else "fail",
        "mode": "native",
        "runtime": metadata,
        "source": {
            "weights": str(args.weights),
            "config_sha256": sha256(args.weights / "config.json"),
            "weight_index_sha256": sha256(args.weights / "model.safetensors.index.json")
            if (args.weights / "model.safetensors.index.json").exists()
            else None,
            "quantization": config.get("quantization_config") or config.get("quantization"),
            "shard_inventory": shard_inventory,
        },
        "required_targets": list(args.targets),
        "targets_complete": complete,
        "targets": rows,
        "interpretation": "all actual quantized targets must pass; partial success is a Gate D failure",
    }
    return report, 0 if passed else 1


def main() -> None:
    args = parse_args()
    try:
        report, exit_code = run(args)
    except Exception as exc:
        report = {
            "status": "fail",
            "mode": args.mode,
            "runtime": runtime_metadata(),
            "exception_class": type(exc).__name__,
            "exception_category": classify_exception(exc),
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    rendered = json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n"
    print(rendered, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.exists():
            raise FileExistsError(f"refusing to overwrite DGRAD result: {args.out}")
        args.out.write_text(rendered, encoding="utf-8")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
