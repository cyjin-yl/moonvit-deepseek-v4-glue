"""Gate D stage 0: quantization discovery + minimal Dgrad reproducer.

Run this on the rented box BEFORE loading the full DeepSeek-V4-Flash-0731.
It answers exactly one question: can the quantized linear layer that the real
checkpoint uses propagate gradients to its INPUT? (The frozen LLM never needs
weight gradients, but the projector can only learn if loss gradients reach the
visual embeddings through every LLM layer.)

"Blackwell has FP4 tensor cores" does NOT imply "this NVFP4 kernel has an
input-gradient path" — this script measures instead of assuming. Stages:

  A. config discovery: print the quantization section of config.json and the
     actual safetensors dtypes of the first shard (no full load).
  B. minimal Dgrad reproducer: instantiate the model on meta device, take the
     first real linear of decoder layer 0, materialize ONLY that module's
     tensors from the checkpoint slice, then forward+backward with
     input.requires_grad=True. PASS iff input.grad is finite and nonzero and
     every module parameter keeps grad=None (frozen contract).

Exit code 0 = Dgrad path exists at layer level; 1 = do not train, escalate to
scenario A'/B. Prints a JSON verdict on the last line for the run log.

Example::

    python tools/gate_d_dgrad.py --weights /root/weights/dsv4f
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def discover_config(weights: Path) -> dict:
    config = json.loads((weights / "config.json").read_text())
    quant = config.get("quantization_config") or config.get("quantization")
    print("== config discovery ==")
    print(f"model_type={config.get('model_type')} torch_dtype={config.get('torch_dtype')}")
    print(f"quantization_config={json.dumps(quant, indent=2, default=str) if quant else None}")
    return config


def discover_shard_dtypes(weights: Path) -> dict:
    from safetensors import safe_open

    index_path = weights / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        first_shard = sorted(set(index["weight_map"].values()))[0]
    else:
        first_shard = "model.safetensors"
    dtypes: dict[str, int] = {}
    with safe_open(str(weights / first_shard), framework="pt") as handle:
        for key in list(handle.keys())[:200]:
            dtype = str(handle.get_slice(key).get_dtype())
            dtypes[dtype] = dtypes.get(dtype, 0) + 1
    print(f"== first shard ({first_shard}) dtypes (first 200 tensors): {dtypes} ==")
    return dtypes


def find_target_linear(root: torch.nn.Module) -> tuple[str, torch.nn.Module]:
    """First module of decoder layer 0 holding a direct weight param/buffer."""

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


def load_module_slice(weights: Path, module_path: str, module: torch.nn.Module) -> dict:
    from safetensors import safe_open

    index_path = weights / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"] if index_path.exists() else None
    prefix = module_path + "."
    if index is not None:
        wanted = {key: shard for key, shard in index.items() if key.startswith(prefix)}
        shards: dict[str, list[str]] = {}
        for key, shard in wanted.items():
            shards.setdefault(shard, []).append(key)
    else:
        shards = {"model.safetensors": None}
    slice_sd: dict[str, torch.Tensor] = {}
    for shard, keys in shards.items():
        with safe_open(str(weights / shard), framework="pt") as handle:
            for key in (keys or [k for k in handle.keys() if k.startswith(prefix)]):
                slice_sd[key[len(prefix):]] = handle.get_tensor(key)
    if not slice_sd:
        raise SystemExit(f"no checkpoint tensors under prefix {prefix!r}")
    incompatible = module.load_state_dict(slice_sd, strict=False, assign=True)
    print(f"== slice load: {len(slice_sd)} tensors; "
          f"missing={list(incompatible.missing_keys)} unexpected={list(incompatible.unexpected_keys)} ==")
    return {"missing": list(incompatible.missing_keys), "unexpected": list(incompatible.unexpected_keys)}


def dgrad_verdict(module: torch.nn.Module, hidden_size: int, dtype: torch.dtype, device: str) -> dict:
    module.requires_grad_(False)
    module.to(device)
    inputs = torch.randn(2, 4, hidden_size, device=device, dtype=dtype, requires_grad=True)
    output = module(inputs)
    if isinstance(output, tuple):
        output = output[0]
    loss = output.float().square().mean()
    loss.backward()
    grad = inputs.grad
    weight_grads = [p.grad for p in module.parameters()]
    verdict = {
        "module_class": type(module).__name__,
        "input_shape": [2, 4, hidden_size],
        "input_grad_exists": grad is not None,
        "input_grad_finite": bool(grad is not None and torch.isfinite(grad).all()),
        "input_grad_nonzero": bool(grad is not None and grad.abs().max() > 0),
        "weight_grads_all_none": all(g is None for g in weight_grads),
        "output_dtype": str(output.dtype),
    }
    verdict["pass"] = (
        verdict["input_grad_finite"] and verdict["input_grad_nonzero"] and verdict["weight_grads_all_none"]
    )
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="local 0731 checkpoint dir")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import transformers

    print(f"torch {torch.__version__} transformers {transformers.__version__}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")

    config = discover_config(args.weights)
    discover_shard_dtypes(args.weights)

    print("== instantiating full model on meta device (no weights) ==")
    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(
            transformers.AutoConfig.from_pretrained(str(args.weights))
        )
    module_path, module = find_target_linear(model)
    print(f"target linear: {module_path} ({type(module).__name__})")

    placement = load_module_slice(args.weights, module_path, module)
    dtype = getattr(torch, config.get("torch_dtype", "bfloat16"))
    verdict = dgrad_verdict(module, config["hidden_size"], dtype, args.device)
    verdict.update({
        "module": module_path,
        "quantization": config.get("quantization_config") or config.get("quantization"),
        "slice_missing_keys": placement["missing"],
        "slice_unexpected_keys": placement["unexpected"],
    })
    print(json.dumps({"gate_d_stage0": verdict}, indent=2, default=str))
    sys.exit(0 if verdict["pass"] else 1)


if __name__ == "__main__":
    main()
