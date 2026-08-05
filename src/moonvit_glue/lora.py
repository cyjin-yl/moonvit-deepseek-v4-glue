"""源码可直接修改的顶部 LoRA 线性层与严格序列化合同。"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    """保留原始 linear，并显式暴露 fp32 A/B 低秩参数。"""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, inputs: Tensor) -> Tensor:
        base_output = self.base(inputs)
        low_rank = torch.nn.functional.linear(inputs.to(torch.float32), self.lora_a)
        delta = torch.nn.functional.linear(low_rank, self.lora_b) * self.scaling
        return base_output + delta.to(dtype=base_output.dtype)


def _resolve_parent(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid module path: {dotted_name!r}")
    parent = root
    for part in parts[:-1]:
        if not hasattr(parent, part):
            raise ValueError(f"LoRA module path does not exist: {dotted_name}")
        parent = getattr(parent, part)
        if not isinstance(parent, nn.Module):
            raise ValueError(f"LoRA path component is not a module: {dotted_name}")
    return parent, parts[-1]


def inject_lora(
    language_model: nn.Module,
    *,
    layer_indices: Sequence[int],
    target_modules: Sequence[str],
    rank: int,
    alpha: float,
    seed: int,
) -> list[str]:
    """只替换预注册 decoder layer 与相对 module path。"""
    layers = getattr(getattr(language_model, "model", None), "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise ValueError("language model must expose model.layers as ModuleList")
    indices = [int(index) for index in layer_indices]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("LoRA layer indices must be non-empty and unique")
    if min(indices) < 0 or max(indices) >= len(layers):
        raise ValueError("LoRA layer index exceeds language-model depth")
    targets = [str(name) for name in target_modules]
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("LoRA target modules must be non-empty and unique")
    replaced = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        for layer_index in sorted(indices):
            layer = layers[layer_index]
            for target_name in targets:
                parent, attribute = _resolve_parent(layer, target_name)
                if not hasattr(parent, attribute):
                    raise ValueError(f"LoRA module path does not exist: {target_name}")
                base = getattr(parent, attribute)
                if not isinstance(base, nn.Linear) or isinstance(base, LoRALinear):
                    raise ValueError(
                        f"LoRA target must be an unwrapped nn.Linear: layer {layer_index} {target_name}"
                    )
                wrapper = LoRALinear(base, rank=rank, alpha=alpha)
                wrapper.to(device=base.weight.device)
                setattr(parent, attribute, wrapper)
                replaced.append(f"model.layers.{layer_index}.{target_name}")
    return replaced


def iter_lora_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    for child in module.modules():
        if isinstance(child, LoRALinear):
            yield child.lora_a
            yield child.lora_b


def freeze_non_lora(module: nn.Module) -> int:
    """冻结完整主干，再只打开 LoRA A/B，并返回可训练参数量。"""
    module.requires_grad_(False)
    parameters = list(iter_lora_parameters(module))
    if not parameters:
        raise ValueError("no LoRA parameters were injected")
    for parameter in parameters:
        parameter.requires_grad_(True)
    return sum(parameter.numel() for parameter in parameters)


def lora_state_dict(module: nn.Module) -> dict[str, Tensor]:
    state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in module.named_parameters()
        if name.endswith(".lora_a") or name.endswith(".lora_b")
    }
    if not state:
        raise ValueError("cannot serialize an empty LoRA state")
    return state


def load_lora_state_dict(module: nn.Module, state: dict[str, Tensor]) -> None:
    current = {
        name: parameter
        for name, parameter in module.named_parameters()
        if name.endswith(".lora_a") or name.endswith(".lora_b")
    }
    if set(current) != set(state):
        missing = sorted(set(current) - set(state))
        unexpected = sorted(set(state) - set(current))
        raise ValueError(f"LoRA state keys mismatch: missing={missing}, unexpected={unexpected}")
    with torch.no_grad():
        for name, parameter in current.items():
            value = state[name]
            if value.shape != parameter.shape:
                raise ValueError(f"LoRA tensor shape mismatch: {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

