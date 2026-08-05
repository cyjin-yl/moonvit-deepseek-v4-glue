"""顶部 LoRA 诊断必须保持零初始扰动与显式可训练参数边界。"""

import torch
from torch import nn

from moonvit_glue.lora import (
    LoRALinear,
    freeze_non_lora,
    inject_lora,
    load_lora_state_dict,
    lora_state_dict,
)


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 2, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer(), TinyLayer(), TinyLayer()])


def test_lora_linear_is_exact_noop_at_initialization():
    base = nn.Linear(5, 3, bias=False)
    wrapped = LoRALinear(base, rank=2, alpha=4.0)
    inputs = torch.randn(2, 5)
    assert torch.equal(wrapped(inputs), base(inputs))
    assert wrapped.lora_a.dtype == torch.float32
    assert wrapped.lora_b.dtype == torch.float32


def test_inject_lora_only_mutates_requested_top_layers_and_targets():
    model = TinyBackbone()
    replaced = inject_lora(
        model,
        layer_indices=[1, 2],
        target_modules=["self_attn.q_proj", "self_attn.v_proj"],
        rank=2,
        alpha=4.0,
        seed=7,
    )
    assert replaced == [
        "model.layers.1.self_attn.q_proj",
        "model.layers.1.self_attn.v_proj",
        "model.layers.2.self_attn.q_proj",
        "model.layers.2.self_attn.v_proj",
    ]
    assert isinstance(model.model.layers[0].self_attn.q_proj, nn.Linear)
    assert isinstance(model.model.layers[1].self_attn.q_proj, LoRALinear)


def test_freeze_non_lora_preserves_only_adapter_gradients():
    model = TinyBackbone()
    inject_lora(
        model,
        layer_indices=[2],
        target_modules=["self_attn.q_proj", "self_attn.o_proj"],
        rank=2,
        alpha=2.0,
        seed=11,
    )
    trainable = freeze_non_lora(model)
    assert trainable == 32
    assert all(
        parameter.requires_grad == ("lora_a" in name or "lora_b" in name)
        for name, parameter in model.named_parameters()
    )
    model.model.layers[2].self_attn.q_proj(torch.randn(3, 4)).sum().backward()
    assert model.model.layers[2].self_attn.q_proj.lora_b.grad is not None
    assert model.model.layers[2].self_attn.q_proj.base.weight.grad is None


def test_lora_state_roundtrip_is_strict_and_excludes_base_weights():
    source = TinyBackbone()
    target = TinyBackbone()
    kwargs = dict(
        layer_indices=[2],
        target_modules=["self_attn.q_proj"],
        rank=2,
        alpha=4.0,
        seed=13,
    )
    inject_lora(source, **kwargs)
    inject_lora(target, **kwargs)
    source.model.layers[2].self_attn.q_proj.lora_b.data.fill_(0.25)
    state = lora_state_dict(source)
    assert set(state) == {
        "model.layers.2.self_attn.q_proj.lora_a",
        "model.layers.2.self_attn.q_proj.lora_b",
    }
    load_lora_state_dict(target, state)
    assert torch.equal(
        target.model.layers[2].self_attn.q_proj.lora_b,
        source.model.layers[2].self_attn.q_proj.lora_b,
    )

