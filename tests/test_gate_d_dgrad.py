"""Gate D stage-0 reproducer: linear discovery and frozen-contract Dgrad verdict."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from gate_d_dgrad import dgrad_verdict, find_target_linear


class _TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.ModuleDict({"q_proj": torch.nn.Linear(8, 8)})
        self.mlp = torch.nn.ModuleDict({"gate_proj": torch.nn.Linear(8, 32)})


class _TinyInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_TinyLayer()])


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _TinyInner()


def test_find_target_linear_picks_first_2d_weight_of_layer0():
    name, module = find_target_linear(_TinyModel())
    assert name == "self_attn.q_proj"
    assert isinstance(module, torch.nn.Linear)


def test_dgrad_verdict_passes_plain_linear_and_keeps_weights_frozen():
    module = torch.nn.Linear(8, 8)
    verdict = dgrad_verdict(module, hidden_size=8, dtype=torch.float32, device="cpu")
    assert verdict["pass"]
    assert verdict["input_grad_nonzero"]
    assert verdict["weight_grads_all_none"]
    assert all(p.grad is None for p in module.parameters())
