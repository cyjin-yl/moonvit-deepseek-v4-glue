"""Gate D 三模式 reproducer 的参考、目标发现与 input-only DGRAD 合同。"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from gate_d_dgrad import (
    InputOnlyDgradFunction,
    candidate_error_metrics,
    dgrad_verdict,
    find_quantized_targets,
    find_target_linear,
    run_reference_mode,
    run_target_probe,
)


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


def test_reference_mode_is_explicitly_a_harness_result():
    result = run_reference_mode(dtype=torch.float32, device="cpu", hidden_size=8, out_size=6, seed=7)
    assert result["mode"] == "reference"
    assert result["quantized_path"] is False
    assert result["verdict"] == "pass"
    assert result["probe"]["input_grad_finite"]
    assert result["probe"]["weight_grads_all_none"]


class FP8Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(6, 8), requires_grad=False)

    def forward(self, inputs):
        return torch.nn.functional.linear(inputs, self.weight)


class FP8Experts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 8
        self.intermediate_dim = 5
        self.gate_up_proj = torch.nn.Parameter(torch.randn(2, 10, 8), requires_grad=False)
        self.gate_up_proj_scale_inv = torch.nn.Parameter(torch.ones(2, 10, 1), requires_grad=False)
        self.down_proj = torch.nn.Parameter(torch.randn(2, 8, 5), requires_grad=False)
        self.down_proj_scale_inv = torch.nn.Parameter(torch.ones(2, 8, 1), requires_grad=False)

    def linear(self, inputs, weight, weight_scale_inv, activation_scale=None):
        del weight_scale_inv, activation_scale
        return torch.nn.functional.linear(inputs, weight)


class _QuantizedRoot(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = FP8Linear()
        self.experts = FP8Experts()


def test_quantized_target_discovery_returns_all_three_required_kinds():
    targets = find_quantized_targets(_QuantizedRoot())
    assert [target.kind for target in targets] == [
        "fp8_linear",
        "fp8_expert_gate_up",
        "fp4_expert_down",
    ]
    assert targets[0].module_path == "attn"
    assert targets[1].module_path == "experts"
    assert targets[2].module_path == "experts"


def test_native_target_probe_records_grad_fn_and_frozen_contract():
    targets = find_quantized_targets(_QuantizedRoot())
    rows = [run_target_probe(target, dtype=torch.float32, device="cpu", seed=11) for target in targets]
    assert all(row["pass"] for row in rows)
    assert all(row["backend"] == "reference" for row in rows)
    assert all(row["backend_attempts"] == ["reference"] for row in rows)
    assert all(row["output_grad_fn"] for row in rows)
    assert all(row["input_grad_nonzero"] for row in rows)
    assert all(row["weight_grads_all_none"] for row in rows)


def test_input_only_dgrad_function_returns_only_input_gradient():
    inputs = torch.randn(3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=False)
    scale = torch.ones(1)

    def forward_op(x, w, _scale, _bias):
        return torch.nn.functional.linear(x, w)

    def dgrad_op(grad_output, w, _scale):
        return grad_output @ w

    output = InputOnlyDgradFunction.apply(inputs, weight, scale, None, forward_op, dgrad_op)
    output.square().sum().backward()
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    torch.nn.functional.linear(reference_inputs, weight).square().sum().backward()

    assert torch.allclose(inputs.grad, reference_inputs.grad)
    assert weight.grad is None
    assert scale.grad is None


def test_candidate_error_metrics_report_alignment():
    reference = torch.tensor([1.0, -2.0, 3.0])
    candidate = reference + torch.tensor([0.01, 0.02, -0.03])
    metrics = candidate_error_metrics(candidate, reference)
    assert abs(metrics["max_abs_error"] - 0.03) < 1e-6
    assert metrics["max_relative_error"] > 0
    assert metrics["cosine_similarity"] > 0.999
