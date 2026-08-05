"""适配 verifier 的 tensor 与 balanced-batch 合同必须显式且封闭。"""

import pytest

from moonvit_glue.adaptation_verification import (
    analysis_contract_fields,
    expected_lora_state_keys,
    validate_balanced_task_history,
)


def test_balanced_comparison_analysis_uses_its_explicit_contract():
    assert analysis_contract_fields(
        {"format_version": "balanced-adaptation-comparison-analysis-v1"}
    ) == ("eval_summary_sha256", "contrasts", "mean_gap")


def test_expected_lora_state_keys_expands_only_a_and_b():
    assert expected_lora_state_keys(
        ["model.layers.22.self_attn.q_proj", "model.layers.23.self_attn.v_proj"]
    ) == {
        "model.layers.22.self_attn.q_proj.lora_a",
        "model.layers.22.self_attn.q_proj.lora_b",
        "model.layers.23.self_attn.v_proj.lora_a",
        "model.layers.23.self_attn.v_proj.lora_b",
    }


def test_balanced_history_requires_every_task_in_every_true_batch():
    history = [
        {"task_counts": {"color": 2, "shape": 2}},
        {"task_counts": {"color": 2, "shape": 2}},
    ]

    assert validate_balanced_task_history(
        history, tasks=["color", "shape"], batch_size=4
    ) == {"color": 4, "shape": 4}

    history[1]["task_counts"]["shape"] = 1
    with pytest.raises(ValueError, match="balanced adaptation task quota drift"):
        validate_balanced_task_history(history, tasks=["color", "shape"], batch_size=4)
