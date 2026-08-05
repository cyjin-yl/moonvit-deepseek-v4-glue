"""适配 verifier 的 LoRA tensor 合同必须显式且封闭。"""

from moonvit_glue.adaptation_verification import expected_lora_state_keys


def test_expected_lora_state_keys_expands_only_a_and_b():
    assert expected_lora_state_keys(
        ["model.layers.22.self_attn.q_proj", "model.layers.23.self_attn.v_proj"]
    ) == {
        "model.layers.22.self_attn.q_proj.lora_a",
        "model.layers.22.self_attn.q_proj.lora_b",
        "model.layers.23.self_attn.v_proj.lora_a",
        "model.layers.23.self_attn.v_proj.lora_b",
    }

