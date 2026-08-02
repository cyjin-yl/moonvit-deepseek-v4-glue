import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

from moonvit_glue.model import VisionCausalLM
from moonvit_glue.projector import PatchMergerProjector, ProjectorConfig


def _tiny_gpt2() -> GPT2LMHeadModel:
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_embd=8,
            n_layer=1,
            n_head=2,
            n_positions=32,
            bos_token_id=1,
            eos_token_id=2,
        )
    )


def _tiny_deepseek() -> DeepseekV4ForCausalLM:
    return DeepseekV4ForCausalLM(
        DeepseekV4Config(
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
    )


def test_generic_backbone_generates_with_expanded_image_tokens():
    model = VisionCausalLM(
        language_model=_tiny_gpt2(),
        projector=PatchMergerProjector(
            ProjectorConfig(vision_width=3, language_width=8, merge_factor=2, projector_width=6)
        ),
        placeholder_token_id=31,
        backbone_kind="generic",
    )
    output = model.generate(
        input_ids=torch.tensor([[1, 31, 7, 8, 2]]),
        image_feature_groups=[torch.randn(2, 2, 3)],
        max_new_tokens=3,
        min_new_tokens=3,
        do_sample=False,
        pad_token_id=2,
    )
    # 5 prompt tokens, the placeholder expands to 2 image tokens, plus 3 new.
    assert output.shape == (1, 5 - 1 + 2 + 3)


def test_deepseek_backbone_generates_with_routing_ids_attached():
    model = VisionCausalLM(
        language_model=_tiny_deepseek(),
        projector=PatchMergerProjector(
            ProjectorConfig(vision_width=3, language_width=32, merge_factor=1, projector_width=8)
        ),
        placeholder_token_id=63,
        backbone_kind="deepseek_v4",
    )
    output = model.generate(
        input_ids=torch.tensor([[1, 63, 5, 2]]),
        image_feature_groups=[torch.randn(2, 1, 3)],
        max_new_tokens=3,
        min_new_tokens=3,
        do_sample=False,
        pad_token_id=0,
    )
    # 4 prompt tokens, the placeholder expands to 2 image tokens, plus 3 new.
    assert output.shape == (1, 4 - 1 + 2 + 3)
    # The expanded routing prefix (placeholder repeated per image token) is kept.
    assert output[0, :5].tolist() == [1, 63, 63, 5, 2]


def test_generate_rejects_ambiguous_image_inputs():
    model = VisionCausalLM(
        language_model=_tiny_gpt2(),
        projector=PatchMergerProjector(
            ProjectorConfig(vision_width=3, language_width=8, merge_factor=2, projector_width=6)
        ),
        placeholder_token_id=31,
        backbone_kind="generic",
    )
    try:
        model.generate(input_ids=torch.tensor([[1, 31, 2]]), max_new_tokens=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when no image input is given")
