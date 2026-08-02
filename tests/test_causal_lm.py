import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

from moonvit_glue.model import VisionCausalLM
from moonvit_glue.projector import PatchMergerProjector, ProjectorConfig


def test_projector_only_training_reaches_projector_through_a_frozen_text_lm():
    lm = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_embd=8,
            n_layer=1,
            n_head=2,
            n_positions=16,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    projector = PatchMergerProjector(
        ProjectorConfig(
            vision_width=3,
            language_width=8,
            merge_factor=2,
            projector_width=6,
        )
    )
    model = VisionCausalLM(
        language_model=lm,
        projector=projector,
        placeholder_token_id=31,
        backbone_kind="generic",
        freeze_language_model=True,
    )

    outputs = model(
        input_ids=torch.tensor([[1, 31, 7, 8, 2]]),
        image_feature_groups=[torch.randn(2, 2, 3)],
        labels=torch.tensor([[1, -100, 7, 8, 2]]),
    )
    outputs.loss.backward()

    assert outputs.logits.shape == (1, 6, 32)
    assert any(parameter.grad is not None for parameter in projector.parameters())
    assert all(parameter.grad is None for parameter in lm.parameters())


def test_deepseek_adapter_keeps_routing_ids_but_overrides_lookup_embeddings():
    class RecordingLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(128, 4)
            self.config = type("Config", (), {"model_type": "deepseek_v4"})()
            self.seen_ids = None
            self.seen_embeddings = None

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, input_ids, **kwargs):
            self.seen_ids = input_ids.detach().clone()
            self.seen_embeddings = self.embedding(input_ids)
            return self.seen_embeddings.sum()

    lm = RecordingLM()
    projector = PatchMergerProjector(
        ProjectorConfig(vision_width=2, language_width=4, merge_factor=1)
    )
    model = VisionCausalLM(
        language_model=lm,
        projector=projector,
        placeholder_token_id=99,
        backbone_kind="deepseek_v4",
    )

    projected = projector([torch.randn(2, 1, 2)])[0]
    expected = torch.stack(
        [
            lm.embedding.weight[5],
            projected[0],
            projected[1],
            lm.embedding.weight[6],
        ]
    ).unsqueeze(0)
    model(input_ids=torch.tensor([[5, 99, 6]]), image_embeddings=[projected])

    assert lm.seen_ids.tolist() == [[5, 99, 99, 6]]
    assert torch.allclose(lm.seen_embeddings, expected)


def test_real_transformers_deepseek_v4_hash_moe_backpropagates_to_projector():
    config = DeepseekV4Config(
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
    lm = DeepseekV4ForCausalLM(config)
    projector = PatchMergerProjector(
        ProjectorConfig(
            vision_width=3,
            language_width=32,
            merge_factor=1,
            projector_width=8,
        )
    )
    model = VisionCausalLM(
        language_model=lm,
        projector=projector,
        placeholder_token_id=63,
        backbone_kind="deepseek_v4",
    )

    outputs = model(
        input_ids=torch.tensor([[1, 63, 5, 2]]),
        image_feature_groups=[torch.randn(2, 1, 3)],
        labels=torch.tensor([[1, -100, 5, 2]]),
    )
    outputs.loss.backward()

    assert outputs.logits.shape == (1, 5, 64)
    assert all(parameter.grad is not None for parameter in projector.parameters())
    assert all(parameter.grad is None for parameter in lm.parameters())
