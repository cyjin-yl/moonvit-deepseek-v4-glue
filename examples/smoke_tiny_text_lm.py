"""Offline smoke test: graft fake visual features onto a tiny text-only LM."""

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from moonvit_glue import PatchMergerProjector, ProjectorConfig, VisionCausalLM


torch.manual_seed(7)
language_model = GPT2LMHeadModel(
    GPT2Config(
        vocab_size=64,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_positions=64,
        bos_token_id=1,
        eos_token_id=2,
    )
)
projector = PatchMergerProjector(
    ProjectorConfig(
        vision_width=8,
        language_width=32,
        merge_factor=4,
        projector_width=32,
    )
)
model = VisionCausalLM(
    language_model=language_model,
    projector=projector,
    placeholder_token_id=63,
    backbone_kind="generic",
)

outputs = model(
    input_ids=torch.tensor([[1, 10, 63, 11, 12, 2]]),
    image_feature_groups=[torch.randn(3, 4, 8)],
    labels=torch.tensor([[1, 10, -100, 11, 12, 2]]),
)
outputs.loss.backward()

projector_grads = sum(parameter.grad is not None for parameter in projector.parameters())
language_grads = sum(parameter.grad is not None for parameter in language_model.parameters())
print(f"loss={outputs.loss.item():.4f}")
print(f"expanded_sequence_length={outputs.logits.shape[1]}")
print(f"projector_parameters_with_grad={projector_grads}")
print(f"language_parameters_with_grad={language_grads}")

