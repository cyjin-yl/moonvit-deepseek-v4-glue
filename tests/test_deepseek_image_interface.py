import torch

from deepseek_interface_screen_v1 import build_model, merge_invariants
from moonvit_glue.merge import expand_image_placeholders


def test_interface_screen_merge_invariants_are_explicit_and_frozen():
    placeholder = 63
    model = build_model(seed=20260805, device=torch.device("cpu"), placeholder_token_id=placeholder)
    input_ids = torch.tensor([[1, placeholder, 5, 7, 2]])
    labels = torch.tensor([[1, -100, 5, 7, 2]])
    text_embeddings = model.language_model.get_input_embeddings()(input_ids)
    merged = expand_image_placeholders(
        input_ids=input_ids,
        text_embeddings=text_embeddings,
        image_embeddings=[torch.zeros(3, 32)],
        placeholder_token_id=placeholder,
        labels=labels,
    )
    checks = merge_invariants(
        merged,
        input_ids=input_ids,
        labels=labels,
        placeholder_token_id=placeholder,
        image_token_count=3,
    )
    assert checks["pass"] is True
    assert merged.routing_input_ids.tolist() == [[1, placeholder, placeholder, placeholder, 5, 7, 2]]
    assert merged.position_ids.tolist() == [[0, 1, 2, 3, 4, 5, 6]]

