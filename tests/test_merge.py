import pytest
import torch

from moonvit_glue.merge import expand_image_placeholders


def test_one_placeholder_expands_to_all_image_tokens_and_masks_their_labels():
    input_ids = torch.tensor([[11, 99, 12]])
    text_embeddings = torch.tensor(
        [[[1.0, 1.0], [9.0, 9.0], [2.0, 2.0]]], requires_grad=True
    )
    image_embeddings = [torch.tensor([[3.0, 3.0], [4.0, 4.0]])]
    labels = torch.tensor([[21, 22, 23]])

    merged = expand_image_placeholders(
        input_ids=input_ids,
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        placeholder_token_id=99,
        labels=labels,
    )

    assert merged.inputs_embeds.tolist() == [
        [[1.0, 1.0], [3.0, 3.0], [4.0, 4.0], [2.0, 2.0]]
    ]
    assert merged.routing_input_ids.tolist() == [[11, 99, 99, 12]]
    assert merged.attention_mask.tolist() == [[1, 1, 1, 1]]
    assert merged.labels.tolist() == [[21, -100, -100, 23]]
    assert merged.position_ids.tolist() == [[0, 1, 2, 3]]


def test_image_count_must_match_placeholder_count():
    with pytest.raises(ValueError, match="1 image placeholder.*0 image feature"):
        expand_image_placeholders(
            input_ids=torch.tensor([[99, 12]]),
            text_embeddings=torch.zeros(1, 2, 4),
            image_embeddings=[],
            placeholder_token_id=99,
        )
