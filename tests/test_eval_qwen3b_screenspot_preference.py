import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from eval_qwen3b_screenspot_preference import (
    read_partial,
    score_candidate_pair,
    supervision_batch,
)
from moonvit_glue.chat_contract import ChatPrompt, ChatSupervision


def _supervision(ids, labels):
    return ChatSupervision(
        input_ids=ids,
        labels=labels,
        prompt_length=2,
        answer_tokens=sum(value != -100 for value in labels),
        prompt=ChatPrompt(
            input_ids=ids[:2],
            template_text_for_audit="prompt",
            semantic_system_prompt="system",
            semantic_user_prompt="user",
            placeholder_count=ids.count(9),
        ),
    )


class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 4)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, input_ids=None, inputs_embeds=None, **_kwargs):
        hidden = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        logits = torch.zeros((*hidden.shape[:2], 20), device=hidden.device)
        return SimpleNamespace(logits=logits)


class _Projector(nn.Module):
    def forward(self, groups):
        return [torch.ones((groups[0].shape[0], 4), device=groups[0].device)]


class _Receiver(nn.Module):
    def forward(self, inputs):
        return inputs


def test_supervision_batch_right_pads_without_scoring_padding():
    rows = [
        _supervision([1, 2, 3], [-100, -100, 3]),
        _supervision([1, 2, 4, 5], [-100, -100, 4, 5]),
    ]
    input_ids, attention_mask, labels = supervision_batch(
        rows, pad_token_id=0, device=torch.device("cpu")
    )
    assert input_ids.tolist() == [[1, 2, 3, 0], [1, 2, 4, 5]]
    assert attention_mask.tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]
    assert labels.tolist() == [[-100, -100, 3, -100], [-100, -100, 4, 5]]


def test_candidate_pair_scores_blind_and_visual_as_batch_two():
    language_model = _TinyLM()
    blind_rows = [
        _supervision([1, 2, 3], [-100, -100, 3]),
        _supervision([1, 2, 4], [-100, -100, 4]),
    ]
    blind = score_candidate_pair(
        language_model=language_model,
        projector=None,
        receiver=None,
        feature_groups=None,
        supervisions=blind_rows,
        placeholder_token_id=9,
        pad_token_id=0,
        device=torch.device("cpu"),
    )
    assert len(blind) == 2
    assert all(row["answer_tokens"] == 1 for row in blind)

    vision_rows = [
        _supervision([1, 9, 2, 3], [-100, -100, -100, 3]),
        _supervision([1, 9, 2, 4], [-100, -100, -100, 4]),
    ]
    visual = score_candidate_pair(
        language_model=language_model,
        projector=_Projector(),
        receiver=_Receiver(),
        feature_groups=[torch.ones((2, 4, 1))],
        supervisions=vision_rows,
        placeholder_token_id=9,
        pad_token_id=0,
        device=torch.device("cpu"),
    )
    assert len(visual) == 2
    assert all(torch.isfinite(torch.tensor(row["logp_mean"])) for row in visual)


def test_preference_resume_rows_must_be_exact_condition_prefix(tmp_path):
    path = tmp_path / "vision.partial.jsonl"
    path.write_text(
        json.dumps({"sample_id": "a", "condition": "vision"}) + "\n",
        encoding="utf-8",
    )
    assert len(
        read_partial(path, expected_ids=["a", "b"], expected_condition="vision")
    ) == 1
    with pytest.raises(ValueError, match="exact prefix"):
        read_partial(path, expected_ids=["b", "a"], expected_condition="vision")
    with pytest.raises(ValueError, match="condition differs"):
        read_partial(path, expected_ids=["a", "b"], expected_condition="blind")
