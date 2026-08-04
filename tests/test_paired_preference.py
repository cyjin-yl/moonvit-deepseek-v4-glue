"""Paired preference exposes internal visual discrimination before generation."""

import torch

from moonvit_glue.paired_preference import (
    answer_logprob_stats,
    build_pair_index,
    summarize_preference_rows,
)


def test_summary_requires_both_pair_variants_to_prefer_their_visual_answer():
    rows = [
        {
            "id": "p1-a",
            "pair_id": "p1",
            "task": "color",
            "correct_margin": 0.8,
            "correct_token_nll": 0.4,
            "counterfactual_token_nll": 1.2,
            "failure": None,
        },
        {
            "id": "p1-b",
            "pair_id": "p1",
            "task": "color",
            "correct_margin": 0.3,
            "correct_token_nll": 0.6,
            "counterfactual_token_nll": 0.9,
            "failure": None,
        },
        {
            "id": "p2-a",
            "pair_id": "p2",
            "task": "color",
            "correct_margin": 0.2,
            "correct_token_nll": 0.7,
            "counterfactual_token_nll": 0.9,
            "failure": None,
        },
        {
            "id": "p2-b",
            "pair_id": "p2",
            "task": "color",
            "correct_margin": -0.1,
            "correct_token_nll": 1.0,
            "counterfactual_token_nll": 0.9,
            "failure": None,
        },
    ]

    summary = summarize_preference_rows(rows)

    assert summary["sample_preference_accuracy"] == {
        "numerator": 3,
        "denominator": 4,
        "value": 0.75,
    }
    assert summary["paired_preference_accuracy"] == {
        "numerator": 1,
        "denominator": 2,
        "value": 0.5,
    }
    assert summary["mean_correct_margin"] == 0.3
    assert summary["median_correct_margin"] == 0.25
    assert summary["by_task"]["color"]["paired_preference_accuracy"]["denominator"] == 2


def test_answer_logprob_is_causal_and_token_normalized_per_sample():
    logits = torch.zeros(2, 3, 5)
    labels = torch.tensor([[-100, 1, 2], [-100, 3, -100]])
    logits[0, 0, 1] = 2.0
    logits[0, 1, 2] = 2.0
    logits[1, 0, 3] = 2.0

    stats = answer_logprob_stats(logits, labels)
    expected = float(torch.log_softmax(torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0]), 0)[1])

    assert stats[0]["answer_tokens"] == 2
    assert stats[1]["answer_tokens"] == 1
    assert abs(stats[0]["logp_sum"] - 2 * expected) < 1e-6
    assert abs(stats[0]["logp_mean"] - expected) < 1e-6
    assert abs(stats[1]["token_normalized_nll"] + expected) < 1e-6


def test_pair_index_exposes_the_other_answer_and_image_without_changing_question():
    records = [
        {
            "id": "p-a",
            "pair_id": "p",
            "pair_variant": "a",
            "question": "color?",
            "answers": ["red"],
            "task": "color",
        },
        {
            "id": "p-b",
            "pair_id": "p",
            "pair_variant": "b",
            "question": "color?",
            "answers": ["blue"],
            "task": "color",
        },
    ]

    index = build_pair_index(records)

    assert index["p-a"]["correct_answer"] == "red"
    assert index["p-a"]["counterfactual_answer"] == "blue"
    assert index["p-a"]["paired_image_id"] == "p-b"
