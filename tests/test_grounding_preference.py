import math

import pytest

from moonvit_glue.grounding_preference import (
    build_counterfactual_targets,
    make_preference_row,
    paired_preference_bootstrap,
    summarize_preference_rows,
    target_click_answer,
)


def _sample(sample_id, box, *, platform="Web", target_type="text"):
    return {
        "sample_id": sample_id,
        "bbox_999_xyxy": box,
        "platform": platform,
        "target_type": target_type,
    }


def _stats(logp_mean, *, tokens=4):
    return {
        "answer_tokens": tokens,
        "logp_sum": logp_mean * tokens,
        "logp_mean": logp_mean,
        "token_normalized_nll": -logp_mean,
    }


def test_grounding_counterfactual_targets_follow_the_frozen_derangement():
    samples = [
        _sample("a", [10.0, 20.0, 30.0, 40.0]),
        _sample("b", [800.0, 700.0, 900.0, 900.0], target_type="icon/widget"),
    ]
    manifest = {
        "samples": samples,
        "shuffled_image_control": {
            "mapping": [
                {"sample_id": "a", "shuffled_image_sample_id": "b"},
                {"sample_id": "b", "shuffled_image_sample_id": "a"},
            ]
        },
    }

    assert target_click_answer(samples[0]) == "click(start_box=[20, 30])"
    targets = build_counterfactual_targets(manifest)
    assert targets["a"] == {
        "correct_answer": "click(start_box=[20, 30])",
        "counterfactual_answer": "click(start_box=[850, 800])",
        "counterfactual_sample_id": "b",
    }
    assert targets["b"]["counterfactual_sample_id"] == "a"


def test_grounding_counterfactual_targets_reject_equal_answers_or_missing_ids():
    samples = [
        _sample("a", [10.0, 20.0, 30.0, 40.0]),
        _sample("b", [10.0, 20.0, 30.0, 40.0]),
    ]
    with pytest.raises(ValueError, match="did not change"):
        build_counterfactual_targets(
            {
                "samples": samples,
                "shuffled_image_control": {
                    "mapping": [
                        {"sample_id": "a", "shuffled_image_sample_id": "b"},
                        {"sample_id": "b", "shuffled_image_sample_id": "a"},
                    ]
                },
            }
        )
    with pytest.raises(ValueError, match="unknown shuffled"):
        build_counterfactual_targets(
            {
                "samples": samples[:1],
                "shuffled_image_control": {
                    "mapping": [
                        {"sample_id": "a", "shuffled_image_sample_id": "missing"}
                    ]
                },
            }
        )


def test_preference_rows_use_token_normalized_margin_and_keep_breakdowns():
    positive = make_preference_row(
        sample=_sample("a", [0, 0, 10, 10], platform="Android"),
        condition="vision",
        input_image_sample_id="a",
        counterfactual_sample_id="b",
        correct_answer="click(start_box=[5, 5])",
        counterfactual_answer="click(start_box=[900, 900])",
        correct_stats=_stats(-0.4, tokens=4),
        counterfactual_stats=_stats(-0.8, tokens=7),
    )
    negative = make_preference_row(
        sample=_sample(
            "b", [800, 800, 900, 900], platform="iOS", target_type="icon/widget"
        ),
        condition="vision",
        input_image_sample_id="b",
        counterfactual_sample_id="a",
        correct_answer="click(start_box=[850, 850])",
        counterfactual_answer="click(start_box=[5, 5])",
        correct_stats=_stats(-1.0),
        counterfactual_stats=_stats(-0.7),
    )
    assert positive["correct_margin"] == pytest.approx(0.4)
    assert positive["correct_preferred"] is True
    assert negative["correct_preferred"] is False

    summary = summarize_preference_rows([positive, negative])
    overall = summary["conditions"]["vision"]["breakdowns"]["overall"]
    assert overall["records"] == 2
    assert overall["preference_count"] == 1
    assert overall["paired_preference_accuracy"] == 0.5
    assert overall["mean_correct_margin"] == pytest.approx(0.05)
    assert summary["conditions"]["vision"]["breakdowns"]["Android"][
        "paired_preference_accuracy"
    ] == 1.0
    assert summary["conditions"]["vision"]["breakdowns"]["icon/widget"][
        "paired_preference_accuracy"
    ] == 0.0


def test_paired_preference_bootstrap_is_deterministic_and_oriented_positive():
    samples = [
        _sample("a", [0, 0, 10, 10]),
        _sample("b", [800, 800, 900, 900]),
        _sample("c", [300, 300, 400, 400]),
    ]
    first = []
    second = []
    for index, sample in enumerate(samples):
        first.append(
            make_preference_row(
                sample=sample,
                condition="vision",
                input_image_sample_id=sample["sample_id"],
                counterfactual_sample_id="x",
                correct_answer="gold",
                counterfactual_answer="counterfactual",
                correct_stats=_stats(-0.2 - index * 0.1),
                counterfactual_stats=_stats(-0.8),
            )
        )
        second.append(
            make_preference_row(
                sample=sample,
                condition="shuffled",
                input_image_sample_id="x",
                counterfactual_sample_id="x",
                correct_answer="gold",
                counterfactual_answer="counterfactual",
                correct_stats=_stats(-0.9),
                counterfactual_stats=_stats(-0.8),
            )
        )

    one = paired_preference_bootstrap(first, second, samples=200, seed=17)
    two = paired_preference_bootstrap(first, second, samples=200, seed=17)
    assert one == two
    assert one["orientation"] == "positive_means_first_condition_better"
    assert one["metrics"]["paired_preference_accuracy"]["improvement"] == 1.0
    assert one["metrics"]["mean_correct_margin"]["improvement"] > 0
    assert all(
        math.isfinite(value)
        for value in one["metrics"]["mean_correct_margin"]["ci95"]
    )

    reversed_rows = list(reversed(second))
    with pytest.raises(ValueError, match="identical sample IDs"):
        paired_preference_bootstrap(first, reversed_rows, samples=10)
