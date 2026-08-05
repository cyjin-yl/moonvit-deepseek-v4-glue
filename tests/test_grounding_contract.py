import math

import pytest

from moonvit_glue.grounding_contract import (
    MAX_CENTER_L2,
    format_click_action,
    paired_bootstrap,
    parse_click_action,
    score_click_prediction,
    summarize_click_scores,
)


def test_click_parser_accepts_only_the_preregistered_action_grammar():
    assert parse_click_action("click(start_box=[12, 987])") == (12, 987)
    assert parse_click_action("\nclick(start_box=[0, 999])\n") == (0, 999)

    invalid = [
        "click(start_box=[12,987])",
        "Click(start_box=[12, 987])",
        "click(start_box=[12.0, 987])",
        "click(start_box=[1000, 987])",
        "answer: click(start_box=[12, 987])",
        "click(start_box=[12, 987]) click(start_box=[13, 988])",
        "[12, 987]",
    ]
    assert all(parse_click_action(text) is None for text in invalid)
    assert format_click_action((12, 987)) == "click(start_box=[12, 987])"


def test_click_scoring_reports_center_threshold_and_box_distances():
    target = [400.0, 400.0, 600.0, 600.0]
    inside = score_click_prediction(
        sample_id="inside",
        prediction="click(start_box=[500, 500])",
        target_box=target,
    )
    assert inside["parse_ok"]
    assert inside["click_in_box"]
    assert inside["center_l2"] == 0.0
    assert inside["bbox_l2"] == 0.0
    assert inside["bbox_l1"] == 0.0
    assert inside["accuracy_at_50"]

    outside = score_click_prediction(
        sample_id="outside",
        prediction="click(start_box=[700, 500])",
        target_box=target,
    )
    assert not outside["click_in_box"]
    assert outside["center_l2"] == 200.0
    assert outside["bbox_l2"] == 100.0
    assert outside["bbox_l1"] == 100.0
    assert not outside["accuracy_at_100"]
    assert outside["accuracy_at_200"]

    unparsed = score_click_prediction(
        sample_id="unparsed",
        prediction="the button is on the right",
        target_box=target,
    )
    assert not unparsed["parse_ok"]
    assert unparsed["center_l2"] is None
    assert unparsed["center_l2_penalized"] == MAX_CENTER_L2


def test_click_summary_keeps_parsed_and_all_sample_denominators_separate():
    rows = [
        score_click_prediction(
            sample_id="a",
            prediction="click(start_box=[500, 500])",
            target_box=[450, 450, 550, 550],
        ),
        score_click_prediction(
            sample_id="b",
            prediction="click(start_box=[700, 500])",
            target_box=[450, 450, 550, 550],
        ),
        score_click_prediction(
            sample_id="c",
            prediction="unparseable",
            target_box=[450, 450, 550, 550],
        ),
    ]
    summary = summarize_click_scores(rows)

    assert summary["total_count"] == 3
    assert summary["parse_count"] == 2
    assert summary["parse_rate"] == pytest.approx(2 / 3)
    assert summary["accuracy_at_50"] == {
        "hit_count": 1,
        "parsed_denominator": 2,
        "parsed_accuracy": 0.5,
        "all_denominator": 3,
        "all_accuracy": pytest.approx(1 / 3),
    }
    assert summary["click_in_box_accuracy"]["parsed_accuracy"] == 0.5
    assert summary["click_in_box_accuracy"]["all_accuracy"] == pytest.approx(1 / 3)
    assert summary["center_distance"]["parsed"]["mean"] == 100.0
    assert summary["center_distance"]["all_penalized"]["mean"] == pytest.approx(
        (0.0 + 200.0 + math.sqrt(2.0) * 999.0) / 3.0
    )


def test_paired_bootstrap_is_deterministic_and_positive_means_first_is_better():
    vision = []
    blind = []
    for index in range(20):
        target = [450.0, 450.0, 550.0, 550.0]
        vision.append(
            score_click_prediction(
                sample_id=f"s{index:02d}",
                prediction="click(start_box=[500, 500])",
                target_box=target,
            )
        )
        blind.append(
            score_click_prediction(
                sample_id=f"s{index:02d}",
                prediction="click(start_box=[700, 500])",
                target_box=target,
            )
        )

    first = paired_bootstrap(vision, blind, samples=200, seed=20260805)
    second = paired_bootstrap(vision, blind, samples=200, seed=20260805)
    assert first == second
    assert first["orientation"] == "positive_means_first_condition_better"
    assert first["metrics"]["accuracy_at_50_all"]["improvement"] == 1.0
    assert first["metrics"]["mean_center_distance_all_penalized"]["improvement"] == 200.0
    assert first["metrics"]["median_center_distance_all_penalized"]["ci95"] == [
        200.0,
        200.0,
    ]

    mismatched = [dict(blind[0], sample_id="other"), *blind[1:]]
    with pytest.raises(ValueError, match="sample IDs"):
        paired_bootstrap(vision, mismatched, samples=10, seed=1)
