import pytest

from moonvit_glue.representation_trajectory import (
    find_collapse_onset,
    summarize_training_windows,
    validate_checkpoint_schedule,
)


def _contract():
    return {
        "conditions": [
            {"name": "step0", "step": 0, "examples_seen": 0, "answer_tokens_seen": 0},
            {"name": "step100", "step": 100, "examples_seen": 800, "answer_tokens_seen": 7000},
            {"name": "step200", "step": 200, "examples_seen": 1600, "answer_tokens_seen": 14000},
        ],
        "gross_collapse_rule": {
            "requires_all": True,
            "current_over_step0_relative_spread_below": 0.25,
            "current_over_step0_effective_rank_below": 0.5,
        },
        "actions": {
            "first_checkpoint_collapsed": "protect_from_start",
            "later_checkpoint_collapsed": "localize_then_protect",
            "never_collapsed": "run_margin",
        },
    }


def _representation(spread, rank):
    return {
        "relative_between_image_spread": spread,
        "effective_rank_participation": rank,
    }


def test_find_collapse_onset_selects_earliest_saved_checkpoint():
    result = find_collapse_onset(
        {
            "step0": _representation(0.8, 20.0),
            "step100": _representation(0.3, 12.0),
            "step200": _representation(0.1, 5.0),
        },
        _contract(),
    )

    assert result["collapse_detected"] is True
    assert result["onset_condition"] == "step200"
    assert result["onset_examples_seen"] == 1600
    assert result["last_precollapse_condition"] == "step100"
    assert result["registered_action"] == "localize_then_protect"


def test_find_collapse_onset_uses_from_start_action_at_first_checkpoint():
    result = find_collapse_onset(
        {
            "step0": _representation(0.8, 20.0),
            "step100": _representation(0.1, 5.0),
            "step200": _representation(0.08, 3.0),
        },
        _contract(),
    )

    assert result["onset_step"] == 100
    assert result["last_precollapse_step"] == 0
    assert result["registered_action"] == "protect_from_start"


def test_training_windows_are_disjoint_and_bound_progress():
    contract = _contract()
    contract["conditions"][1].update(step=2, examples_seen=16, answer_tokens_seen=140)
    contract["conditions"][2].update(step=4, examples_seen=32, answer_tokens_seen=290)
    history = [
        {"step": 1, "examples_seen": 8, "answer_tokens_seen": 70, "effective_epochs": 0.1, "loss": 4.0, "gradient_norm_before_clip": 8.0, "learning_rate": 0.001},
        {"step": 2, "examples_seen": 16, "answer_tokens_seen": 140, "effective_epochs": 0.2, "loss": 2.0, "gradient_norm_before_clip": 4.0, "learning_rate": 0.001},
        {"step": 3, "examples_seen": 24, "answer_tokens_seen": 220, "effective_epochs": 0.3, "loss": 3.0, "gradient_norm_before_clip": 2.0, "learning_rate": 0.001},
        {"step": 4, "examples_seen": 32, "answer_tokens_seen": 290, "effective_epochs": 0.4, "loss": 1.0, "gradient_norm_before_clip": 1.0, "learning_rate": 0.001},
    ]

    result = summarize_training_windows(history, contract["conditions"])

    assert result["step100"]["loss"]["mean"] == pytest.approx(3.0)
    assert result["step200"]["loss"]["mean"] == pytest.approx(2.0)
    assert result["step200"]["loss"]["cumulative_mean"] == pytest.approx(2.5)
    assert result["step200"]["gradient_norm_before_clip"]["maximum"] == 2.0


@pytest.mark.parametrize(
    "conditions",
    [
        [{"name": "step100", "step": 100, "examples_seen": 1, "answer_tokens_seen": 1}],
        [
            {"name": "step0", "step": 0, "examples_seen": 0, "answer_tokens_seen": 0},
            {"name": "step0", "step": 100, "examples_seen": 1, "answer_tokens_seen": 1},
        ],
    ],
)
def test_checkpoint_schedule_fails_closed(conditions):
    with pytest.raises(ValueError):
        validate_checkpoint_schedule(conditions)
