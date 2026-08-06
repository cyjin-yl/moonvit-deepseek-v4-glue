"""Checkpoint 感知的表示塌缩轨迹判定。"""

from __future__ import annotations

import statistics
from typing import Any, Sequence

from moonvit_glue.representation_retention import decide_representation_action


def validate_checkpoint_schedule(conditions: Sequence[dict[str, Any]]) -> None:
    """确保 step0 与后续 checkpoint 构成严格递增且唯一的冻结日程。"""

    if len(conditions) < 2:
        raise ValueError("trajectory requires step0 and at least one trained checkpoint")
    names = [str(row["name"]) for row in conditions]
    steps = [int(row["step"]) for row in conditions]
    if names[0] != "step0" or steps[0] != 0:
        raise ValueError("trajectory must begin with step0")
    if len(set(names)) != len(names) or len(set(steps)) != len(steps):
        raise ValueError("trajectory conditions must be unique")
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError("trajectory steps must increase strictly")
    for row in conditions:
        if int(row["examples_seen"]) < 0 or int(row["answer_tokens_seen"]) < 0:
            raise ValueError("trajectory progress counters must be non-negative")


def summarize_training_windows(
    history: Sequence[dict[str, Any]], conditions: Sequence[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """按冻结 checkpoint 边界汇总不重叠的训练区间。"""

    validate_checkpoint_schedule(conditions)
    trained = list(conditions[1:])
    if not history:
        raise ValueError("training history is empty")
    observed_steps = [int(row["step"]) for row in history]
    if observed_steps != list(range(1, observed_steps[-1] + 1)):
        raise ValueError("training history steps are not contiguous from one")
    if observed_steps[-1] != int(trained[-1]["step"]):
        raise ValueError("training history does not end at the final checkpoint")

    result: dict[str, dict[str, Any]] = {}
    previous = 0
    for checkpoint in trained:
        stop = int(checkpoint["step"])
        rows = [row for row in history if previous < int(row["step"]) <= stop]
        if len(rows) != stop - previous:
            raise ValueError(f"training history window is incomplete: {previous}:{stop}")
        final = rows[-1]
        if int(final["examples_seen"]) != int(checkpoint["examples_seen"]):
            raise ValueError(f"examples_seen differs at step {stop}")
        if int(final["answer_tokens_seen"]) != int(checkpoint["answer_tokens_seen"]):
            raise ValueError(f"answer_tokens_seen differs at step {stop}")
        losses = [float(row["loss"]) for row in rows]
        gradients = [float(row["gradient_norm_before_clip"]) for row in rows]
        cumulative = [float(row["loss"]) for row in history if int(row["step"]) <= stop]
        result[str(checkpoint["name"])] = {
            "step_start_exclusive": previous,
            "step_end_inclusive": stop,
            "optimizer_steps": len(rows),
            "examples_seen": int(final["examples_seen"]),
            "answer_tokens_seen": int(final["answer_tokens_seen"]),
            "effective_epochs": float(final["effective_epochs"]),
            "loss": {
                "first": losses[0],
                "last": losses[-1],
                "minimum": min(losses),
                "median": statistics.median(losses),
                "mean": statistics.fmean(losses),
                "maximum": max(losses),
                "cumulative_mean": statistics.fmean(cumulative),
            },
            "gradient_norm_before_clip": {
                "last": gradients[-1],
                "median": statistics.median(gradients),
                "mean": statistics.fmean(gradients),
                "maximum": max(gradients),
            },
            "learning_rates": sorted({float(row["learning_rate"]) for row in rows}),
        }
        previous = stop
    return result


def find_collapse_onset(
    representations: dict[str, dict[str, Any]], contract: dict[str, Any]
) -> dict[str, Any]:
    """返回首个同时触发两个冻结门槛的 checkpoint。"""

    conditions = list(contract["conditions"])
    validate_checkpoint_schedule(conditions)
    step0 = representations["step0"]
    decisions: dict[str, dict[str, Any]] = {}
    onset_index: int | None = None
    # retention 判定器只需要两个占位 action；轨迹级 action 在确定 onset 后选择。
    decision_contract = {
        **contract,
        "actions": {
            "gross_collapse": "gross_collapse",
            "diversity_retained": "diversity_retained",
        },
    }
    for index, checkpoint in enumerate(conditions[1:], start=1):
        name = str(checkpoint["name"])
        decision = decide_representation_action(
            step0, representations[name], decision_contract
        )
        decisions[name] = decision
        if onset_index is None and bool(decision["gross_collapse"]):
            onset_index = index

    if onset_index is None:
        action = contract["actions"]["never_collapsed"]
        onset = None
        previous = conditions[-1]
    else:
        onset = conditions[onset_index]
        previous = conditions[onset_index - 1]
        action_key = (
            "first_checkpoint_collapsed" if onset_index == 1 else "later_checkpoint_collapsed"
        )
        action = contract["actions"][action_key]
    return {
        "collapse_detected": onset is not None,
        "onset_condition": None if onset is None else str(onset["name"]),
        "onset_step": None if onset is None else int(onset["step"]),
        "onset_examples_seen": None if onset is None else int(onset["examples_seen"]),
        "last_precollapse_condition": str(previous["name"]),
        "last_precollapse_step": int(previous["step"]),
        "registered_action": action,
        "decisions": decisions,
    }
