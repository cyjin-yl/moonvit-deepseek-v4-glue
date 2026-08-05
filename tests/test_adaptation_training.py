from pathlib import Path
import sys

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from train_shape_adaptation import (
    balanced_epoch_indices,
    epoch_indices_for_strategy,
    global_random_epoch_indices,
    maybe_resume_projector_optimizer,
    projector_representation_anchor_loss,
    read_training_order_window,
    resolve_projector_config_source,
)
from eval_shape_adaptation import (
    evaluation_projector_dtype,
    filter_adaptation_states,
    read_evaluation_records,
    select_adaptation_state_ids,
    take_complete_pair_limit_per_task,
)
from analyze_adaptation_compare import (
    endpoint_direction,
    latest_adaptation_state,
    matched_adaptation_states,
    ordered_adaptation_states,
    trajectory_peak_summary,
)
from compare_adaptation_checkpoints import paired_run_metric_rows
from interpolate_projector_checkpoints import interpolate_state_dict
from analyze_projector_interpolation import select_interpolation_candidate
from verify_projector_interpolation import endpoint_equivalence
from analyze_projector_retention import select_retention_candidate


def test_balanced_epoch_indices_keeps_every_batch_task_balanced() -> None:
    records = [
        {"id": f"{task}-{index}", "task": task}
        for task in ("color", "shape")
        for index in range(4)
    ]

    order = balanced_epoch_indices(
        records,
        tasks=["color", "shape"],
        batch_size=4,
        generator=torch.Generator().manual_seed(7),
    )

    assert sorted(order) == list(range(8))
    for start in range(0, len(order), 4):
        tasks = [records[index]["task"] for index in order[start : start + 4]]
        assert tasks.count("color") == 2
        assert tasks.count("shape") == 2


def test_balanced_epoch_indices_rejects_nondivisible_batch() -> None:
    records = [
        {"id": f"{task}-{index}", "task": task}
        for task in ("color", "shape")
        for index in range(4)
    ]

    with pytest.raises(ValueError, match="divisible"):
        balanced_epoch_indices(
            records,
            tasks=["color", "shape"],
            batch_size=3,
            generator=torch.Generator().manual_seed(7),
        )


def test_global_random_epoch_indices_uses_every_record_once() -> None:
    records = [
        {"id": f"{task}-{index}", "task": task}
        for task in ("color", "shape")
        for index in range(6)
    ]

    order = global_random_epoch_indices(
        records,
        generator=torch.Generator().manual_seed(7),
    )

    assert sorted(order) == list(range(12))
    assert order != list(range(12))


def test_order_strategy_changes_only_the_batch_assignment() -> None:
    records = [
        {"id": f"{task}-{index}", "task": task}
        for task in ("color", "shape")
        for index in range(6)
    ]

    balanced = epoch_indices_for_strategy(
        records,
        strategy="balanced_stratified",
        tasks=["color", "shape"],
        batch_size=4,
        generator=torch.Generator().manual_seed(7),
    )
    global_random = epoch_indices_for_strategy(
        records,
        strategy="global_random",
        tasks=["color", "shape"],
        batch_size=4,
        generator=torch.Generator().manual_seed(7),
    )

    assert sorted(balanced) == sorted(global_random) == list(range(12))
    assert balanced != global_random
    assert all(
        {records[index]["task"] for index in balanced[start : start + 4]}
        == {"color", "shape"}
        for start in range(0, len(balanced), 4)
    )


def test_order_strategy_rejects_an_unknown_value() -> None:
    with pytest.raises(ValueError, match="unsupported adaptation order strategy"):
        epoch_indices_for_strategy(
            [{"id": "one", "task": "shape"}],
            strategy="curriculum",
            tasks=["shape"],
            batch_size=1,
            generator=torch.Generator().manual_seed(7),
        )


def test_projector_config_source_can_differ_from_weight_checkpoint(tmp_path: Path) -> None:
    base = tmp_path / "balanced-step100"
    source = tmp_path / "gate-b-step500"

    assert resolve_projector_config_source(
        {"base_projector": str(base), "projector_config_source": str(source)}
    ) == source
    assert resolve_projector_config_source({"base_projector": str(base)}) == base


def test_projector_continuation_can_restore_optimizer_state(tmp_path: Path) -> None:
    source_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    source_optimizer = torch.optim.AdamW([source_parameter], lr=0.01)
    source_parameter.grad = torch.tensor([2.0])
    source_optimizer.step()
    checkpoint = tmp_path / "training_state.pt"
    torch.save({"step": 100, "optimizer": source_optimizer.state_dict()}, checkpoint)

    target_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    target_optimizer = torch.optim.AdamW([target_parameter], lr=0.01)
    provenance = maybe_resume_projector_optimizer(
        optimizer=target_optimizer,
        arm={"kind": "projector", "resume_optimizer": True},
        base_projector=tmp_path,
        device=torch.device("cpu"),
    )

    assert provenance["restored"] is True
    assert provenance["source_step"] == 100
    assert target_optimizer.state_dict()["state"]


def test_lora_arm_never_inherits_projector_optimizer(tmp_path: Path) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)

    provenance = maybe_resume_projector_optimizer(
        optimizer=optimizer,
        arm={"kind": "lora", "resume_optimizer": True},
        base_projector=tmp_path,
        device=torch.device("cpu"),
    )

    assert provenance == {"restored": False, "source": None, "source_step": None}


def test_resume_reads_the_exact_requested_training_order_window(tmp_path: Path) -> None:
    records = [{"id": f"sample-{index}"} for index in range(6)]
    order_path = tmp_path / "training_order.jsonl"
    order_path.write_text(
        "".join(
            [
                '{"step":1,"ids":["sample-0","sample-1"]}\n',
                '{"step":2,"ids":["sample-2","sample-3"]}\n',
                '{"step":3,"ids":["sample-4","sample-5"]}\n',
            ]
        ),
        encoding="utf-8",
    )

    batches, provenance = read_training_order_window(
        order_path,
        records,
        start_step=1,
        end_step=3,
        batch_size=2,
    )

    assert batches == [[2, 3], [4, 5]]
    assert provenance["steps"] == [2, 3]
    assert provenance["first_batch_ids"] == ["sample-2", "sample-3"]


def test_resume_rejects_a_missing_source_step(tmp_path: Path) -> None:
    records = [{"id": f"sample-{index}"} for index in range(4)]
    order_path = tmp_path / "training_order.jsonl"
    order_path.write_text(
        '{"step":1,"ids":["sample-0","sample-1"]}\n'
        '{"step":3,"ids":["sample-2","sample-3"]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="steps mismatch"):
        read_training_order_window(
            order_path,
            records,
            start_step=1,
            end_step=3,
            batch_size=2,
        )


def test_projector_anchor_uses_only_preregistered_tasks() -> None:
    current = [
        torch.tensor([[1.0, 1.0]]),
        torch.tensor([[3.0, 3.0]]),
    ]
    reference = [
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[0.0, 0.0]]),
    ]
    records = [{"task": "count"}, {"task": "spatial"}]

    loss, selected = projector_representation_anchor_loss(
        current,
        reference,
        records,
        anchor_tasks={"count", "shape"},
    )

    assert float(loss) == pytest.approx(1.0)
    assert selected == 1


def test_projector_anchor_rejects_a_batch_without_anchor_tasks() -> None:
    with pytest.raises(ValueError, match="no selected records"):
        projector_representation_anchor_loss(
            [torch.tensor([[1.0]])],
            [torch.tensor([[0.0]])],
            [{"task": "spatial"}],
            anchor_tasks={"count"},
        )


def test_multitask_evaluation_can_read_a_complete_manifest_without_ids(
    tmp_path: Path,
) -> None:
    data = tmp_path / "selection.jsonl"
    data.write_text(
        "".join(
            f'{{"id":"{task}-{index}","task":"{task}"}}\n'
            for task in ("color", "shape")
            for index in range(2)
        ),
        encoding="utf-8",
    )

    records = read_evaluation_records(data, expected_records=4)

    assert [row["task"] for row in records] == ["color", "color", "shape", "shape"]


def test_multitask_evaluation_checks_the_manifest_denominator(tmp_path: Path) -> None:
    data = tmp_path / "selection.jsonl"
    data.write_text('{"id":"only"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="denominator"):
        read_evaluation_records(data, expected_records=2)


def test_endpoint_screen_keeps_frozen_and_requested_steps() -> None:
    states = [
        {"id": "frozen-base", "kind": "frozen", "adaptation_step": 0},
        {"id": "lora-step25", "kind": "lora", "adaptation_step": 25},
        {"id": "lora-step100", "kind": "lora", "adaptation_step": 100},
        {"id": "projector-step25", "kind": "projector", "adaptation_step": 25},
        {"id": "projector-step100", "kind": "projector", "adaptation_step": 100},
    ]

    filtered = filter_adaptation_states(states, [100])

    assert [row["id"] for row in filtered] == [
        "frozen-base",
        "lora-step100",
        "projector-step100",
    ]


def test_interpolation_screen_can_keep_only_projector_states() -> None:
    states = [
        {"id": "frozen-base", "kind": "frozen", "adaptation_step": 0},
        {"id": "lora-step25", "kind": "lora", "adaptation_step": 25},
        {"id": "projector-interp25", "kind": "projector", "adaptation_step": 25},
        {"id": "projector-interp50", "kind": "projector", "adaptation_step": 50},
        {"id": "projector-interp75", "kind": "projector", "adaptation_step": 75},
    ]

    filtered = filter_adaptation_states(
        states, [25, 50, 75], requested_kinds=["projector"]
    )

    assert [row["id"] for row in filtered] == [
        "frozen-base",
        "projector-interp25",
        "projector-interp50",
        "projector-interp75",
    ]


def test_sentinel_screen_selects_exact_reference_and_current_states() -> None:
    states = [
        {"id": "frozen-base", "kind": "frozen", "adaptation_step": 0},
        {"id": "exchange-step50", "kind": "projector", "adaptation_step": 50},
        {"id": "ordinary-step75", "kind": "projector", "adaptation_step": 75},
    ]

    selected = select_adaptation_state_ids(
        states, ["exchange-step50", "ordinary-step75"]
    )

    assert [row["id"] for row in selected] == [
        "exchange-step50",
        "ordinary-step75",
    ]
    with pytest.raises(ValueError, match="absent"):
        select_adaptation_state_ids(states, ["missing-step"])


def test_endpoint_direction_requires_two_task_level_generation_wins() -> None:
    decisions = {
        "shape": {
            "generation_lora_minus_projector_ci95_low": 0.01,
            "generation_lora_minus_projector_ci95_high": 0.20,
        },
        "spatial": {
            "generation_lora_minus_projector_ci95_low": 0.02,
            "generation_lora_minus_projector_ci95_high": 0.15,
        },
        "ocr": {
            "generation_lora_minus_projector_ci95_low": -0.04,
            "generation_lora_minus_projector_ci95_high": 0.05,
        },
    }

    assert endpoint_direction(decisions) == "language_upper_stack_supported"


def test_evaluation_can_match_canonical_projector_precision() -> None:
    assert evaluation_projector_dtype(
        {
            "projector_dtype": "float32",
            "evaluation": {"projector_dtype": "bfloat16"},
        }
    ) == "bfloat16"


def test_balanced_screen_keeps_equal_complete_pairs_per_task() -> None:
    records = [
        {
            "id": f"{task}-{pair}-{variant}",
            "task": task,
            "pair_id": f"{task}-{pair}",
        }
        for task in ("color", "shape")
        for pair in range(3)
        for variant in ("a", "b")
    ]

    selected = take_complete_pair_limit_per_task(records, 4)

    assert len(selected) == 8
    assert {task: sum(row["task"] == task for row in selected) for task in ("color", "shape")} == {
        "color": 4,
        "shape": 4,
    }


def test_trajectory_analysis_uses_the_latest_endpoint() -> None:
    assert latest_adaptation_state(
        ["frozen-base", "lora-step25", "lora-step100", "lora-step50"],
        "lora-step",
    ) == "lora-step100"


def test_trajectory_analysis_pairs_equal_arm_steps() -> None:
    assert matched_adaptation_states(
        [
            "frozen-base",
            "lora-step25",
            "lora-step100",
            "projector-step25",
            "projector-step100",
        ],
        "lora-step",
        "projector-step",
    ) == [
        ("lora-step25", "projector-step25", 25),
        ("lora-step100", "projector-step100", 100),
    ]


def test_trajectory_analysis_orders_numeric_steps() -> None:
    assert ordered_adaptation_states(
        ["lora-step100", "lora-step25", "lora-step50"], "lora-step"
    ) == ["lora-step25", "lora-step50", "lora-step100"]


def test_cross_run_comparison_keeps_identical_complete_pair_ids() -> None:
    def rows(state: str, scores: tuple[float, float]) -> list[dict]:
        return [
            {
                "state": state,
                "condition": "vision",
                "task": "shape",
                "id": f"sample-{index}",
                "pair_id": "pair-1",
                "pair_variant": variant,
                "correct_margin": score,
                "failure": None,
            }
            for index, (variant, score) in enumerate(zip(("a", "b"), scores))
        ]

    early, late = paired_run_metric_rows(
        rows("projector-step50", (-1.0, 1.0)),
        rows("projector-step100", (1.0, 2.0)),
        early_state="projector-step50",
        late_state="projector-step100",
        condition="vision",
        task="shape",
        metric="paired_preference",
    )
    assert early == [{"id": "pair-1", "score": 0.0}]
    assert late == [{"id": "pair-1", "score": 1.0}]


def test_flat_trajectory_is_not_reported_as_a_nonmonotonic_peak() -> None:
    summary = trajectory_peak_summary(
        [
            {"state": "lora-step25", "mean": 0.0},
            {"state": "lora-step50", "mean": 0.0},
            {"state": "lora-step100", "mean": 0.0},
        ],
        "lora-step",
    )
    assert summary["latest_state"] == "lora-step100"
    assert summary["nonmonotonic_peak"] is False


def test_projector_interpolation_reproduces_endpoints_and_midpoint() -> None:
    early = {"weight": torch.tensor([1.0, 3.0])}
    late = {"weight": torch.tensor([5.0, 7.0])}

    start = interpolate_state_dict(early, late, alpha=0.0)
    middle = interpolate_state_dict(early, late, alpha=0.5)
    end = interpolate_state_dict(early, late, alpha=1.0)

    assert torch.equal(start["weight"], early["weight"])
    assert torch.equal(middle["weight"], torch.tensor([3.0, 5.0]))
    assert torch.equal(end["weight"], late["weight"])


def test_interpolation_candidate_requires_retention_and_worst_task_gain() -> None:
    states = {
        "projector-interp000": {
            "alpha": 0.0,
            "preference": {
                "color": 0.60,
                "coordinate": 0.40,
                "count": 0.40,
                "ocr": 0.10,
                "shape": 0.80,
                "spatial": 0.70,
            },
            "generation_macro": 0.20,
        },
        "projector-interp050": {
            "alpha": 0.5,
            "preference": {
                "color": 0.65,
                "coordinate": 0.50,
                "count": 0.36,
                "ocr": 0.20,
                "shape": 0.76,
                "spatial": 0.80,
            },
            "generation_macro": 0.23,
        },
        "projector-interp100": {
            "alpha": 1.0,
            "preference": {
                "color": 0.70,
                "coordinate": 0.55,
                "count": 0.10,
                "ocr": 0.15,
                "shape": 0.45,
                "spatial": 1.00,
            },
            "generation_macro": 0.25,
        },
    }

    decision = select_interpolation_candidate(states)

    assert decision["selected_state"] == "projector-interp050"
    assert decision["targeted_merge_pass"] is True


def test_interpolation_endpoint_verifier_allows_only_state_metadata_to_differ() -> None:
    reference = [
        {
            "state": "projector-step50",
            "condition": "vision",
            "id": "sample-a",
            "pair_id": "pair-1",
            "pair_variant": "a",
            "task": "shape",
            "correct_margin": 1.25,
        }
    ]
    interpolation = [
        {
            **reference[0],
            "state": "projector-interp000",
            "interpolation_alpha": 0.0,
        }
    ]

    assert endpoint_equivalence(
        interpolation,
        reference,
        interpolation_state="projector-interp000",
        reference_state="projector-step50",
        value_fields=("correct_margin",),
    ) == 1


def test_retention_candidate_must_keep_old_tasks_and_gain_new_tasks() -> None:
    states = {
        "frozen-base": {
            "preference": {
                "color": 0.58,
                "coordinate": 0.44,
                "count": 0.42,
                "ocr": 0.18,
                "shape": 0.80,
                "spatial": 0.74,
            },
            "generation_macro": 0.23,
        },
        "resume-control": {
            "preference": {
                "color": 0.74,
                "coordinate": 0.54,
                "count": 0.12,
                "ocr": 0.22,
                "shape": 0.48,
                "spatial": 1.00,
            },
            "generation_macro": 0.26,
        },
        "anchor-good": {
            "weight": 0.001,
            "preference": {
                "color": 0.66,
                "coordinate": 0.50,
                "count": 0.40,
                "ocr": 0.20,
                "shape": 0.78,
                "spatial": 0.90,
            },
            "generation_macro": 0.27,
        },
        "anchor-static": {
            "weight": 0.01,
            "preference": {
                "color": 0.58,
                "coordinate": 0.44,
                "count": 0.42,
                "ocr": 0.18,
                "shape": 0.80,
                "spatial": 0.74,
            },
            "generation_macro": 0.23,
        },
    }

    decision = select_retention_candidate(states)

    assert decision["selected_state"] == "anchor-good"
    assert decision["targeted_retention_pass"] is True
    static = next(row for row in decision["candidates"] if row["state"] == "anchor-static")
    assert static["gains_coordinate_spatial"] is False
