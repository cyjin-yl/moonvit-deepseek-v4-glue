from pathlib import Path
import sys

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from train_shape_adaptation import (
    balanced_epoch_indices,
    maybe_resume_projector_optimizer,
    resolve_projector_config_source,
)
from eval_shape_adaptation import (
    evaluation_projector_dtype,
    filter_adaptation_states,
    read_evaluation_records,
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
