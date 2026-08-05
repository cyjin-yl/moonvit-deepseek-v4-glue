from pathlib import Path
import sys

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from train_shape_adaptation import balanced_epoch_indices


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
