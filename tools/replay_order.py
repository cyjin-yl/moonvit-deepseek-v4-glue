"""把冻结训练窗口变换成保持 counterfactual pair 的 matched replay 顺序。"""

from __future__ import annotations

import random
from collections import Counter, defaultdict


def _flat(batches: list[list[int]]) -> list[int]:
    return [index for batch in batches for index in batch]


def _complete_pairs(
    records: list[dict], indices: list[int], *, task: str
) -> list[tuple[str, tuple[int, int]]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        record = records[index]
        if str(record["task"]) == task:
            grouped[str(record["pair_id"])].append(index)
    complete = []
    for pair_id, members in grouped.items():
        variants = {str(records[index]["pair_variant"]) for index in members}
        if len(members) == 2 and len(variants) == 2:
            complete.append((pair_id, (members[0], members[1])))
    return sorted(complete, key=lambda item: item[0])


def _record_counts(records: list[dict], indices: list[int]) -> dict[str, int]:
    return dict(sorted(Counter(str(records[index]["task"]) for index in indices).items()))


def transform_replay_batches(
    records: list[dict],
    *,
    source_batches: list[list[int]],
    history_batches: list[list[int]],
    tasks: list[str],
    replay_tasks: list[str],
    pairs_per_task_per_window: int,
    window_batch_count: int,
    seed: int,
) -> tuple[list[list[int]], dict]:
    """用历史完整 pair 替换当前窗口中的 donor pair。

    变换保留 batch 数、每个 batch 的大小和总 examples。重放 pair 只来自历史窗口，
    donor pair 只来自对应当前窗口；同一历史 pair 在一次 run 内不会重复使用。
    """

    if not source_batches or window_batch_count <= 0:
        raise ValueError("replay needs non-empty source batches and a positive window size")
    if len(source_batches) % window_batch_count:
        raise ValueError("source batch count must be divisible by replay window size")
    if len(set(tasks)) != len(tasks):
        raise ValueError("replay task universe contains duplicates")
    replay_tasks = [str(task) for task in replay_tasks]
    if len(set(replay_tasks)) != len(replay_tasks) or not set(replay_tasks) <= set(tasks):
        raise ValueError("replay tasks must be a unique subset of configured tasks")
    if len(replay_tasks) >= len(tasks):
        raise ValueError("replay needs at least one non-replay donor task")
    if replay_tasks and pairs_per_task_per_window <= 0:
        raise ValueError("replay pairs per task per window must be positive")
    batch_sizes = {len(batch) for batch in source_batches}
    if len(batch_sizes) != 1:
        raise ValueError("source replay batches have inconsistent sizes")
    source_indices = _flat(source_batches)
    history_indices = _flat(history_batches)
    if len(set(source_indices)) != len(source_indices):
        raise ValueError("source replay window repeats record IDs")
    if set(source_indices) & set(history_indices):
        raise ValueError("history and source replay windows must be disjoint")

    original = [list(batch) for batch in source_batches]
    if not replay_tasks:
        return original, {
            "status": "valid",
            "strategy": "complete-pair-window-replay-v1",
            "replay_tasks": [],
            "pairs_per_task_per_window": pairs_per_task_per_window,
            "window_batch_count": window_batch_count,
            "windows": [],
            "added_ids": [],
            "removed_ids": [],
            "added_records_by_task": {},
            "removed_records_by_task": {},
            "all_replacements_are_complete_pairs": True,
            "examples_preserved": True,
        }

    rng = random.Random(seed)
    replay_pool = {}
    for task in replay_tasks:
        candidates = _complete_pairs(records, history_indices, task=task)
        rng.shuffle(candidates)
        required = pairs_per_task_per_window * (
            len(source_batches) // window_batch_count
        )
        if len(candidates) < required:
            raise ValueError(
                f"task {task} has {len(candidates)} complete replay pairs; {required} required"
            )
        replay_pool[task] = candidates

    transformed = [list(batch) for batch in source_batches]
    donor_tasks = [task for task in tasks if task not in set(replay_tasks)]
    added_indices: list[int] = []
    removed_indices: list[int] = []
    window_rows = []
    replay_cursor = {task: 0 for task in replay_tasks}
    window_count = len(source_batches) // window_batch_count
    for window_index in range(window_count):
        batch_start = window_index * window_batch_count
        batch_end = batch_start + window_batch_count
        window_batches = transformed[batch_start:batch_end]
        window_indices = _flat(window_batches)
        window_added: list[int] = []
        replay_pairs_by_task = {}
        for task in replay_tasks:
            start = replay_cursor[task]
            end = start + pairs_per_task_per_window
            selected = replay_pool[task][start:end]
            replay_cursor[task] = end
            replay_pairs_by_task[task] = [pair_id for pair_id, _ in selected]
            window_added.extend(index for _, members in selected for index in members)

        total_donor_pairs = pairs_per_task_per_window * len(replay_tasks)
        quotient, remainder = divmod(total_donor_pairs, len(donor_tasks))
        rotated = donor_tasks[window_index % len(donor_tasks) :] + donor_tasks[: window_index % len(donor_tasks)]
        donor_pair_targets = {
            task: quotient + (position < remainder)
            for position, task in enumerate(rotated)
        }
        window_removed: list[int] = []
        donor_pairs_by_task = {}
        for task in donor_tasks:
            candidates = _complete_pairs(records, window_indices, task=task)
            rng.shuffle(candidates)
            required = int(donor_pair_targets[task])
            if len(candidates) < required:
                raise ValueError(
                    f"task {task} has {len(candidates)} removable complete pairs in window "
                    f"{window_index}; {required} required"
                )
            selected = candidates[:required]
            donor_pairs_by_task[task] = [pair_id for pair_id, _ in selected]
            window_removed.extend(index for _, members in selected for index in members)
        if len(window_added) != len(window_removed):
            raise AssertionError("replay replacement cardinality drifted")

        positions = {
            index: position for position, index in enumerate(window_indices)
        }
        replacement_positions = sorted(positions[index] for index in window_removed)
        rng.shuffle(window_added)
        for position, replacement in zip(replacement_positions, window_added, strict=True):
            batch_offset, inside_batch = divmod(position, len(window_batches[0]))
            transformed[batch_start + batch_offset][inside_batch] = replacement
        added_indices.extend(window_added)
        removed_indices.extend(window_removed)
        window_rows.append(
            {
                "window_index": window_index,
                "source_batch_start": batch_start,
                "source_batch_end_exclusive": batch_end,
                "replay_pairs_by_task": replay_pairs_by_task,
                "donor_pairs_by_task": donor_pairs_by_task,
                "before_records_by_task": _record_counts(records, window_indices),
                "after_records_by_task": _record_counts(
                    records, _flat(transformed[batch_start:batch_end])
                ),
            }
        )

    final_indices = _flat(transformed)
    if len(final_indices) != len(source_indices):
        raise AssertionError("replay changed the number of training examples")
    added_ids = [str(records[index]["id"]) for index in added_indices]
    removed_ids = [str(records[index]["id"]) for index in removed_indices]
    return transformed, {
        "status": "valid",
        "strategy": "complete-pair-window-replay-v1",
        "seed": seed,
        "replay_tasks": replay_tasks,
        "pairs_per_task_per_window": pairs_per_task_per_window,
        "window_batch_count": window_batch_count,
        "windows": window_rows,
        "added_ids": added_ids,
        "removed_ids": removed_ids,
        "added_records_by_task": _record_counts(records, added_indices),
        "removed_records_by_task": _record_counts(records, removed_indices),
        "source_records_by_task": _record_counts(records, source_indices),
        "final_records_by_task": _record_counts(records, final_indices),
        "all_replacements_are_complete_pairs": True,
        "examples_preserved": len(final_indices) == len(source_indices),
    }
