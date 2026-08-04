"""为 checkpoint 轨迹评测生成抗泄漏数据 manifest。"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .trajectory_metrics import derangement_indices


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _absolute_image(record: dict, source: Path) -> dict:
    row = dict(record)
    if row.get("image"):
        row["image"] = str((source.parent / row["image"]).resolve())
    return row


def configured_conditions(dataset: dict, checkpoint_id: str) -> list[str]:
    """解析预注册的基础条件与 checkpoint 专属控制矩阵。"""

    conditions = [str(value) for value in dataset["conditions"]]
    for extension in dataset.get("checkpoint_condition_extensions", []):
        if checkpoint_id in {str(value) for value in extension["checkpoint_ids"]}:
            for condition in extension["conditions"]:
                value = str(condition)
                if value not in conditions:
                    conditions.append(value)
    return conditions


def make_pair_stratified_subset(
    records: Iterable[dict],
    *,
    pairs_per_task: int,
    seed: int,
) -> list[dict]:
    """按任务稳定哈希排序，选择完整最小对。"""

    if pairs_per_task < 1:
        raise ValueError("pairs_per_task must be positive")
    materialized = list(records)
    by_pair: dict[str, list[dict]] = {}
    for row in materialized:
        pair_id = str(row["pair_id"])
        by_pair.setdefault(pair_id, []).append(row)
    task_pairs: dict[str, list[str]] = {}
    for pair_id, pair_rows in by_pair.items():
        tasks = {str(row["task"]) for row in pair_rows}
        variants = {str(row["pair_variant"]) for row in pair_rows}
        if len(pair_rows) != 2 or variants != {"a", "b"} or len(tasks) != 1:
            raise ValueError(f"invalid minimal pair: {pair_id}")
        task = next(iter(tasks))
        task_pairs.setdefault(task, []).append(pair_id)

    selected_pairs: set[str] = set()
    for task in sorted(task_pairs):
        candidates = task_pairs[task]
        if len(candidates) < pairs_per_task:
            raise ValueError(
                f"task {task} has {len(candidates)} pairs, needs {pairs_per_task}"
            )
        ranked = sorted(
            candidates,
            key=lambda pair_id: hashlib.sha256(
                f"{seed}:{task}:{pair_id}".encode("utf-8")
            ).digest(),
        )
        selected_pairs.update(ranked[:pairs_per_task])
    return [row for row in materialized if str(row["pair_id"]) in selected_pairs]


def make_stratified_control_records(
    records: Iterable[dict],
    *,
    seed: int,
    split: str,
    group_key: str = "benchmark",
) -> list[dict]:
    """生成确定性的组内错图与 patch permutation 控制。"""

    materialized = list(records)
    by_id = {str(row["id"]): row for row in materialized}
    if len(by_id) != len(materialized):
        raise ValueError("control records require unique sample IDs")
    groups: dict[str, list[str]] = {}
    for row in materialized:
        group = str(row[group_key])
        groups.setdefault(group, []).append(str(row["id"]))

    shuffled_by_id: dict[str, str] = {}
    for group_index, group in enumerate(sorted(groups)):
        identifiers = sorted(groups[group])
        if len(identifiers) < 2:
            raise ValueError(f"control stratum has fewer than two records: {group}")
        order = derangement_indices(
            len(identifiers), seed=int(seed) + group_index * 1_000_003
        )
        shuffled_by_id.update(
            {sample_id: identifiers[other] for sample_id, other in zip(identifiers, order)}
        )

    output = []
    for sample_id in sorted(by_id):
        patch_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{sample_id}:patch".encode("utf-8")).digest()[:8],
            "big",
        ) % (2**63 - 1)
        output.append(
            {
                "id": sample_id,
                "split": split,
                "group": str(by_id[sample_id][group_key]),
                "shuffled_image_id": shuffled_by_id[sample_id],
                "patch_permutation": {
                    "algorithm": "torch.randperm",
                    "seed": patch_seed,
                },
            }
        )
    return output


def make_shape_matched_blank_controls(
    records: Iterable[dict],
    controls: Iterable[dict],
    feature_cache_records: Iterable[dict],
    *,
    output_dir: Path | str,
    blank_rgb: tuple[int, int, int] = (255, 255, 255),
) -> tuple[list[dict], list[dict]]:
    """为每种特征网格生成一张 blank 图，并绑定到对应样本。"""

    from PIL import Image

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite shape-matched blanks: {output}")
    images = output / "images"
    images.mkdir(parents=True)
    records_by_id = {str(row["id"]): row for row in records}
    controls_by_id = {str(row["id"]): dict(row) for row in controls}
    if set(records_by_id) != set(controls_by_id):
        raise ValueError("blank controls require identical data/control sample IDs")
    shape_by_id = {
        str(row["id"]): tuple(int(value) for value in row["feature_shape"])
        for row in feature_cache_records
        if row.get("status") == "ok"
    }
    if set(shape_by_id) != set(records_by_id):
        raise ValueError("feature-cache record coverage does not match benchmark data")

    ids_by_shape: dict[tuple[int, ...], list[str]] = {}
    for sample_id, shape in shape_by_id.items():
        ids_by_shape.setdefault(shape, []).append(sample_id)
    blank_records = []
    blank_id_by_shape = {}
    for shape in sorted(ids_by_shape):
        representative_id = sorted(ids_by_shape[shape])[0]
        source_image = Path(str(records_by_id[representative_id]["image"]))
        with Image.open(source_image) as source:
            size = source.size
        shape_key = "x".join(str(value) for value in shape)
        blank_id = f"control:benchmark:blank:{shape_key}"
        blank_path = images / f"blank-{shape_key}.png"
        Image.new("RGB", size, blank_rgb).save(blank_path)
        blank_id_by_shape[shape] = blank_id
        blank_records.append(
            {
                "id": blank_id,
                "image": str(blank_path.resolve()),
                "question": "control image",
                "answers": ["n/a"],
                "metric": "exact_match",
                "expected_feature_shape": list(shape),
                "representative_sample_id": representative_id,
                "source_pixel_size": list(size),
                "image_sha256": _sha256(blank_path),
            }
        )

    updated = []
    for sample_id in sorted(records_by_id):
        row = controls_by_id[sample_id]
        shape = shape_by_id[sample_id]
        row["blank_image_id"] = blank_id_by_shape[shape]
        row["blank_expected_feature_shape"] = list(shape)
        updated.append(row)
    return updated, blank_records


def prepare_trajectory_data(
    *,
    eval_files: Iterable[Path | str],
    train_file: Path | str,
    output_dir: Path | str,
    heldout_count: int = 32,
) -> dict:
    """生成偶数行 selection half 与历史末尾 N 条 heldout。"""

    if heldout_count < 1:
        raise ValueError("heldout_count must be positive")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite trajectory data: {output}")
    output.mkdir(parents=True)
    eval_paths = [Path(path).resolve() for path in eval_files]
    train_path = Path(train_file).resolve()
    if not eval_paths:
        raise ValueError("at least one evaluation JSONL is required")

    selection: list[dict] = []
    benchmark_counts: Counter[str] = Counter()
    sources = []
    for path in sorted(eval_paths):
        benchmark = path.stem
        source_count = 0
        with path.open("r", encoding="utf-8") as stream:
            for original_index, line in enumerate(stream):
                if not line.strip() or original_index % 2:
                    continue
                row = _absolute_image(json.loads(line), path)
                row["benchmark"] = benchmark
                row["_original_index"] = original_index
                row["_selection_parity"] = "even"
                selection.append(row)
                benchmark_counts[benchmark] += 1
                source_count += 1
        sources.append({
            "path": str(path),
            "sha256": _sha256(path),
            "selection_records": source_count,
        })

    tail: deque[dict] = deque(maxlen=heldout_count)
    with train_path.open("r", encoding="utf-8") as stream:
        for original_index, line in enumerate(stream):
            if line.strip():
                row = _absolute_image(json.loads(line), train_path)
                row["_original_index"] = original_index
                row["_heldout_rule"] = f"historical_last_{heldout_count}"
                tail.append(row)
    heldout = list(tail)
    if len(heldout) != heldout_count:
        raise ValueError(f"training file has fewer than {heldout_count} records")

    selection_path = output / "benchmark_selection.jsonl"
    heldout_path = output / "historical_heldout.jsonl"
    _write_jsonl(selection_path, selection)
    _write_jsonl(heldout_path, heldout)
    with (output / "benchmark_counts.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["benchmark", "selection_records"])
        writer.writeheader()
        for benchmark in sorted(benchmark_counts):
            writer.writerow({"benchmark": benchmark, "selection_records": benchmark_counts[benchmark]})

    heldout_sources = Counter(str(row.get("source") or "unknown") for row in heldout)
    manifest = {
        "format_version": "checkpoint-trajectory-data-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "discipline": {
            "selection_rule": "zero-based even original row indices only",
            "final_half_rule": "zero-based odd original row indices",
            "final_half_materialized": False,
            "final_half_scored": False,
            "heldout_rule": f"historical training split tail, last {heldout_count} rows",
        },
        "counts": {
            "benchmark_selection": len(selection),
            "benchmark_selection_by_name": dict(sorted(benchmark_counts.items())),
            "historical_heldout": len(heldout),
            "historical_heldout_by_source": dict(sorted(heldout_sources.items())),
        },
        "sources": {
            "benchmarks": sources,
            "training": {"path": str(train_path), "sha256": _sha256(train_path)},
        },
        "files": {
            selection_path.name: {"bytes": selection_path.stat().st_size, "sha256": _sha256(selection_path)},
            heldout_path.name: {"bytes": heldout_path.stat().st_size, "sha256": _sha256(heldout_path)},
        },
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
