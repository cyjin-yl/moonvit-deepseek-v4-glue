"""独立校验 checkpoint 轨迹 run 的完整性。"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .trajectory_data import configured_conditions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON in {path.name}:{line_number}") from error


def _finite(value, label: str) -> None:
    if value is not None and isinstance(value, (int, float)) and not math.isfinite(float(value)):
        raise ValueError(f"non-finite value in {label}: {value}")


def verify_generation_selection_manifest(
    dataset: dict,
    canonical_ids: set[str],
    provenance: dict,
) -> None:
    """把原始生成 ID 绑定到预注册 selection manifest。"""

    manifest_value = dataset.get("generation_selection_manifest")
    if not manifest_value:
        return
    manifest_path = Path(manifest_value)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _sha256(manifest_path) != provenance.get(
        "generation_selection_manifest_sha256"
    ):
        raise ValueError("generation selection manifest SHA-256 mismatch")
    if manifest.get("logical_dataset_sha256") != dataset.get(
        "logical_dataset_sha256"
    ):
        raise ValueError("generation selection logical dataset SHA-256 mismatch")
    selected_ids = {str(value) for value in manifest["selection"]["sample_ids"]}
    if selected_ids != canonical_ids:
        raise ValueError("generation raw IDs differ from selection manifest")
    if len(selected_ids) != int(dataset["expected_records"]):
        raise ValueError("generation selection manifest denominator mismatch")
    data_path = Path(dataset["data"])
    expected_file = manifest["files"].get(data_path.name)
    if expected_file is None or _sha256(data_path) != expected_file.get("sha256"):
        raise ValueError("generation selection data SHA-256 mismatch")
    if provenance.get("data_sha256") != expected_file.get("sha256"):
        raise ValueError("generation selection provenance data SHA-256 mismatch")


def trajectory_dataset_provenance(summary: dict, dataset_name: str) -> dict:
    """读取 runner 固定在 summary 顶层的 dataset provenance。"""

    return summary.get("datasets", {}).get(dataset_name, {})


def verify_trajectory_run(root: Path | str) -> dict:
    """校验哈希、精确条件分母、别名与有限 loss。"""

    root = Path(root)
    config_path = root / "CONFIG.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads((root / "SUMMARY.json").read_text(encoding="utf-8"))
    if summary.get("metadata", {}).get("final_half_scored") is not False:
        raise ValueError("trajectory summary does not prove final_half_scored=false")
    if _sha256(config_path) != summary.get("metadata", {}).get("config_sha256"):
        raise ValueError("trajectory config SHA-256 mismatch")

    for filename, expected in summary["raw_files"].items():
        path = root / filename
        if not path.is_file() or path.stat().st_size != int(expected["bytes"]) or _sha256(path) != expected["sha256"]:
            raise ValueError(f"raw file hash mismatch: {filename}")

    checkpoint_ids = [str(row["id"]) for row in config["checkpoints"]]
    if len(checkpoint_ids) != len(set(checkpoint_ids)):
        raise ValueError("duplicate checkpoint ids in config")
    aliases = {str(row["id"]): str(row["source"]) for row in config.get("aliases", [])}
    if set(checkpoint_ids) | set(aliases) != set(summary["checkpoints"]):
        raise ValueError("summary checkpoint/alias coverage mismatch")
    for alias, source in aliases.items():
        if source not in checkpoint_ids or summary["checkpoints"][alias].get("alias_of") != source:
            raise ValueError(f"invalid checkpoint alias: {alias}")

    expected_combinations = {
        (checkpoint, str(dataset["name"]), str(condition)): int(dataset["expected_records"])
        for checkpoint in checkpoint_ids
        for dataset in config["datasets"]
        for condition in configured_conditions(dataset, checkpoint)
    }
    generation_counts: Counter[tuple[str, str, str]] = Counter()
    generation_ids: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    generation_by_key: dict[tuple[str, str, str, str], dict] = {}
    generation_keys: set[tuple[str, str, str, str]] = set()
    generation_failures = 0
    generation_rows = 0
    for row in _rows(root / "records.jsonl"):
        key3 = (str(row["checkpoint"]), str(row["dataset"]), str(row["condition"]))
        key4 = (*key3, str(row["id"]))
        if key3 not in expected_combinations:
            raise ValueError(f"unexpected generation condition: {key3}")
        if key4 in generation_keys:
            raise ValueError(f"duplicate generation row: {key4}")
        generation_keys.add(key4)
        generation_by_key[key4] = row
        generation_counts[key3] += 1
        generation_ids[key3].add(str(row["id"]))
        generation_rows += 1
        generation_failures += bool(row.get("failure"))
        _finite(row.get("score"), f"generation score {key4}")
        _finite(row.get("wall_seconds"), f"generation wall_seconds {key4}")
    if generation_counts != Counter(expected_combinations):
        missing = {
            key: expected - generation_counts.get(key, 0)
            for key, expected in expected_combinations.items()
            if generation_counts.get(key, 0) != expected
        }
        raise ValueError(f"generation raw denominators mismatch: {missing}")
    for dataset in config["datasets"]:
        dataset_name = str(dataset["name"])
        canonical = generation_ids[
            (
                checkpoint_ids[0],
                dataset_name,
                configured_conditions(dataset, checkpoint_ids[0])[0],
            )
        ]
        verify_generation_selection_manifest(
            dataset,
            canonical,
            trajectory_dataset_provenance(summary, dataset_name),
        )
        for checkpoint in checkpoint_ids:
            for condition in configured_conditions(dataset, checkpoint):
                identifiers = generation_ids[(checkpoint, dataset_name, condition)]
                if identifiers != canonical:
                    raise ValueError(
                        "generation condition sample IDs mismatch: "
                        f"{checkpoint}/{dataset_name}/{condition}"
                    )
        if dataset.get("kind") == "synthetic":
            for checkpoint in checkpoint_ids:
                pair_members: dict[str, set[str]] = defaultdict(set)
                for sample_id in canonical:
                    row = generation_by_key[
                        (checkpoint, dataset_name, "vision", sample_id)
                    ]
                    pair_members[str(row["pair_id"])].add(sample_id)
                for condition in configured_conditions(dataset, checkpoint):
                    for sample_id in canonical:
                        row = generation_by_key[
                            (checkpoint, dataset_name, condition, sample_id)
                        ]
                        source_id = row.get("visual_source_id")
                        if condition == "blind" and source_id is not None:
                            raise ValueError("blind generation row has a visual source")
                        if condition in {
                            "vision",
                            "patch_permutation",
                            "background_matched_aux",
                        } and source_id != sample_id:
                            raise ValueError(f"visual source mismatch: {condition}/{sample_id}")
                        if condition == "shuffled_image" and source_id == sample_id:
                            raise ValueError(f"shuffle fixed point in raw generation: {sample_id}")
                        if condition == "paired_counterfactual_image":
                            members = pair_members[str(row["pair_id"])]
                            if members != {sample_id, str(source_id)}:
                                raise ValueError(
                                    f"paired counterfactual source mismatch: {sample_id}"
                                )

    heldout = config["heldout_shuffle_loss"]
    expected_shuffle = int(heldout["expected_records"])
    repeats = int(heldout["shuffle_repeats"])
    shuffle_counts: Counter[str] = Counter()
    shuffle_keys: set[tuple[str, str]] = set()
    shuffle_failures = 0
    shuffle_rows = 0
    for row in _rows(root / "shuffle_loss_records.jsonl"):
        checkpoint = str(row["checkpoint"])
        key = (checkpoint, str(row["id"]))
        if checkpoint not in checkpoint_ids:
            raise ValueError(f"unexpected shuffle checkpoint: {checkpoint}")
        if key in shuffle_keys:
            raise ValueError(f"duplicate shuffle row: {key}")
        shuffle_keys.add(key)
        shuffle_counts[checkpoint] += 1
        shuffle_rows += 1
        shuffle_failures += bool(row.get("failure"))
        losses = row.get("shuffled_losses", [])
        if row.get("failure") is None and len(losses) != repeats:
            raise ValueError(f"shuffle repeat count mismatch: {key}")
        for field in ("true_loss", "mean_shuffled_loss", "delta", "wall_seconds"):
            _finite(row.get(field), f"shuffle {field} {key}")
        for value in losses:
            _finite(value, f"shuffle repeat {key}")
    expected_shuffle_counts = Counter({checkpoint: expected_shuffle for checkpoint in checkpoint_ids})
    if shuffle_counts != expected_shuffle_counts:
        raise ValueError(f"shuffle raw denominators mismatch: {dict(shuffle_counts)}")

    failure_events = sum(1 for _ in _rows(root / "failures.jsonl"))
    return {
        "status": "valid",
        "run_status": summary.get("status"),
        "generation_rows_verified": generation_rows,
        "generation_failures": generation_failures,
        "shuffle_rows_verified": shuffle_rows,
        "shuffle_failures": shuffle_failures,
        "failure_events": failure_events,
        "unique_checkpoints_verified": len(checkpoint_ids),
        "aliases_verified": len(aliases),
        "final_half_scored": False,
        "raw_files_verified": len(summary["raw_files"]),
    }
