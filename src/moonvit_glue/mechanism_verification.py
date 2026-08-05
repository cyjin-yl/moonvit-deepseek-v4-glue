"""package 4 表示、probe 与 activation-patching 产物的独立验证。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file

from .mechanism_probe import select_complete_task_pairs


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def expected_representation_keys(
    *, hidden_state_count: int, poolings: Sequence[str]
) -> set[str]:
    keys = {"labels", "source_labels", "shape_logits"}
    for family in ("tower", "projector"):
        keys.update(f"{family}_{pooling}" for pooling in poolings)
    for index in range(hidden_state_count):
        keys.add(f"layer_{index:02d}_assistant")
        keys.add(f"layer_{index:02d}_image_mean")
    return keys


def validate_visual_source(
    condition: str,
    sample_id: str,
    source_id: str,
    mate_id: str,
    control: dict,
) -> None:
    if condition in {"vision", "patch_permutation"}:
        expected = sample_id
    elif condition == "paired_counterfactual_image":
        expected = mate_id
    elif condition == "shuffled_image":
        expected = str(control["shuffled_image_id"])
    else:
        raise ValueError(f"unknown mechanism condition: {condition}")
    if source_id != expected:
        raise ValueError(
            f"mechanism visual source mismatch: {condition}/{sample_id}: "
            f"expected {expected}, got {source_id}"
        )


def validate_pair_permutation_rows(rows: Sequence[dict]) -> None:
    """主 probe 证据要求每个 vision cell 都有 pair-unit permutation null。"""
    vision = [row for row in rows if row["condition"] == "vision"]
    if not vision or any(
        row.get("pair_permutation_samples") in (None, "", "0") for row in vision
    ):
        raise ValueError("vision probe rows are missing pair-permutation nulls")


def _pair_index(records: list[dict]) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for record in records:
        grouped.setdefault(str(record["pair_id"]), []).append(str(record["id"]))
    result = {}
    for pair_id, ids in grouped.items():
        if len(ids) != 2:
            raise ValueError(f"verification pair {pair_id!r} is incomplete")
        result[ids[0]], result[ids[1]] = ids[1], ids[0]
    return result


def _check_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"artifact hash mismatch: {path}: {actual} != {expected}")


def verify_representations(run: Path) -> dict:
    summary = json.loads((run / "SUMMARY.json").read_text(encoding="utf-8"))
    config = json.loads((run / "CONFIG.json").read_text(encoding="utf-8"))
    if summary.get("status") != "valid" or summary["metadata"].get("final_half_scored"):
        raise ValueError("representation run is not valid selection-only output")
    if adapter := config.get("language_adapter"):
        directory = Path(adapter["directory"])
        _check_hash(directory / "adapter_config.json", adapter["config_sha256"])
        _check_hash(directory / "lora.safetensors", adapter["weights_sha256"])
    if override := config.get("projector_checkpoint_override"):
        directory = Path(override["directory"])
        _check_hash(directory / "projector.safetensors", override["weights_sha256"])
    dataset = config["dataset"]
    classes = [str(value) for value in dataset["classes"]]
    class_index = {answer: index for index, answer in enumerate(classes)}
    train_all = select_complete_task_pairs(
        read_jsonl(Path(dataset["train_data"])), str(dataset["task"])
    )
    selection_all = select_complete_task_pairs(
        read_jsonl(Path(dataset["selection_data"])), str(dataset["task"])
    )
    controls = {str(row["id"]): row for row in read_jsonl(Path(dataset["controls"]))}
    by_id = {str(row["id"]): row for row in train_all + selection_all}
    mate = _pair_index(selection_all)
    screening_pairs = config.get("screening_override", {}).get("limit_pairs")
    train_records = (
        train_all[: int(screening_pairs) * 2] if screening_pairs else train_all
    )
    selection_records = (
        selection_all[: int(screening_pairs) * 2] if screening_pairs else selection_all
    )
    if len(train_records) != int(summary["train_records"]):
        raise ValueError("representation train denominator mismatch")
    if len(selection_records) != int(summary["selection_records"]):
        raise ValueError("representation selection denominator mismatch")
    poolings = [str(value) for value in config["extraction"]["projector_poolings"]]
    expected_keys = expected_representation_keys(hidden_state_count=25, poolings=poolings)
    conditions = [str(value) for value in config["extraction"]["conditions"]]
    checkpoints = [str(row["id"]) for row in config["checkpoints"]]
    if set(summary["checkpoints"]) != set(checkpoints):
        raise ValueError("representation checkpoint set mismatch")
    verified_files = 0
    verified_rows = 0
    metadata_hash_by_cell: dict[str, str] = {}
    for checkpoint in checkpoints:
        cells = summary["checkpoints"][checkpoint]["cells"]
        expected_cells = {"train/vision"} | {
            f"selection/{condition}" for condition in conditions
        }
        if set(cells) != expected_cells:
            raise ValueError("representation cell set mismatch")
        for cell, entry in cells.items():
            split, condition = cell.split("/", 1)
            tensor_path = run / entry["tensor_file"]
            metadata_path = run / entry["metadata_file"]
            _check_hash(tensor_path, entry["tensor_sha256"])
            _check_hash(metadata_path, entry["metadata_sha256"])
            if tensor_path.stat().st_size != int(entry["tensor_bytes"]):
                raise ValueError("representation tensor byte count mismatch")
            if metadata_path.stat().st_size != int(entry["metadata_bytes"]):
                raise ValueError("representation metadata byte count mismatch")
            tensors = load_file(str(tensor_path), device="cpu")
            metadata = read_jsonl(metadata_path)
            expected_records = train_records if split == "train" else selection_records
            if set(tensors) != expected_keys or len(metadata) != len(expected_records):
                raise ValueError("representation tensor keys or row denominator mismatch")
            if int(entry["tensor_keys"]) != len(expected_keys):
                raise ValueError("representation tensor key count mismatch")
            expected_ids = [str(row["id"]) for row in expected_records]
            if [str(row["id"]) for row in metadata] != expected_ids:
                raise ValueError("representation metadata ID order mismatch")
            labels = tensors["labels"].long()
            source_labels = tensors["source_labels"].long()
            if labels.shape != (len(metadata),) or source_labels.shape != labels.shape:
                raise ValueError("representation label tensor shape mismatch")
            for key, tensor in tensors.items():
                if tensor.shape[0] != len(metadata):
                    raise ValueError(f"representation leading dimension mismatch: {key}")
                if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"representation tensor contains non-finite values: {key}")
            for index, row in enumerate(metadata):
                sample_id = str(row["id"])
                source_id = str(row["source_id"])
                if int(labels[index]) != class_index[str(by_id[sample_id]["answers"][0])]:
                    raise ValueError("representation target label mismatch")
                if int(source_labels[index]) != class_index[str(by_id[source_id]["answers"][0])]:
                    raise ValueError("representation source label mismatch")
                if split == "selection":
                    validate_visual_source(
                        condition,
                        sample_id,
                        source_id,
                        mate[sample_id],
                        controls[sample_id],
                    )
                elif condition != "vision" or source_id != sample_id:
                    raise ValueError("train representation source mismatch")
            cell_key = f"{split}/{condition}"
            previous = metadata_hash_by_cell.setdefault(cell_key, entry["metadata_sha256"])
            if previous != entry["metadata_sha256"]:
                raise ValueError("representation metadata drifted across checkpoints")
            verified_files += 2
            verified_rows += len(metadata)
    return {
        "status": "valid",
        "representation_files_verified": verified_files,
        "representation_rows_verified": verified_rows,
        "checkpoints_verified": len(checkpoints),
        "tensor_keys_per_cell": len(expected_keys),
        "final_half_scored": False,
    }


def verify_probe_analysis(run: Path, source_run: Path) -> dict:
    summary = json.loads((run / "SUMMARY.json").read_text(encoding="utf-8"))
    source_summary = source_run / "SUMMARY.json"
    if summary.get("status") != "valid" or summary.get("final_half_scored"):
        raise ValueError("probe analysis is not valid selection-only output")
    if summary["source_summary_sha256"] != sha256(source_summary):
        raise ValueError("probe analysis source summary hash mismatch")
    for name, entry in summary["files"].items():
        path = run / name
        _check_hash(path, entry["sha256"])
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError("probe analysis file byte count mismatch")
    metrics = read_csv(run / "probe_metrics.csv")
    native = read_csv(run / "native_logit_lens.csv")
    intervals = read_csv(run / "probe_intervals.csv")
    predictions = read_jsonl(run / "probe_predictions.jsonl")
    if len(metrics) != int(summary["metric_rows"]):
        raise ValueError("probe metric row count mismatch")
    if len(native) != int(summary["native_rows"]):
        raise ValueError("native readout row count mismatch")
    if len(intervals) != int(summary["interval_rows"]):
        raise ValueError("probe interval row count mismatch")
    if len(predictions) != int(summary["prediction_rows"]):
        raise ValueError("probe prediction row count mismatch")
    source = json.loads(source_summary.read_text(encoding="utf-8"))
    expected_records = int(source["selection_records"])
    if any(int(row["records"]) != expected_records for row in metrics + native):
        raise ValueError("probe analysis selection denominator mismatch")
    # v2 之后 vision 行必须带完整 pair-permutation null；旧 v1 可保留但不能作主显著性证据。
    validate_pair_permutation_rows(metrics)
    return {
        "status": "valid",
        "probe_files_verified": len(summary["files"]),
        "probe_metric_rows_verified": len(metrics),
        "probe_prediction_rows_verified": len(predictions),
        "vision_pair_permutation_null_verified": True,
        "final_half_scored": False,
    }


def verify_activation_patching(run: Path) -> dict:
    summary = json.loads((run / "SUMMARY.json").read_text(encoding="utf-8"))
    if summary.get("status") != "valid" or summary["metadata"].get("final_half_scored"):
        raise ValueError("activation patch run is not valid selection-only output")
    for name, entry in summary["files"].items():
        path = run / name
        _check_hash(path, entry["sha256"])
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError("activation patch file byte count mismatch")
    rows = read_jsonl(run / "patching_records.jsonl")
    curves = read_csv(run / "patching_curve.csv")
    if len(rows) != int(summary["raw_rows"]) or len(curves) != int(summary["curve_rows"]):
        raise ValueError("activation patch raw/curve denominator mismatch")
    checkpoints = set(summary["checkpoints"])
    scan_layers = set(map(int, summary["scan_layers"]))
    negative_layers = set(map(int, summary["negative_control_layers"]))
    expected_cells = {
        (checkpoint, intervention, layer)
        for checkpoint in checkpoints
        for intervention in ("correct_image_span", "correct_assistant")
        for layer in scan_layers
    }
    expected_cells |= {
        (checkpoint, intervention, layer)
        for checkpoint in checkpoints
        for intervention in ("wrong_label_donor_image_span", "zero_image_span")
        for layer in negative_layers
    }
    expected_cells |= {
        (checkpoint, intervention, -1)
        for checkpoint in checkpoints
        for intervention in ("input_clean_center", "input_clean_outer", "input_clean_full")
    }
    actual_cells = {
        (str(row["checkpoint"]), str(row["intervention"]), int(row["layer_index"]))
        for row in rows
    }
    if actual_cells != expected_cells:
        raise ValueError("activation patch intervention cell set mismatch")
    expected_per_cell = int(summary["records"])
    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for row in rows:
        key = (str(row["checkpoint"]), str(row["intervention"]), int(row["layer_index"]))
        grouped.setdefault(key, []).append(row)
        for field in (
            "clean_margin",
            "counterfactual_margin",
            "wrong_label_margin",
            "patched_margin",
            "effect_vs_counterfactual",
        ):
            if not math.isfinite(float(row[field])):
                raise ValueError("activation patch row contains non-finite margin")
    if any(len(cell_rows) != expected_per_cell for cell_rows in grouped.values()):
        raise ValueError("activation patch cell denominator mismatch")
    for checkpoint in checkpoints:
        full = grouped[(checkpoint, "input_clean_full", -1)]
        if max(abs(float(row["patched_margin"]) - float(row["clean_margin"])) for row in full) > 1e-6:
            raise ValueError("full input replacement does not reproduce clean margin")
        by_id = {str(row["id"]): float(row["clean_margin"]) for row in full}
        counter = {str(row["id"]): float(row["counterfactual_margin"]) for row in full}
        mate = {str(row["id"]): str(row["counterfactual_source_id"]) for row in full}
        error = max(abs(counter[sample_id] + by_id[mate[sample_id]]) for sample_id in by_id)
        if error > 5e-3:
            raise ValueError("activation patch paired margin antisymmetry failed")
        final_layer = int(summary.get("language_layers", 24)) - 1
        if final_layer in scan_layers:
            final_assistant = grouped[(checkpoint, "correct_assistant", final_layer)]
            if max(
                abs(float(row["patched_margin"]) - float(row["clean_margin"]))
                for row in final_assistant
            ) > 1e-6:
                raise ValueError("final-layer assistant patch does not reproduce clean margin")
    return {
        "status": "valid",
        "patching_files_verified": len(summary["files"]),
        "patching_rows_verified": len(rows),
        "patching_cells_verified": len(grouped),
        "final_half_scored": False,
    }
