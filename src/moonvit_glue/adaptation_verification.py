"""顶部 LoRA 诊断训练、评测与分析产物的独立验证。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file

from .metrics import normalize_answer


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def expected_lora_state_keys(resolved_modules: Sequence[str]) -> set[str]:
    return {
        f"{module}.{suffix}"
        for module in resolved_modules
        for suffix in ("lora_a", "lora_b")
    }


def _check_file(path: Path, entry: dict) -> None:
    if path.stat().st_size != int(entry["bytes"]) or sha256(path) != entry["sha256"]:
        raise ValueError(f"adaptation artifact hash/bytes mismatch: {path}")


def validate_balanced_task_history(
    history: Sequence[dict], *, tasks: Sequence[str], batch_size: int
) -> dict[str, int]:
    """确认每个真实 batch 都保持完全相同的任务配额。"""

    task_names = [str(task) for task in tasks]
    if not task_names or batch_size % len(task_names) != 0:
        raise ValueError("balanced adaptation batch/task contract is invalid")
    quota = batch_size // len(task_names)
    totals = {task: 0 for task in task_names}
    for row in history:
        counts = {str(key): int(value) for key, value in row.get("task_counts", {}).items()}
        if set(counts) != set(task_names) or any(counts[task] != quota for task in task_names):
            raise ValueError("balanced adaptation task quota drift")
        for task in task_names:
            totals[task] += counts[task]
    return totals


def verify_training_run(run: Path) -> dict:
    summary = json.loads((run / "SUMMARY.json").read_text(encoding="utf-8"))
    if summary.get("status") != "valid" or summary.get("final_half_scored"):
        raise ValueError("adaptation training run is not valid")
    for name, entry in summary["files"].items():
        _check_file(run / name, entry)
    history = read_jsonl(run / "train_history.jsonl")
    order = read_jsonl(run / "training_order.jsonl")
    if len(history) != int(summary["steps"]) or len(order) != len(history):
        raise ValueError("adaptation training history denominator mismatch")
    batch_size = int(summary["batch_size"])
    for index, (history_row, order_row) in enumerate(zip(history, order, strict=True), 1):
        if int(history_row["step"]) != index or int(order_row["step"]) != index:
            raise ValueError("adaptation training step order drift")
        if int(history_row["examples_seen"]) != index * batch_size:
            raise ValueError("adaptation examples_seen accounting mismatch")
        if len(order_row["ids"]) != batch_size:
            raise ValueError("adaptation true-batch ID count mismatch")
        for field in ("loss", "gradient_norm_before_clip", "step_wall_seconds"):
            if not math.isfinite(float(history_row[field])):
                raise ValueError("adaptation history contains non-finite values")
    if int(summary["examples_seen"]) != len(history) * batch_size:
        raise ValueError("adaptation summary examples_seen mismatch")
    task_examples = None
    if summary.get("tasks"):
        task_examples = validate_balanced_task_history(
            history, tasks=summary["tasks"], batch_size=batch_size
        )
        if task_examples != {
            str(task): int(count)
            for task, count in summary["train_records_by_task"].items()
        }:
            raise ValueError("balanced adaptation per-task epoch denominator mismatch")
    verified_tensors = 0
    for key, embedded_manifest in summary["checkpoints"].items():
        directory = run / "checkpoints" / key
        disk_manifest = json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))
        if disk_manifest != embedded_manifest:
            raise ValueError("adaptation checkpoint manifest drift")
        if int(disk_manifest["examples_seen"]) != int(disk_manifest["step"]) * batch_size:
            raise ValueError("checkpoint examples_seen accounting mismatch")
        for name, entry in disk_manifest["files"].items():
            _check_file(directory / name, entry)
        if summary["kind"] == "lora":
            tensors = load_file(str(directory / "lora.safetensors"), device="cpu")
            if set(tensors) != expected_lora_state_keys(summary["resolved_lora_modules"]):
                raise ValueError("LoRA checkpoint tensor keys mismatch")
            if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
                raise ValueError("LoRA checkpoint contains non-finite values")
            if int(disk_manifest["step"]) == 0 and any(
                bool(value.ne(0).any()) for name, value in tensors.items() if name.endswith("lora_b")
            ):
                raise ValueError("LoRA step-0 B matrix is not zero")
            verified_tensors += len(tensors)
        else:
            tensors = load_file(str(directory / "projector.safetensors"), device="cpu")
            if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
                raise ValueError("projector checkpoint contains non-finite values")
            verified_tensors += len(tensors)
    return {
        "status": "valid",
        "kind": summary["kind"],
        "steps_verified": len(history),
        "examples_verified": int(summary["examples_seen"]),
        "checkpoints_verified": len(summary["checkpoints"]),
        "checkpoint_tensors_verified": verified_tensors,
        "task_examples_verified": task_examples,
        "final_half_scored": False,
    }


def verify_evaluation(run: Path) -> dict:
    summary = json.loads((run / "SUMMARY.json").read_text(encoding="utf-8"))
    config = json.loads((run / "CONFIG.json").read_text(encoding="utf-8"))
    if summary.get("status") != "valid" or summary.get("final_half_scored"):
        raise ValueError("adaptation evaluation is not valid")
    for name, entry in summary["files"].items():
        _check_file(run / name, entry)
    for state in config["evaluation_states"]:
        if sha256(Path(state["lora"])) != state["lora_sha256"]:
            raise ValueError("evaluation LoRA source hash mismatch")
        if sha256(Path(state["projector"])) != state["projector_sha256"]:
            raise ValueError("evaluation projector source hash mismatch")
    preference = read_jsonl(run / "preference_records.jsonl")
    generation = read_jsonl(run / "generation_records.jsonl")
    if len(preference) != int(summary["preference_rows"]):
        raise ValueError("adaptation preference row denominator mismatch")
    if len(generation) != int(summary["generation_rows"]):
        raise ValueError("adaptation generation row denominator mismatch")
    states = {str(row["id"]) for row in config["evaluation_states"]}
    expected_preference_cells = {
        (state, condition)
        for state in states
        for condition in ("vision", "paired_counterfactual_image", "shuffled_image")
    }
    expected_generation_cells = {
        (state, condition)
        for state in states
        for condition in ("vision", "paired_counterfactual_image")
    }
    grouped_preference: dict[tuple[str, str], list[dict]] = defaultdict(list)
    grouped_generation: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in preference:
        grouped_preference[(str(row["state"]), str(row["condition"]))].append(row)
        if row.get("failure") is not None:
            raise ValueError("adaptation preference contains a failure")
        expected_margin = float(row["correct_logp"]) - float(row["counterfactual_logp"])
        if abs(float(row["correct_margin"]) - expected_margin) > 1e-8:
            raise ValueError("adaptation preference margin mismatch")
    for row in generation:
        grouped_generation[(str(row["state"]), str(row["condition"]))].append(row)
        if row.get("failure") is not None:
            raise ValueError("adaptation generation contains a failure")
        expected = normalize_answer(str(row["answers"][0]))
        predicted = normalize_answer(str(row["prediction"]))
        if bool(row["correct"]) != (expected == predicted):
            raise ValueError("adaptation generation correctness mismatch")
    if set(grouped_preference) != expected_preference_cells:
        raise ValueError("adaptation preference cell set mismatch")
    if set(grouped_generation) != expected_generation_cells:
        raise ValueError("adaptation generation cell set mismatch")
    preference_records = int(summary["teacher_forced_records_per_cell"])
    generation_records = int(summary["generation_records_per_cell"])
    if any(len(rows) != preference_records for rows in grouped_preference.values()):
        raise ValueError("adaptation preference cell denominator mismatch")
    if any(len(rows) != generation_records for rows in grouped_generation.values()):
        raise ValueError("adaptation generation cell denominator mismatch")
    for rows in [*grouped_preference.values(), *grouped_generation.values()]:
        pairs: dict[str, int] = defaultdict(int)
        for row in rows:
            pairs[str(row["pair_id"])] += 1
        if any(count != 2 for count in pairs.values()):
            raise ValueError("adaptation evaluation contains an incomplete pair")
    return {
        "status": "valid",
        "states_verified": len(states),
        "preference_rows_verified": len(preference),
        "generation_rows_verified": len(generation),
        "preference_cells_verified": len(grouped_preference),
        "generation_cells_verified": len(grouped_generation),
        "final_half_scored": False,
    }


def verify_analysis(run: Path, evaluation: Path) -> dict:
    summary = json.loads((run / "SUMMARY.json").read_text(encoding="utf-8"))
    if summary.get("status") != "valid" or summary.get("final_half_scored"):
        raise ValueError("adaptation analysis is not valid")
    if summary["source_summary_sha256"] != sha256(evaluation / "SUMMARY.json"):
        raise ValueError("adaptation analysis source hash mismatch")
    for name, entry in summary["files"].items():
        _check_file(run / name, entry)
    with (run / "adaptation_contrasts.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != int(summary["contrast_rows"]):
        raise ValueError("adaptation contrast row denominator mismatch")
    if any(not all(math.isfinite(float(row[field])) for field in ("mean_gap", "ci95_low", "ci95_high")) for row in rows):
        raise ValueError("adaptation contrast contains non-finite values")
    return {
        "status": "valid",
        "contrast_rows_verified": len(rows),
        "final_half_scored": False,
    }
