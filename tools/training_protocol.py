"""Reproducible measurement and validation protocol for alignment training."""

from __future__ import annotations

import hashlib
import random
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Sequence, TypeVar

from moonvit_glue.metrics import normalize_answer

T = TypeVar("T")


def records_manifest_sha256(records: Sequence[dict]) -> str:
    """Hash logical dataset rows without re-reading large embedded image payloads."""

    logical_rows = [
        {key: value for key, value in record.items() if key != "image_bytes"}
        for record in records
    ]
    logical_rows.sort(key=lambda record: str(record.get("id")))
    encoded = json.dumps(
        logical_rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_batch_semantics(
    *,
    micro_batch_size: int,
    gradient_accumulation_steps: int | None,
    legacy_batch_size: int | None,
) -> dict[str, int | bool]:
    """Resolve honest batch terminology while preserving the old CLI alias."""

    if micro_batch_size != 1:
        raise ValueError(
            "true batched forward is not implemented; use micro_batch_size=1 and "
            "gradient accumulation until variable-length batching lands"
        )
    if gradient_accumulation_steps is not None and legacy_batch_size is not None:
        if gradient_accumulation_steps != legacy_batch_size:
            raise ValueError("--batch-size conflicts with --gradient-accumulation-steps")
    legacy_used = gradient_accumulation_steps is None and legacy_batch_size is not None
    if gradient_accumulation_steps is not None:
        accumulation = gradient_accumulation_steps
    elif legacy_batch_size is not None:
        accumulation = legacy_batch_size
    else:
        accumulation = 4
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    return {
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": micro_batch_size * accumulation,
        "legacy_batch_size_used": legacy_used,
    }


def restore_progress_counts(
    *,
    start_step: int,
    last_history: dict,
    effective_batch_size: int,
    batch_semantics_explicit: bool,
) -> dict[str, int | bool]:
    """Restore honest counters, refusing to guess legacy checkpoint batch size."""

    if start_step < 0 or effective_batch_size <= 0:
        raise ValueError("start_step must be non-negative and effective_batch_size positive")
    if "examples_seen" in last_history:
        examples_seen = int(last_history["examples_seen"])
    elif start_step:
        if not batch_semantics_explicit:
            raise ValueError(
                "legacy checkpoint lacks examples_seen; pass its original "
                "--gradient-accumulation-steps (or deprecated --batch-size) explicitly"
            )
        examples_seen = start_step * effective_batch_size
    else:
        examples_seen = 0
    answer_complete = start_step == 0 or "answer_tokens_seen" in last_history
    return {
        "examples_seen": examples_seen,
        "answer_tokens_seen": int(last_history.get("answer_tokens_seen", 0)),
        "answer_token_accounting_complete": answer_complete,
    }


def canonical_source(record: dict) -> str:
    source = str(record.get("source") or "unknown")
    if source.startswith("textvqa"):
        return "textvqa"
    if source.startswith("docvqa"):
        return "docvqa"
    if source.startswith("showui"):
        return "showui"
    if source in {"train", "art", "sft_art"}:
        return "art"
    return source


def _build_validation_manifest(records: Sequence[dict], total_samples: int, seed: int) -> dict:
    if total_samples <= 0:
        raise ValueError("validation total_samples must be positive")
    groups: dict[str, list[dict]] = {}
    seen_ids: set[str] = set()
    for record in records:
        record_id = str(record.get("id") or "")
        if not record_id:
            raise ValueError("every training record needs a stable id")
        if record_id in seen_ids:
            raise ValueError(f"duplicate training record id: {record_id}")
        seen_ids.add(record_id)
        groups.setdefault(canonical_source(record), []).append(record)
    sources = sorted(groups)
    if total_samples < len(sources):
        raise ValueError("validation total must cover every source")
    base, remainder = divmod(total_samples, len(sources))
    selected: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for index, source in enumerate(sources):
        quota = base + int(index < remainder)
        candidates = sorted(groups[source], key=lambda record: str(record["id"]))
        if len(candidates) < quota:
            raise ValueError(f"source {source} has {len(candidates)} records, needs {quota}")
        rng = random.Random(f"{seed}:{source}")
        chosen = sorted(rng.sample(candidates, quota), key=lambda record: str(record["id"]))
        selected.extend({"id": str(record["id"]), "source": source} for record in chosen)
        counts[source] = quota
    return {
        "version": 1,
        "seed": seed,
        "total_samples": total_samples,
        "counts_by_source": counts,
        "records": selected,
    }


def prepare_validation_split(
    records: Sequence[dict],
    *,
    manifest_path: str | Path,
    total_samples: int,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    """Create or reuse an ID-pinned, source-stratified validation manifest."""

    path = Path(manifest_path)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = _build_validation_manifest(records, total_samples, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    by_id = {str(record.get("id")): record for record in records}
    manifest_ids = [entry["id"] for entry in manifest["records"]]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("validation manifest contains duplicate ids")
    missing = [record_id for record_id in manifest_ids if record_id not in by_id]
    if missing:
        raise ValueError(f"validation ids missing from dataset: {missing[:3]}")
    validation = [by_id[record_id] for record_id in manifest_ids]
    for record, entry in zip(validation, manifest["records"]):
        if canonical_source(record) != entry["source"]:
            raise ValueError(f"validation source mismatch for {entry['id']}")
    held_out = set(manifest_ids)
    training = [record for record in records if str(record.get("id")) not in held_out]
    return training, validation, manifest


def summarize_validation_losses(
    records: Sequence[dict],
    *,
    true_losses: Sequence[float],
    shuffled_loss_runs: Sequence[Sequence[float]],
    shuffled_id_runs: Sequence[Sequence[str]],
) -> dict:
    """Summarize true-vs-deranged losses overall and independently by source."""

    count = len(records)
    if count == 0 or len(true_losses) != count:
        raise ValueError("true losses must match non-empty validation records")
    if len(shuffled_loss_runs) != len(shuffled_id_runs) or not shuffled_loss_runs:
        raise ValueError("shuffle loss and id runs must be non-empty and aligned")
    for losses, ids in zip(shuffled_loss_runs, shuffled_id_runs):
        if len(losses) != count or len(ids) != count:
            raise ValueError("every shuffle run must match validation record count")

    def metrics(indices: list[int]) -> dict[str, float | int]:
        true_loss = mean(float(true_losses[index]) for index in indices)
        shuffled_means = [
            mean(float(run[index]) for index in indices) for run in shuffled_loss_runs
        ]
        deltas = [shuffled - true_loss for shuffled in shuffled_means]
        return {
            "count": len(indices),
            "true_loss": true_loss,
            "shuffled_loss_mean": mean(shuffled_means),
            "shuffle_delta_mean": mean(deltas),
            "shuffle_delta_std": pstdev(deltas),
        }

    source_indices: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        source_indices.setdefault(canonical_source(record), []).append(index)
    shuffle_runs = []
    record_ids = [str(record["id"]) for record in records]
    for run_index, (losses, image_ids) in enumerate(
        zip(shuffled_loss_runs, shuffled_id_runs), start=1
    ):
        shuffled_mean = mean(float(loss) for loss in losses)
        shuffle_runs.append({
            "run": run_index,
            "shuffled_loss": shuffled_mean,
            "delta": shuffled_mean - mean(float(loss) for loss in true_losses),
            "pairs": [
                {"record_id": record_id, "image_id": str(image_id)}
                for record_id, image_id in zip(record_ids, image_ids)
            ],
        })
    return {
        "std_kind": "population_across_derangements",
        "overall": metrics(list(range(count))),
        "by_source": {
            source: metrics(indices) for source, indices in sorted(source_indices.items())
        },
        "shuffle_runs": shuffle_runs,
    }


def make_derangements(items: Sequence[T], *, repeats: int, seed: int) -> list[list[T]]:
    """Return seeded Sattolo permutations, guaranteeing no fixed points."""

    if len(items) < 2:
        raise ValueError("derangement requires at least two items")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    rng = random.Random(seed)
    results: list[list[T]] = []
    for _ in range(repeats):
        indices = list(range(len(items)))
        for index in range(len(indices) - 1, 0, -1):
            replacement = rng.randrange(index)
            indices[index], indices[replacement] = indices[replacement], indices[index]
        results.append([items[index] for index in indices])
    return results


@dataclass(frozen=True)
class SupervisionChoice:
    raw_answers: list[str]
    canonical_answer: str
    selected_answer: str
    normalization_rule: str


def select_supervision(
    answers: Sequence[str],
    *,
    rule: str = "canonical",
    rng: random.Random | None = None,
) -> SupervisionChoice:
    """Select a reproducible teacher answer while retaining answer provenance."""

    raw_answers = [str(answer) for answer in answers if str(answer).strip()]
    if not raw_answers:
        raise ValueError("at least one non-empty answer is required")
    if len(raw_answers) == 1:
        answer = raw_answers[0].strip()
        return SupervisionChoice(raw_answers, answer, answer, "single_answer_passthrough")

    normalized = [normalize_answer(answer) for answer in raw_answers]
    counts = Counter(normalized)
    first_index = {answer: normalized.index(answer) for answer in counts}
    canonical = max(counts, key=lambda answer: (counts[answer], -first_index[answer]))
    if rule == "canonical":
        selected = canonical
        normalization_rule = "vqa_normalized_majority"
    elif rule == "random":
        if rng is None:
            raise ValueError("random answer selection requires an explicit RNG")
        selected = rng.choice(raw_answers).strip()
        normalization_rule = "seeded_random_acceptable_answer"
    else:
        raise ValueError(f"unknown answer selection rule: {rule}")
    return SupervisionChoice(raw_answers, canonical, selected, normalization_rule)


@dataclass
class TrainingProgress:
    """Count actual work independently of the overloaded word ``step``."""

    total_training_examples: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    optimizer_steps: int = 0
    examples_seen: int = 0
    answer_tokens_seen: int = 0

    def __post_init__(self) -> None:
        for name in (
            "total_training_examples",
            "micro_batch_size",
            "gradient_accumulation_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    @property
    def effective_epochs(self) -> float:
        return self.examples_seen / self.total_training_examples

    def record_microbatch(self, *, examples: int, answer_tokens: int) -> None:
        if examples <= 0 or answer_tokens < 0:
            raise ValueError("examples must be positive and answer_tokens non-negative")
        self.examples_seen += examples
        self.answer_tokens_seen += answer_tokens

    def record_optimizer_step(self) -> None:
        self.optimizer_steps += 1

    def snapshot(self) -> dict[str, int | float]:
        return {
            "optimizer_steps": self.optimizer_steps,
            "examples_seen": self.examples_seen,
            "answer_tokens_seen": self.answer_tokens_seen,
            "effective_epochs": self.effective_epochs,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
        }
