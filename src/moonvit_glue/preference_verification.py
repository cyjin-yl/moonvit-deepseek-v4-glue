"""独立校验 teacher-forced paired-preference run 的完整性。"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify_preference_run(run_dir: Path | str) -> dict:
    """校验精确矩阵覆盖、pair 完整性、有限数值与文件哈希。"""

    run = Path(run_dir)
    config_path = run / "CONFIG.json"
    summary_path = run / "SUMMARY.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["metadata"].get("final_half_scored") is not False:
        raise ValueError("preference run must attest final_half_scored=false")
    if _sha256(config_path) != summary["metadata"].get("config_sha256"):
        raise ValueError("config SHA-256 mismatch")

    for filename, expected in summary["raw_files"].items():
        path = run / filename
        if not path.exists():
            raise ValueError(f"missing raw file: {filename}")
        if path.stat().st_size != int(expected["bytes"]) or _sha256(path) != expected["sha256"]:
            raise ValueError(f"raw file hash mismatch: {filename}")

    checkpoint_ids = [str(row["id"]) for row in config["checkpoints"]]
    aliases = {str(row["id"]): str(row["source"]) for row in config.get("aliases", [])}
    if set(checkpoint_ids) | set(aliases) != set(summary["checkpoints"]):
        raise ValueError("checkpoint or alias coverage mismatch")
    for alias, source in aliases.items():
        if summary["checkpoints"][alias].get("alias_of") != source:
            raise ValueError(f"alias provenance mismatch: {alias}")

    dataset = config["synthetic"]
    conditions = [str(value) for value in dataset["conditions"]]
    expected_records = int(dataset["expected_records"])
    rows = _jsonl(run / "preference_records.jsonl")
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    finite_fields = (
        "correct_logp_mean",
        "counterfactual_logp_mean",
        "correct_token_nll",
        "counterfactual_token_nll",
        "correct_margin",
    )
    for row in rows:
        checkpoint = str(row["checkpoint"])
        condition = str(row["condition"])
        sample_id = str(row["id"])
        key = (checkpoint, condition, sample_id)
        if key in seen:
            raise ValueError(f"duplicate preference row: {key}")
        seen.add(key)
        if checkpoint not in checkpoint_ids or condition not in conditions:
            raise ValueError(f"unexpected preference cell: {checkpoint}/{condition}")
        if row.get("failure") is None:
            for field in finite_fields:
                if not math.isfinite(float(row[field])):
                    raise ValueError(f"non-finite preference value: {field}")
            if int(row["correct_answer_tokens"]) < 1 or int(row["counterfactual_answer_tokens"]) < 1:
                raise ValueError("answer token count must be positive")
            if bool(row["preference_correct"]) != (float(row["correct_margin"]) > 0):
                raise ValueError("preference flag disagrees with correct margin")
        cells[(checkpoint, condition)].append(row)

    expected_cells = {(checkpoint, condition) for checkpoint in checkpoint_ids for condition in conditions}
    if set(cells) != expected_cells:
        raise ValueError("preference checkpoint-condition coverage mismatch")
    canonical_ids: set[str] | None = None
    pairs_per_cell: int | None = None
    for cell in sorted(expected_cells):
        cell_rows = cells[cell]
        if len(cell_rows) != expected_records:
            raise ValueError(f"preference denominator mismatch for {cell}")
        identifiers = {str(row["id"]) for row in cell_rows}
        if canonical_ids is None:
            canonical_ids = identifiers
        elif identifiers != canonical_ids:
            raise ValueError(f"preference condition sample IDs mismatch for {cell}")
        pairs: dict[str, list[dict]] = defaultdict(list)
        for row in cell_rows:
            pairs[str(row["pair_id"])].append(row)
        malformed = [pair_id for pair_id, pair in pairs.items() if len(pair) != 2]
        if malformed:
            raise ValueError(f"incomplete preference pairs for {cell}: {malformed[:3]}")
        if pairs_per_cell is None:
            pairs_per_cell = len(pairs)
        elif len(pairs) != pairs_per_cell:
            raise ValueError(f"preference pair denominator drift for {cell}")

    if "paired_counterfactual_image" in conditions:
        rows_by_key = {
            (str(row["checkpoint"]), str(row["condition"]), str(row["id"])): row
            for row in rows
        }
        assert canonical_ids is not None
        for checkpoint in checkpoint_ids:
            pair_members: dict[str, set[str]] = defaultdict(set)
            for sample_id in canonical_ids:
                row = rows_by_key[(checkpoint, "vision", sample_id)]
                pair_members[str(row["pair_id"])].add(sample_id)
            for condition in conditions:
                for sample_id in canonical_ids:
                    row = rows_by_key[(checkpoint, condition, sample_id)]
                    source_id = row.get("visual_source_id")
                    if condition == "blind" and source_id is not None:
                        raise ValueError("blind preference row has a visual source")
                    if condition in {
                        "vision",
                        "patch_permutation",
                        "background_matched_aux",
                    } and source_id != sample_id:
                        raise ValueError(f"preference visual source mismatch: {condition}/{sample_id}")
                    if condition == "shuffled_image" and source_id == sample_id:
                        raise ValueError(f"preference shuffle fixed point: {sample_id}")
                    if condition == "paired_counterfactual_image":
                        members = pair_members[str(row["pair_id"])]
                        if members != {sample_id, str(source_id)}:
                            raise ValueError(
                                f"preference paired source mismatch: {sample_id}"
                            )

    return {
        "status": "valid",
        "rows_verified": len(rows),
        "cells_verified": len(cells),
        "pairs_per_cell": pairs_per_cell,
        "aliases_verified": len(aliases),
    }
