#!/usr/bin/env python3
"""为机制实验冻结完整 task pair 与预注册的 activation-patching 子集。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from moonvit_glue.mechanism_probe import select_complete_task_pairs
from tools_common import load_records


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranked_pairs(records: list[dict], seed: int, count: int) -> list[str]:
    pair_ids = sorted({str(record["pair_id"]) for record in records})
    ranked = sorted(
        pair_ids,
        key=lambda pair_id: hashlib.sha256(f"{seed}:{pair_id}".encode()).hexdigest(),
    )
    if count <= 0 or count > len(ranked):
        raise ValueError("patch pair count falls outside available complete pairs")
    return ranked[:count]


def write_ids(path: Path, records: list[dict], *, source: Path, task: str) -> None:
    payload = {
        "format_version": "mechanism-record-ids-v1",
        "source": str(source.resolve()),
        "source_sha256": file_sha256(source),
        "task": task,
        "records": [{"id": str(record["id"])} for record in records],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite mechanism selection: {args.out}")
    args.out.mkdir(parents=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    task = str(dataset["task"])
    train_path = Path(dataset["train_data"])
    selection_path = Path(dataset["selection_data"])
    train = select_complete_task_pairs(load_records(train_path), task)
    selection = select_complete_task_pairs(load_records(selection_path), task)
    if len(train) != int(dataset["expected_train_records"]):
        raise ValueError("mechanism train denominator mismatch")
    if len(selection) != int(dataset["expected_selection_records"]):
        raise ValueError("mechanism selection denominator mismatch")
    train_pairs = {str(record["pair_id"]) for record in train}
    selection_pairs = {str(record["pair_id"]) for record in selection}
    train_ids = {str(record["id"]) for record in train}
    selection_ids = {str(record["id"]) for record in selection}
    if train_pairs & selection_pairs or train_ids & selection_ids:
        raise ValueError("mechanism train/selection identities overlap")

    patch_config = config["activation_patching"]
    patch_pairs = ranked_pairs(
        selection,
        int(patch_config["pair_selection_seed"]),
        int(patch_config["pair_count"]),
    )
    patch_pair_set = set(patch_pairs)
    patch_records = [record for record in selection if str(record["pair_id"]) in patch_pair_set]
    write_ids(args.out / "train_ids.json", train, source=train_path, task=task)
    write_ids(args.out / "selection_ids.json", selection, source=selection_path, task=task)
    write_ids(
        args.out / "patching_selection_ids.json",
        patch_records,
        source=selection_path,
        task=task,
    )
    classes = [str(value) for value in dataset["classes"]]
    manifest = {
        "format_version": "mechanism-selection-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "classes": classes,
        "train_records": len(train),
        "train_pairs": len(train_pairs),
        "selection_records": len(selection),
        "selection_pairs": len(selection_pairs),
        "patching_records": len(patch_records),
        "patching_pairs": len(patch_pairs),
        "patching_pair_ids": patch_pairs,
        "answer_counts": {
            "train": dict(Counter(str(record["answers"][0]) for record in train)),
            "selection": dict(Counter(str(record["answers"][0]) for record in selection)),
            "patching": dict(Counter(str(record["answers"][0]) for record in patch_records)),
        },
        "train_selection_id_overlap": 0,
        "train_selection_pair_overlap": 0,
        "selection_rule": "sha256(seed:pair_id), retain both variants",
        "selection_seed": int(patch_config["pair_selection_seed"]),
        "logical_dataset_sha256": str(dataset["logical_dataset_sha256"]),
        "files": {},
        "final_half_scored": False,
    }
    for name in ("train_ids.json", "selection_ids.json", "patching_selection_ids.json"):
        path = args.out / name
        manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    (args.out / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
