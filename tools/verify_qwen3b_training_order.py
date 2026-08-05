#!/usr/bin/env python3
"""独立重哈希冻结的 3B 训练顺序、原图与实际监督目标。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from moonvit_glue.grounding_contract import parse_click_action
from moonvit_glue.training_order import (
    GROUNDING_ENRICHED_SELECTION_RULE,
    PREFIX_SELECTION_RULE,
    canonical_training_target,
    grounding_enriched_source_indices,
    verify_training_order_manifest,
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _logical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def _selection_matches_registered_rule(
    manifest: dict[str, Any], source_records: list[dict[str, Any]]
) -> bool:
    try:
        selection = manifest["selection"]
        if selection.get("shuffle") is not False:
            return False
        if selection.get("holdout_removed") is not False:
            return False
        source_indices = [
            int(row["source_row_index"]) for row in manifest["records"]
        ]
        rule = str(selection.get("rule"))
        if rule == PREFIX_SELECTION_RULE:
            return source_indices == list(range(len(source_indices)))
        if rule != GROUNDING_ENRICHED_SELECTION_RULE:
            return False
        if selection.get("within_route_order") != "frozen_source_order":
            return False
        if selection.get("merge_rule") != "alternate_grounding_then_short_answer":
            return False
        expected = grounding_enriched_source_indices(
            source_records,
            grounding_examples=int(selection["grounding_examples"]),
            short_answer_examples=int(selection["short_answer_examples"]),
        )
        return source_indices == expected
    except (KeyError, TypeError, ValueError):
        return False


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite verification: {args.out}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_records = _load_jsonl(args.data)
    entries = manifest["records"]
    image_root = args.data.resolve().parent

    record_mismatches: list[str] = []
    image_mismatches: list[str] = []
    target_mismatches: list[str] = []
    matched_image_bytes = 0
    for offset, entry in enumerate(entries):
        source_index = int(entry["source_row_index"])
        if source_index < 0 or source_index >= len(source_records):
            record_mismatches.append(f"{offset}:source_row_out_of_range")
            image_mismatches.append(f"{offset}:source_row_out_of_range")
            target_mismatches.append(f"{offset}:source_row_out_of_range")
            continue
        record = source_records[source_index]
        record_id = str(entry["id"])
        if (
            str(record.get("id")) != record_id
            or _logical_sha256(record) != str(entry["record_sha256"])
            or hashlib.sha256(
                str(record.get("question") or "").encode("utf-8")
            ).hexdigest()
            != str(entry["question_sha256"])
            or _logical_sha256(record.get("answers"))
            != str(entry["answers_sha256"])
        ):
            record_mismatches.append(record_id)

        expected_target, expected_transform = canonical_training_target(record)
        target = str(entry["target_answer"])
        if (
            target != expected_target
            or str(entry["target_transform"]) != expected_transform
            or hashlib.sha256(target.encode("utf-8")).hexdigest()
            != str(entry["target_answer_sha256"])
            or (
                str(entry["prompt_route"]) == "grounding"
                and parse_click_action(target) is None
            )
        ):
            target_mismatches.append(record_id)

        image_path = (image_root / str(entry["image"])).resolve()
        try:
            image_path.relative_to(image_root)
            encoded = image_path.read_bytes()
            with Image.open(image_path) as image:
                width, height = image.size
            image_matches = (
                len(encoded) == int(entry["image_bytes"])
                and hashlib.sha256(encoded).hexdigest()
                == str(entry["image_sha256"])
                and width == int(entry["image_width"])
                and height == int(entry["image_height"])
            )
        except (FileNotFoundError, OSError, ValueError):
            image_matches = False
            encoded = b""
        if image_matches:
            matched_image_bytes += len(encoded)
        else:
            image_mismatches.append(record_id)

        if offset == 0 or (offset + 1) % 100 == 0 or offset + 1 == len(entries):
            print(f"[{offset + 1}/{len(entries)}] independently verified", flush=True)

    source_counts = dict(sorted(Counter(row["source"] for row in entries).items()))
    route_counts = dict(
        sorted(Counter(row["prompt_route"] for row in entries).items())
    )
    transform_counts = dict(
        sorted(Counter(row["target_transform"] for row in entries).items())
    )
    checks = {
        "manifest_internal_verifier": verify_training_order_manifest(manifest),
        "contract_sha256": _file_sha256(args.contract)
        == str(manifest["contract_sha256"]),
        "data_sha256": _file_sha256(args.data) == str(manifest["data"]["sha256"]),
        "data_bytes": args.data.stat().st_size == int(manifest["data"]["bytes"]),
        "data_row_count": len(source_records)
        == int(manifest["data"]["total_records"]),
        "records_sha256": _logical_sha256(entries)
        == str(manifest["records_sha256"]),
        "record_id_and_logical_hashes": not record_mismatches,
        "image_sha256_bytes_and_dimensions": not image_mismatches,
        "teacher_targets_and_transforms": not target_mismatches,
        "source_counts": source_counts == manifest["source_counts"],
        "prompt_route_counts": route_counts == manifest["prompt_route_counts"],
        "target_transform_counts": transform_counts
        == manifest["target_transform_counts"],
        "selection_matches_registered_rule": _selection_matches_registered_rule(
            manifest, source_records
        ),
        "no_training_result_or_final_half": (
            manifest["training_results_exist"] is False
            and manifest["final_half_scored"] is False
        ),
    }
    status = "valid" if all(checks.values()) else "invalid"
    verification = {
        "schema_version": "qwen3b-training-order-independent-verification-v1",
        "status": status,
        "manifest_file_sha256": _file_sha256(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "records_sha256": manifest["records_sha256"],
        "checked_records": len(entries),
        "matched_records": len(entries) - len(record_mismatches),
        "matched_images": len(entries) - len(image_mismatches),
        "declared_image_bytes": sum(int(row["image_bytes"]) for row in entries),
        "matched_image_bytes": matched_image_bytes,
        "matched_targets": len(entries) - len(target_mismatches),
        "source_counts": source_counts,
        "prompt_route_counts": route_counts,
        "target_transform_counts": transform_counts,
        "checks": checks,
        "record_mismatches": record_mismatches,
        "image_mismatches": image_mismatches,
        "target_mismatches": target_mismatches,
        "training_results_exist": False,
        "final_half_scored": False,
        "paid_resources_used": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2), flush=True)
    if status != "valid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
