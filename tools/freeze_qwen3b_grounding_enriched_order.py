#!/usr/bin/env python3
"""冻结固定预算的 grounding-enriched 3B 训练顺序。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from moonvit_glue.training_order import (
    GROUNDING_ENRICHED_SELECTION_RULE,
    build_training_order_manifest,
    grounding_enriched_source_indices,
    verify_training_order_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"training row {line_number} is not an object")
            records.append(value)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--grounding-examples", required=True, type=int)
    parser.add_argument("--short-answer-examples", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite training order: {args.out}")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    records = _load_jsonl(args.data)
    examples_seen = args.grounding_examples + args.short_answer_examples
    source_indices = grounding_enriched_source_indices(
        records,
        grounding_examples=args.grounding_examples,
        short_answer_examples=args.short_answer_examples,
    )

    def progress(done: int, total: int) -> None:
        if done == 1 or done == total or done % 100 == 0:
            print(f"[{done}/{total}] froze record and image identity", flush=True)

    manifest = build_training_order_manifest(
        data_path=args.data,
        contract=contract,
        contract_sha256=_sha256(args.contract),
        examples_seen=examples_seen,
        progress=progress,
        source_indices=source_indices,
        selection_rule=GROUNDING_ENRICHED_SELECTION_RULE,
        selection_metadata={
            "grounding_examples": args.grounding_examples,
            "short_answer_examples": args.short_answer_examples,
            "within_route_order": "frozen_source_order",
            "merge_rule": "alternate_grounding_then_short_answer",
        },
    )
    if not verify_training_order_manifest(manifest):
        raise RuntimeError("generated training order failed its internal verifier")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    print(
        json.dumps(
            {
                "status": "frozen",
                "out": str(args.out),
                "selection_rule": manifest["selection"]["rule"],
                "manifest_sha256": manifest["manifest_sha256"],
                "records_sha256": manifest["records_sha256"],
                "examples_seen": manifest["selection"]["examples_seen"],
                "optimizer_steps": manifest["selection"]["optimizer_steps"],
                "source_counts": manifest["source_counts"],
                "prompt_route_counts": manifest["prompt_route_counts"],
                "target_transform_counts": manifest["target_transform_counts"],
                "unique_image_sha256": manifest["unique_image_sha256"],
                "training_results_exist": False,
                "final_half_scored": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
