"""冻结 Qwen 代理的 text-only language-retention selection。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from moonvit_glue.screenspot_contract import seal_manifest, verify_manifest

SEED = "20260805"
MMLU_REPO = "TIGER-Lab/MMLU-Pro"
MMLU_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
MMLU_PATH = "data/test-00000-of-00001.parquet"
MMLU_BYTES = 4_144_185
MMLU_SHA256 = "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8"
GSM_REPO = "openai/gsm8k"
GSM_REVISION = "740312add88f781978c0658806c59bc2815b9866"
GSM_PATH = "main/test-00000-of-00001.parquet"
GSM_BYTES = 419_088
GSM_SHA256 = "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"
MMLU_PER_CATEGORY = 10
GSM_COUNT = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(namespace: str, identity: str) -> str:
    return hashlib.sha256(f"{SEED}\0{namespace}\0{identity}".encode()).hexdigest()


def _verify_file(path: Path, *, size: int, digest: str) -> None:
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise ValueError(f"source file contract mismatch: {path}")


def _gsm_reference(answer: str) -> str:
    match = re.search(r"####\s*(.+?)\s*$", answer)
    if match is None:
        raise ValueError(f"GSM8K answer has no final marker: {answer!r}")
    return match.group(1).replace(",", "").strip()


def build_manifest(mmlu_path: Path, gsm_path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required") from error

    _verify_file(mmlu_path, size=MMLU_BYTES, digest=MMLU_SHA256)
    _verify_file(gsm_path, size=GSM_BYTES, digest=GSM_SHA256)
    mmlu_rows = pq.read_table(mmlu_path).to_pylist()
    gsm_rows = pq.read_table(gsm_path).to_pylist()

    by_category: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for source_index, row in enumerate(mmlu_rows):
        by_category[str(row["category"])].append((source_index, row))
    selected_mmlu: list[dict[str, Any]] = []
    for category in sorted(by_category):
        ordered = sorted(
            by_category[category],
            key=lambda item: _rank(
                "mmlu-pro-within-category", f"{item[1]['question_id']}\0{item[1]['question']}"
            ),
        )
        for source_index, row in ordered[:MMLU_PER_CATEGORY]:
            selected_mmlu.append(
                {
                    "sample_id": f"language-mmlu-pro-{int(row['question_id']):05d}",
                    "source_row_index": source_index,
                    "question_id": int(row["question_id"]),
                    "question": row["question"],
                    "options": list(row["options"]),
                    "answer": row["answer"],
                    "answer_index": int(row["answer_index"]),
                    "category": category,
                    "source_subject": row["src"],
                }
            )
    selected_mmlu.sort(key=lambda row: _rank("mmlu-pro-order", row["sample_id"]))
    for order, row in enumerate(selected_mmlu):
        row["evaluation_order"] = order

    ordered_gsm = sorted(
        enumerate(gsm_rows),
        key=lambda item: _rank("gsm8k-selection", f"{item[0]}\0{item[1]['question']}"),
    )[:GSM_COUNT]
    selected_gsm = []
    for order, (source_index, row) in enumerate(ordered_gsm):
        selected_gsm.append(
            {
                "sample_id": f"language-gsm8k-{source_index:04d}",
                "source_row_index": source_index,
                "evaluation_order": order,
                "question": row["question"],
                "answer_raw": row["answer"],
                "answer_final": _gsm_reference(row["answer"]),
            }
        )

    manifest = {
        "schema_version": "language-retention-contract-v1",
        "name": "language_retention_v1",
        "frozen_on": "2026-08-05",
        "seed": SEED,
        "immutability_rule": "once committed, membership and order must not change",
        "sources": {
            "mmlu_pro": {
                "repo": MMLU_REPO,
                "resolved_revision": MMLU_REVISION,
                "split": "test",
                "path": MMLU_PATH,
                "bytes": MMLU_BYTES,
                "sha256": MMLU_SHA256,
                "source_count": len(mmlu_rows),
            },
            "gsm8k": {
                "repo": GSM_REPO,
                "resolved_revision": GSM_REVISION,
                "config": "main",
                "split": "test",
                "path": GSM_PATH,
                "bytes": GSM_BYTES,
                "sha256": GSM_SHA256,
                "source_count": len(gsm_rows),
            },
        },
        "selection": {
            "mmlu_pro": {
                "method": "stable SHA-256 rank, 10 rows per category",
                "count": len(selected_mmlu),
                "category_counts": dict(
                    sorted(Counter(row["category"] for row in selected_mmlu).items())
                ),
            },
            "gsm8k": {
                "method": "stable SHA-256 rank over the full main test split",
                "count": len(selected_gsm),
            },
            "total_count": len(selected_mmlu) + len(selected_gsm),
        },
        "evaluation": {
            "condition": "text-only; no image placeholder or image tokens",
            "comparisons": ["candidate-minus-step0", "candidate-minus-previous-best"],
            "mmlu_pro": {
                "metric": "strict exact option letter",
                "generation_max_new_tokens": 8,
                "prompt": "Answer the multiple-choice question. Return exactly one option letter and no other text.",
            },
            "gsm8k": {
                "metric": "exact normalized final numeric answer",
                "generation_max_new_tokens": 256,
                "prompt": "Solve the problem. End with exactly: Final answer: <number>",
            },
            "common": {
                "do_sample": False,
                "temperature": 0.0,
                "report_teacher_forced_answer_nll": True,
            },
        },
        "samples": {
            "mmlu_pro": selected_mmlu,
            "gsm8k": selected_gsm,
        },
    }
    return seal_manifest(manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmlu-pro", type=Path, required=True)
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.mmlu_pro, args.gsm8k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reread = json.loads(args.out.read_text(encoding="utf-8"))
    if not verify_manifest(reread):
        raise RuntimeError("language-retention manifest failed self-verification")
    print(
        json.dumps(
            {
                "path": str(args.out),
                "manifest_sha256": manifest["manifest_sha256"],
                "selection": manifest["selection"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
