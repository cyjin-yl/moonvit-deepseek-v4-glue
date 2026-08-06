#!/usr/bin/env python3
"""冻结 Qwen3B projector health probe 的样本、图片与 MoonViT feature 绑定。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from moonvit_glue import FeatureCache
from moonvit_glue.screenspot_contract import seal_manifest, verify_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """按 dtype、shape 与连续内存字节绑定 feature，避免只 hash 文件路径。"""

    value = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--teacher-forced-count", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not verify_manifest(source):
        raise ValueError("source ScreenSpot manifest self-verification failed")
    if int(args.seed) != 20260805:
        raise ValueError("health probe seed is frozen at 20260805")
    if args.teacher_forced_count <= 0:
        raise ValueError("teacher-forced probe count must be positive")

    cache_manifest_path = args.feature_cache / "MANIFEST.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    cache = FeatureCache(args.feature_cache)
    cache_by_id = {str(row["id"]): row for row in cache_manifest["records"]}
    samples = list(source["samples"])
    if len(samples) < args.teacher_forced_count:
        raise ValueError("teacher-forced count exceeds frozen probe size")
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        sample_id = str(sample["sample_id"])
        cache_row = cache_by_id.get(sample_id)
        if cache_row is None:
            raise ValueError(f"feature cache is missing frozen probe sample: {sample_id}")
        groups = cache.get(sample_id, device="cpu", dtype=torch.float32)
        feature = groups[0]
        if list(feature.shape) != [int(value) for value in cache_row["feature_shape"]]:
            raise ValueError(f"feature shape differs for frozen probe sample: {sample_id}")
        rows.append(
            {
                "probe_index": index,
                "sample_id": sample_id,
                "platform": str(sample["platform"]),
                "target_type": str(sample["target_type"]),
                "image_sha256": str(sample["image_sha256"]),
                "feature_sha256": tensor_sha256(feature),
                "feature_shape": list(feature.shape),
                "cache_record_sha256": hashlib.sha256(
                    json.dumps(cache_row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            }
        )

    payload = {
        "format_version": "qwen3b-projector-health-probe-v1",
        "name": "qwen3b_projector_health_probe_v1",
        "label": "Frozen representation probe; excluded from training",
        "seed": int(args.seed),
        "source_dataset_manifest_file_sha256": sha256_file(args.manifest),
        "source_dataset_manifest_sha256": source["manifest_sha256"],
        "feature_cache_manifest_file_sha256": sha256_file(cache_manifest_path),
        "feature_cache_manifest_sha256": cache_manifest.get("manifest_sha256"),
        "count": len(rows),
        "teacher_forced_probe_count": int(args.teacher_forced_count),
        "teacher_forced_sample_ids": [
            row["sample_id"] for row in rows[: args.teacher_forced_count]
        ],
        "samples": rows,
        "immutability_rule": "sample IDs, order, image hashes and feature hashes are frozen after commit",
        "paid_resources_used": False,
    }
    sealed = seal_manifest(payload)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "PROBE_MANIFEST.json").write_text(
        json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
