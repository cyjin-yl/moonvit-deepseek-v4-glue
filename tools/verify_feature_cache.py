#!/usr/bin/env python3
"""重哈希所有 shard，并逐条读取冻结视觉特征缓存。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from moonvit_glue import FeatureCache
from moonvit_glue.training_order import verify_training_order_manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_training_order_binding(
    cache_manifest: dict, training_order_manifest: dict
) -> dict:
    """逐行检查 cache 顺序、图像身份和 content-address alias。"""

    cache_records = cache_manifest["records"]
    order_records = training_order_manifest["records"]
    if len(cache_records) != len(order_records):
        raise ValueError("cache and training order record counts differ")
    first_id_by_image: dict[str, str] = {}
    aliased = 0
    maximum_tokens = 0
    for cache_record, order_record in zip(cache_records, order_records, strict=True):
        record_id = str(order_record["id"])
        comparable = ("id", "image_sha256", "image_width", "image_height")
        if any(cache_record[key] != order_record[key] for key in comparable):
            raise ValueError(f"cache record differs from training order: {record_id}")
        token_count = int(cache_record["feature_shape"][0])
        maximum_tokens = max(maximum_tokens, token_count)
        if token_count > int(
            training_order_manifest["feature_cache"]["max_visual_tokens"]
        ):
            raise ValueError(f"cached visual tokens exceed contract: {record_id}")
        image_sha256 = str(cache_record["image_sha256"])
        source_id = first_id_by_image.get(image_sha256)
        if source_id is None:
            first_id_by_image[image_sha256] = record_id
            if "alias_of" in cache_record:
                raise ValueError(f"first image occurrence cannot be an alias: {record_id}")
        else:
            aliased += 1
            if str(cache_record.get("alias_of")) != source_id:
                raise ValueError(f"cache alias is not first-occurrence canonical: {record_id}")
    if len(first_id_by_image) != int(training_order_manifest["unique_image_sha256"]):
        raise ValueError("cache unique image count differs from training order")
    if aliased != int(cache_manifest.get("aliased_records", -1)):
        raise ValueError("cache alias count differs from its manifest")
    return {
        "records_matched": len(cache_records),
        "unique_images_matched": len(first_id_by_image),
        "aliased_records_matched": aliased,
        "maximum_visual_tokens": maximum_tokens,
    }


def verify_feature_cache(
    root: Path,
    *,
    expected_count: int | None = None,
    training_order_manifest_path: Path | None = None,
    expected_git_sha: str | None = None,
) -> dict:
    root = root.resolve()
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache = FeatureCache(root)
    if expected_count is not None and len(cache) != expected_count:
        raise ValueError(f"feature cache count mismatch: {len(cache)} != {expected_count}")

    known_shards = {str(row["path"]): row for row in manifest["shards"]}
    for name, expected in known_shards.items():
        path = root / name
        if path.stat().st_size != int(expected["bytes"]):
            raise ValueError(f"feature shard byte count mismatch: {name}")
        if sha256(path) != str(expected["sha256"]):
            raise ValueError(f"feature shard SHA-256 mismatch: {name}")

    records_by_id = {str(record["id"]): record for record in manifest["records"]}
    if len(records_by_id) != len(manifest["records"]):
        raise ValueError("feature cache contains duplicate record IDs")
    values = 0
    unique_values = 0
    spans: set[tuple[str, int, int]] = set()
    aliases = 0
    for record in manifest["records"]:
        if str(record["shard"]) not in known_shards:
            raise ValueError(f"record points outside manifest shards: {record['id']}")
        span = (str(record["shard"]), int(record["start"]), int(record["end"]))
        if "alias_of" in record:
            aliases += 1
            source = records_by_id.get(str(record["alias_of"]))
            if source is None or "alias_of" in source:
                raise ValueError(f"invalid canonical alias source: {record['id']}")
            comparable = (
                "image_sha256",
                "image_width",
                "image_height",
                "feature_shape",
                "dtype",
                "shard",
                "start",
                "end",
            )
            if any(record[key] != source[key] for key in comparable):
                raise ValueError(f"alias differs from source feature identity: {record['id']}")
        feature = cache.get(str(record["id"]))[0]
        if list(feature.shape) != list(record["feature_shape"]):
            raise ValueError(f"feature shape mismatch: {record['id']}")
        if not bool(torch.isfinite(feature).all()):
            raise ValueError(f"non-finite cached feature: {record['id']}")
        values += feature.numel()
        if span not in spans:
            unique_values += feature.numel()
            spans.add(span)

    if "unique_feature_spans" in manifest and len(spans) != int(
        manifest["unique_feature_spans"]
    ):
        raise ValueError("unique feature span count differs from manifest")
    if "aliased_records" in manifest and aliases != int(manifest["aliased_records"]):
        raise ValueError("aliased record count differs from manifest")

    order_binding = None
    if training_order_manifest_path is not None:
        order_path = training_order_manifest_path.resolve()
        order_manifest = json.loads(order_path.read_text(encoding="utf-8"))
        if not verify_training_order_manifest(order_manifest):
            raise ValueError("training order manifest is invalid")
        if manifest.get("training_order_manifest_sha256") != order_manifest.get(
            "manifest_sha256"
        ):
            raise ValueError("cache training-order self-hash binding differs")
        if manifest.get("training_order_records_sha256") != order_manifest.get(
            "records_sha256"
        ):
            raise ValueError("cache training-order records hash binding differs")
        if manifest.get("training_order_manifest_file_sha256") != sha256(order_path):
            raise ValueError("cache training-order file hash binding differs")
        order_binding = compare_training_order_binding(manifest, order_manifest)

    runtime_sources = manifest.get("runtime_source_files") or []
    if expected_git_sha is not None:
        if manifest.get("git_sha") != expected_git_sha:
            raise ValueError("cache runner Git SHA differs from expected commit")
        if manifest.get("git_tracked_worktree_clean") is not True:
            raise ValueError("cache runner did not attest a clean tracked worktree")
        if not runtime_sources:
            raise ValueError("cache runner did not record runtime source files")
    for source in runtime_sources:
        source_path = Path(source["path"])
        if not source_path.is_file() or sha256(source_path) != str(source["sha256"]):
            raise ValueError(f"cache runtime source differs: {source_path}")

    return {
        "status": "valid",
        "cache": str(root),
        "manifest_sha256": sha256(manifest_path),
        "records_verified": len(manifest["records"]),
        "shards_verified": len(known_shards),
        "values_verified": values,
        "unique_values_verified": unique_values,
        "unique_feature_spans": len(spans),
        "aliased_records": aliases,
        "total_shard_bytes": sum(int(row["bytes"]) for row in known_shards.values()),
        "runner_git_sha": manifest.get("git_sha"),
        "runner_tracked_worktree_clean": manifest.get("git_tracked_worktree_clean"),
        "runtime_source_files_verified": len(runtime_sources),
        "training_order_binding": order_binding,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--training-order-manifest", type=Path)
    parser.add_argument("--expected-git-sha")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify_feature_cache(
        args.cache,
        expected_count=args.expected_count,
        training_order_manifest_path=args.training_order_manifest,
        expected_git_sha=args.expected_git_sha,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
