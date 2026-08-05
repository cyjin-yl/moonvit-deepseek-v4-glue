#!/usr/bin/env python3
"""重哈希所有 shard，并逐条读取冻结视觉特征缓存。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from moonvit_glue import FeatureCache


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_feature_cache(root: Path, *, expected_count: int | None = None) -> dict:
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify_feature_cache(args.cache, expected_count=args.expected_count)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
