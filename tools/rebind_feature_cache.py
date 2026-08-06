#!/usr/bin/env python3
"""把已验证的视觉特征缓存绑定到另一份等序训练清单。

该工具只改写 ``MANIFEST.json`` 的训练顺序绑定字段，并用硬链接复用原有
``features-*.safetensors``。视觉塔、图片身份和记录顺序必须完全相同；任何
不一致都在创建目标目录前拒绝。硬链接失败也会直接失败，绝不退化为复制。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from moonvit_glue.training_order import verify_training_order_manifest

from verify_feature_cache import compare_training_order_binding


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _records_sha256(records: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_shard_name(value: Any) -> str:
    name = str(value)
    path = Path(name)
    if not name or path.is_absolute() or path.name != name or ".." in path.parts:
        raise ValueError(f"cache shard path must be a plain relative filename: {name!r}")
    return name


def _validate_source_cache(source: Path) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    manifest_path = source / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"source cache manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cache_format_version") != 1:
        raise ValueError("source cache uses an unsupported cache format")
    records = manifest.get("records")
    shards = manifest.get("shards")
    if not isinstance(records, list) or not isinstance(shards, list):
        raise ValueError("source cache manifest lacks records or shards")
    if manifest.get("records_sha256") != _records_sha256(records):
        raise ValueError("source cache records SHA-256 mismatch")
    if manifest.get("count") is not None and int(manifest["count"]) != len(records):
        raise ValueError("source cache count differs from records")

    shard_by_name: dict[str, dict[str, Any]] = {}
    for row in shards:
        if not isinstance(row, dict):
            raise ValueError("source cache shard entry is not an object")
        name = _safe_shard_name(row.get("path"))
        if name in shard_by_name:
            raise ValueError(f"duplicate source shard: {name}")
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"source cache shard is missing: {path}")
        expected_bytes = int(row.get("bytes", -1))
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"source shard byte count mismatch: {name}")
        actual_sha = sha256(path)
        if actual_sha != str(row.get("sha256")):
            raise ValueError(f"source shard SHA-256 mismatch: {name}")
        shard_by_name[name] = row

    records_by_id: dict[str, dict[str, Any]] = {}
    aliases = 0
    spans: set[tuple[str, int, int]] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("source cache record is not an object")
        record_id = str(record.get("id", ""))
        if not record_id or record_id in records_by_id:
            raise ValueError(f"source cache has duplicate/empty record id: {record_id!r}")
        records_by_id[record_id] = record
        name = _safe_shard_name(record.get("shard"))
        if name not in shard_by_name:
            raise ValueError(f"record points outside source shards: {record_id}")
        start, end = int(record.get("start", -1)), int(record.get("end", -1))
        shape = record.get("feature_shape")
        if start < 0 or end <= start or not isinstance(shape, list) or len(shape) != 3:
            raise ValueError(f"invalid source feature span: {record_id}")
        if end - start != int(shape[0]):
            raise ValueError(f"source feature span/shape mismatch: {record_id}")
        spans.add((name, start, end))
        if "alias_of" in record:
            aliases += 1
            source_record = records_by_id.get(str(record["alias_of"]))
            if source_record is None or "alias_of" in source_record:
                raise ValueError(f"invalid source cache alias: {record_id}")
            for key in (
                "image_sha256", "image_width", "image_height", "feature_shape",
                "dtype", "shard", "start", "end",
            ):
                if record.get(key) != source_record.get(key):
                    raise ValueError(f"source alias differs from canonical span: {record_id}")
    if manifest.get("aliased_records") is not None and aliases != int(manifest["aliased_records"]):
        raise ValueError("source alias count differs from manifest")
    if manifest.get("unique_feature_spans") is not None and len(spans) != int(manifest["unique_feature_spans"]):
        raise ValueError("source unique feature span count differs from manifest")
    return manifest, sha256(manifest_path), list(shards)


def _current_git_sha() -> str | None:
    repo = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def rebind_feature_cache(
    source_cache: str | Path,
    target_training_order: str | Path,
    out: str | Path,
    *,
    current_git_sha: str | None = None,
) -> dict[str, Any]:
    """Create a hard-linked cache with a new, verified training-order binding."""

    source = Path(source_cache).resolve()
    order_path = Path(target_training_order).resolve()
    destination = Path(out).resolve()
    # ``Path.exists`` is false for a dangling symlink; treating it as occupied
    # keeps the no-overwrite contract unambiguous.
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    if destination == source:
        raise ValueError("output cache cannot be the source cache")
    if not order_path.is_file():
        raise FileNotFoundError(f"target training order is missing: {order_path}")
    order = json.loads(order_path.read_text(encoding="utf-8"))
    if not verify_training_order_manifest(order):
        raise ValueError("target training order manifest has an invalid self-hash or schema")

    source_manifest, source_manifest_sha, source_shards = _validate_source_cache(source)
    # This checks exactly id, image SHA, dimensions and list order, while retaining
    # the source cache's canonical/alias spans.
    binding = compare_training_order_binding(source_manifest, order)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.rebind-", dir=destination.parent))
    linked: list[dict[str, Any]] = []
    try:
        for shard in source_shards:
            name = _safe_shard_name(shard["path"])
            source_path = source / name
            target_path = temporary / name
            try:
                os.link(source_path, target_path, follow_symlinks=False)
            except OSError as error:
                raise OSError(
                    f"hard-linking source shard failed; refusing to copy: {name}"
                ) from error
            source_stat = source_path.stat()
            target_stat = target_path.stat()
            samefile = os.path.samefile(source_path, target_path)
            if not samefile or source_stat.st_ino != target_stat.st_ino:
                raise RuntimeError(f"hard-link inode evidence failed: {name}")
            linked.append(
                {
                    "path": name,
                    "source_path": str(source_path),
                    "target_path": str(destination / name),
                    "source_inode": int(source_stat.st_ino),
                    "target_inode": int(target_stat.st_ino),
                    "source_device": int(source_stat.st_dev),
                    "target_device": int(target_stat.st_dev),
                    "samefile": True,
                    "bytes": int(source_stat.st_size),
                    "sha256": str(shard["sha256"]),
                }
            )

        target_manifest = dict(source_manifest)
        target_manifest["training_order_manifest"] = str(order_path)
        target_manifest["training_order_manifest_file_sha256"] = sha256(order_path)
        target_manifest["training_order_manifest_sha256"] = str(order["manifest_sha256"])
        target_manifest["training_order_records_sha256"] = str(order["records_sha256"])
        provenance = {
            "schema_version": "moonvit-cache-rebinding-v1",
            "source_cache_resolved_path": str(source),
            "source_manifest_file_sha256": source_manifest_sha,
            "target_training_order_resolved_path": str(order_path),
            "target_training_order_manifest_file_sha256": target_manifest[
                "training_order_manifest_file_sha256"
            ],
            "target_training_order_manifest_sha256": target_manifest[
                "training_order_manifest_sha256"
            ],
            "target_training_order_records_sha256": target_manifest[
                "training_order_records_sha256"
            ],
            "current_git_sha": current_git_sha or _current_git_sha(),
            "tool_sha256": sha256(Path(__file__).resolve()),
            "linked_shards": linked,
            "records_binding": binding,
        }
        target_manifest["cache_rebinding"] = provenance
        (temporary / "MANIFEST.json").write_text(
            json.dumps(target_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary = {
            "status": "valid",
            "operation": "rebind_feature_cache",
            "source_cache": str(source),
            "target_cache": str(destination),
            "target_training_order_manifest": str(order_path),
            "source_manifest_file_sha256": source_manifest_sha,
            "target_manifest_file_sha256": sha256(temporary / "MANIFEST.json"),
            "target_training_order_manifest_sha256": target_manifest[
                "training_order_manifest_sha256"
            ],
            "target_training_order_records_sha256": target_manifest[
                "training_order_records_sha256"
            ],
            "records_verified": binding["records_matched"],
            "shards_linked": len(linked),
            "all_shards_hard_linked": all(row["samefile"] for row in linked),
            "source_git_sha": source_manifest.get("git_sha"),
            "current_git_sha": provenance["current_git_sha"],
            "tool_sha256": provenance["tool_sha256"],
            "linked_shards": linked,
        }
        (temporary / "SUMMARY.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing output: {destination}")
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    summary["target_cache"] = str(destination)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True, type=Path)
    parser.add_argument("--target-training-order", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--current-git-sha")
    args = parser.parse_args()
    print(
        json.dumps(
            rebind_feature_cache(
                args.source_cache,
                args.target_training_order,
                args.out,
                current_git_sha=args.current_git_sha,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
