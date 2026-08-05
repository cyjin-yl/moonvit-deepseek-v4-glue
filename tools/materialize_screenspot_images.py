#!/usr/bin/env python3
"""从冻结 ScreenSpot parquet 提取 manifest 指定图像，供统一 cache runner 使用。"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image

from moonvit_glue.screenspot_contract import verify_manifest


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def git_tracked_worktree_clean() -> bool:
    unstaged = subprocess.run(["git", "diff", "--quiet", "--"], check=False)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--"], check=False
    )
    return unstaged.returncode == 0 and staged.returncode == 0


def materialized_record(sample: dict[str, Any], image_path: str) -> dict[str, Any]:
    """生成 cache/eval 共用 JSONL 行，保留正式评分所需身份。"""

    return {
        "id": str(sample["sample_id"]),
        "image": str(image_path),
        "question": str(sample["instruction"]),
        "gt_box": [float(value) for value in sample["bbox_999_xyxy"]],
        "platform": str(sample["platform"]),
        "target_type": str(sample["target_type"]),
        "image_sha256": str(sample["image_sha256"]),
        "image_width": int(sample["image_width"]),
        "image_height": int(sample["image_height"]),
        "source_parquet": str(sample["source_parquet"]),
        "source_row_index": int(sample["source_row_index"]),
    }


def verify_source_shards(source_dir: Path, manifest: dict[str, Any]) -> list[dict]:
    verified = []
    for row in manifest["dataset"]["source_shards"]:
        path = source_dir / str(row["path"])
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"ScreenSpot source shard byte count differs: {path}")
        digest = sha256_file(path)
        if digest != str(row["sha256"]):
            raise ValueError(f"ScreenSpot source shard SHA-256 differs: {path}")
        verified.append(
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}
        )
    return verified


def extract_images(
    source_dir: Path, out: Path, samples: list[dict[str, Any]]
) -> dict[str, str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required to materialize ScreenSpot") from error

    targets: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for sample in samples:
        targets.setdefault(str(sample["source_parquet"]), {}).setdefault(
            int(sample["source_row_index"]), []
        ).append(sample)
    image_dir = out / "images"
    image_dir.mkdir(parents=True)
    relative_by_id: dict[str, str] = {}
    written_sha: set[str] = set()
    for relative_parquet, rows_by_index in sorted(targets.items()):
        parquet = pq.ParquetFile(source_dir / relative_parquet)
        offset = 0
        pending = set(rows_by_index)
        for batch in parquet.iter_batches(batch_size=32, columns=["image"]):
            upper = offset + batch.num_rows
            selected = sorted(index for index in pending if offset <= index < upper)
            for row_index in selected:
                raw = batch.slice(row_index - offset, 1).to_pylist()[0]
                encoded = raw["image"]["bytes"]
                if not isinstance(encoded, bytes):
                    raise ValueError("ScreenSpot parquet image bytes are absent")
                digest = hashlib.sha256(encoded).hexdigest()
                with Image.open(io.BytesIO(encoded)) as image:
                    image_size = tuple(int(value) for value in image.size)
                for sample in rows_by_index[row_index]:
                    if len(encoded) != int(sample["image_bytes"]):
                        raise ValueError(f"ScreenSpot image bytes differ: {sample['sample_id']}")
                    if digest != str(sample["image_sha256"]):
                        raise ValueError(f"ScreenSpot image SHA-256 differs: {sample['sample_id']}")
                    if image_size != (
                        int(sample["image_width"]),
                        int(sample["image_height"]),
                    ):
                        raise ValueError(
                            f"ScreenSpot image dimensions differ: {sample['sample_id']}"
                        )
                    relative = f"images/{digest}.bin"
                    if digest not in written_sha:
                        (out / relative).write_bytes(encoded)
                        written_sha.add(digest)
                    relative_by_id[str(sample["sample_id"])] = relative
                pending.remove(row_index)
            offset = upper
            if not pending:
                break
        if pending:
            raise ValueError(
                f"ScreenSpot source rows are absent from {relative_parquet}: {sorted(pending)}"
            )
    return relative_by_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite ScreenSpot materialization: {args.out}")
    if args.require_clean_git and not git_tracked_worktree_clean():
        raise RuntimeError("tracked Git worktree is dirty; materialization is refused")
    started = time.perf_counter()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not verify_manifest(manifest):
        raise ValueError("ScreenSpot manifest self-hash verification failed")
    args.out.mkdir(parents=True)
    source_shards = verify_source_shards(args.source_dir, manifest)
    samples = list(manifest["samples"])
    relative_by_id = extract_images(args.source_dir, args.out, samples)
    records = [
        materialized_record(sample, relative_by_id[str(sample["sample_id"])])
        for sample in samples
    ]
    data_path = args.out / "screenspot.jsonl"
    with data_path.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    image_paths = sorted((args.out / "images").iterdir())
    summary = {
        "format_version": "screenspot-materialization-v1",
        "git_sha": git_sha(),
        "git_tracked_worktree_clean": git_tracked_worktree_clean(),
        "dataset_name": manifest["name"],
        "dataset_manifest": str(args.manifest.resolve()),
        "dataset_manifest_file_sha256": sha256_file(args.manifest),
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "source_shards": source_shards,
        "records": len(records),
        "records_sha256": canonical_sha256(records),
        "data_file": data_path.name,
        "data_file_bytes": data_path.stat().st_size,
        "data_file_sha256": sha256_file(data_path),
        "unique_images": len(image_paths),
        "image_bytes": sum(path.stat().st_size for path in image_paths),
        "wall_seconds": time.perf_counter() - started,
        "paid_resources_used": False,
    }
    (args.out / "MATERIALIZATION.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
