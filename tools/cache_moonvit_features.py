"""Build an auditable, sharded cache for a frozen MoonViT tower."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from moonvit_glue import FeatureCacheWriter, load_moonvit_v2_encoder
from tools_common import load_record_image, load_records, record_image_bytes
from training_protocol import records_manifest_sha256


def emit(message: str) -> None:
    """Progress output must never turn a valid cache row into a data failure."""

    try:
        print(message, flush=True)
    except BrokenPipeError:
        pass


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def select_records(
    records: list[dict],
    ids_manifest: Path | None,
    record_slice: str | None,
    limit: int | None,
    shuffle_seed: int | None,
) -> list[dict]:
    if ids_manifest:
        manifest = json.loads(ids_manifest.read_text(encoding="utf-8"))
        entries = manifest.get("records", manifest)
        ids = [str(entry["id"] if isinstance(entry, dict) else entry) for entry in entries]
        by_id = {str(record["id"]): record for record in records}
        missing = [sample_id for sample_id in ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"IDs missing from dataset: {missing[:3]}")
        records = [by_id[sample_id] for sample_id in ids]
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(records)
    if record_slice == "even":
        records = records[::2]
    elif record_slice == "odd":
        records = records[1::2]
    if limit is not None:
        records = records[:limit]
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--moonvit-v2-weights", required=True, type=Path)
    parser.add_argument("--moonvit-v2-attn", default="eager",
                        choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-image-side", type=int, default=448)
    parser.add_argument("--ids-manifest", type=Path, default=None)
    parser.add_argument("--record-slice", choices=["even", "odd"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="Shuffle before --limit (matches train_overfit selection)")
    parser.add_argument("--shard-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite feature cache: {args.out}")
    args.out.mkdir(parents=True)
    started = time.time()
    records = select_records(
        load_records(args.data),
        args.ids_manifest,
        args.record_slice,
        args.limit,
        args.shuffle_seed,
    )
    if not records:
        raise ValueError("feature cache selection is empty")
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    weights_sha256 = sha256_file(args.moonvit_v2_weights)
    tower = load_moonvit_v2_encoder(
        args.moonvit_v2_weights,
        attn_implementation=args.moonvit_v2_attn,
        torch_dtype=dtype,
    ).to(device)
    config_json = json.dumps(
        tower.model.config.to_dict(), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    metadata = {
        "cache_format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "data_path": str(args.data.resolve()),
        "data_records_manifest_sha256": records_manifest_sha256(records),
        "ids_manifest": str(args.ids_manifest.resolve()) if args.ids_manifest else None,
        "record_slice": args.record_slice,
        "shuffle_seed": args.shuffle_seed,
        "max_image_side": args.max_image_side,
        "moonvit_architecture": type(tower.model).__name__,
        "moonvit_config_sha256": hashlib.sha256(config_json).hexdigest(),
        "moonvit_weights_path": str(args.moonvit_v2_weights.resolve()),
        "moonvit_weights_sha256": weights_sha256,
        "moonvit_attention": args.moonvit_v2_attn,
        "vision_width": tower.vision_width,
        "merge_factor": tower.merge_factor,
        "feature_storage_device": "cpu",
    }
    writer = FeatureCacheWriter(
        args.out, cache_metadata=metadata, shard_size=args.shard_size
    )
    raw_path = args.out / "cache_records.jsonl"
    failures_path = args.out / "failures.jsonl"
    failure_count = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with raw_path.open("w", encoding="utf-8") as raw_stream, failures_path.open(
        "w", encoding="utf-8"
    ) as failure_stream:
        for index, record in enumerate(records):
            sample_started = time.time()
            try:
                payload = record_image_bytes(record, args.data.parent)
                image = load_record_image(record, args.data.parent)
                original_size = image.size
                if args.max_image_side:
                    from PIL import Image

                    image.thumbnail(
                        (args.max_image_side, args.max_image_side), Image.Resampling.LANCZOS
                    )
                inputs = tower.preprocess(image)
                features = tower(**inputs)
                if len(features) != 1:
                    raise ValueError(f"expected one feature group, got {len(features)}")
                writer.add(
                    sample_id=str(record["id"]),
                    feature=features[0],
                    image_sha256=hashlib.sha256(payload).hexdigest(),
                    image_size=original_size,
                )
                row = {
                    "index": index,
                    "id": str(record["id"]),
                    "status": "ok",
                    "feature_shape": list(features[0].shape),
                    "dtype": str(features[0].dtype).removeprefix("torch."),
                    "wall_seconds": time.time() - sample_started,
                }
                raw_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                raw_stream.flush()
                emit(f"[{index + 1}/{len(records)}] {row['id']} {row['feature_shape']}")
            except Exception as exc:
                failure_count += 1
                failure = {
                    "index": index,
                    "id": str(record.get("id", index)),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failure_stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
                failure_stream.flush()
                emit(f"[failed] {failure['id']}: {exc}")
    manifest = writer.close()
    summary = {
        "requested": len(records),
        "cached": manifest["count"],
        "failed": failure_count,
        "wall_seconds": time.time() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "manifest": "MANIFEST.json",
        "records": raw_path.name,
        "failures": failures_path.name,
    }
    (args.out / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    emit(json.dumps(summary, ensure_ascii=False, indent=2))
    if failure_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
