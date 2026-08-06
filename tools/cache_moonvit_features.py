"""Build an auditable, sharded cache for a frozen MoonViT tower."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

import moonvit_glue.feature_cache as feature_cache_module
import moonvit_glue.training_order as training_order_module
from moonvit_glue import FeatureCacheWriter, MoonViTEncoder, load_moonvit_v2_encoder
from moonvit_glue.training_order import (
    load_ordered_records,
    verify_training_order_manifest,
)
from tools_common import load_records, record_image_bytes
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


def resolve_v1_snapshot(model_id: str, revision: str | None) -> tuple[Path, dict[str, str]]:
    """Resolve and hash the pinned V1 snapshot before loading remote code.

    A cache used for an architecture comparison must carry a weight identity,
    even when the model was loaded through ``AutoModel``.  The snapshot itself
    is never copied into the repository; only its deterministic file hashes are
    written to the cache manifest.
    """

    candidate = Path(model_id)
    if candidate.is_dir():
        snapshot = candidate.resolve()
    else:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                model_id,
                revision=revision,
                local_files_only=bool(os.environ.get("HF_HUB_OFFLINE")),
            )
        ).resolve()
    files = {}
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file() or path.name.endswith(".lock"):
            continue
        relative = path.relative_to(snapshot).as_posix()
        files[relative] = sha256_file(path)
    weight_files = [
        name for name in files
        if name.endswith(".safetensors") or name.endswith(".bin")
    ]
    if not weight_files:
        raise FileNotFoundError(f"no V1 weight file in snapshot: {snapshot}")
    digest = hashlib.sha256()
    for name in weight_files:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(files[name].encode("ascii"))
        digest.update(b"\n")
    files["__weights_manifest__"] = digest.hexdigest()
    return snapshot, files


def git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def git_tracked_worktree_clean() -> bool:
    """只检查 tracked/index 修改；实验输出等 untracked 文件不影响代码身份。"""

    unstaged = subprocess.run(["git", "diff", "--quiet", "--"], check=False)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet", "--"], check=False)
    return unstaged.returncode == 0 and staged.returncode == 0


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


def resolve_cache_selection(
    *,
    data_path: Path,
    training_order_manifest: Path | None,
    ids_manifest: Path | None,
    record_slice: str | None,
    limit: int | None,
    shuffle_seed: int | None,
) -> tuple[list[dict], dict | None]:
    """选择普通 cache 记录，或严格复现已冻结的训练顺序。"""

    if training_order_manifest is not None:
        if any(
            value is not None
            for value in (ids_manifest, record_slice, limit, shuffle_seed)
        ):
            raise ValueError(
                "--training-order-manifest cannot be combined with IDs, slice, limit, or shuffle"
            )
        manifest = json.loads(training_order_manifest.read_text(encoding="utf-8"))
        if not verify_training_order_manifest(manifest):
            raise ValueError("training order manifest is invalid")
        return load_ordered_records(data_path=data_path, manifest=manifest), manifest

    records = select_records(
        load_records(data_path),
        ids_manifest,
        record_slice,
        limit,
        shuffle_seed,
    )
    return records, None


def validate_training_order_cache_contract(
    manifest: dict,
    *,
    max_image_side: int,
    moonvit_weights_sha256: str,
) -> None:
    """阻止 cache 分辨率或视觉塔身份偏离冻结训练合同。"""

    expected = manifest["feature_cache"]
    if int(expected["max_image_side"]) != int(max_image_side):
        raise ValueError("max image side differs from the training order manifest")
    if str(expected["moonvit_weights_sha256"]) != str(moonvit_weights_sha256):
        raise ValueError("MoonViT SHA-256 differs from the training order manifest")


def validate_training_order_image(
    entry: dict,
    *,
    record_id: str,
    payload: bytes,
    image_size: tuple[int, int],
) -> None:
    """在 tower forward 前重验一条训练记录的原图身份。"""

    if str(entry["id"]) != str(record_id):
        raise ValueError(f"training order record ID differs: {record_id}")
    if len(payload) != int(entry["image_bytes"]):
        raise ValueError(f"training image byte count differs: {record_id}")
    if hashlib.sha256(payload).hexdigest() != str(entry["image_sha256"]):
        raise ValueError(f"training image SHA-256 differs: {record_id}")
    if tuple(int(value) for value in image_size) != (
        int(entry["image_width"]),
        int(entry["image_height"]),
    ):
        raise ValueError(f"training image dimensions differ: {record_id}")


def validate_training_order_feature_shape(
    manifest: dict, *, record_id: str, feature_shape: tuple[int, ...]
) -> None:
    """确保 resize 后的视觉 token 数没有越过训练预算。"""

    if len(feature_shape) != 3:
        raise ValueError(f"unexpected MoonViT feature rank: {record_id}")
    maximum = int(manifest["feature_cache"]["max_visual_tokens"])
    if int(feature_shape[0]) > maximum:
        raise ValueError(f"visual token count exceeds training contract: {record_id}")


def binding_manifest_metadata(path: Path | None) -> dict[str, str | None]:
    """记录通用评测 manifest 的文件身份与自声明身份。"""

    if path is None:
        return {
            "binding_manifest": None,
            "binding_manifest_file_sha256": None,
            "binding_manifest_sha256": None,
            "binding_manifest_name": None,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("manifest_sha256"), str) or not isinstance(
        payload.get("name"), str
    ):
        raise ValueError("binding manifest requires string name and manifest_sha256")
    return {
        "binding_manifest": str(path.resolve()),
        "binding_manifest_file_sha256": sha256_file(path),
        "binding_manifest_sha256": str(payload["manifest_sha256"]),
        "binding_manifest_name": str(payload["name"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--vision-tower",
        choices=["v1", "v2"],
        default="v2",
        help=(
            "v1 loads the standalone MoonViT-SO-400M tower used by the "
            "community GLM-5.2V path; v2 loads the vendored Kimi-K3 tower."
        ),
    )
    parser.add_argument(
        "--moonvit-model",
        default="moonshotai/MoonViT-SO-400M",
        help="HF model ID or local snapshot for --vision-tower v1",
    )
    parser.add_argument("--moonvit-revision", default=None)
    parser.add_argument(
        "--moonvit-snapshot",
        type=Path,
        default=None,
        help="Optional already-resolved V1 snapshot; avoids re-resolving a HF repo",
    )
    parser.add_argument(
        "--moonvit-runtime-snapshot",
        type=Path,
        default=None,
        help=(
            "Optional real-file V1 snapshot used only for Transformers loading; "
            "the pinned --moonvit-model/revision remain the cache identity"
        ),
    )
    parser.add_argument(
        "--moonvit-v2-weights",
        type=Path,
        default=None,
        help="Extracted MoonViT-V2 safetensors (required with --vision-tower v2)",
    )
    parser.add_argument("--moonvit-v2-attn", default="eager",
                        choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-image-side", type=int, default=448)
    parser.add_argument("--training-order-manifest", type=Path, default=None)
    parser.add_argument("--binding-manifest", type=Path, default=None,
                        help="Bind a general evaluation cache to a frozen manifest")
    parser.add_argument("--ids-manifest", type=Path, default=None)
    parser.add_argument("--record-slice", choices=["even", "odd"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="Shuffle before --limit (matches train_overfit selection)")
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


def load_tower(args: argparse.Namespace, *, dtype: torch.dtype):
    """Load either frozen tower while keeping one cache format.

    The V1 and V2 encoders deliberately expose the same ``preprocess`` and
    ``[tokens, merge, width]`` interface.  Keeping the branch here prevents
    downstream trainers from silently changing their cache reader or image
    order when comparing the community V1 structure with our V2 path.
    """

    if args.vision_tower == "v2":
        if args.moonvit_v2_weights is None:
            raise ValueError("--vision-tower v2 requires --moonvit-v2-weights")
        return load_moonvit_v2_encoder(
            args.moonvit_v2_weights,
            attn_implementation=args.moonvit_v2_attn,
            torch_dtype=dtype,
        )
    # ``snapshot_download`` 的 snapshot 目录含有指向 blobs 的符号链接；
    # Transformers dynamic-module loader 在直接接收这个本地路径时会把相对
    # import 解析到 blob 哈希目录，导致 configuration_moonvit.py 丢失。正式
    # 缓存可传入一个由 ``cp -L`` 物化的 real-file snapshot，模型 ID/revision
    # 仍写入 manifest 作为社区身份，权重 snapshot 仍独立哈希。
    model_id = str(args.moonvit_runtime_snapshot or args.moonvit_model)
    revision = None if args.moonvit_runtime_snapshot is not None else args.moonvit_revision
    return MoonViTEncoder.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
    )


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite feature cache: {args.out}")
    if args.require_clean_git and not git_tracked_worktree_clean():
        raise RuntimeError("tracked Git worktree is dirty; cache provenance would be false")
    started = time.time()
    records, training_order = resolve_cache_selection(
        data_path=args.data,
        training_order_manifest=args.training_order_manifest,
        ids_manifest=args.ids_manifest,
        record_slice=args.record_slice,
        limit=args.limit,
        shuffle_seed=args.shuffle_seed,
    )
    if not records:
        raise ValueError("feature cache selection is empty")
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    if args.vision_tower == "v2":
        if args.moonvit_v2_weights is None:
            raise ValueError("--vision-tower v2 requires --moonvit-v2-weights")
        weights_path = args.moonvit_v2_weights
    else:
        # Resolve the snapshot explicitly so a formal V1 cache carries a
        # reproducible weight identity instead of the old null placeholder.
        snapshot_path, snapshot_files = resolve_v1_snapshot(
            str(args.moonvit_snapshot or args.moonvit_model), args.moonvit_revision
        )
        weights_path = None
        if args.moonvit_snapshot is None:
            args.moonvit_snapshot = snapshot_path
        weights_sha256 = snapshot_files["__weights_manifest__"]
    if args.vision_tower == "v2":
        snapshot_files = {}
        weights_sha256 = sha256_file(weights_path)
    if training_order is not None:
        validate_training_order_cache_contract(
            training_order,
            max_image_side=args.max_image_side,
            moonvit_weights_sha256=weights_sha256,
        )
    args.out.mkdir(parents=True)
    tower = load_tower(args, dtype=dtype).to(device)
    config_json = json.dumps(
        tower.model.config.to_dict(), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    runtime_source_paths = (
        Path(__file__).resolve(),
        Path(feature_cache_module.__file__).resolve(),
        Path(training_order_module.__file__).resolve(),
    )
    metadata = {
        "cache_format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "git_tracked_worktree_clean": git_tracked_worktree_clean(),
        "runtime_source_files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in runtime_source_paths
        ],
        "data_path": str(args.data.resolve()),
        "data_records_manifest_sha256": records_manifest_sha256(records),
        "ids_manifest": str(args.ids_manifest.resolve()) if args.ids_manifest else None,
        "training_order_manifest": (
            str(args.training_order_manifest.resolve())
            if args.training_order_manifest
            else None
        ),
        "training_order_manifest_file_sha256": (
            sha256_file(args.training_order_manifest)
            if args.training_order_manifest
            else None
        ),
        "training_order_manifest_sha256": (
            training_order["manifest_sha256"] if training_order else None
        ),
        "training_order_records_sha256": (
            training_order["records_sha256"] if training_order else None
        ),
        **binding_manifest_metadata(args.binding_manifest),
        "record_slice": args.record_slice,
        "shuffle_seed": args.shuffle_seed,
        "max_image_side": args.max_image_side,
        "max_visual_tokens": (
            int(training_order["feature_cache"]["max_visual_tokens"])
            if training_order
            else None
        ),
        "vision_tower": args.vision_tower,
        "moonvit_model": args.moonvit_model if args.vision_tower == "v1" else None,
        "moonvit_revision": args.moonvit_revision if args.vision_tower == "v1" else None,
        "moonvit_snapshot": str(args.moonvit_snapshot.resolve())
        if args.moonvit_snapshot is not None
        else None,
        "moonvit_runtime_snapshot": (
            str(args.moonvit_runtime_snapshot.resolve())
            if args.moonvit_runtime_snapshot is not None
            else None
        ),
        "moonvit_snapshot_files": snapshot_files if args.vision_tower == "v1" else None,
        "moonvit_architecture": type(tower.model).__name__,
        "moonvit_config_sha256": hashlib.sha256(config_json).hexdigest(),
        "moonvit_weights_path": str(weights_path.resolve()) if weights_path else None,
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
    tower_forward_count = 0
    reused_by_image_sha256 = 0
    content_sources: dict[str, str] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with raw_path.open("w", encoding="utf-8") as raw_stream, failures_path.open(
        "w", encoding="utf-8"
    ) as failure_stream:
        for index, record in enumerate(records):
            sample_started = time.time()
            try:
                payload = record_image_bytes(record, args.data.parent)
                with Image.open(io.BytesIO(payload)) as encoded_image:
                    image = encoded_image.convert("RGB")
                original_size = tuple(int(value) for value in image.size)
                image_sha256 = hashlib.sha256(payload).hexdigest()
                if training_order is not None:
                    validate_training_order_image(
                        training_order["records"][index],
                        record_id=str(record["id"]),
                        payload=payload,
                        image_size=original_size,
                    )
                source_id = content_sources.get(image_sha256)
                if source_id is not None:
                    entry = writer.add_alias(
                        sample_id=str(record["id"]),
                        source_sample_id=source_id,
                        image_sha256=image_sha256,
                        image_size=original_size,
                    )
                    reused_by_image_sha256 += 1
                    row = {
                        "index": index,
                        "id": str(record["id"]),
                        "status": "ok",
                        "feature_shape": list(entry["feature_shape"]),
                        "dtype": str(entry["dtype"]),
                        "feature_reused": True,
                        "alias_of": source_id,
                        "wall_seconds": time.time() - sample_started,
                    }
                    raw_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    raw_stream.flush()
                    emit(
                        f"[{index + 1}/{len(records)}] {row['id']} "
                        f"{row['feature_shape']} reused={source_id}"
                    )
                    continue
                if args.max_image_side:
                    image.thumbnail(
                        (args.max_image_side, args.max_image_side), Image.Resampling.LANCZOS
                    )
                inputs = tower.preprocess(image)
                features = tower(**inputs)
                if len(features) != 1:
                    raise ValueError(f"expected one feature group, got {len(features)}")
                if training_order is not None:
                    validate_training_order_feature_shape(
                        training_order,
                        record_id=str(record["id"]),
                        feature_shape=tuple(int(value) for value in features[0].shape),
                    )
                writer.add(
                    sample_id=str(record["id"]),
                    feature=features[0],
                    image_sha256=image_sha256,
                    image_size=original_size,
                )
                content_sources[image_sha256] = str(record["id"])
                tower_forward_count += 1
                row = {
                    "index": index,
                    "id": str(record["id"]),
                    "status": "ok",
                    "feature_shape": list(features[0].shape),
                    "dtype": str(features[0].dtype).removeprefix("torch."),
                    "feature_reused": False,
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
        "tower_forwards": tower_forward_count,
        "reused_by_image_sha256": reused_by_image_sha256,
        "unique_image_sha256": len(content_sources),
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
