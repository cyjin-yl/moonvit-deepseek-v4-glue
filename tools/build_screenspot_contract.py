"""冻结 ScreenSpot 完整公共测试集与 50 条 GLM-format 对齐子集。

该工具只接受预注册的 ``bevaya/ScreenSpot`` revision 和三份 parquet。
大型图像继续保存在本地 parquet；Git manifest 保存每张图的原始字节 SHA-256、
source row、bbox、类别与确定性 shuffled-image derangement。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from moonvit_glue.screenspot_contract import (
    canonical_platform,
    canonical_target_type,
    deterministic_image_derangement,
    normalize_screenspot_bbox,
    seal_manifest,
    stable_stratified_subset,
    verify_manifest,
)

DATASET_REPO = "bevaya/ScreenSpot"
DATASET_REVISION = "0be08781e2e188582f6131625ae1598d443b4d5d"
SELECTION_SEED = "20260805"
BBOX_SOURCE_FORMAT = "fractional_xyxy"
EXPECTED_TOTAL = 1_272
EXPECTED_SHARDS = (
    {
        "path": "data/test-00000-of-00003.parquet",
        "bytes": 134_512_659,
        "sha256": "ff06d312270eecc9d9ac968a51ceb0bc54f80cfda691ccccb8ebf4b4e5faa8fb",
    },
    {
        "path": "data/test-00001-of-00003.parquet",
        "bytes": 198_971_508,
        "sha256": "d48b8275f9dcff56f9c1bfedafbeb018cc200cbb649f8a92299cba0a9130891a",
    },
    {
        "path": "data/test-00002-of-00003.parquet",
        "bytes": 268_832_649,
        "sha256": "a28a1e9f027003730c1375550bfdd3b212d0c0b5c24b338140309a2af3674e5b",
    },
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_shards(source_dir: Path) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for expected in EXPECTED_SHARDS:
        path = source_dir / expected["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(
                f"source shard mismatch for {path}: bytes={size}, sha256={digest}"
            )
        verified.append(dict(expected))
    return verified


def _image_record(value: dict[str, Any]) -> dict[str, Any]:
    encoded = value.get("bytes")
    if not isinstance(encoded, bytes):
        raise ValueError("ScreenSpot parquet image must contain encoded bytes")
    with Image.open(io.BytesIO(encoded)) as image:
        width, height = image.size
        image_format = image.format
        mode = image.mode
    return {
        "image_sha256": hashlib.sha256(encoded).hexdigest(),
        "image_bytes": len(encoded),
        "image_width": width,
        "image_height": height,
        "image_format": image_format,
        "image_mode": mode,
        "image_source_path": value.get("path"),
    }


def load_records(source_dir: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required to read the frozen parquet shards") from error

    records: list[dict[str, Any]] = []
    global_index = 0
    columns = ["file_name", "bbox", "instruction", "data_type", "data_source", "image"]
    for shard in EXPECTED_SHARDS:
        relative_path = shard["path"]
        parquet = pq.ParquetFile(source_dir / relative_path)
        shard_row_index = 0
        for batch in parquet.iter_batches(batch_size=16, columns=columns):
            for raw in batch.to_pylist():
                image = _image_record(raw["image"])
                platform = canonical_platform(raw["data_source"])
                target_type = canonical_target_type(raw["data_type"])
                sample_id = f"screenspot-{DATASET_REVISION[:8]}-{global_index:04d}"
                bbox = normalize_screenspot_bbox(
                    raw["bbox"],
                    width=image["image_width"],
                    height=image["image_height"],
                    source_format=BBOX_SOURCE_FORMAT,
                )
                records.append(
                    {
                        "source_global_index": global_index,
                        "sample_id": sample_id,
                        "source_row_id": f"{relative_path}:{shard_row_index:04d}",
                        "source_parquet": relative_path,
                        "source_row_index": shard_row_index,
                        "file_name": raw["file_name"],
                        "instruction": raw["instruction"],
                        "bbox_source": [float(value) for value in raw["bbox"]],
                        "bbox_source_format": BBOX_SOURCE_FORMAT,
                        "bbox_999_xyxy": bbox,
                        "data_source_raw": raw["data_source"],
                        "platform": platform,
                        "data_type_raw": raw["data_type"],
                        "target_type": target_type,
                        **image,
                    }
                )
                global_index += 1
                shard_row_index += 1
    if len(records) != EXPECTED_TOTAL:
        raise ValueError(f"expected {EXPECTED_TOTAL} rows, found {len(records)}")
    return records


def _strata_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        f"{record['platform']}::{record['target_type']}" for record in records
    )
    return dict(sorted(counts.items()))


def _platform_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(record["platform"] for record in records).items()))


def _type_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(record["target_type"] for record in records).items()))


def make_manifest(
    *,
    name: str,
    label: str,
    records: list[dict[str, Any]],
    shards: list[dict[str, Any]],
    selection: dict[str, Any],
    frozen_on: str,
) -> dict[str, Any]:
    ordered_records = [
        {**record, "evaluation_order": index}
        for index, record in enumerate(records)
    ]
    derangement = deterministic_image_derangement(
        ordered_records, seed=SELECTION_SEED
    )
    manifest = {
        "schema_version": "screenspot-community-contract-v1",
        "name": name,
        "label": label,
        "frozen_on": frozen_on,
        "immutability_rule": "once committed, sample membership and order must not change",
        "dataset": {
            "repo": DATASET_REPO,
            "resolved_revision": DATASET_REVISION,
            "split": "test",
            "config": "default",
            "upstream_repository": "njucckevin/SeeClick",
            "source_bbox_format": BBOX_SOURCE_FORMAT,
            "source_shards": shards,
            "public_test_total_count": EXPECTED_TOTAL,
        },
        "selection": selection,
        "counts": {
            "total": len(ordered_records),
            "by_platform": _platform_counts(ordered_records),
            "by_target_type": _type_counts(ordered_records),
            "by_platform_and_target_type": _strata_counts(ordered_records),
        },
        "coordinate_contract": {
            "prediction_scale": [0, 999],
            "origin": "top-left",
            "target_bbox_format": "xyxy",
            "canonical_output": "click(start_box=[x, y])",
        },
        "shuffled_image_control": {
            "method": "stable hash order followed by the first image-SHA-safe cyclic offset",
            "seed": SELECTION_SEED,
            "mapping": [
                {
                    "sample_id": record["sample_id"],
                    "shuffled_image_sample_id": derangement[record["sample_id"]],
                }
                for record in ordered_records
            ],
        },
        "samples": ordered_records,
    }
    return seal_manifest(manifest)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reread = json.loads(path.read_text(encoding="utf-8"))
    if not verify_manifest(reread):
        raise RuntimeError(f"written manifest failed self-verification: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="directory containing data/test-0000*-of-00003.parquet",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frozen-on", default="2026-08-05")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    shards = verify_source_shards(source_dir)
    records = load_records(source_dir)

    full = make_manifest(
        name="screenspot_public_test_v1",
        label="complete public ScreenSpot test set",
        records=records,
        shards=shards,
        selection={
            "kind": "complete_public_test",
            "method": "all 1,272 public test rows in frozen parquet order",
            "seed": None,
        },
        frozen_on=args.frozen_on,
    )
    subset = stable_stratified_subset(records, size=50, seed=SELECTION_SEED)
    glm50 = make_manifest(
        name="screenspot_glm50_v1",
        label="GLM-format metric-aligned public subset",
        records=subset,
        shards=shards,
        selection={
            "kind": "metric_aligned_public_subset",
            "method": "stable SHA-256 rank, balanced over platform × text/icon-widget",
            "seed": SELECTION_SEED,
            "requested_count": 50,
            "disclaimer": "This is not the community private 50-sample validation set.",
        },
        frozen_on=args.frozen_on,
    )

    full_path = args.out_dir / "screenspot_public_test_v1" / "MANIFEST.json"
    subset_path = args.out_dir / "screenspot_glm50_v1" / "MANIFEST.json"
    write_manifest(full_path, full)
    write_manifest(subset_path, glm50)
    print(
        json.dumps(
            {
                "full": {
                    "path": str(full_path),
                    "count": len(records),
                    "manifest_sha256": full["manifest_sha256"],
                },
                "glm50": {
                    "path": str(subset_path),
                    "count": len(subset),
                    "manifest_sha256": glm50["manifest_sha256"],
                    "strata": _strata_counts(subset),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
