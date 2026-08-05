"""ScreenSpot 运行时图像映射与特征 cache 绑定校验。"""

from __future__ import annotations

from typing import Any, Mapping


def shuffled_image_mapping(manifest: Mapping[str, Any]) -> dict[str, str]:
    """返回预注册 derangement，并拒绝自映射、重复目标或同图替换。"""

    samples = list(manifest.get("samples", []))
    sample_by_id = {str(row["sample_id"]): row for row in samples}
    if len(sample_by_id) != len(samples):
        raise ValueError("ScreenSpot manifest contains duplicate sample IDs")
    rows = list(manifest.get("shuffled_image_control", {}).get("mapping", []))
    mapping = {
        str(row["sample_id"]): str(row["shuffled_image_sample_id"])
        for row in rows
    }
    expected_ids = [str(row["sample_id"]) for row in samples]
    if len(mapping) != len(rows) or set(mapping) != set(expected_ids):
        raise ValueError("ScreenSpot shuffled mapping does not cover every sample once")
    targets = list(mapping.values())
    if len(set(targets)) != len(targets) or set(targets) != set(expected_ids):
        raise ValueError("ScreenSpot shuffled mapping is not a derangement permutation")
    for sample_id, shuffled_id in mapping.items():
        if sample_id == shuffled_id:
            raise ValueError("ScreenSpot shuffled mapping contains a fixed point")
        if (
            str(sample_by_id[sample_id]["image_sha256"])
            == str(sample_by_id[shuffled_id]["image_sha256"])
        ):
            raise ValueError("ScreenSpot shuffled mapping reuses the same image content")
    return mapping


def validate_screenspot_feature_cache(
    dataset_manifest: Mapping[str, Any],
    cache_manifest: Mapping[str, Any],
    *,
    dataset_manifest_file_sha256: str,
    max_image_side: int,
    max_visual_tokens: int,
    moonvit_weights_sha256: str,
) -> dict[str, int]:
    """把 cache 的顺序、图像身份和形状锁到固定 ScreenSpot manifest。"""

    samples = list(dataset_manifest.get("samples", []))
    records = list(cache_manifest.get("records", []))
    if int(cache_manifest.get("count", -1)) != len(samples) or len(records) != len(
        samples
    ):
        raise ValueError("ScreenSpot feature cache count differs from its manifest")
    expected_binding = {
        "binding_manifest_file_sha256": str(dataset_manifest_file_sha256),
        "binding_manifest_sha256": str(dataset_manifest["manifest_sha256"]),
        "binding_manifest_name": str(dataset_manifest["name"]),
        "max_image_side": int(max_image_side),
        "moonvit_weights_sha256": str(moonvit_weights_sha256),
        "vision_width": 1024,
        "merge_factor": 4,
    }
    for key, expected in expected_binding.items():
        if cache_manifest.get(key) != expected:
            raise ValueError(f"ScreenSpot feature cache binding differs: {key}")

    maximum_tokens = 0
    aliases = 0
    for index, (sample, record) in enumerate(zip(samples, records, strict=True)):
        expected = {
            "id": str(sample["sample_id"]),
            "image_sha256": str(sample["image_sha256"]),
            "image_width": int(sample["image_width"]),
            "image_height": int(sample["image_height"]),
            "dtype": "float32",
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(
                    f"ScreenSpot feature cache row {index} differs: {key}"
                )
        shape = [int(value) for value in record.get("feature_shape", [])]
        if len(shape) != 3 or shape[1:] != [4, 1024] or shape[0] <= 0:
            raise ValueError(f"ScreenSpot feature cache shape differs at row {index}")
        if shape[0] > int(max_visual_tokens):
            raise ValueError(
                f"ScreenSpot feature cache exceeds visual-token budget at row {index}"
            )
        maximum_tokens = max(maximum_tokens, shape[0])
        aliases += int("alias_of" in record)

    if int(cache_manifest.get("aliased_records", -1)) != aliases:
        raise ValueError("ScreenSpot feature cache alias count differs")
    if int(cache_manifest.get("unique_feature_spans", -1)) != len(samples) - aliases:
        raise ValueError("ScreenSpot feature cache unique-span count differs")
    shuffled_image_mapping(dataset_manifest)
    return {
        "records": len(samples),
        "aliased_records": aliases,
        "unique_feature_spans": len(samples) - aliases,
        "maximum_visual_tokens": maximum_tokens,
    }
