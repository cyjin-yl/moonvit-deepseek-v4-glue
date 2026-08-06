"""ScreenSpot 运行时图像映射与特征 cache 绑定校验。"""

from __future__ import annotations

from typing import Any, Mapping


def validate_feature_cache_interface(
    cache_manifest: Mapping[str, Any],
    *,
    vision_width: int,
    merge_factor: int,
    max_image_side: int | None = None,
    max_visual_tokens: int | None = None,
    vision_tower: str | None = None,
    moonvit_model: str | None = None,
    moonvit_revision: str | None = None,
    moonvit_weights_sha256: str | None = None,
    require_tower_identity: bool = False,
) -> dict[str, Any]:
    """Validate the architecture-facing portion of a feature-cache manifest.

    V1 and V2 intentionally share the cache transport format while producing
    different trailing feature widths.  Older V2 manifests predate explicit
    tower identity fields, so identity checks remain opt-in; shape and
    resolution checks are always enforced.  Architecture-control contracts
    should set ``require_tower_identity=True`` and provide a pinned revision or
    weight hash.
    """

    width = int(vision_width)
    merge = int(merge_factor)
    if width <= 0 or merge <= 0:
        raise ValueError("vision width and merge factor must be positive")

    expected_metadata: dict[str, Any] = {
        "vision_width": width,
        "merge_factor": merge,
    }
    if max_image_side is not None:
        expected_metadata["max_image_side"] = int(max_image_side)
    if vision_tower is not None:
        expected_metadata["vision_tower"] = str(vision_tower)
    if moonvit_model is not None:
        expected_metadata["moonvit_model"] = str(moonvit_model)
    if moonvit_revision is not None:
        expected_metadata["moonvit_revision"] = str(moonvit_revision)
    if moonvit_weights_sha256 is not None:
        expected_metadata["moonvit_weights_sha256"] = str(moonvit_weights_sha256)

    identity_fields = {
        "vision_tower",
        "moonvit_model",
        "moonvit_revision",
        "moonvit_weights_sha256",
    }
    for key, expected in expected_metadata.items():
        actual = cache_manifest.get(key)
        if key in identity_fields and actual is None and not require_tower_identity:
            # Keep compatibility with the original V2 cache manifests, which
            # only carried the extracted weight hash.
            continue
        if actual != expected:
            raise ValueError(f"feature cache interface binding differs: {key}")

    if require_tower_identity:
        if not isinstance(cache_manifest.get("vision_tower"), str):
            raise ValueError("feature cache is missing vision tower identity")
        if moonvit_revision is None and moonvit_weights_sha256 is None:
            raise ValueError(
                "architecture-control cache requires a pinned revision or weight hash"
            )
        if moonvit_revision is not None and not isinstance(
            cache_manifest.get("moonvit_revision"), str
        ):
            raise ValueError("feature cache is missing MoonViT resolved revision")

    records = list(cache_manifest.get("records", []))
    maximum_tokens = 0
    for index, row in enumerate(records):
        shape = [int(value) for value in row.get("feature_shape", [])]
        if len(shape) != 3 or shape[1:] != [merge, width] or shape[0] <= 0:
            raise ValueError(f"feature cache interface shape differs at row {index}")
        if max_visual_tokens is not None and shape[0] > int(max_visual_tokens):
            raise ValueError(
                f"feature cache interface exceeds visual-token budget at row {index}"
            )
        if row.get("dtype") != "float32":
            raise ValueError(f"feature cache interface dtype differs at row {index}")
        maximum_tokens = max(maximum_tokens, shape[0])
    return {
        "vision_width": width,
        "merge_factor": merge,
        "records": len(records),
        "maximum_visual_tokens": maximum_tokens,
    }


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
    moonvit_weights_sha256: str | None = None,
    vision_width: int = 1024,
    merge_factor: int = 4,
    vision_tower: str | None = None,
    moonvit_model: str | None = None,
    moonvit_revision: str | None = None,
    require_tower_identity: bool = False,
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
        "vision_width": int(vision_width),
        "merge_factor": int(merge_factor),
    }
    if moonvit_weights_sha256 is not None:
        expected_binding["moonvit_weights_sha256"] = str(moonvit_weights_sha256)
    if vision_tower is not None:
        expected_binding["vision_tower"] = str(vision_tower)
    if moonvit_model is not None:
        expected_binding["moonvit_model"] = str(moonvit_model)
    if moonvit_revision is not None:
        expected_binding["moonvit_revision"] = str(moonvit_revision)
    for key, expected in expected_binding.items():
        actual = cache_manifest.get(key)
        if (
            key in {"moonvit_weights_sha256", "vision_tower", "moonvit_model", "moonvit_revision"}
            and actual is None
            and not require_tower_identity
        ):
            continue
        if actual != expected:
            raise ValueError(f"ScreenSpot feature cache binding differs: {key}")
    if require_tower_identity:
        if not isinstance(cache_manifest.get("vision_tower"), str):
            raise ValueError("ScreenSpot feature cache is missing vision tower identity")
        if moonvit_revision is None and moonvit_weights_sha256 is None:
            raise ValueError(
                "ScreenSpot architecture-control cache requires a pinned revision or weight hash"
            )

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
        if (
            len(shape) != 3
            or shape[1:] != [int(merge_factor), int(vision_width)]
            or shape[0] <= 0
        ):
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
