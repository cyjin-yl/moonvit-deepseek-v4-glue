import copy

import pytest

from moonvit_glue.screenspot_runtime import (
    shuffled_image_mapping,
    validate_screenspot_feature_cache,
)


def _dataset() -> dict:
    return {
        "name": "screenspot_glm50_v1",
        "manifest_sha256": "m" * 64,
        "samples": [
            {
                "sample_id": "a",
                "image_sha256": "a" * 64,
                "image_width": 100,
                "image_height": 200,
            },
            {
                "sample_id": "b",
                "image_sha256": "b" * 64,
                "image_width": 300,
                "image_height": 400,
            },
        ],
        "shuffled_image_control": {
            "mapping": [
                {"sample_id": "a", "shuffled_image_sample_id": "b"},
                {"sample_id": "b", "shuffled_image_sample_id": "a"},
            ]
        },
    }


def _cache() -> dict:
    return {
        "count": 2,
        "binding_manifest_file_sha256": "f" * 64,
        "binding_manifest_sha256": "m" * 64,
        "binding_manifest_name": "screenspot_glm50_v1",
        "max_image_side": 1024,
        "moonvit_weights_sha256": "v" * 64,
        "vision_width": 1024,
        "merge_factor": 4,
        "aliased_records": 0,
        "unique_feature_spans": 2,
        "records": [
            {
                "id": "a",
                "image_sha256": "a" * 64,
                "image_width": 100,
                "image_height": 200,
                "dtype": "float32",
                "feature_shape": [20, 4, 1024],
            },
            {
                "id": "b",
                "image_sha256": "b" * 64,
                "image_width": 300,
                "image_height": 400,
                "dtype": "float32",
                "feature_shape": [40, 4, 1024],
            },
        ],
    }


def test_shuffled_mapping_is_an_image_safe_permutation():
    assert shuffled_image_mapping(_dataset()) == {"a": "b", "b": "a"}
    fixed = _dataset()
    fixed["shuffled_image_control"]["mapping"][0]["shuffled_image_sample_id"] = "a"
    with pytest.raises(ValueError, match="derangement"):
        shuffled_image_mapping(fixed)


def test_screenspot_cache_is_bound_to_exact_order_images_and_budget():
    assert validate_screenspot_feature_cache(
        _dataset(),
        _cache(),
        dataset_manifest_file_sha256="f" * 64,
        max_image_side=1024,
        max_visual_tokens=1369,
        moonvit_weights_sha256="v" * 64,
    ) == {
        "records": 2,
        "aliased_records": 0,
        "unique_feature_spans": 2,
        "maximum_visual_tokens": 40,
    }

    reordered = _cache()
    reordered["records"].reverse()
    with pytest.raises(ValueError, match="row 0"):
        validate_screenspot_feature_cache(
            _dataset(),
            reordered,
            dataset_manifest_file_sha256="f" * 64,
            max_image_side=1024,
            max_visual_tokens=1369,
            moonvit_weights_sha256="v" * 64,
        )

    oversized = copy.deepcopy(_cache())
    oversized["records"][1]["feature_shape"][0] = 1370
    with pytest.raises(ValueError, match="visual-token budget"):
        validate_screenspot_feature_cache(
            _dataset(),
            oversized,
            dataset_manifest_file_sha256="f" * 64,
            max_image_side=1024,
            max_visual_tokens=1369,
            moonvit_weights_sha256="v" * 64,
        )
