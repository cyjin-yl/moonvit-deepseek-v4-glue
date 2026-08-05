import hashlib
from pathlib import Path
import sys

import pytest
import torch

from moonvit_glue.feature_cache import FeatureCacheWriter


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from verify_feature_cache import compare_training_order_binding, verify_feature_cache


def test_verify_feature_cache_rehashes_shards_and_reads_every_record(tmp_path: Path):
    writer = FeatureCacheWriter(
        tmp_path / "cache",
        cache_metadata={"cache_format_version": 1, "max_image_side": 256},
        shard_size=1,
    )
    writer.add(
        sample_id="a",
        feature=torch.ones(2, 4, 3),
        image_sha256="1" * 64,
        image_size=(16, 16),
    )
    writer.add(
        sample_id="b",
        feature=torch.zeros(1, 4, 3),
        image_sha256="2" * 64,
        image_size=(16, 16),
    )
    writer.close()

    result = verify_feature_cache(tmp_path / "cache", expected_count=2)

    assert result["status"] == "valid"
    assert result["records_verified"] == 2
    assert result["shards_verified"] == 2
    assert result["values_verified"] == 36
    assert result["unique_values_verified"] == 36
    assert result["unique_feature_spans"] == 2
    assert result["aliased_records"] == 0
    assert result["training_order_binding"] is None


def test_verify_feature_cache_validates_content_address_aliases(tmp_path: Path):
    writer = FeatureCacheWriter(
        tmp_path / "cache",
        cache_metadata={"cache_format_version": 1, "max_image_side": 448},
        shard_size=8,
    )
    writer.add(
        sample_id="first",
        feature=torch.ones(2, 4, 3),
        image_sha256="1" * 64,
        image_size=(16, 16),
    )
    writer.add_alias(
        sample_id="second",
        source_sample_id="first",
        image_sha256="1" * 64,
        image_size=(16, 16),
    )
    writer.close()

    result = verify_feature_cache(tmp_path / "cache", expected_count=2)

    assert result["records_verified"] == 2
    assert result["values_verified"] == 48
    assert result["unique_values_verified"] == 24
    assert result["unique_feature_spans"] == 1
    assert result["aliased_records"] == 1


def test_verify_feature_cache_strict_provenance_rehashes_runtime_source(
    tmp_path: Path,
):
    source = tmp_path / "runner.py"
    source.write_text("print('frozen')\n", encoding="utf-8")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    writer = FeatureCacheWriter(
        tmp_path / "cache",
        cache_metadata={
            "cache_format_version": 1,
            "git_sha": "a" * 40,
            "git_tracked_worktree_clean": True,
            "runtime_source_files": [
                {"path": str(source), "sha256": source_sha256}
            ],
        },
        shard_size=1,
    )
    writer.add(
        sample_id="a",
        feature=torch.ones(1, 4, 3),
        image_sha256="1" * 64,
        image_size=(16, 16),
    )
    writer.close()

    result = verify_feature_cache(
        tmp_path / "cache", expected_count=1, expected_git_sha="a" * 40
    )
    assert result["runtime_source_files_verified"] == 1

    source.write_text("print('tampered')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime source differs"):
        verify_feature_cache(
            tmp_path / "cache", expected_count=1, expected_git_sha="a" * 40
        )


def test_compare_training_order_binding_checks_exact_alias_and_budget():
    order = {
        "feature_cache": {"max_visual_tokens": 256},
        "unique_image_sha256": 1,
        "records": [
            {
                "id": "first",
                "image_sha256": "1" * 64,
                "image_width": 16,
                "image_height": 16,
            },
            {
                "id": "second",
                "image_sha256": "1" * 64,
                "image_width": 16,
                "image_height": 16,
            },
        ],
    }
    cache = {
        "aliased_records": 1,
        "records": [
            {
                **order["records"][0],
                "feature_shape": [256, 4, 1024],
                "shard": "features-00000.safetensors",
                "start": 0,
                "end": 256,
            },
            {
                **order["records"][1],
                "feature_shape": [256, 4, 1024],
                "shard": "features-00000.safetensors",
                "start": 0,
                "end": 256,
                "alias_of": "first",
            },
        ],
    }

    assert compare_training_order_binding(cache, order) == {
        "records_matched": 2,
        "unique_images_matched": 1,
        "aliased_records_matched": 1,
        "maximum_visual_tokens": 256,
    }
    cache["records"][1]["alias_of"] = "wrong"
    with pytest.raises(ValueError, match="first-occurrence canonical"):
        compare_training_order_binding(cache, order)
