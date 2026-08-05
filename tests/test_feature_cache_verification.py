from pathlib import Path
import sys

import torch

from moonvit_glue.feature_cache import FeatureCacheWriter


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from verify_feature_cache import verify_feature_cache


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
