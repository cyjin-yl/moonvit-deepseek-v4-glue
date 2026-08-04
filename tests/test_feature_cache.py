"""Frozen MoonViT feature caches preserve tensors and audit provenance."""

import json

import pytest
import torch

from moonvit_glue.feature_cache import FeatureCache, FeatureCacheWriter


def test_variable_length_features_round_trip_with_provenance(tmp_path):
    root = tmp_path / "cache"
    writer = FeatureCacheWriter(
        root,
        cache_metadata={
            "cache_format_version": 1,
            "moonvit_config_sha256": "c" * 64,
            "moonvit_weights_sha256": "w" * 64,
        },
        shard_size=2,
    )
    first = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    second = torch.arange(40, dtype=torch.float32).reshape(5, 2, 4)
    writer.add(
        sample_id="first",
        feature=first,
        image_sha256="a" * 64,
        image_size=(32, 24),
    )
    writer.add(
        sample_id="second",
        feature=second,
        image_sha256="b" * 64,
        image_size=(48, 40),
    )
    manifest = writer.close()

    cache = FeatureCache(root)

    assert cache.metadata["moonvit_weights_sha256"] == "w" * 64
    assert torch.equal(cache.get("first")[0], first)
    assert torch.equal(cache.get("second")[0], second)
    assert manifest["count"] == 2
    assert manifest["records_sha256"]
    assert manifest["records"][0] == {
        "id": "first",
        "image_sha256": "a" * 64,
        "image_width": 32,
        "image_height": 24,
        "feature_shape": [3, 2, 4],
        "dtype": "float32",
        "shard": "features-00000.safetensors",
        "start": 0,
        "end": 3,
    }


def test_reader_rejects_tampered_record_manifest(tmp_path):
    root = tmp_path / "cache"
    writer = FeatureCacheWriter(
        root,
        cache_metadata={"cache_format_version": 1},
        shard_size=1,
    )
    writer.add(
        sample_id="original",
        feature=torch.zeros(2, 4, 3),
        image_sha256="a" * 64,
        image_size=(16, 16),
    )
    writer.close()
    path = root / "MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["records"][0]["id"] = "tampered"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="records SHA-256 mismatch"):
        FeatureCache(root)
