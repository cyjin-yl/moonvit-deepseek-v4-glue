"""Frozen MoonViT feature caches preserve tensors and audit provenance."""

import json
from pathlib import Path

import pytest
import torch

import moonvit_glue.feature_cache as feature_cache_module
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


def test_content_address_alias_reuses_one_feature_span(tmp_path):
    root = tmp_path / "cache"
    writer = FeatureCacheWriter(
        root,
        cache_metadata={"cache_format_version": 1},
        shard_size=8,
    )
    feature = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    writer.add(
        sample_id="first",
        feature=feature,
        image_sha256="a" * 64,
        image_size=(32, 24),
    )
    writer.add_alias(
        sample_id="second",
        source_sample_id="first",
        image_sha256="a" * 64,
        image_size=(32, 24),
    )
    manifest = writer.close()

    assert len(manifest["shards"]) == 1
    assert manifest["unique_feature_spans"] == 1
    assert manifest["aliased_records"] == 1
    assert manifest["records"][1]["alias_of"] == "first"
    assert manifest["records"][0]["start"] == manifest["records"][1]["start"]
    assert manifest["records"][0]["end"] == manifest["records"][1]["end"]
    cache = FeatureCache(root)
    assert torch.equal(cache.get("first")[0], feature)
    assert torch.equal(cache.get("second")[0], feature)


def test_content_address_alias_requires_identical_image_identity(tmp_path):
    writer = FeatureCacheWriter(
        tmp_path / "cache",
        cache_metadata={"cache_format_version": 1},
        shard_size=8,
    )
    writer.add(
        sample_id="first",
        feature=torch.zeros(1, 2, 4),
        image_sha256="a" * 64,
        image_size=(32, 24),
    )

    with pytest.raises(ValueError, match="image SHA-256"):
        writer.add_alias(
            sample_id="second",
            source_sample_id="first",
            image_sha256="b" * 64,
            image_size=(32, 24),
        )


def test_reader_keeps_a_bounded_lru_of_shards(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    writer = FeatureCacheWriter(
        root,
        cache_metadata={"cache_format_version": 1},
        shard_size=1,
    )
    for index, sample_id in enumerate(("a", "b", "c")):
        writer.add(
            sample_id=sample_id,
            feature=torch.full((1, 2, 4), float(index)),
            image_sha256=str(index) * 64,
            image_size=(16, 16),
        )
    writer.close()

    original_load = feature_cache_module.load_file
    loads = []

    def counted_load(path, *, device):
        loads.append(Path(path).name)
        return original_load(path, device=device)

    monkeypatch.setattr(feature_cache_module, "load_file", counted_load)
    cache = FeatureCache(root, max_loaded_shards=2)
    for sample_id in ("a", "b", "a", "c", "b"):
        cache.get(sample_id)

    assert loads == [
        "features-00000.safetensors",
        "features-00001.safetensors",
        "features-00002.safetensors",
        "features-00001.safetensors",
    ]
