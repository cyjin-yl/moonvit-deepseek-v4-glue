"""Feature-cache CLI keeps transport failures separate from data failures."""

import hashlib
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import cache_moonvit_features


def test_progress_broken_pipe_does_not_fail_a_cache_record(monkeypatch):
    def broken_print(*_args, **_kwargs):
        raise BrokenPipeError("detached client")

    monkeypatch.setattr("builtins.print", broken_print)

    cache_moonvit_features.emit("valid row")


def test_load_tower_requires_v2_weights(monkeypatch):
    args = Namespace(
        vision_tower="v2",
        moonvit_v2_weights=None,
        moonvit_v2_attn="eager",
        moonvit_model="unused",
        moonvit_revision=None,
    )
    with pytest.raises(ValueError, match="requires --moonvit-v2-weights"):
        cache_moonvit_features.load_tower(args, dtype=cache_moonvit_features.torch.float32)


def test_load_tower_routes_v1_to_standalone_loader(monkeypatch):
    sentinel = object()
    seen = {}

    def fake_loader(model_id, *, revision, torch_dtype):
        seen.update(model_id=model_id, revision=revision, torch_dtype=torch_dtype)
        return sentinel

    monkeypatch.setattr(cache_moonvit_features.MoonViTEncoder, "from_pretrained", fake_loader)
    args = Namespace(
        vision_tower="v1",
        moonvit_v2_weights=None,
        moonvit_v2_attn="eager",
        moonvit_model="moonshotai/MoonViT-SO-400M",
        moonvit_revision="a889d399ff2306053e4e28d499d3b8f97d3e5007",
    )
    assert cache_moonvit_features.load_tower(args, dtype=cache_moonvit_features.torch.float32) is sentinel
    assert seen == {
        "model_id": "moonshotai/MoonViT-SO-400M",
        "revision": "a889d399ff2306053e4e28d499d3b8f97d3e5007",
        "torch_dtype": cache_moonvit_features.torch.float32,
    }


def test_training_order_mode_rejects_secondary_selection(tmp_path):
    with pytest.raises(ValueError, match="cannot be combined"):
        cache_moonvit_features.resolve_cache_selection(
            data_path=tmp_path / "train.jsonl",
            training_order_manifest=tmp_path / "ORDER.json",
            ids_manifest=None,
            record_slice=None,
            limit=4,
            shuffle_seed=None,
        )


def test_training_order_cache_contract_checks_resolution_and_tower_hash():
    manifest = {
        "feature_cache": {
            "max_image_side": 448,
            "moonvit_weights_sha256": "a" * 64,
        }
    }
    cache_moonvit_features.validate_training_order_cache_contract(
        manifest,
        max_image_side=448,
        moonvit_weights_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="max image side"):
        cache_moonvit_features.validate_training_order_cache_contract(
            manifest,
            max_image_side=640,
            moonvit_weights_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="MoonViT SHA-256"):
        cache_moonvit_features.validate_training_order_cache_contract(
            manifest,
            max_image_side=448,
            moonvit_weights_sha256="b" * 64,
        )


def test_training_order_image_identity_is_checked_before_forward():
    payload = b"frozen image bytes"
    entry = {
        "id": "sample",
        "image_sha256": hashlib.sha256(payload).hexdigest(),
        "image_bytes": len(payload),
        "image_width": 32,
        "image_height": 24,
    }
    cache_moonvit_features.validate_training_order_image(
        entry,
        record_id="sample",
        payload=payload,
        image_size=(32, 24),
    )

    with pytest.raises(ValueError, match="image SHA-256"):
        cache_moonvit_features.validate_training_order_image(
            entry,
            record_id="sample",
            payload=b"broken image bytes",
            image_size=(32, 24),
        )


def test_training_order_feature_shape_enforces_visual_token_budget():
    manifest = {"feature_cache": {"max_visual_tokens": 256}}
    cache_moonvit_features.validate_training_order_feature_shape(
        manifest,
        record_id="ok",
        feature_shape=(256, 4, 1024),
    )
    with pytest.raises(ValueError, match="visual token count"):
        cache_moonvit_features.validate_training_order_feature_shape(
            manifest,
            record_id="too-many",
            feature_shape=(257, 4, 1024),
        )


def test_binding_manifest_metadata_hashes_file_and_records_identity(tmp_path):
    path = tmp_path / "MANIFEST.json"
    path.write_text(
        '{"name":"screenspot_glm50_v1","manifest_sha256":"abc"}\n',
        encoding="utf-8",
    )
    result = cache_moonvit_features.binding_manifest_metadata(path)
    assert result["binding_manifest"] == str(path.resolve())
    assert result["binding_manifest_name"] == "screenspot_glm50_v1"
    assert result["binding_manifest_sha256"] == "abc"
    assert len(result["binding_manifest_file_sha256"]) == 64
