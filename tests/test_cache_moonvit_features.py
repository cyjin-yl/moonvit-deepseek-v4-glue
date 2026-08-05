"""Feature-cache CLI keeps transport failures separate from data failures."""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import cache_moonvit_features


def test_progress_broken_pipe_does_not_fail_a_cache_record(monkeypatch):
    def broken_print(*_args, **_kwargs):
        raise BrokenPipeError("detached client")

    monkeypatch.setattr("builtins.print", broken_print)

    cache_moonvit_features.emit("valid row")


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
