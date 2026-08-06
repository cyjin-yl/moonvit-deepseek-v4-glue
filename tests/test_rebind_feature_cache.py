import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from moonvit_glue.feature_cache import FeatureCacheWriter
from moonvit_glue.training_order import (
    _logical_sha256,
    _manifest_sha256,
    build_training_order_manifest,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from rebind_feature_cache import rebind_feature_cache  # noqa: E402
from verify_feature_cache import verify_feature_cache  # noqa: E402
import rebind_feature_cache as rebind_module  # noqa: E402


def _make_order(tmp_path: Path) -> Path:
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    for index in range(2):
        image_name = f"image-{index}.png"
        Image.new("RGB", (20 + index, 30 + index), (index, 5, 9)).save(
            image_root / image_name
        )
        rows.append(
            {
                "id": f"row-{index}",
                "source": "showui_desktop",
                "image": f"images/{image_name}",
                "question": "where?",
                "answers": [f"click(start_box=[{100 + index}, {200 + index}])"],
            }
        )
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    contract = {
        "datasets": {
            "training_pack": {
                "order_is_frozen": True,
                "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "records": len(rows),
            }
        },
        "training_budget": {
            "examples_seen_checkpoints": [2],
            "optimizer_steps_checkpoints": [1],
            "micro_batch_size": 1,
            "gradient_accumulation": 2,
            "real_global_batch": 2,
            "effective_epochs_denominator": 2,
        },
        "vision_tower": {"name": "fixture", "extracted_weights_sha256": "0" * 64},
        "image_preprocessing": {"train_max_image_side": 448, "train_max_visual_tokens": 256},
    }
    manifest = build_training_order_manifest(
        data_path=data_path,
        contract=contract,
        contract_sha256="1" * 64,
        examples_seen=2,
    )
    order_path = tmp_path / "target-order" / "MANIFEST.json"
    order_path.parent.mkdir()
    order_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return order_path


def _make_source_cache(tmp_path: Path, order_path: Path) -> Path:
    order = json.loads(order_path.read_text())
    source = tmp_path / "source-cache"
    writer = FeatureCacheWriter(
        source,
        cache_metadata={
            "cache_format_version": 1,
            "max_image_side": 448,
            "max_visual_tokens": 256,
            "training_order_manifest": "/old/order/MANIFEST.json",
            "training_order_manifest_file_sha256": "2" * 64,
            "training_order_manifest_sha256": "3" * 64,
            "training_order_records_sha256": "4" * 64,
            "git_sha": "a" * 40,
            "runtime_source_files": [],
        },
        shard_size=1,
    )
    for index, row in enumerate(order["records"]):
        writer.add(
            sample_id=row["id"],
            feature=torch.full((2 + index, 4, 3), float(index + 1)),
            image_sha256=row["image_sha256"],
            image_size=(row["image_width"], row["image_height"]),
        )
    writer.close()
    return source


def test_rebind_reuses_shards_and_existing_verifier_accepts_target(tmp_path: Path):
    order_path = _make_order(tmp_path)
    source = _make_source_cache(tmp_path, order_path)
    target = tmp_path / "target-cache"

    summary = rebind_feature_cache(source, order_path, target, current_git_sha="b" * 40)
    assert summary["status"] == "valid"
    assert summary["all_shards_hard_linked"] is True
    assert summary["current_git_sha"] == "b" * 40
    assert len(summary["linked_shards"]) == 2

    source_manifest = json.loads((source / "MANIFEST.json").read_text())
    target_manifest = json.loads((target / "MANIFEST.json").read_text())
    order = json.loads(order_path.read_text())
    assert target_manifest["git_sha"] == source_manifest["git_sha"]
    assert target_manifest["records_sha256"] == source_manifest["records_sha256"]
    assert target_manifest["training_order_manifest_sha256"] == order["manifest_sha256"]
    assert target_manifest["training_order_records_sha256"] == order["records_sha256"]
    assert target_manifest["cache_rebinding"]["source_manifest_file_sha256"] == (
        hashlib.sha256((source / "MANIFEST.json").read_bytes()).hexdigest()
    )
    for row in target_manifest["shards"]:
        assert os.path.samefile(source / row["path"], target / row["path"])
        evidence = next(item for item in summary["linked_shards"] if item["path"] == row["path"])
        assert evidence["source_inode"] == evidence["target_inode"]
        assert evidence["samefile"] is True

    result = verify_feature_cache(target, expected_count=2, training_order_manifest_path=order_path)
    assert result["status"] == "valid"
    assert result["training_order_binding"]["records_matched"] == 2
    assert json.loads((target / "SUMMARY.json").read_text())["status"] == "valid"


def test_rebind_rejects_target_record_mismatch_even_with_rehashed_order(tmp_path: Path):
    order_path = _make_order(tmp_path)
    source = _make_source_cache(tmp_path, order_path)
    order = json.loads(order_path.read_text())
    order["records"][1]["image_sha256"] = "f" * 64
    order["records_sha256"] = _logical_sha256(order["records"])
    order["manifest_sha256"] = _manifest_sha256(order)
    mismatched = tmp_path / "mismatched-order" / "MANIFEST.json"
    mismatched.parent.mkdir()
    mismatched.write_text(json.dumps(order, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="cache record differs from training order"):
        rebind_feature_cache(source, mismatched, tmp_path / "should-not-exist")
    assert not (tmp_path / "should-not-exist").exists()


def test_rebind_refuses_existing_output_before_touching_source(tmp_path: Path):
    order_path = _make_order(tmp_path)
    source = _make_source_cache(tmp_path, order_path)
    target = tmp_path / "existing"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        rebind_feature_cache(source, order_path, target)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_rebind_hardlink_failure_never_falls_back_to_copy(tmp_path: Path, monkeypatch):
    order_path = _make_order(tmp_path)
    source = _make_source_cache(tmp_path, order_path)
    target = tmp_path / "failed-cache"

    def fail_link(*args, **kwargs):
        raise OSError("cross-device link")

    monkeypatch.setattr(rebind_module.os, "link", fail_link)
    with pytest.raises(OSError, match="refusing to copy"):
        rebind_feature_cache(source, order_path, target)
    assert not target.exists()
