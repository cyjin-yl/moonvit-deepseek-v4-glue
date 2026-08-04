"""Trajectory verification rejects missing, duplicate, and non-finite raw rows."""

import hashlib
import json

import pytest

from moonvit_glue.trajectory_verification import (
    trajectory_dataset_provenance,
    verify_generation_selection_manifest,
    verify_trajectory_run,
)


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generation_selection_manifest_binds_raw_ids_and_data_hash(tmp_path):
    data_path = tmp_path / "selection.jsonl"
    _jsonl(data_path, [{"id": "a"}, {"id": "b"}])
    manifest_path = tmp_path / "MANIFEST.json"
    manifest = {
        "logical_dataset_sha256": "logical-sha",
        "selection": {"sample_ids": ["a", "b"]},
        "files": {
            data_path.name: {"sha256": _hash(data_path)},
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    dataset = {
        "data": str(data_path),
        "expected_records": 2,
        "logical_dataset_sha256": "logical-sha",
        "generation_selection_manifest": str(manifest_path),
    }
    provenance = {
        "data_sha256": _hash(data_path),
        "generation_selection_manifest_sha256": _hash(manifest_path),
    }

    verify_generation_selection_manifest(dataset, {"a", "b"}, provenance)
    with pytest.raises(ValueError, match="raw IDs"):
        verify_generation_selection_manifest(dataset, {"a", "c"}, provenance)


def test_dataset_provenance_comes_from_summary_top_level():
    summary = {
        "metadata": {"datasets": {"synthetic": {"wrong": True}}},
        "datasets": {"synthetic": {"data_sha256": "expected"}},
    }
    assert trajectory_dataset_provenance(summary, "synthetic") == {
        "data_sha256": "expected"
    }


def test_verifier_checks_exact_raw_denominators_and_hashes(tmp_path):
    config = {
        "checkpoints": [{"id": "c0"}],
        "aliases": [{"id": "current", "source": "c0"}],
        "datasets": [{"name": "synthetic", "expected_records": 2, "conditions": ["vision", "blind"]}],
        "heldout_shuffle_loss": {"expected_records": 2, "shuffle_repeats": 2},
    }
    (tmp_path / "CONFIG.json").write_text(json.dumps(config), encoding="utf-8")
    records = [
        {"checkpoint": "c0", "dataset": "synthetic", "condition": condition, "id": sample_id, "score": 0.0, "failure": None}
        for condition in ("vision", "blind") for sample_id in ("a", "b")
    ]
    shuffle = [
        {"checkpoint": "c0", "id": sample_id, "true_loss": 1.0, "shuffled_losses": [1.1, 1.2], "mean_shuffled_loss": 1.15, "delta": 0.15, "failure": None}
        for sample_id in ("a", "b")
    ]
    _jsonl(tmp_path / "records.jsonl", records)
    _jsonl(tmp_path / "shuffle_loss_records.jsonl", shuffle)
    (tmp_path / "failures.jsonl").write_text("", encoding="utf-8")
    summary = {
        "status": "valid",
        "metadata": {
            "final_half_scored": False,
            "config_sha256": _hash(tmp_path / "CONFIG.json"),
        },
        "checkpoints": {"c0": {}, "current": {"alias_of": "c0"}},
        "raw_files": {
            name: {"bytes": (tmp_path / name).stat().st_size, "sha256": _hash(tmp_path / name)}
            for name in ("records.jsonl", "shuffle_loss_records.jsonl", "failures.jsonl")
        },
    }
    (tmp_path / "SUMMARY.json").write_text(json.dumps(summary), encoding="utf-8")

    verified = verify_trajectory_run(tmp_path)
    assert verified["status"] == "valid"
    assert verified["generation_rows_verified"] == 4
    assert verified["shuffle_rows_verified"] == 2

    _jsonl(tmp_path / "records.jsonl", records + [records[0]])
    try:
        verify_trajectory_run(tmp_path)
    except ValueError as error:
        assert "hash mismatch" in str(error) or "duplicate" in str(error)
    else:
        raise AssertionError("duplicate raw row was accepted")


def test_verifier_rejects_config_changed_after_the_run(tmp_path):
    config = {
        "checkpoints": [{"id": "c0"}],
        "datasets": [
            {"name": "synthetic", "expected_records": 1, "conditions": ["vision"]}
        ],
        "heldout_shuffle_loss": {"expected_records": 1, "shuffle_repeats": 1},
    }
    config_path = tmp_path / "CONFIG.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    _jsonl(
        tmp_path / "records.jsonl",
        [
            {
                "checkpoint": "c0",
                "dataset": "synthetic",
                "condition": "vision",
                "id": "a",
                "score": 0.0,
                "failure": None,
            }
        ],
    )
    _jsonl(
        tmp_path / "shuffle_loss_records.jsonl",
        [
            {
                "checkpoint": "c0",
                "id": "a",
                "true_loss": 1.0,
                "shuffled_losses": [1.1],
                "mean_shuffled_loss": 1.1,
                "delta": 0.1,
                "failure": None,
            }
        ],
    )
    (tmp_path / "failures.jsonl").write_text("", encoding="utf-8")
    summary = {
        "status": "valid",
        "metadata": {
            "final_half_scored": False,
            "config_sha256": _hash(config_path),
        },
        "checkpoints": {"c0": {}},
        "raw_files": {
            name: {
                "bytes": (tmp_path / name).stat().st_size,
                "sha256": _hash(tmp_path / name),
            }
            for name in ("records.jsonl", "shuffle_loss_records.jsonl", "failures.jsonl")
        },
    }
    (tmp_path / "SUMMARY.json").write_text(json.dumps(summary), encoding="utf-8")
    config["datasets"][0]["expected_records"] = 2
    config_path.write_text(json.dumps(config), encoding="utf-8")

    try:
        verify_trajectory_run(tmp_path)
    except ValueError as error:
        assert "config SHA-256 mismatch" in str(error)
    else:
        raise AssertionError("post-run config mutation was accepted")


def test_verifier_rejects_condition_sample_id_drift(tmp_path):
    config = {
        "checkpoints": [{"id": "c0"}],
        "datasets": [
            {
                "name": "synthetic",
                "expected_records": 2,
                "conditions": ["vision", "blind"],
            }
        ],
        "heldout_shuffle_loss": {"expected_records": 1, "shuffle_repeats": 1},
    }
    config_path = tmp_path / "CONFIG.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    records = [
        {
            "checkpoint": "c0",
            "dataset": "synthetic",
            "condition": condition,
            "id": sample_id,
            "score": 0.0,
            "failure": None,
        }
        for condition, identifiers in (("vision", ("a", "b")), ("blind", ("a", "c")))
        for sample_id in identifiers
    ]
    _jsonl(tmp_path / "records.jsonl", records)
    _jsonl(
        tmp_path / "shuffle_loss_records.jsonl",
        [
            {
                "checkpoint": "c0",
                "id": "a",
                "true_loss": 1.0,
                "shuffled_losses": [1.1],
                "mean_shuffled_loss": 1.1,
                "delta": 0.1,
                "failure": None,
            }
        ],
    )
    (tmp_path / "failures.jsonl").write_text("", encoding="utf-8")
    summary = {
        "status": "valid",
        "metadata": {
            "final_half_scored": False,
            "config_sha256": _hash(config_path),
        },
        "checkpoints": {"c0": {}},
        "raw_files": {
            name: {
                "bytes": (tmp_path / name).stat().st_size,
                "sha256": _hash(tmp_path / name),
            }
            for name in ("records.jsonl", "shuffle_loss_records.jsonl", "failures.jsonl")
        },
    }
    (tmp_path / "SUMMARY.json").write_text(json.dumps(summary), encoding="utf-8")

    try:
        verify_trajectory_run(tmp_path)
    except ValueError as error:
        assert "condition sample IDs mismatch" in str(error)
    else:
        raise AssertionError("condition-specific sample ID drift was accepted")
