import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "experiments"
    / "qwen3b_community_eval_20260805"
    / "feature_cache_4k_v1"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def test_qwen3b_feature_cache_is_exact_clean_4k_order():
    summary = _load(PACKAGE / "SUMMARY.json")
    manifest = _load(PACKAGE / "MANIFEST.json")
    verification = _load(PACKAGE / "INDEPENDENT_VERIFICATION.json")

    assert summary == {
        "requested": 4000,
        "cached": 4000,
        "failed": 0,
        "tower_forwards": 3534,
        "reused_by_image_sha256": 466,
        "unique_image_sha256": 3534,
        "wall_seconds": 503.5900933742523,
        "peak_gpu_memory_bytes": 1_949_755_904,
        "manifest": "MANIFEST.json",
        "records": "cache_records.jsonl",
        "failures": "failures.jsonl",
    }
    assert manifest["git_sha"] == "1e4c4000a88b02761abc5051ea78af9b2c7d4142"
    assert manifest["git_tracked_worktree_clean"] is True
    assert manifest["training_order_manifest_sha256"] == (
        "ddca738e366f37237354bb011bdff1a00d010bdf256ef9101a6adbf35ab9c2fd"
    )
    assert manifest["training_order_records_sha256"] == (
        "61fa7360208b90bb791914c27801cc90d702d579155d798d39ae4e400f7f315e"
    )
    assert manifest["count"] == 4000
    assert manifest["unique_feature_spans"] == 3534
    assert manifest["aliased_records"] == 466
    assert manifest["max_image_side"] == 448
    assert manifest["max_visual_tokens"] == 256
    assert manifest["vision_width"] == 1024
    assert manifest["merge_factor"] == 4
    assert len(manifest["records"]) == 4000
    assert len(manifest["shards"]) == 111
    assert sum(row["bytes"] for row in manifest["shards"]) == 10_372_103_792

    assert verification["status"] == "valid"
    assert verification["manifest_sha256"] == _sha256(PACKAGE / "MANIFEST.json")
    assert verification["records_verified"] == 4000
    assert verification["shards_verified"] == 111
    assert verification["values_verified"] == 2_921_816_064
    assert verification["unique_values_verified"] == 2_593_021_952
    assert verification["unique_feature_spans"] == 3534
    assert verification["aliased_records"] == 466
    assert verification["runtime_source_files_verified"] == 3
    assert verification["training_order_binding"] == {
        "records_matched": 4000,
        "unique_images_matched": 3534,
        "aliased_records_matched": 466,
        "maximum_visual_tokens": 256,
    }


def test_qwen3b_feature_cache_preserves_full_root_and_invalid_attempt():
    remote = _load(PACKAGE / "REMOTE_ARTIFACT_MANIFEST.json")
    assert remote["git_base_sha"] == "a9bd07b97e9bfc11ae4b82d69584a88b8799a646"
    assert remote["file_count"] == 118
    assert remote["total_bytes"] == 10_374_552_697
    assert len(
        [name for name in remote["files"] if name.startswith("features-")]
    ) == 111
    assert remote["final_half_scored"] is False

    failure_root = PACKAGE / "failures" / "attempt01_uncommitted_runner"
    failure = _load(failure_root / "SUMMARY.json")
    failure_remote = _load(failure_root / "REMOTE_ARTIFACT_MANIFEST.json")
    assert failure["status"] == "invalid"
    assert failure["git_tracked_worktree_clean"] is False
    assert failure["logged_records"] == 1128
    assert failure["written_shards"] == 33
    assert failure["final_cache_manifest_written"] is False
    assert failure["training_use_allowed"] is False
    assert failure["capability_claim_allowed"] is False
    assert failure["run_log_sha256"] == _sha256(failure_root / "run.log")
    assert failure["cache_records_sha256"] == _sha256(
        failure_root / "cache_records.jsonl"
    )
    assert failure_remote["file_count"] == 37
    assert failure_remote["total_bytes"] == 3_110_105_586
    assert len(
        [name for name in failure_remote["files"] if name.startswith("features-")]
    ) == 33


def test_qwen3b_feature_cache_curated_manifest_matches_package_files():
    artifact_manifest_path = PACKAGE / "ARTIFACT_MANIFEST.json"
    artifact_manifest = _load(artifact_manifest_path)
    actual = {
        path.relative_to(PACKAGE).as_posix(): path
        for path in PACKAGE.rglob("*")
        if path.is_file() and path != artifact_manifest_path
    }
    assert set(actual) == set(artifact_manifest["files"])
    assert artifact_manifest["file_count"] == len(actual)
    assert artifact_manifest["total_bytes"] == sum(
        path.stat().st_size for path in actual.values()
    )
    for relative, path in actual.items():
        declared = artifact_manifest["files"][relative]
        assert declared["bytes"] == path.stat().st_size
        assert declared["sha256"] == _sha256(path)
    assert artifact_manifest["final_half_scored"] is False
