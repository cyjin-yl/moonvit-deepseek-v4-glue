import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "experiments"
    / "qwen3b_community_eval_20260805"
    / "feature_cache_grounding_enriched_4k_v1"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def test_grounding_enriched_cache_is_exact_clean_order():
    summary = _load(PACKAGE / "SUMMARY.json")
    manifest = _load(PACKAGE / "MANIFEST.json")
    verification = _load(PACKAGE / "INDEPENDENT_VERIFICATION.json")

    assert summary == {
        "requested": 4000,
        "cached": 4000,
        "failed": 0,
        "tower_forwards": 2013,
        "reused_by_image_sha256": 1987,
        "unique_image_sha256": 2013,
        "wall_seconds": 299.14158940315247,
        "peak_gpu_memory_bytes": 1_947_973_120,
        "manifest": "MANIFEST.json",
        "records": "cache_records.jsonl",
        "failures": "failures.jsonl",
    }
    assert _sha256(PACKAGE / "MANIFEST.json") == (
        "1f035374a0db01c8347dfcb2fb6aa6c5e4aad7128ccaa4e169e17f784a6c8a41"
    )
    assert manifest["git_sha"] == (
        "aa933ca1cf5a9386f60cd67658083d4e79b2b376"
    )
    assert manifest["git_tracked_worktree_clean"] is True
    assert manifest["training_order_manifest_sha256"] == (
        "d632ecc2c9bc216a552f240e87b9733904b67dcfe30c489a62ba03df25370bf1"
    )
    assert manifest["training_order_records_sha256"] == (
        "f3c3dec199a30927fb715b2d4fbc890baa8e3f3a456b56707f46490164b915ab"
    )
    assert manifest["count"] == 4000
    assert manifest["unique_feature_spans"] == 2013
    assert manifest["aliased_records"] == 1987
    assert manifest["max_image_side"] == 448
    assert manifest["max_visual_tokens"] == 256
    assert len(manifest["records"]) == 4000
    assert len(manifest["shards"]) == 63
    assert sum(row["bytes"] for row in manifest["shards"]) == 5_943_468_912

    assert verification["status"] == "valid"
    assert verification["manifest_sha256"] == _sha256(PACKAGE / "MANIFEST.json")
    assert verification["records_verified"] == 4000
    assert verification["shards_verified"] == 63
    assert verification["values_verified"] == 2_742_976_512
    assert verification["unique_values_verified"] == 1_485_864_960
    assert verification["unique_feature_spans"] == 2013
    assert verification["aliased_records"] == 1987
    assert verification["training_order_binding"] == {
        "records_matched": 4000,
        "unique_images_matched": 2013,
        "aliased_records_matched": 1987,
        "maximum_visual_tokens": 256,
    }


def test_grounding_enriched_cache_remote_and_package_hashes():
    remote = _load(PACKAGE / "REMOTE_ARTIFACT_MANIFEST.json")
    assert remote["git_base_sha"] == (
        "aa933ca1cf5a9386f60cd67658083d4e79b2b376"
    )
    assert remote["file_count"] == 70
    assert remote["total_bytes"] == 5_946_091_225
    assert len(
        [name for name in remote["files"] if name.startswith("features-")]
    ) == 63
    assert remote["final_half_scored"] is False

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
