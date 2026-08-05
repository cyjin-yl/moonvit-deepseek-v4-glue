import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments" / "qwen3b_community_eval_20260805" / "training_order_v1"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def test_qwen3b_training_order_is_exact_pre_result_prefix():
    manifest = _load(PACKAGE / "MANIFEST.json")
    verification = _load(PACKAGE / "INDEPENDENT_VERIFICATION.json")

    assert manifest["manifest_sha256"] == (
        "ddca738e366f37237354bb011bdff1a00d010bdf256ef9101a6adbf35ab9c2fd"
    )
    assert manifest["selection"] == {
        "effective_epochs": 4000 / 59198,
        "effective_epochs_denominator": 59198,
        "examples_seen": 4000,
        "gradient_accumulation": 8,
        "holdout_removed": False,
        "micro_batch_size": 1,
        "optimizer_steps": 500,
        "real_global_batch": 8,
        "rule": "first_n_rows_preserve_source_order",
        "shuffle": False,
        "subset_passes": 1.0,
    }
    assert manifest["source_counts"] == {
        "docvqa_train": 1160,
        "showui_desktop": 339,
        "textvqa_train": 1985,
        "train": 516,
    }
    assert manifest["prompt_route_counts"] == {
        "grounding": 339,
        "short_answer": 3661,
    }
    assert manifest["target_transform_counts"] == {
        "legacy_click_spacing_to_canonical": 339,
        "single_answer_passthrough": 1198,
        "vqa_normalized_majority": 2461,
        "vqa_raw_majority_empty_normalization_fallback": 2,
    }
    assert manifest["training_results_exist"] is False
    assert manifest["final_half_scored"] is False

    assert verification["status"] == "valid"
    assert verification["matched_records"] == 4000
    assert verification["matched_images"] == 4000
    assert verification["matched_targets"] == 4000
    assert verification["declared_image_bytes"] == 1_523_324_154
    assert verification["matched_image_bytes"] == 1_523_324_154
    assert all(verification["checks"].values())
    assert verification["paid_resources_used"] is False


def test_qwen3b_training_order_keeps_failures_and_manifest_matches_files():
    first = _load(
        PACKAGE / "failures" / "attempt01_legacy_click_format" / "SUMMARY.json"
    )
    second = _load(
        PACKAGE / "failures" / "attempt02_empty_normalized_target" / "SUMMARY.json"
    )
    assert first["status"] == second["status"] == "invalid"
    assert first["manifest_written"] is second["manifest_written"] is False
    assert first["capability_claim_allowed"] is False
    assert second["capability_claim_allowed"] is False

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
