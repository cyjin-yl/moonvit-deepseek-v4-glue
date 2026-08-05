import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "experiments"
    / "qwen3b_community_eval_20260805"
    / "training_order_grounding_enriched_4k_v1"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def test_grounding_enriched_order_is_exact_pre_result_selection():
    manifest = _load(PACKAGE / "MANIFEST.json")
    verification = _load(PACKAGE / "INDEPENDENT_VERIFICATION.json")

    assert manifest["manifest_sha256"] == (
        "d632ecc2c9bc216a552f240e87b9733904b67dcfe30c489a62ba03df25370bf1"
    )
    assert manifest["records_sha256"] == (
        "f3c3dec199a30927fb715b2d4fbc890baa8e3f3a456b56707f46490164b915ab"
    )
    assert manifest["selection"] == {
        "effective_epochs": 4000 / 59198,
        "effective_epochs_denominator": 59198,
        "examples_seen": 4000,
        "gradient_accumulation": 8,
        "grounding_examples": 2000,
        "holdout_removed": False,
        "merge_rule": "alternate_grounding_then_short_answer",
        "micro_batch_size": 1,
        "optimizer_steps": 500,
        "real_global_batch": 8,
        "rule": "first_n_per_route_alternate_grounding_then_short_answer",
        "short_answer_examples": 2000,
        "shuffle": False,
        "subset_passes": 1.0,
        "within_route_order": "frozen_source_order",
    }
    assert manifest["source_counts"] == {
        "docvqa_train": 649,
        "showui_desktop": 2000,
        "textvqa_train": 1080,
        "train": 271,
    }
    assert manifest["prompt_route_counts"] == {
        "grounding": 2000,
        "short_answer": 2000,
    }
    assert [row["prompt_route"] for row in manifest["records"]] == [
        route
        for _ in range(2000)
        for route in ("grounding", "short_answer")
    ]
    for route in ("grounding", "short_answer"):
        source_indices = [
            row["source_row_index"]
            for row in manifest["records"]
            if row["prompt_route"] == route
        ]
        assert source_indices == sorted(source_indices)

    assert verification["status"] == "valid"
    assert verification["matched_records"] == 4000
    assert verification["matched_images"] == 4000
    assert verification["matched_targets"] == 4000
    assert verification["declared_image_bytes"] == 1_255_969_179
    assert verification["matched_image_bytes"] == 1_255_969_179
    assert all(verification["checks"].values())
    assert verification["paid_resources_used"] is False


def test_grounding_enriched_order_remote_and_package_hashes():
    remote = _load(PACKAGE / "REMOTE_ARTIFACT_MANIFEST.json")
    assert remote["runner_git_sha"] == (
        "c43c161a084b9446d35da12ad667d2fe42e4f3a7"
    )
    assert remote["training_results_exist"] is False
    assert remote["final_half_scored"] is False
    assert remote["paid_resources_used"] is False
    for relative, declared in remote["files"].items():
        path = PACKAGE / relative
        assert path.stat().st_size == declared["bytes"]
        assert _sha256(path) == declared["sha256"]

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
