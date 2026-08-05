import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "qwen3b_community_eval_20260805"
FULL = EXPERIMENT / "screenspot_public_4k_v1"
PREFERENCE = EXPERIMENT / "screenspot_glm50_preference_4k_v1"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def test_complete_public_screenspot_rejects_the_4k_candidate():
    generation = _load(FULL / "evaluation" / "SUMMARY.json")
    scores = _load(FULL / "scoring" / "SUMMARY.json")
    decision = _load(FULL / "DECISION.json")
    remote = _load(FULL / "REMOTE_ARTIFACT_MANIFEST.json")
    vision = scores["conditions"]["vision"]["breakdowns"]["overall"]
    vision_blind = scores["comparisons"]["vision-minus-blind"]["metrics"]
    vision_shuffled = scores["comparisons"]["vision-minus-shuffled"]["metrics"]

    assert scores["formal_complete"] is True
    assert scores["sample_count"] == 1272
    assert vision["parse_count"] == 1227
    assert vision["parse_rate"] == 0.964622641509434
    assert vision["accuracy_at_50"]["all_accuracy"] == 0.01729559748427673
    assert vision["accuracy_at_100"]["all_accuracy"] == 0.04874213836477988
    assert vision["accuracy_at_200"]["all_accuracy"] == 0.1179245283018868
    assert vision["click_in_box_accuracy"]["all_accuracy"] == 0.026729559748427674
    assert vision["center_distance"]["all_penalized"]["mean"] == 565.1814817616176
    assert vision_blind["mean_center_distance_all_penalized"]["ci95"] == [
        -185.67552431259801,
        -154.17474650570574,
    ]
    assert vision_shuffled["click_in_box_all"]["ci95"] == [
        -0.007861635220125784,
        0.0062893081761006275,
    ]
    assert decision["decision"] == "reject_current_candidate"
    assert decision["real_grounding_improvement_rule_passed"] is False
    assert decision["checkpoint_enters_deepseek_candidate_list"] is False

    for condition, declared in generation["prediction_files"].items():
        path = FULL / "evaluation" / "predictions" / f"{condition}.jsonl"
        assert path.stat().st_size == declared["bytes"]
        assert _sha256(path) == declared["sha256"]
        assert _sha256(path) == scores["inputs"]["predictions"][condition]["sha256"]
    scorer_manifest = _load(FULL / "scoring" / "ARTIFACT_MANIFEST.json")
    for declared in scorer_manifest["files"]:
        path = FULL / "scoring" / declared["path"]
        assert path.stat().st_size == declared["bytes"]
        assert _sha256(path) == declared["sha256"]
    assert remote["evaluation_summary_sha256"] == _sha256(
        FULL / "evaluation" / "SUMMARY.json"
    )
    assert remote["scoring_summary_sha256"] == _sha256(
        FULL / "scoring" / "SUMMARY.json"
    )
    assert remote["feature_cache_manifest_sha256"] == _sha256(
        FULL / "feature_cache" / "MANIFEST.json"
    )


def test_teacher_forced_preference_locates_image_agnostic_answer_learning():
    summary = _load(PREFERENCE / "formal" / "SUMMARY.json")
    decision = _load(PREFERENCE / "DECISION.json")
    remote = _load(PREFERENCE / "REMOTE_ARTIFACT_MANIFEST.json")
    conditions = summary["preference_summary"]["conditions"]
    comparisons = summary["comparisons"]

    assert summary["status"] == "valid"
    assert summary["formal_preference_complete"] is True
    assert summary["records"] == 50
    assert conditions["vision"]["breakdowns"]["overall"][
        "paired_preference_accuracy"
    ] == 0.46
    assert conditions["blind"]["breakdowns"]["overall"][
        "paired_preference_accuracy"
    ] == 0.56
    assert conditions["shuffled"]["breakdowns"]["overall"][
        "paired_preference_accuracy"
    ] == 0.52
    assert comparisons["vision-minus-shuffled"]["metrics"][
        "mean_correct_margin"
    ]["ci95"] == [-0.012874255196952114, -0.00185716262885502]
    assert comparisons["current-candidate-minus-previous-best"]["metrics"][
        "mean_correct_token_nll"
    ]["ci95"] == [1.1371311920422777, 1.438924557388047]
    assert decision["decision"] == "no_content_specific_visual_readout"
    assert decision["generation_only_failure_supported"] is False
    assert decision["format_and_coordinate_prior_learning_supported"] is True
    assert decision["extend_same_stream_supported"] is False
    for condition, declared in summary["preference_files"].items():
        path = PREFERENCE / "formal" / "preferences" / f"{condition}.jsonl"
        assert path.stat().st_size == declared["bytes"]
        assert _sha256(path) == declared["sha256"]
    assert remote["formal_summary_sha256"] == _sha256(
        PREFERENCE / "formal" / "SUMMARY.json"
    )
    assert remote["formal_run_config_sha256"] == _sha256(
        PREFERENCE / "formal" / "RUN_CONFIG.json"
    )


def test_full_grounding_packages_preserve_failures_and_match_manifests():
    resume_failure = _load(FULL / "failures" / "resume_attempt" / "FAILURE.json")
    alias_failure = _load(
        PREFERENCE / "failures" / "alias_summary_attempt" / "FAILURE.json"
    )
    assert resume_failure["stage"] == "cache_and_frozen_file_verification"
    assert resume_failure["exception_type"] == "FileNotFoundError"
    assert alias_failure["stage"] == "paired_bootstrap"
    assert alias_failure["exception_type"] == "ValueError"

    for package in (FULL, PREFERENCE):
        artifact_manifest_path = package / "ARTIFACT_MANIFEST.json"
        artifact_manifest = _load(artifact_manifest_path)
        actual = {
            path.relative_to(package).as_posix(): path
            for path in package.rglob("*")
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
