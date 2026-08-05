import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "qwen3b_community_eval_20260805"
TRAINING = EXPERIMENT / "training_4k_v1"
GROUNDING = EXPERIMENT / "screenspot_glm50_4k_v1"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def test_qwen3b_fixed_budget_training_is_exact_and_independently_verified():
    summary = _load(TRAINING / "SUMMARY.json")
    verification = _load(TRAINING / "INDEPENDENT_VERIFICATION.json")

    assert summary["status"] == "valid"
    assert summary["formal_training_complete"] is True
    assert summary["optimizer_steps"] == 500
    assert summary["examples_seen"] == 4000
    assert summary["answer_tokens_seen"] == 21_532
    assert summary["micro_batch_size"] == 1
    assert summary["gradient_accumulation"] == 8
    assert summary["real_global_batch"] == 8
    assert summary["qwen_parameter_count"] == 3_085_938_688
    assert summary["qwen_trainable_parameter_count"] == 0
    assert summary["projector_parameter_count"] == 33_564_672
    assert summary["peak_gpu_memory_bytes"] == 8_979_616_768
    assert summary["visual_ability_established"] is False
    assert summary["capability_claim_allowed"] is False

    assert verification["status"] == "verified"
    assert verification["runner_git_sha"] == (
        "97e9c03403b2673e68f2ed7fc421630add8a9d3a"
    )
    assert verification["verifier_git_sha"] == (
        "075f3e5889aa445a9ca748bb8ecfc21ba96abacc"
    )
    assert len(verification["checkpoints"]) == 5
    assert verification["checkpoint_total_bytes"] == 2_351_006_545
    assert verification["final_training_state_step"] == 500
    assert verification["final_optimizer_parameter_states"] == 6
    assert verification["final_projector_sha256"] == (
        "566830f3b6f85f5aa66b13566054022bcffce3660d5b2210fc5ee192834ca89f"
    )


def test_qwen3b_glm50_rejects_the_4k_candidate_on_causal_controls():
    scores = _load(GROUNDING / "scoring" / "SUMMARY.json")
    decision = _load(GROUNDING / "DECISION.json")
    vision = scores["conditions"]["vision"]["breakdowns"]["overall"]
    blind = scores["conditions"]["blind"]["breakdowns"]["overall"]
    step0 = scores["conditions"]["step0"]["breakdowns"]["overall"]
    vision_blind = scores["comparisons"]["vision-minus-blind"]["metrics"]
    candidate_step0 = scores["comparisons"][
        "current-candidate-minus-previous-best"
    ]["metrics"]

    assert scores["formal_complete"] is True
    assert scores["sample_count"] == 50
    assert vision["parse_rate"] == 0.96
    assert vision["accuracy_at_50"]["all_accuracy"] == 0.02
    assert vision["accuracy_at_100"]["all_accuracy"] == 0.04
    assert vision["accuracy_at_200"]["all_accuracy"] == 0.16
    assert vision["click_in_box_accuracy"]["all_accuracy"] == 0.04
    assert vision["center_distance"]["all_penalized"]["mean"] == (
        554.5317116043587
    )
    assert blind["click_in_box_accuracy"]["all_accuracy"] == 0.12
    assert step0["click_in_box_accuracy"]["all_accuracy"] == 0.10

    assert vision_blind["click_in_box_all"]["improvement"] == (
        -0.07999999999999999
    )
    assert vision_blind["mean_center_distance_all_penalized"]["ci95"] == [
        -246.69635950689488,
        -89.23801476100289,
    ]
    assert candidate_step0["mean_center_distance_all_penalized"]["ci95"] == [
        -246.73778344855054,
        -75.49653441259227,
    ]
    assert decision["decision"] == "reject_current_candidate"
    assert decision["previous_best_after_decision"] == "step0"
    assert decision["real_grounding_improvement_rule_passed"] is False
    assert decision["checkpoint_enters_deepseek_candidate_list"] is False


def test_qwen3b_4k_packages_preserve_failures_and_match_curated_manifests():
    failure = _load(TRAINING / "failures" / "verifier_attempt01" / "SUMMARY.json")
    assert failure["status"] == "invalid"
    assert failure["exception_type"] == "KeyError"
    assert failure["accepted_verification_written"] is False
    assert failure["training_artifacts_modified"] is False
    assert failure["retry_status"] == "verified"

    for package in (TRAINING, GROUNDING):
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
