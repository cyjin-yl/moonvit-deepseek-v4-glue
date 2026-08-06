import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "experiments"
    / "qwen3b_community_eval_20260805"
    / "representation_retention_v1"
)
FORMAL = PACKAGE / "formal"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_manifest_rehashes_complete_raw_result():
    manifest = _load(FORMAL / "ARTIFACT_MANIFEST.json")
    assert manifest["file_count"] == 10
    assert manifest["total_bytes"] == 8_105_867
    assert manifest["final_half_scored"] is False
    for relative, expected in manifest["files"].items():
        path = FORMAL / relative
        assert path.stat().st_size == expected["bytes"]
        assert _sha256(path) == expected["sha256"]


def test_registered_action_matches_independently_recomputed_collapse():
    summary = _load(FORMAL / "SUMMARY.json")
    verification = _load(FORMAL / "INDEPENDENT_VERIFICATION.json")
    decision = _load(PACKAGE / "DECISION.json")

    assert summary["status"] == "valid"
    assert summary["formal_analysis_complete"] is True
    assert summary["sample_count"] == 50
    assert verification["status"] == "verified"
    assert verification["pooled_tensor_count"] == 5
    assert verification["pairwise_rows"] == 6_125
    assert verification["per_sample_rows"] == 50
    assert summary["registered_action"] == verification["registered_action"]
    assert decision["registered_action"] == summary["registered_action"]

    projector = summary["decisions"]["projector_4096"]
    receiver = summary["decisions"]["fixed_receiver_2048"]
    assert projector["gross_collapse"] is True
    assert receiver["gross_collapse"] is True
    assert projector["relative_spread_ratio"] == pytest.approx(0.13839551812134085)
    assert projector["effective_rank_ratio"] == pytest.approx(0.08585614520107673)
    assert receiver["relative_spread_ratio"] == pytest.approx(0.1372408079249148)
    assert receiver["effective_rank_ratio"] == pytest.approx(0.08457489147863204)
    assert summary["visual_ability_established"] is False
    assert summary["paid_resources_used"] is False
    assert summary["final_half_scored"] is False


def test_result_localizes_common_direction_collapse_before_receiver():
    summary = _load(FORMAL / "SUMMARY.json")
    projector = summary["stage_summaries"]["projector_4096"]
    step0 = projector["step0"]["representation"]
    current = projector["current_candidate"]["representation"]

    assert step0["effective_rank_participation"] == pytest.approx(13.279701449469266)
    assert current["effective_rank_participation"] == pytest.approx(1.1401439758725824)
    assert step0["top1_variance_fraction"] == pytest.approx(0.1747579506841463)
    assert current["top1_variance_fraction"] == pytest.approx(0.9345874604527952)
    assert current["sample_rms"] > 700 * step0["sample_rms"]
    assert (
        projector["current_candidate"]["mean_within_image_rms"]
        > 100 * projector["step0"]["mean_within_image_rms"]
    )

    projector_geometry = summary["geometry_comparisons"]["projector_4096"]
    receiver_geometry = summary["geometry_comparisons"]["fixed_receiver_2048"]
    assert projector_geometry["linear_cka"] == pytest.approx(0.435616695301796)
    assert receiver_geometry["linear_cka"] == pytest.approx(0.4277521079263767)


def test_driver_recovery_was_scoped_and_checksum_bound():
    recovery = _load(PACKAGE / "RUNTIME_RECOVERY.json")
    assert recovery["observed_before_run"]["loaded_kernel_module_version"] == "580.159.04"
    assert recovery["recovery"]["system_files_modified"] is False
    assert recovery["recovery"]["rebooted"] is False
    assert recovery["recovery"]["gpu_clients_stopped"] is False
    assert recovery["recovery"]["official_metadata_rpm_sha256_matched"] is True
    assert recovery["post_recovery_checks"]["torch_cuda_available"] is True
    assert recovery["post_recovery_checks"]["cuda_tensor_roundtrip"] is True


def test_failed_verifier_attempt_and_repair_are_preserved():
    failure = (FORMAL / "verification_failed_order_attempt_01.log").read_text(
        encoding="utf-8"
    )
    repair = _load(PACKAGE / "POST_RESULT_VERIFIER_REPAIR.json")
    assert "ValueError: pairwise geometry rows differ" in failure
    assert "exit_code=1" in failure
    assert repair["analysis_source_changed"] is False
    assert repair["analysis_result_changed"] is False
    assert repair["decision_rule_changed"] is False


def test_failed_cross_platform_manifest_attempt_and_repair_are_preserved():
    failure = (FORMAL / "full_suite_failed_crlf_attempt_01.log").read_text(
        encoding="utf-8"
    )
    repair = _load(PACKAGE / "POST_RESULT_MANIFEST_REPAIR.json")
    assert "assert 1438 == 1481" in failure
    assert repair["failed_full_suite"]["passed"] == 348
    assert repair["failed_full_suite"]["failed"] == 1
    assert repair["raw_result_changed"] is False
    assert repair["scientific_decision_changed"] is False

    final_log = (FORMAL / "full_suite_final.log").read_text(encoding="utf-8")
    assert "full_suite_collected=347" in final_log
    assert "full_suite_passed=347" in final_log
    assert "full_suite_failed=0" in final_log
