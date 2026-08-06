import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "experiments"
    / "qwen3b_community_eval_20260805"
    / "representation_trajectory_v1"
)
FORMAL = PACKAGE / "formal"
RETENTION_FORMAL = (
    ROOT
    / "experiments"
    / "qwen3b_community_eval_20260805"
    / "representation_retention_v1"
    / "formal"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_manifest_rehashes_complete_raw_trajectory():
    manifest = _load(FORMAL / "ARTIFACT_MANIFEST.json")

    assert manifest["file_count"] == 10
    assert manifest["total_bytes"] == 20_662_737
    assert manifest["final_half_scored"] is False
    for relative, expected in manifest["files"].items():
        path = FORMAL / relative
        assert path.stat().st_size == expected["bytes"]
        assert _sha256(path) == expected["sha256"]


def test_package_manifest_rehashes_result_contract_chart_and_failure_record():
    manifest = _load(PACKAGE / "ARTIFACT_MANIFEST.json")

    assert manifest["file_count"] == 17
    assert manifest["total_bytes"] == 20_683_664
    assert manifest["final_half_scored"] is False
    for relative, expected in manifest["files"].items():
        path = PACKAGE / relative
        assert path.stat().st_size == expected["bytes"]
        assert _sha256(path) == expected["sha256"]

    full_suite = (FORMAL / "full_suite_final.log").read_text(encoding="utf-8")
    assert "full_suite_collected=361" in full_suite
    assert "full_suite_passed=361" in full_suite
    assert "full_suite_failed=0" in full_suite


def test_registered_onset_is_first_saved_trained_checkpoint():
    summary = _load(FORMAL / "SUMMARY.json")
    verification = _load(FORMAL / "INDEPENDENT_VERIFICATION.json")
    decision = _load(PACKAGE / "DECISION.json")

    assert summary["status"] == "valid"
    assert summary["formal_analysis_complete"] is True
    assert summary["condition_names"] == [
        "step0",
        "step100",
        "step200",
        "step300",
        "step400",
        "step500",
    ]
    assert summary["registered_onset"]["onset_step"] == 100
    assert summary["registered_onset"]["onset_examples_seen"] == 800
    assert summary["registered_onset"]["last_precollapse_step"] == 0
    assert summary["registered_action"] == decision["registered_action"]
    assert verification["status"] == "verified"
    assert verification["pooled_tensor_count"] == 13
    assert verification["pairwise_rows"] == 15_925
    assert verification["per_sample_rows"] == 50
    assert verification["training_history_rows"] == 500
    assert summary["visual_ability_established"] is False
    assert summary["paid_resources_used"] is False
    assert summary["final_half_scored"] is False


def test_step100_collapse_precedes_loss_plateau():
    summary = _load(FORMAL / "SUMMARY.json")
    projector = summary["stage_summaries"]["projector_4096"]
    onset = summary["collapse_onsets"]["projector_4096"]

    assert onset["onset_step"] == 100
    assert onset["decisions"]["step100"]["relative_spread_ratio"] == pytest.approx(
        0.12984915495775887
    )
    assert onset["decisions"]["step100"]["effective_rank_ratio"] == pytest.approx(
        0.07720574719395511
    )
    assert projector["step100"]["representation"]["sample_rms"] == pytest.approx(
        35.740097890246744
    )
    assert projector["step100"]["representation"][
        "top1_variance_fraction"
    ] == pytest.approx(0.9875544517314816)
    assert summary["training_windows"]["step100"]["loss"]["mean"] == pytest.approx(
        3.916363775571808
    )


def test_step500_exactly_reproduces_package15n_endpoint():
    trajectory = _load(FORMAL / "SUMMARY.json")
    retention = _load(RETENTION_FORMAL / "SUMMARY.json")

    for stage in ("projector_4096", "fixed_receiver_2048"):
        assert trajectory["stage_summaries"][stage]["step500"] == retention[
            "stage_summaries"
        ][stage]["current_candidate"]
        assert trajectory["geometry_comparisons"][stage]["step500"] == retention[
            "geometry_comparisons"
        ][stage]
