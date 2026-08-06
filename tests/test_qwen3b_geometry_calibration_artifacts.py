import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = (
    ROOT
    / "experiments"
    / "qwen3b_community_eval_20260805"
    / "geometry_repair_screen_v1"
    / "calibration"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_calibration_manifest_rehashes_raw_result_and_failure_record():
    manifest = _load(CALIBRATION / "ARTIFACT_MANIFEST.json")

    assert manifest["file_count"] == 8
    assert manifest["total_bytes"] == 541_891
    assert manifest["final_half_scored"] is False
    for relative, expected in manifest["files"].items():
        path = CALIBRATION / relative
        assert path.stat().st_size == expected["bytes"]
        assert _sha256(path) == expected["sha256"]


def test_calibration_is_independently_verified_and_fixes_lambda_values():
    summary = _load(CALIBRATION / "SUMMARY.json")
    verification = _load(CALIBRATION / "INDEPENDENT_VERIFICATION.json")

    assert summary["status"] == "valid"
    assert summary["formal_calibration_complete"] is True
    assert verification["status"] == "verified"
    assert verification["geometry"] == summary["geometry"]
    assert verification["per_parameter_gradients"] == summary[
        "per_parameter_gradients"
    ]
    assert summary["unweighted_auxiliary_gradient_norm"] == pytest.approx(
        3.8781849596907043
    )
    assert summary["recorded_ce_gradient_norm_before_clip"] == pytest.approx(
        0.7901650667190552
    )
    assert summary["derived_arms"] == {
        "control": {"target_gradient_ratio": 0.0, "lambda": 0.0},
        "ratio005": {
            "target_gradient_ratio": 0.05,
            "lambda": pytest.approx(0.01018730507868909),
        },
        "ratio020": {
            "target_gradient_ratio": 0.2,
            "lambda": pytest.approx(0.04074922031475636),
        },
        "ratio080": {
            "target_gradient_ratio": 0.8,
            "lambda": pytest.approx(0.16299688125902545),
        },
    }
    assert summary["capability_claim_allowed"] is False
    assert summary["visual_ability_established"] is False
    assert summary["paid_resources_used"] is False
    assert summary["final_half_scored"] is False


def test_failed_log_pipeline_is_preserved_without_replacing_gpu_result():
    failure = _load(
        CALIBRATION
        / "failures"
        / "attempt01_tee_before_directory"
        / "FAILURE.json"
    )

    assert failure["status"] == "orchestration_failure_preserved"
    assert failure["exit_code"] == 1
    assert failure["rerun_required"] is False
    assert failure["paid_resources_used"] is False
    assert failure["final_half_scored"] is False
