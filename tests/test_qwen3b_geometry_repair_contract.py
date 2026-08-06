import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from calibrate_qwen3b_geometry_regularization import (  # noqa: E402
    SUMMARY_BINDING_KEYS,
    calibration_summary_bindings,
)
CONFIG = ROOT / "configs" / "qwen3b-geometry-repair-screen-v1.json"
PACKAGE = (
    ROOT
    / "experiments"
    / "qwen3b_community_eval_20260805"
    / "geometry_repair_screen_v1"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_geometry_screen_budget_lambda_derivation_and_selection_are_frozen():
    config = _load(CONFIG)
    preregistration = _load(PACKAGE / "PREREGISTRATION.json")

    assert config["frozen_before_calibration_and_screen"] is True
    assert config["calibration"]["target_gradient_ratios"] == [0.05, 0.2, 0.8]
    assert config["screen"]["optimizer_steps"] == 100
    assert config["screen"]["examples_seen"] == 800
    assert [row["name"] for row in config["screen"]["arms"]] == [
        "control",
        "ratio005",
        "ratio020",
        "ratio080",
    ]
    assert config["screen"]["representation_guard"] == {
        "requires_all": True,
        "current_over_step0_relative_spread_at_least": 0.25,
        "current_over_step0_effective_rank_at_least": 0.5,
    }
    assert preregistration["calibration"]["lambda_values_known_at_freeze"] is False
    assert preregistration["calibration_result_exists_at_freeze"] is False
    assert preregistration["screen_result_exists_at_freeze"] is False
    assert preregistration["paid_resources_used"] is False
    assert preregistration["final_half_scored"] is False


def test_geometry_preregistration_binds_runtime_source_bytes():
    preregistration = _load(PACKAGE / "PREREGISTRATION.json")
    for relative, expected in preregistration["source_files"].items():
        path = ROOT / relative
        assert path.stat().st_size == expected["bytes"]
        assert _sha256(path) == expected["sha256"]


def test_calibration_summary_exposes_every_trainer_binding():
    run_config = {key: f"value-{index}" for index, key in enumerate(SUMMARY_BINDING_KEYS)}

    assert calibration_summary_bindings(run_config) == run_config


def test_pre_result_binding_failure_is_archived_with_no_training_result():
    failure_root = PACKAGE / "failures" / "attempt01_calibration_binding"
    failure = _load(failure_root / "FAILURE.json")
    manifest = _load(failure_root / "ARTIFACT_MANIFEST.json")

    assert failure["status"] == "failed"
    assert failure["stage"] == "tokenizer_config_supervision"
    assert failure["paid_resources_used"] is False
    assert failure["final_half_scored"] is False
    assert manifest["file_count"] == 7
    assert manifest["final_half_scored"] is False
    for relative, expected in manifest["files"].items():
        path = failure_root / relative
        assert path.stat().st_size == expected["bytes"]
        assert _sha256(path) == expected["sha256"]


def test_test_collection_failure_is_archived():
    failure_root = PACKAGE / "failures" / "attempt02_test_import"
    failure = _load(failure_root / "FAILURE.json")
    manifest = _load(failure_root / "ARTIFACT_MANIFEST.json")

    assert failure["status"] == "test_collection_failure_preserved"
    assert failure["checkpoint_or_capability_result_created"] is False
    assert manifest["file_count"] == 2
    for relative, expected in manifest["files"].items():
        path = failure_root / relative
        assert path.stat().st_size == expected["bytes"]
        assert _sha256(path) == expected["sha256"]
