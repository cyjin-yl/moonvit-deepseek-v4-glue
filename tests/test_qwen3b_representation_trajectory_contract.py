import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "qwen3b-representation-trajectory-v1.json"
PACKAGE = (
    ROOT
    / "experiments"
    / "qwen3b_community_eval_20260805"
    / "representation_trajectory_v1"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_trajectory_schedule_and_onset_rule_are_frozen_before_result():
    config = _load(CONFIG)
    preregistration = _load(PACKAGE / "PREREGISTRATION.json")

    assert config["frozen_before_result"] is True
    assert [row["step"] for row in config["conditions"]] == [0, 100, 200, 300, 400, 500]
    assert [row["examples_seen"] for row in config["conditions"]] == [0, 800, 1600, 2400, 3200, 4000]
    assert config["decision_stage"] == "fixed_receiver_2048"
    assert config["gross_collapse_rule"] == {
        "requires_all": True,
        "current_over_step0_relative_spread_below": 0.25,
        "current_over_step0_effective_rank_below": 0.5,
    }
    assert preregistration["known_endpoint_evidence_at_freeze"]["step"] == 500
    assert preregistration["trajectory_result_exists_at_freeze"] is False
    assert preregistration["paid_resources_used"] is False
    assert preregistration["final_half_scored"] is False


def test_preregistration_binds_runtime_source_bytes():
    preregistration = _load(PACKAGE / "PREREGISTRATION.json")
    for relative, expected in preregistration["source_files"].items():
        path = ROOT / relative
        assert path.stat().st_size == expected["bytes"]
        assert _sha256(path) == expected["sha256"]


def test_pre_result_implementation_repair_preserves_registered_decision():
    repair = _load(PACKAGE / "PRE_RESULT_IMPLEMENTATION_REPAIR.json")
    failure_log = PACKAGE / repair["failure_log"]["path"]

    assert failure_log.stat().st_size == repair["failure_log"]["bytes"]
    assert _sha256(failure_log) == repair["failure_log"]["sha256"]
    assert repair["checkpoint_schedule_changed"] is False
    assert repair["gross_collapse_thresholds_changed"] is False
    assert repair["onset_rule_changed"] is False
    assert repair["gpu_analysis_started_before_repair"] is False
    assert repair["trajectory_result_existed_before_repair"] is False
