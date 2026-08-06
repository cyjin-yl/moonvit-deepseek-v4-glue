import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "qwen3b-representation-retention-v1.json"
PACKAGE = (
    ROOT
    / "experiments"
    / "qwen3b_community_eval_20260805"
    / "representation_retention_v1"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_representation_retention_rule_is_frozen_before_results():
    config = _load(CONFIG)
    preregistration = _load(PACKAGE / "PREREGISTRATION.json")
    assert config["frozen_before_result"] is True
    assert config["dataset_name"] == "screenspot_glm50_v1"
    assert config["sample_count"] == 50
    assert config["conditions"] == ["step0", "current_candidate"]
    assert config["decision_stage"] == "fixed_receiver_2048"
    assert config["gross_collapse_rule"] == {
        "requires_all": True,
        "current_over_step0_relative_spread_below": 0.25,
        "current_over_step0_effective_rank_below": 0.5,
    }
    assert "no image-only target-coordinate probe" in config["interpretation_boundary"]
    assert preregistration["result_exists_at_freeze"] is False
    assert preregistration["paid_resources_used"] is False
    assert preregistration["final_half_scored"] is False


def test_preregistration_and_post_result_repair_bind_runtime_source_bytes():
    preregistration = _load(PACKAGE / "PREREGISTRATION.json")
    repair = _load(PACKAGE / "POST_RESULT_VERIFIER_REPAIR.json")
    for relative, expected in preregistration["source_files"].items():
        path = ROOT / relative
        if relative == "tools/verify_qwen3b_representation_retention.py":
            assert repair["failed_verifier"] == {
                **expected,
                "failure": repair["failed_verifier"]["failure"],
            }
            assert path.stat().st_size == repair["corrected_verifier"]["bytes"]
            assert _sha256(path) == repair["corrected_verifier"]["sha256"]
        else:
            assert path.stat().st_size == expected["bytes"]
            assert _sha256(path) == expected["sha256"]
    assert repair["analysis_source_changed"] is False
    assert repair["analysis_result_changed"] is False
    assert repair["decision_rule_changed"] is False


def test_preresult_package_manifest_rehashes_every_declared_artifact():
    manifest = _load(PACKAGE / "ARTIFACT_MANIFEST.json")
    assert manifest["final_half_scored"] is False
    assert manifest["file_count"] == 17
    assert manifest["total_bytes"] == sum(
        expected["bytes"] for expected in manifest["files"].values()
    )
    for relative, expected in manifest["files"].items():
        path = PACKAGE / relative
        assert path.stat().st_size == expected["bytes"]
        assert _sha256(path) == expected["sha256"]
