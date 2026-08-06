import hashlib
import json
from pathlib import Path

from tools.verify_qwen3b_projector_structure import effective_output_norm


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "qwen3b-projector-structure-screen-v1.json"
VARIANTS = (
    ROOT
    / "experiments"
    / "qwen3b_community_eval_20260805"
    / "projector_structure_screen_v1"
    / "initializations"
)
PREREG = (
    ROOT
    / "experiments"
    / "qwen3b_community_eval_20260805"
    / "projector_structure_screen_v1"
    / "PREREGISTRATION.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_structure_contract_is_frozen_and_migration_safe():
    contract = _load(CONTRACT)
    assert contract["frozen_before_any_structure_result"] is True
    assert contract["base_projector"]["canonical_output_width"] == 4096
    assert contract["base_projector"]["same_weights_for_all_arms"] is True
    assert contract["boundary"]["affine_parameters"] is False
    assert contract["boundary"]["projector_parameter_count_unchanged"] is True
    assert contract["budget"]["auto_stop_and_rollback"] is True
    assert contract["selection"]["final_20_step_ce_ratio_to_baseline_max"] == 1.25
    assert contract["paid_resources_used"] is False
    assert contract["final_half_scored"] is False


def test_structure_variant_configs_match_frozen_hashes():
    contract = _load(CONTRACT)
    arms = {row["name"]: row for row in contract["arms"]}
    for arm_name, directory in (
        ("post_layernorm", VARIANTS / "layernorm"),
        ("post_rmsnorm", VARIANTS / "rmsnorm"),
    ):
        config_path = directory / "projector_config.json"
        config = _load(config_path)
        assert config["output_norm"] == arms[arm_name]["output_norm"]
        assert _sha256(config_path) == arms[arm_name]["config_sha256"]
        assert config["language_width"] == 4096
        assert config["vision_width"] == 1024


def test_legacy_step0_config_defaults_to_no_output_norm():
    assert effective_output_norm({"language_width": 4096}) == "none"
    assert effective_output_norm({"output_norm": "layernorm"}) == "layernorm"


def test_structure_preregistration_binds_contract_and_source():
    contract = _load(CONTRACT)
    prereg = _load(PREREG)
    drift = _load(
        PREREG.parent / "PREREGISTRATION_SOURCE_DRIFT_20260807.json"
    )
    assert prereg["frozen_before_any_structure_result"] is True
    assert prereg["contract_sha256"] == _sha256(CONTRACT)
    source = ROOT / prereg["runner_source"]
    if _sha256(source) != prereg["runner_source_sha256"]:
        current = drift["files"][prereg["runner_source"]]
        assert current["frozen_sha256"] == prereg["runner_source_sha256"]
        assert current["current_sha256"] == _sha256(source)
    else:
        assert prereg["runner_source_sha256"] == _sha256(source)
    assert prereg["arms"] == [row["name"] for row in contract["arms"]]
    assert prereg["paid_resources_used"] is False
    assert prereg["final_half_scored"] is False


def test_pre_result_structure_test_failure_is_archived_without_gpu_result():
    failure_root = (
        ROOT
        / "experiments"
        / "qwen3b_community_eval_20260805"
        / "projector_structure_screen_v1"
        / "failures"
        / "attempt01_rmsnorm_test_tolerance"
    )
    failure = _load(failure_root / "FAILURE.json")
    repair = _load(failure_root / "REPAIR_RECORD.json")
    assert failure["gpu_started"] is False
    assert failure["optimizer_step_created"] is False
    assert failure["checkpoint_or_capability_result_created"] is False
    assert repair["contract_changed"] is False
    assert repair["screen_budget_changed"] is False


def test_pre_result_verifier_failure_is_archived_without_gpu_result():
    failure_root = (
        ROOT
        / "experiments"
        / "qwen3b_community_eval_20260805"
        / "projector_structure_screen_v1"
        / "failures"
        / "attempt02_legacy_config_verifier"
    )
    failure = _load(failure_root / "FAILURE.json")
    repair = _load(failure_root / "REPAIR_RECORD.json")
    assert failure["gpu_started"] is False
    assert failure["optimizer_step_created"] is False
    assert failure["checkpoint_or_capability_result_created"] is False
    assert repair["contract_changed"] is False
    assert repair["screen_budget_changed"] is False
