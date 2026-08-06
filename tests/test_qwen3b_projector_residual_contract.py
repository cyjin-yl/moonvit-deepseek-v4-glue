import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "qwen3b-projector-residual-screen-v1.json"
PACKAGE = ROOT / "experiments" / "qwen3b_community_eval_20260805" / "projector_residual_screen_v1"
PREREG = PACKAGE / "PREREGISTRATION.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_residual_contract_is_frozen_and_migration_safe():
    contract = _load(CONTRACT)
    assert contract["frozen_before_any_residual_result"] is True
    assert contract["base_projector"]["canonical_output_width"] == 4096
    assert contract["boundary"]["zero_init_output_equals_step0"] is True
    assert contract["boundary"]["gated_output_equals_step0"] is True
    assert contract["budget"]["auto_stop_and_rollback"] is True
    assert contract["selection"]["final_20_step_ce_ratio_to_baseline_max"] == 1.25
    assert contract["paid_resources_used"] is False
    assert contract["final_half_scored"] is False


def test_residual_variant_configs_and_preregistration_hashes_are_bound():
    contract = _load(CONTRACT)
    arms = {row["name"]: row for row in contract["arms"]}
    for arm_name, directory in (
        ("zero_init_residual", PACKAGE / "initializations" / "zero_init_residual"),
        ("gated_residual", PACKAGE / "initializations" / "gated_residual"),
    ):
        config_path = directory / "projector_config.json"
        config = _load(config_path)
        assert config["residual_mode"] == arms[arm_name]["residual_mode"]
        assert _sha256(config_path) == arms[arm_name]["config_sha256"]
        assert config["language_width"] == 4096
        assert config["vision_width"] == 1024
    prereg = _load(PREREG)
    assert prereg["frozen_before_any_residual_result"] is True
    assert prereg["contract_sha256"] == _sha256(CONTRACT)
    assert prereg["runner_source_sha256"] == _sha256(ROOT / prereg["runner_source"])
    assert prereg["initialization_tool_sha256"] == _sha256(ROOT / prereg["initialization_tool"])
    assert prereg["verifier_tool_sha256"] == _sha256(ROOT / prereg["verifier_tool"])
    assert prereg["arms"] == [row["name"] for row in contract["arms"]]
    assert prereg["paid_resources_used"] is False


def test_scalar_hash_failure_is_archived_without_gpu_result():
    failure_root = PACKAGE / "failures" / "attempt01_scalar_gate_hash"
    failure = _load(failure_root / "FAILURE.json")
    repair = _load(failure_root / "REPAIR_RECORD.json")
    assert failure["gpu_started"] is False
    assert failure["optimizer_step_created"] is False
    assert failure["checkpoint_or_capability_result_created"] is False
    assert repair["contract_changed"] is False
    assert repair["screen_budget_changed"] is False


def test_initialization_pointer_is_complete_and_bound():
    pointer = _load(PACKAGE / "INITIALIZATION_ARTIFACT_POINTER.json")
    assert pointer["complete_copy"] is True
    assert pointer["file_count"] == 7
    assert pointer["total_bytes"] == 402740644
    assert pointer["manifest_sha256"] == "533b281af2f3b43009a006a60b04effc23eb57147e99e89d75f2226efd061d85"
    assert pointer["paid_resources_used"] is False
