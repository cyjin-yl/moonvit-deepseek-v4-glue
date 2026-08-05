import json
from collections import Counter
from pathlib import Path

from moonvit_glue.grounding_evaluation import REQUIRED_CONDITIONS
from moonvit_glue.screenspot_contract import verify_manifest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "qwen2.5-3b-community-eval-v1.json"


def test_qwen3b_contract_uses_pure_text_model_and_keeps_4096_boundary():
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    proxy = contract["proxy_model"]
    assert proxy["repo"] == "Qwen/Qwen2.5-3B-Instruct"
    assert proxy["resolved_revision"] == "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    assert proxy["architecture"] == "Qwen2ForCausalLM"
    assert proxy["has_vision_config"] is False
    assert proxy["hidden_size"] == 2048
    assert all(item["sha256"] for item in proxy["files"])

    assert contract["canonical_projector"]["output_width"] == 4096
    initialization = contract["canonical_projector"]["initialization_contract"]
    assert initialization["step0"]["weights_sha256"] == (
        "efd942e0d8cbece08d3a8fab5d192eb4a2772211817ac5e733aa0f55aebb06b0"
    )
    assert initialization["random_projector"]["weights_sha256"] == (
        "7bd4aacfa7cfbd3ba9e44337d873a66a5b2bbd77636c30c36febc47a07ddfc44"
    )
    assert initialization["step0"]["seed"] != initialization["random_projector"]["seed"]
    assert initialization["publication"]["commit"] == (
        "65639da5988c0fc12d152bc68d3888b35d90a010"
    )
    assert initialization["publication"]["all_five_files_verified"] is True
    receiver = contract["qwen_proxy_receiver"]
    assert receiver["input_width"] == 4096
    assert receiver["output_width"] == 2048
    assert receiver["trainable_parameter_count"] == 0
    assert receiver["deepseek_action"] == "discard"

    runtime = contract["v100_runtime"]
    assert runtime["compute_capability"] == [7, 0]
    assert runtime["language_compute_dtype"] == "float16"
    assert runtime["projector_master_dtype"] == "float32"
    assert contract["training_budget"]["language_dtype"] == "float16"


def test_qwen3b_contract_locks_conditions_budgets_and_claim_seeds():
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert tuple(contract["evaluation_conditions"] + contract["checkpoint_roles"]) == (
        "vision",
        "blind",
        "shuffled",
        "step0",
        "random_projector",
        "previous_best",
        "current_candidate",
    )
    assert set(REQUIRED_CONDITIONS) == set(
        contract["evaluation_conditions"] + contract["checkpoint_roles"]
    )
    assert contract["training_budget"]["examples_seen_checkpoints"] == [
        4000,
        8000,
        16000,
        32000,
        64000,
    ]
    assert contract["training_budget"]["optimizer_steps_checkpoints"] == [
        500,
        1000,
        2000,
        4000,
        8000,
    ]
    assert contract["fairness"]["claim_seeds"] == [20260805, 20260806, 20260807]
    assert contract["paired_bootstrap"]["samples"] == 2000
    assert contract["paired_bootstrap"]["seed"] == 20260805


def test_frozen_screenspot_manifests_are_present_self_hashed_and_balanced():
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    full_path = ROOT / contract["datasets"]["screenspot_full"]["manifest"]
    glm_path = ROOT / contract["datasets"]["screenspot_glm50"]["manifest"]
    full = json.loads(full_path.read_text(encoding="utf-8"))
    glm = json.loads(glm_path.read_text(encoding="utf-8"))

    assert verify_manifest(full)
    assert verify_manifest(glm)
    assert full["manifest_sha256"] == contract["datasets"]["screenspot_full"][
        "manifest_sha256"
    ]
    assert glm["manifest_sha256"] == contract["datasets"]["screenspot_glm50"][
        "manifest_sha256"
    ]
    assert len(full["samples"]) == 1272
    assert len(glm["samples"]) == 50
    assert glm["label"] == "GLM-format metric-aligned public subset"
    counts = Counter((row["platform"], row["target_type"]) for row in glm["samples"])
    assert len(counts) == 10
    assert set(counts.values()) == {5}


def test_frozen_proxy_receiver_artifact_matches_contract_hashes():
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    receiver = contract["qwen_proxy_receiver"]
    root = ROOT / receiver["artifact"]
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    weights = next(item for item in manifest["files"] if item["path"].endswith(".safetensors"))
    assert manifest["trainable_parameter_count"] == 0
    assert manifest["permutation_is_valid"]
    assert weights["sha256"] == receiver["buffer_sha256"]


def test_frozen_projector_initializations_are_self_hashed_and_match_contract():
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    spec = contract["canonical_projector"]["initialization_contract"]
    manifest = json.loads((ROOT / spec["manifest"]).read_text(encoding="utf-8"))
    assert verify_manifest(manifest)
    assert manifest["manifest_sha256"] == spec["manifest_sha256"]
    by_role = {role["role"]: role for role in manifest["roles"]}
    for role_name in ("step0", "random_projector"):
        role = by_role[role_name]
        expected = spec[role_name]
        assert role["seed"] == expected["seed"]
        assert role["tensor_state_sha256"] == expected["tensor_state_sha256"]
        weights = next(
            item for item in role["files"] if item["path"].endswith("projector.safetensors")
        )
        assert weights["bytes"] == expected["weights_bytes"]
        assert weights["sha256"] == expected["weights_sha256"]


def test_language_retention_manifest_is_frozen_before_results():
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    spec = contract["datasets"]["language_retention"]
    manifest = json.loads((ROOT / spec["manifest"]).read_text(encoding="utf-8"))
    assert verify_manifest(manifest)
    assert manifest["manifest_sha256"] == spec["manifest_sha256"]
    assert manifest["selection"]["mmlu_pro"]["count"] == 140
    assert set(manifest["selection"]["mmlu_pro"]["category_counts"].values()) == {10}
    assert manifest["selection"]["gsm8k"]["count"] == 100
    assert manifest["selection"]["total_count"] == 240
