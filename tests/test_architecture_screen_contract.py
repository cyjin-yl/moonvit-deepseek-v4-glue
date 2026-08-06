import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_architecture_screen_binds_canonical_boundary_and_distinct_towers():
    path = ROOT / "configs/qwen2.5-3b-community-architecture-screen-v1.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    assert contract["schema_version"] == "qwen25-3b-community-architecture-screen-v1"
    assert contract["shared_boundary"] == {
        "canonical_projector_output_width": 4096,
        "qwen_receiver_input_width": 4096,
        "qwen_receiver_output_width": 2048,
        "receiver_is_parameter_free": True,
        "receiver_artifact": "experiments/qwen3b_community_eval_20260805/contract/proxy_receiver",
        "same_receiver_for_all_arms": True,
        "deepseek_action": "discard_qwen_receiver",
    }
    arms = contract["arms"]
    assert arms["local_v1_family_proxy"]["vision_tower"]["vision_width"] == 1152
    assert arms["local_v2_exact_k3"]["vision_tower"]["vision_width"] == 1024
    assert all(
        arm["projector"]["output_width"] == 4096
        and arm["projector"]["initialization_seed"] == 20260805
        and arm["projector"]["random_projector_seed"] == 20260806
        for arm in arms.values()
    )
    for arm in arms.values():
        config_path = ROOT / arm["projector"]["config_path"]
        assert _sha(config_path) == arm["projector"]["config_sha256"]
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert int(config["language_width"]) == 4096


def test_architecture_screen_does_not_reuse_direct_2048_draft():
    contract = json.loads(
        (ROOT / "configs/qwen2.5-3b-community-architecture-screen-v1.json").read_text(
            encoding="utf-8"
        )
    )
    for arm in contract["arms"].values():
        assert int(arm["projector"]["output_width"]) == 4096
        assert "2048" not in str(arm["projector"].get("config_path", ""))
