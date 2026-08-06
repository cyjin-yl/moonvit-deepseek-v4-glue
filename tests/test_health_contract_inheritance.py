import json
from pathlib import Path

from train_qwen3b_proxy import load_health_contract


ROOT = Path(__file__).resolve().parents[1]


def test_v1_health_contract_inherits_frozen_guards_and_overrides_probe_identity():
    contract = load_health_contract(
        ROOT / "configs/qwen3b-projector-health-v1-local-v1-family.json"
    )
    assert contract["architecture_id"] == "local_v1_family_proxy"
    assert contract["probe_schedule"]["initial_steps"] == [0, 1, 2, 5, 10, 20, 30, 50, 75, 100]
    assert contract["guards"]["hard"] == {
        "relative_spread_ratio_min": 0.25,
        "effective_rank_ratio_min": 0.50,
        "applies_to": ["projector", "receiver"],
        "selection_requires_all": True,
    }
    assert "local_v1_family_proxy/health_probe_manifest_v1" in contract["probe_manifest"]["file"]
