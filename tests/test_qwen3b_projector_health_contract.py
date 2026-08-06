import hashlib
import json
from pathlib import Path

import pytest

from moonvit_glue.training_health import validate_health_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "qwen3b-projector-health-v1.json"
PROBE = (
    ROOT
    / "experiments"
    / "qwen3b_community_eval_20260805"
    / "health_probe_v1"
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_health_contract_freezes_schedule_and_guards():
    contract = _load(CONFIG)
    validate_health_contract(contract)
    assert contract["canonical_projector_width"] == 4096
    assert contract["probe_schedule"]["initial_steps"] == [
        0,
        1,
        2,
        5,
        10,
        20,
        30,
        50,
        75,
        100,
    ]
    assert contract["guards"]["hard"]["relative_spread_ratio_min"] == 0.25
    assert contract["guards"]["hard"]["effective_rank_ratio_min"] == 0.5


def test_probe_manifest_self_hash_and_artifact_inventory():
    manifest_path = PROBE / "PROBE_MANIFEST.json"
    manifest = _load(manifest_path)
    expected = manifest["manifest_sha256"]
    payload = dict(manifest)
    payload.pop("manifest_sha256")
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == expected
    assert len(manifest["samples"]) == 50
    assert len({row["sample_id"] for row in manifest["samples"]}) == 50
    assert all(len(row["feature_sha256"]) == 64 for row in manifest["samples"])

    artifact = _load(PROBE / "ARTIFACT_MANIFEST.json")
    for relative, expected_file in artifact["files"].items():
        path = PROBE / relative
        assert path.stat().st_size == expected_file["bytes"]
        assert _sha(path) == expected_file["sha256"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("format_version", "wrong"),
        ("canonical_projector_width", 2048),
    ],
)
def test_health_contract_rejects_mutation(field, value):
    contract = _load(CONFIG)
    contract[field] = value
    with pytest.raises(ValueError):
        validate_health_contract(contract)
