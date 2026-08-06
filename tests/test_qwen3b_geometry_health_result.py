import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "experiments"
    / "qwen3b_community_eval_20260805"
    / "geometry_repair_screen_hf_v1"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_control_health_result_preserves_early_stop_boundary():
    result = _load(PACKAGE / "control" / "RESULT.json")
    assert result["arm"] == "control"
    assert result["geometry_lambda"] == 0.0
    assert result["optimizer_steps_completed"] == 2
    assert result["stop_and_rollback"]["collapse_onset_interval"] == [1, 2]
    assert result["stop_and_rollback"]["restored_step"] == 1
    assert result["trajectory"][-1]["auto_stop"] is True
    assert result["trajectory"][-1]["critical_reasons"] == [
        "projector_rms_rising_spread_falling",
        "receiver_rms_rising_spread_falling",
    ]
    assert result["interpretation"]["visual_ability_established"] is False
    assert result["interpretation"]["previous_best_promoted"] is False


def test_curated_health_artifact_manifest_rehashes():
    root = PACKAGE
    manifest = _load(root / "ARTIFACT_MANIFEST.json")
    assert manifest["final_half_scored"] is False
    assert manifest["file_count"] == len(manifest["files"])
    for relative, expected in manifest["files"].items():
        path = root / relative
        assert path.is_file(), relative
        assert path.stat().st_size == expected["bytes"], relative
        assert _sha256(path) == expected["sha256"], relative


def test_independent_verifier_and_raw_pointer_are_bound():
    verifier = _load(PACKAGE / "control" / "HEALTH_VERIFIER.json")
    pointer = _load(PACKAGE / "control" / "RAW_ARTIFACT_POINTER.json")
    assert verifier["status"] == "verified"
    assert verifier["probe_steps"] == [0, 1, 2]
    assert verifier["auto_stopped"] is True
    assert pointer["complete_raw_copy"] is True
    assert pointer["local_rehash"]["mismatches"] == 0
    assert pointer["health_manifest_file_count"] == 22
    assert pointer["health_manifest_total_bytes"] == 1141300055


def test_four_arm_screen_has_no_passing_lambda_and_cancels_expansion():
    decision = _load(PACKAGE / "DECISION.json")
    assert [row["name"] for row in decision["arms"]] == [
        "control",
        "ratio005",
        "ratio020",
        "ratio080",
    ]
    assert [row["lambda"] for row in decision["arms"]] == [
        0.0,
        0.01018730507868909,
        0.04074922031475636,
        0.16299688125902545,
    ]
    assert decision["selection"]["passing_arms"] == []
    assert decision["selection"]["selected_arm"] is None
    assert decision["selection"]["full_500_step_expansion"] == "cancelled"

    for row in decision["arms"]:
        arm = row["name"]
        result = _load(PACKAGE / arm / "RESULT.json")
        verifier = _load(PACKAGE / arm / "HEALTH_VERIFIER.json")
        pointer = _load(PACKAGE / arm / "RAW_ARTIFACT_POINTER.json")
        assert row["status"] == "auto_stopped_by_projector_health_guard"
        assert row["collapse_onset_interval"] == [1, 2]
        assert result["stop_and_rollback"]["collapse_onset_interval"] == [1, 2]
        assert verifier["status"] == "verified"
        assert verifier["probe_steps"] == [0, 1, 2]
        assert pointer["local_rehash"]["mismatches"] == 0
        assert _sha256(PACKAGE / arm / "HEALTH_ARTIFACT_MANIFEST.json") == pointer[
            "health_manifest_sha256"
        ]
