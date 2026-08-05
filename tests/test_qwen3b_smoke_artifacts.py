import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments" / "qwen3b_community_eval_20260805" / "smoke_v1"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_qwen3b_smoke_keeps_failure_and_limits_success_claims():
    failed = _load_json(
        PACKAGE / "failures" / "attempt01_checkpoint_verifier" / "FAILURE.json"
    )
    summary = _load_json(PACKAGE / "valid_retry2" / "SUMMARY.json")
    verification = _load_json(PACKAGE / "VERIFICATION.json")

    assert failed["status"] == "invalid"
    assert failed["failed_at_stage"] == "checkpoint_save_restore"
    assert failed["capability_claim_allowed"] is False

    assert summary["status"] == "valid"
    assert summary["final_half_scored"] is False
    assert summary["model"]["architecture"] == "Qwen2ForCausalLM"
    assert summary["model"]["parameter_count"] == 3_085_938_688
    assert summary["model"]["verified_file_count"] == 9
    assert summary["moonvit"]["weights_sha256"] == (
        "01436a95939965185bb853ddf984e09c00f597b9c2f6708ba302ffbaf75ced24"
    )
    assert summary["claims"] == {
        "checkpoint_round_trip_exact": True,
        "deepseek_transfer": "transferable_with_runtime_validation",
        "qwen_parameter_gradients_absent": True,
        "real_image_gradient_reaches_projector": True,
        "real_moonvit_image_forward": True,
        "real_qwen3b_load": True,
        "reason": "step0 generation is an engineering smoke, not a trained or paired benchmark",
        "visual_ability_established": False,
    }
    assert summary["generation"]["vision_prediction"] == summary["generation"][
        "blind_prediction"
    ]

    assert verification["status"] == "valid"
    assert verification["canonical_remote_artifact_rehash"] == {
        "declared_bytes": 470_235_478,
        "declared_files": 13,
        "matched_bytes": 470_235_478,
        "matched_files": 13,
    }
    assert verification["capability_claim_allowed"] is False
    assert verification["paid_resources_used"] is False


def test_qwen3b_smoke_package_manifest_matches_checked_in_files():
    manifest_path = PACKAGE / "ARTIFACT_MANIFEST.json"
    manifest = _load_json(manifest_path)
    expected = manifest["files"]
    actual = {
        path.relative_to(PACKAGE).as_posix(): path
        for path in PACKAGE.rglob("*")
        if path.is_file() and path != manifest_path
    }

    assert set(actual) == set(expected)
    assert manifest["file_count"] == len(expected)
    assert manifest["total_bytes"] == sum(path.stat().st_size for path in actual.values())
    for relative, path in actual.items():
        assert path.stat().st_size == expected[relative]["bytes"]
        assert _sha256(path) == expected[relative]["sha256"]
    assert manifest["final_half_scored"] is False
