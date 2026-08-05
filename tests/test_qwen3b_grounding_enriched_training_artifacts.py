import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "experiments"
    / "qwen3b_community_eval_20260805"
    / "training_grounding_enriched_4k_v1"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def test_grounding_enriched_training_is_exact_and_verified():
    summary = _load(PACKAGE / "SUMMARY.json")
    supervision = _load(PACKAGE / "SUPERVISION_SUMMARY.json")
    verification = _load(PACKAGE / "INDEPENDENT_VERIFICATION.json")

    assert summary["status"] == "valid"
    assert summary["formal_training_complete"] is True
    assert summary["optimizer_steps"] == 500
    assert summary["examples_seen"] == 4000
    assert summary["answer_tokens_seen"] == 36_589
    assert summary["micro_batch_size"] == 1
    assert summary["gradient_accumulation"] == 8
    assert summary["real_global_batch"] == 8
    assert summary["qwen_parameter_count"] == 3_085_938_688
    assert summary["qwen_trainable_parameter_count"] == 0
    assert summary["projector_parameter_count"] == 33_564_672
    assert summary["peak_gpu_memory_bytes"] == 8_973_374_976
    assert summary["loss_first"] == 4.144003733992577
    assert summary["loss_last"] == 1.915631964802742
    assert summary["visual_ability_established"] is False
    assert summary["capability_claim_allowed"] is False
    assert supervision["route_counts"] == {
        "grounding": 2000,
        "short_answer": 2000,
    }
    assert supervision["answer_tokens_total"] == 36_589

    assert verification["status"] == "verified"
    assert verification["runner_git_sha"] == (
        "f0afdae9d475a87dba39a6204657bdc2ca28e307"
    )
    assert verification["verifier_git_sha"] == (
        "f0afdae9d475a87dba39a6204657bdc2ca28e307"
    )
    assert len(verification["checkpoints"]) == 5
    assert verification["checkpoint_total_bytes"] == 2_351_007_317
    assert verification["final_training_state_step"] == 500
    assert verification["final_optimizer_parameter_states"] == 6
    assert verification["final_projector_sha256"] == (
        "62f69393dd3d157446db05a7060942bdfbd23f834bbe2e6782560431b5773df4"
    )
    assert verification["step0_projector_sha256"] == (
        "efd942e0d8cbece08d3a8fab5d192eb4a2772211817ac5e733aa0f55aebb06b0"
    )


def test_grounding_enriched_training_remote_and_package_hashes():
    remote = _load(PACKAGE / "REMOTE_ARTIFACT_MANIFEST.json")
    assert remote["git_base_sha"] == (
        "f0afdae9d475a87dba39a6204657bdc2ca28e307"
    )
    assert remote["file_count"] == 40
    assert remote["total_bytes"] == 2_353_629_390
    assert len(
        [name for name in remote["files"] if name.startswith("checkpoints/")]
    ) == 30
    assert remote["final_half_scored"] is False

    artifact_manifest_path = PACKAGE / "ARTIFACT_MANIFEST.json"
    artifact_manifest = _load(artifact_manifest_path)
    actual = {
        path.relative_to(PACKAGE).as_posix(): path
        for path in PACKAGE.rglob("*")
        if path.is_file() and path != artifact_manifest_path
    }
    assert set(actual) == set(artifact_manifest["files"])
    assert artifact_manifest["file_count"] == len(actual)
    assert artifact_manifest["total_bytes"] == sum(
        path.stat().st_size for path in actual.values()
    )
    for relative, path in actual.items():
        declared = artifact_manifest["files"][relative]
        assert declared["bytes"] == path.stat().st_size
        assert declared["sha256"] == _sha256(path)
    assert artifact_manifest["final_half_scored"] is False
