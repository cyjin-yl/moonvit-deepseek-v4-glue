import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "experiments"
    / "qwen3b_community_eval_20260805"
    / "screenspot_glm50_preference_grounding_enriched_4k_v1"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _overall(summary: dict, condition: str) -> dict:
    return summary["preference_summary"]["conditions"][condition]["breakdowns"][
        "overall"
    ]


def test_grounding_enriched_preference_rejects_candidate():
    summary = _load(PACKAGE / "formal" / "SUMMARY.json")
    decision = _load(PACKAGE / "DECISION.json")

    assert summary["status"] == "valid"
    assert summary["formal_preference_complete"] is True
    assert summary["records"] == 50
    assert summary["bootstrap"] == {"samples": 2000, "seed": 20260805}
    assert summary["qwen_parameter_count"] == 3_085_938_688
    assert summary["qwen_trainable_parameter_count"] == 0
    assert summary["peak_gpu_memory_bytes"] == 7_652_064_768
    assert _overall(summary, "vision")["paired_preference_accuracy"] == 0.52
    assert _overall(summary, "blind")["paired_preference_accuracy"] == 0.56
    assert _overall(summary, "shuffled")["paired_preference_accuracy"] == 0.54
    assert _overall(summary, "step0")["paired_preference_accuracy"] == 0.54
    assert _overall(summary, "random_projector")["paired_preference_accuracy"] == 0.5
    assert _overall(summary, "vision")["mean_correct_token_nll"] == (
        1.0591498362887037
    )

    vision_shuffle = summary["comparisons"]["vision-minus-shuffled"]["metrics"]
    assert vision_shuffle["paired_preference_accuracy"]["improvement"] == (
        -0.020000000000000018
    )
    assert vision_shuffle["paired_preference_accuracy"]["ci95"] == [
        -0.06000000000000005,
        0.0,
    ]
    assert vision_shuffle["mean_correct_logp"]["ci95"] == [
        -0.00578621568129612,
        0.002342096596847069,
    ]
    current_step0 = summary["comparisons"][
        "current-candidate-minus-previous-best"
    ]["metrics"]
    assert current_step0["mean_correct_token_nll"]["improvement"] == (
        1.4485429812874986
    )
    assert current_step0["mean_correct_token_nll"]["ci95"] == [
        1.297928403741711,
        1.6069774260918297,
    ]
    assert decision["decision"] == "reject_at_paired_preference_gate"
    assert decision["paired_preference_gate_passed"] is False
    assert decision["visual_ability_established"] is False
    assert decision["checkpoint_enters_deepseek_candidate_list"] is False
    assert decision["previous_best_after_decision"] == "step0"
    assert decision["paid_resources_used"] is False


def test_grounding_enriched_preference_aliases_and_manifests_are_exact():
    formal = PACKAGE / "formal"
    run_config = _load(formal / "RUN_CONFIG.json")
    summary = _load(formal / "SUMMARY.json")
    assert run_config["runner_git_sha"] == (
        "f09d28b8792a5ac287f7719a8a8c71959105f742"
    )
    assert run_config["projector_sources"]["current_candidate"][
        "weights_sha256"
    ] == "62f69393dd3d157446db05a7060942bdfbd23f834bbe2e6782560431b5773df4"
    assert summary["condition_aliases"] == {
        "previous_best": "step0",
        "previous_best_shuffled": "step0_shuffled",
        "shuffled": "current_candidate_shuffled",
        "vision": "current_candidate",
    }
    for alias, source in summary["condition_aliases"].items():
        assert (formal / "preferences" / f"{alias}.jsonl").read_bytes() == (
            formal / "preferences" / f"{source}.jsonl"
        ).read_bytes()

    remote = _load(PACKAGE / "REMOTE_ARTIFACT_MANIFEST.json")
    assert remote["git_base_sha"] == (
        "f09d28b8792a5ac287f7719a8a8c71959105f742"
    )
    assert remote["file_count"] == 14
    assert remote["total_bytes"] == 685_140
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
