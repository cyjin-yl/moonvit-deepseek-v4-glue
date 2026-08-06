import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "experiments"
    / "qwen3b_community_eval_20260805"
    / "screenspot_glm50_grounding_enriched_4k_v1"
)
TRAINING = (
    ROOT
    / "experiments"
    / "qwen3b_community_eval_20260805"
    / "training_grounding_enriched_4k_v1"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _overall(summary: dict, condition: str) -> dict:
    return summary["conditions"][condition]["breakdowns"]["overall"]


def test_grounding_enriched_generation_rejects_candidate():
    summary = _load(PACKAGE / "scoring" / "SUMMARY.json")
    decision = _load(PACKAGE / "DECISION.json")
    vision = _overall(summary, "vision")
    blind = _overall(summary, "blind")
    shuffled = _overall(summary, "shuffled")
    step0 = _overall(summary, "step0")

    assert summary["formal_complete"] is True
    assert summary["sample_count"] == 50
    assert summary["bootstrap"] == {
        "samples": 2000,
        "scope": "overall; category point estimates are reported without category CIs",
        "seed": 20260805,
    }
    assert (vision["parse_rate"], blind["parse_rate"], shuffled["parse_rate"]) == (
        1.0,
        1.0,
        1.0,
    )
    assert vision["accuracy_at_50"]["all_accuracy"] == 0.02
    assert vision["accuracy_at_100"]["all_accuracy"] == 0.02
    assert vision["accuracy_at_200"]["all_accuracy"] == 0.14
    assert vision["click_in_box_accuracy"]["all_accuracy"] == 0.06
    assert vision["center_distance"]["all_penalized"]["mean"] == (
        502.0642943579057
    )
    assert blind["click_in_box_accuracy"]["all_accuracy"] == 0.12
    assert shuffled["click_in_box_accuracy"]["all_accuracy"] == 0.06
    assert step0["click_in_box_accuracy"]["all_accuracy"] == 0.10

    vision_blind = summary["comparisons"]["vision-minus-blind"]["metrics"]
    assert vision_blind["click_in_box_all"]["improvement"] == -0.06
    assert vision_blind["mean_center_distance_all_penalized"]["ci95"] == [
        -171.64269963295538,
        -44.58921650586644,
    ]
    vision_shuffle = summary["comparisons"]["vision-minus-shuffled"]["metrics"]
    assert vision_shuffle["click_in_box_all"]["improvement"] == 0.0
    assert vision_shuffle["click_in_box_all"]["ci95"] == [0.0, 0.0]
    assert vision_shuffle["mean_center_distance_all_penalized"]["ci95"] == [
        -3.543580960072036,
        3.2127602607886194,
    ]
    assert decision["decision"] == "reject_current_candidate"
    assert decision["community_metric_aligned_baseline_reached"] is False
    assert decision["generation_causal_gate_passed"] is False
    assert decision["checkpoint_enters_deepseek_candidate_list"] is False
    assert decision["previous_best_after_decision"] == "step0"


def test_output_collapse_is_not_the_training_label_mode():
    predictions = PACKAGE / "evaluation" / "predictions"
    vision = _rows(predictions / "vision.jsonl")
    shuffled = _rows(predictions / "shuffled.jsonl")
    vision_counts = Counter(row["prediction"] for row in vision)
    shuffled_counts = Counter(row["prediction"] for row in shuffled)
    assert len(vision_counts) == 6
    assert vision_counts.most_common(1) == [("click(start_box=[125, 345])", 31)]
    assert len(shuffled_counts) == 9
    assert shuffled_counts.most_common(1) == [("click(start_box=[125, 345])", 23)]
    assert sum(
        left["prediction"] == right["prediction"]
        for left, right in zip(vision, shuffled, strict=True)
    ) == 30

    pattern = re.compile(r"click\(start_box=\[(\d+), (\d+)\]\)")
    training_points = []
    for row in _rows(TRAINING / "SUPERVISION_RECORDS.jsonl"):
        if row["prompt_route"] != "grounding":
            continue
        match = pattern.fullmatch(row["target_answer"])
        assert match is not None
        training_points.append(tuple(map(int, match.groups())))
    assert len(training_points) == 2000
    assert len(set(training_points)) == 1066
    assert training_points.count((125, 345)) == 0
    assert sum(x == 125 for x, _ in training_points) == 2
    assert sum(y == 345 for _, y in training_points) == 4

    analysis = _load(PACKAGE / "OUTPUT_COLLAPSE_ANALYSIS.json")
    assert analysis["vision"]["mode_count"] == 31
    assert analysis["grounding_training_targets"]["exact_125_345_count"] == 0
    assert analysis["paid_resources_used"] is False


def test_generation_raw_artifacts_and_manifests_are_exact():
    evaluation = PACKAGE / "evaluation"
    scoring = PACKAGE / "scoring"
    eval_summary = _load(evaluation / "SUMMARY.json")
    eval_config = _load(evaluation / "RUN_CONFIG.json")
    assert eval_summary["status"] == "valid"
    assert eval_summary["formal_generation_complete"] is True
    assert eval_summary["peak_gpu_memory_bytes"] == 7_245_852_672
    assert eval_config["runner_git_sha"] == (
        "4c9cea914e614aaa8725181339a80ae94f109093"
    )
    assert eval_config["projector_sources"]["current_candidate"][
        "weights_sha256"
    ] == "62f69393dd3d157446db05a7060942bdfbd23f834bbe2e6782560431b5773df4"
    assert (evaluation / "predictions" / "vision.jsonl").read_bytes() == (
        evaluation / "predictions" / "current_candidate.jsonl"
    ).read_bytes()
    assert (evaluation / "predictions" / "step0.jsonl").read_bytes() == (
        evaluation / "predictions" / "previous_best.jsonl"
    ).read_bytes()
    assert (scoring / "scores" / "vision.jsonl").read_bytes() == (
        scoring / "scores" / "current_candidate.jsonl"
    ).read_bytes()

    remote = _load(PACKAGE / "REMOTE_ARTIFACT_MANIFEST.json")
    assert remote["evaluation_file_count"] == 10
    assert remote["evaluation_total_bytes"] == 173_819
    assert remote["scoring_file_count"] == 9
    assert remote["scoring_total_bytes"] == 392_104
    assert remote["paid_resources_used"] is False
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
