import json

import pytest

from moonvit_glue.grounding_evaluation import (
    REQUIRED_CONDITIONS,
    evaluate_conditions,
    read_prediction_jsonl,
)
from moonvit_glue.screenspot_contract import seal_manifest


def _manifest():
    samples = []
    platforms = ["Android", "iOS", "Windows", "Web"]
    target_types = ["text", "icon/widget", "text", "icon/widget"]
    for index in range(4):
        samples.append(
            {
                "sample_id": f"sample-{index}",
                "evaluation_order": index,
                "platform": platforms[index],
                "target_type": target_types[index],
                "bbox_999_xyxy": [450.0, 450.0, 550.0, 550.0],
            }
        )
    return seal_manifest(
        {
            "schema_version": "screenspot-community-contract-v1",
            "name": "test",
            "samples": samples,
        }
    )


def _predictions(text):
    return [
        {"sample_id": f"sample-{index}", "prediction": text}
        for index in range(4)
    ]


def test_evaluator_runs_all_registered_conditions_breakdowns_and_pairs():
    good = _predictions("click(start_box=[500, 500])")
    far = _predictions("click(start_box=[800, 800])")
    unparsed = _predictions("cannot locate")
    conditions = {
        "vision": good,
        "blind": far,
        "shuffled": unparsed,
        "random_projector": far,
        "step0": far,
        "previous_best": far,
        "current_candidate": good,
    }
    result = evaluate_conditions(
        _manifest(), conditions, bootstrap_samples=40, bootstrap_seed=20260805
    )

    assert tuple(result["condition_order"]) == REQUIRED_CONDITIONS
    assert result["conditions"]["vision"]["breakdowns"]["overall"][
        "click_in_box_accuracy"
    ]["all_accuracy"] == 1.0
    assert result["conditions"]["vision"]["breakdowns"]["target_type"]["text"][
        "total_count"
    ] == 2
    assert result["conditions"]["vision"]["breakdowns"]["platform"]["Web"][
        "total_count"
    ] == 1
    assert result["comparisons"]["vision-minus-blind"]["metrics"][
        "click_in_box_all"
    ]["ci95"] == [1.0, 1.0]
    assert result["comparisons"]["current-candidate-minus-previous-best"][
        "metrics"
    ]["mean_center_distance_all_penalized"]["improvement"] > 0


def test_evaluator_rejects_missing_formal_condition_and_non_alias_candidate():
    good = _predictions("click(start_box=[500, 500])")
    conditions = {name: good for name in REQUIRED_CONDITIONS[:-1]}
    with pytest.raises(ValueError, match="missing required conditions"):
        evaluate_conditions(_manifest(), conditions, bootstrap_samples=10)

    conditions = {name: good for name in REQUIRED_CONDITIONS}
    conditions["current_candidate"] = _predictions("click(start_box=[501, 500])")
    with pytest.raises(ValueError, match="current_candidate"):
        evaluate_conditions(_manifest(), conditions, bootstrap_samples=10)


def test_prediction_jsonl_requires_exact_manifest_order(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"sample_id": "sample-1", "prediction": "x"},
                {"sample_id": "sample-0", "prediction": "y"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exact manifest order"):
        read_prediction_jsonl(path, ["sample-0", "sample-1"])
