import json

import pytest

from analyze_stripped_screenspot_categories import (
    build_summary,
    category_ids,
    validate_and_index,
)


def _manifest():
    samples = []
    mapping = []
    platforms = ("Android", "iOS", "Windows", "macOS", "Web")
    for index, platform in enumerate(platforms * 2):
        sample_id = f"s{index}"
        shuffled_id = f"s{(index + 1) % 10}"
        samples.append(
            {
                "sample_id": sample_id,
                "platform": platform,
                "target_type": "text" if index % 2 == 0 else "icon/widget",
            }
        )
        mapping.append(
            {"sample_id": sample_id, "shuffled_image_sample_id": shuffled_id}
        )
    return {
        "name": "screenspot_public_test_v1",
        "samples": samples,
        "shuffled_image_control": {"mapping": mapping},
    }


def _rows(manifest):
    mapping = {
        row["sample_id"]: row["shuffled_image_sample_id"]
        for row in manifest["shuffled_image_control"]["mapping"]
    }
    rows = []
    for condition in ("vision", "blind", "shuffled", "random_projector"):
        for sample in manifest["samples"]:
            success = condition == "vision"
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "condition": condition,
                    "shuffled_sample_id": mapping[sample["sample_id"]],
                    "parse_ok": True,
                    "accuracy_at_50": success,
                    "accuracy_at_100": success,
                    "accuracy_at_200": success,
                    "click_in_box": success,
                    "center_l2_penalized": 1.0 if success else 10.0,
                    "center_l2": 1.0 if success else 10.0,
                    "bbox_l2": 0.0 if success else 9.0,
                    "bbox_l2_penalized": 0.0 if success else 9.0,
                    "bbox_l1": 0.0 if success else 9.0,
                    "bbox_l1_penalized": 0.0 if success else 9.0,
                }
            )
    return rows


def test_category_contract_and_positive_causal_improvement():
    manifest = _manifest()
    summary = build_summary(
        manifest, _rows(manifest), bootstrap_samples=2_000, bootstrap_seed=20260805
    )
    assert summary["category_order"] == [
        "overall",
        "text",
        "icon/widget",
        "Android",
        "iOS",
        "Windows",
        "macOS",
        "Web",
    ]
    overall = summary["categories"]["overall"]
    assert overall["conditions"]["vision"]["click_in_box_accuracy"]["all_accuracy"] == 1.0
    paired = overall["paired"]["vision_minus_shuffled"]
    assert paired["click_in_box_accuracy"]["improvement_ci95_lower"] == 1.0
    assert paired["mean_center_distance"]["improvement_ci95_lower"] == 9.0


def test_duplicate_or_bad_shuffle_provenance_is_rejected():
    manifest = _manifest()
    rows = _rows(manifest)
    rows[0]["shuffled_sample_id"] = "wrong"
    with pytest.raises(ValueError, match="shuffled provenance"):
        validate_and_index(manifest, rows)


def test_all_frozen_categories_are_nonempty():
    groups = category_ids(_manifest())
    assert all(groups.values())
