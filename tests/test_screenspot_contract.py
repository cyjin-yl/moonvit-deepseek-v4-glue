from collections import Counter

import pytest

from moonvit_glue.screenspot_contract import (
    deterministic_image_derangement,
    normalize_screenspot_bbox,
    seal_manifest,
    stable_stratified_subset,
    verify_manifest,
)


def _records():
    rows = []
    for platform in ("Android", "Web"):
        for target_type in ("text", "icon/widget"):
            for index in range(10):
                rows.append(
                    {
                        "sample_id": f"{platform}-{target_type}-{index}",
                        "platform": platform,
                        "target_type": target_type,
                        "image_sha256": f"sha-{platform}-{target_type}-{index}",
                    }
                )
    return rows


def test_frozen_public_mirror_bbox_is_fractional_xyxy_on_the_999_scale():
    assert normalize_screenspot_bbox(
        [0.1, 0.2, 0.15, 0.26],
        width=1000,
        height=1000,
        source_format="fractional_xyxy",
    ) == pytest.approx(
        [99.9, 199.8, 149.85, 259.74]
    )


def test_original_screenspot_pixel_xywh_can_be_normalized_explicitly():
    assert normalize_screenspot_bbox(
        [100, 200, 50, 60],
        width=1000,
        height=1000,
        source_format="pixel_xywh",
    ) == pytest.approx([99.9, 199.8, 149.85, 259.74])


def test_glm_subset_selection_is_stable_and_stratified():
    rows = _records()
    selected = stable_stratified_subset(rows, size=8, seed="20260805")
    selected_reversed = stable_stratified_subset(
        list(reversed(rows)), size=8, seed="20260805"
    )

    assert [row["sample_id"] for row in selected] == [
        row["sample_id"] for row in selected_reversed
    ]
    counts = Counter((row["platform"], row["target_type"]) for row in selected)
    assert set(counts.values()) == {2}


def test_deterministic_derangement_has_no_fixed_or_same_image_points():
    rows = _records()
    first = deterministic_image_derangement(rows, seed="20260805")
    second = deterministic_image_derangement(list(reversed(rows)), seed="20260805")
    assert first == second

    by_id = {row["sample_id"]: row for row in rows}
    for sample_id, shuffled_id in first.items():
        assert sample_id != shuffled_id
        assert by_id[sample_id]["image_sha256"] != by_id[shuffled_id]["image_sha256"]


def test_manifest_seal_detects_any_post_freeze_change():
    manifest = seal_manifest({"schema_version": "test-v1", "samples": _records()[:2]})
    assert verify_manifest(manifest)

    changed = dict(manifest)
    changed["schema_version"] = "tampered"
    assert not verify_manifest(changed)
