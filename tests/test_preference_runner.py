"""Screening overrides must remain explicit in the copied run config."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from eval_paired_preference import apply_screening_overrides, condition_source_id


def test_screening_override_prunes_checkpoints_aliases_and_records():
    config = {
        "checkpoints": [{"id": "c0"}, {"id": "c1"}],
        "aliases": [{"id": "current", "source": "c1"}],
        "synthetic": {"expected_records": 2400},
    }

    screened = apply_screening_overrides(config, limit=4, checkpoint_ids=["c0"])

    assert [row["id"] for row in screened["checkpoints"]] == ["c0"]
    assert screened["aliases"] == []
    assert screened["synthetic"]["limit"] == 4
    assert screened["synthetic"]["expected_records"] == 4
    assert screened["screening_override"] == {
        "record_limit": 4,
        "checkpoint_ids": ["c0"],
    }


def test_condition_source_ids_match_the_actual_visual_intervention():
    pair_index = {"a": {"paired_image_id": "b"}}
    control = {"shuffled_image_id": "z"}

    assert condition_source_id("blind", "a", "selection", control, pair_index) is None
    assert condition_source_id("vision", "a", "selection", control, pair_index) == "a"
    assert condition_source_id("patch_permutation", "a", "selection", control, pair_index) == "a"
    assert condition_source_id("background_matched_aux", "a", "selection", control, pair_index) == "a"
    assert condition_source_id("blank", "a", "selection", control, pair_index) == "control:selection:blank"
    assert condition_source_id("same_image", "a", "selection", control, pair_index) == "control:selection:same"
    assert condition_source_id("shuffled_image", "a", "selection", control, pair_index) == "z"
    assert condition_source_id(
        "paired_counterfactual_image", "a", "selection", control, pair_index
    ) == "b"
