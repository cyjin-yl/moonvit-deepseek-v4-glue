"""机制包 verifier 必须独立检查 tensor 键和实际视觉来源。"""

import pytest

from moonvit_glue.mechanism_verification import (
    expected_representation_keys,
    validate_pair_permutation_rows,
    validate_visual_source,
)


def test_expected_representation_keys_cover_all_hidden_states():
    keys = expected_representation_keys(
        hidden_state_count=3,
        poolings=["global_mean", "spatial_2x2"],
    )
    assert keys == {
        "labels",
        "source_labels",
        "shape_logits",
        "tower_global_mean",
        "tower_spatial_2x2",
        "projector_global_mean",
        "projector_spatial_2x2",
        "layer_00_assistant",
        "layer_00_image_mean",
        "layer_01_assistant",
        "layer_01_image_mean",
        "layer_02_assistant",
        "layer_02_image_mean",
    }


def test_validate_visual_source_checks_each_causal_condition():
    control = {"shuffled_image_id": "shuffled"}
    validate_visual_source("vision", "sample", "sample", "mate", control)
    validate_visual_source("patch_permutation", "sample", "sample", "mate", control)
    validate_visual_source(
        "paired_counterfactual_image", "sample", "mate", "mate", control
    )
    validate_visual_source("shuffled_image", "sample", "shuffled", "mate", control)
    with pytest.raises(ValueError, match="visual source mismatch"):
        validate_visual_source("paired_counterfactual_image", "sample", "sample", "mate", control)


def test_validate_pair_permutation_rows_rejects_uncalibrated_vision_cell():
    validate_pair_permutation_rows(
        [{"condition": "vision", "pair_permutation_samples": "2000"}]
    )
    with pytest.raises(ValueError, match="pair-permutation"):
        validate_pair_permutation_rows(
            [{"condition": "vision", "pair_permutation_samples": "0"}]
        )
