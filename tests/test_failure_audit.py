"""Failure audits use deterministic predicates without cherry-picking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import pytest

from audit_perception_failures import classify_sample, require_checkpoint_rows


def test_failure_audit_classifies_generation_and_internal_evidence_cases():
    generation = {
        "vision": {"correct": True, "normalized_prediction": "red"},
        "blind": {"correct": False, "normalized_prediction": "blue"},
        "patch_permutation": {"correct": False, "normalized_prediction": "blue"},
        "background_matched_aux": {
            "correct": False,
            "normalized_prediction": "blue",
        },
    }
    preference = {"correct_margin": 0.4, "failure": None}

    labels = classify_sample(generation, preference)

    assert labels == {
        "vision_success_blind_failure",
        "patch_permutation_flip",
        "background_prediction_flip",
    }

    generation["vision"] = {"correct": False, "normalized_prediction": "green"}
    labels = classify_sample(generation, preference)
    assert "teacher_forced_positive_generation_failure" in labels
    assert "vision_and_blind_failure" in labels


def test_failure_audit_rejects_unknown_checkpoint_instead_of_empty_valid_audit():
    with pytest.raises(ValueError, match="no synthetic generation rows"):
        require_checkpoint_rows("step1500", [], [{"id": "sample"}])

    with pytest.raises(ValueError, match="no vision preference rows"):
        require_checkpoint_rows("step-001500", [{"id": "sample"}], [])
