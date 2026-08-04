"""Reproducible accounting and validation protocol for alignment runs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from training_protocol import (
    TrainingProgress,
    make_derangements,
    prepare_validation_split,
    records_manifest_sha256,
    resolve_batch_semantics,
    restore_progress_counts,
    select_supervision,
    summarize_validation_losses,
)


def test_progress_reports_real_examples_tokens_epochs_and_batch_semantics():
    progress = TrainingProgress(
        total_training_examples=100,
        micro_batch_size=1,
        gradient_accumulation_steps=4,
    )
    for answer_tokens in (2, 3, 4, 5):
        progress.record_microbatch(examples=1, answer_tokens=answer_tokens)
    progress.record_optimizer_step()

    assert progress.snapshot() == {
        "optimizer_steps": 1,
        "examples_seen": 4,
        "answer_tokens_seen": 14,
        "effective_epochs": 0.04,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 4,
    }


def test_canonical_supervision_uses_normalized_majority_and_preserves_provenance():
    choice = select_supervision(
        ["Power!", "power", " power ", "POWER", "other"],
        rule="canonical",
    )

    assert choice.selected_answer == "power"
    assert choice.canonical_answer == "power"
    assert choice.raw_answers == ["Power!", "power", " power ", "POWER", "other"]
    assert choice.normalization_rule == "vqa_normalized_majority"


def test_seeded_derangements_are_reproducible_permutations_without_fixed_points():
    items = ["a", "b", "c", "d", "e"]

    first = make_derangements(items, repeats=10, seed=17)
    second = make_derangements(items, repeats=10, seed=17)

    assert first == second
    assert len(first) == 10
    for shuffled in first:
        assert sorted(shuffled) == items
        assert all(original != replacement for original, replacement in zip(items, shuffled))


def test_validation_manifest_is_stratified_persisted_and_reused(tmp_path):
    records = [
        {"id": f"{source}-{index}", "source": source, "answers": ["x"]}
        for source in ("textvqa_train", "docvqa_train", "showui_desktop", "train")
        for index in range(5)
    ]
    manifest_path = tmp_path / "validation.json"

    train, validation, manifest = prepare_validation_split(
        records,
        manifest_path=manifest_path,
        total_samples=8,
        seed=23,
    )
    _, reused, reused_manifest = prepare_validation_split(
        list(reversed(records)),
        manifest_path=manifest_path,
        total_samples=8,
        seed=999,
    )

    assert manifest_path.exists()
    assert manifest == reused_manifest
    assert [record["id"] for record in validation] == [record["id"] for record in reused]
    assert len(train) == 12
    assert manifest["counts_by_source"] == {
        "art": 2,
        "docvqa": 2,
        "showui": 2,
        "textvqa": 2,
    }


def test_validation_summary_reports_per_source_shuffle_mean_std_and_pair_ids():
    records = [
        {"id": "t0", "source": "textvqa_train"},
        {"id": "t1", "source": "textvqa_train"},
        {"id": "d0", "source": "docvqa_train"},
        {"id": "d1", "source": "docvqa_train"},
    ]
    summary = summarize_validation_losses(
        records,
        true_losses=[1.0, 3.0, 2.0, 4.0],
        shuffled_loss_runs=[[2.0, 4.0, 3.0, 5.0], [4.0, 6.0, 5.0, 7.0]],
        shuffled_id_runs=[
            ["d0", "d1", "t0", "t1"],
            ["d1", "d0", "t1", "t0"],
        ],
    )

    assert summary["overall"]["true_loss"] == 2.5
    assert summary["overall"]["shuffle_delta_mean"] == 2.0
    assert summary["overall"]["shuffle_delta_std"] == 1.0
    assert summary["by_source"]["textvqa"]["true_loss"] == 2.0
    assert summary["by_source"]["docvqa"]["shuffle_delta_mean"] == 2.0
    assert summary["shuffle_runs"][0]["pairs"][0] == {
        "record_id": "t0",
        "image_id": "d0",
    }


def test_batch_semantics_expose_legacy_accumulation_and_reject_fake_microbatching():
    assert resolve_batch_semantics(
        micro_batch_size=1,
        gradient_accumulation_steps=None,
        legacy_batch_size=8,
    ) == {
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 8,
        "legacy_batch_size_used": True,
    }

    import pytest

    with pytest.raises(ValueError, match="true batched forward is not implemented"):
        resolve_batch_semantics(
            micro_batch_size=2,
            gradient_accumulation_steps=4,
            legacy_batch_size=None,
        )
    with pytest.raises(ValueError, match="must be positive"):
        resolve_batch_semantics(
            micro_batch_size=1,
            gradient_accumulation_steps=0,
            legacy_batch_size=None,
        )


def test_legacy_resume_requires_explicit_original_accumulation_for_honest_counts():
    import pytest

    with pytest.raises(ValueError, match="legacy checkpoint lacks examples_seen"):
        restore_progress_counts(
            start_step=2_000,
            last_history={},
            effective_batch_size=4,
            batch_semantics_explicit=False,
        )

    assert restore_progress_counts(
        start_step=2_000,
        last_history={},
        effective_batch_size=8,
        batch_semantics_explicit=True,
    ) == {
        "examples_seen": 16_000,
        "answer_tokens_seen": 0,
        "answer_token_accounting_complete": False,
    }
    assert restore_progress_counts(
        start_step=2_000,
        last_history={"examples_seen": 15_992, "answer_tokens_seen": 31_984},
        effective_batch_size=8,
        batch_semantics_explicit=False,
    ) == {
        "examples_seen": 15_992,
        "answer_tokens_seen": 31_984,
        "answer_token_accounting_complete": True,
    }


def test_records_manifest_hash_is_order_stable_and_content_sensitive():
    records = [
        {
            "id": "b",
            "source": "docvqa_train",
            "image": "b.png",
            "question": "Read this",
            "answers": ["one", "1"],
            "image_bytes": b"large payload deliberately excluded",
        },
        {
            "id": "a",
            "source": "textvqa_train",
            "image": "a.png",
            "question": "What color?",
            "answers": ["red"],
        },
    ]

    digest = records_manifest_sha256(records)

    assert digest == records_manifest_sha256(list(reversed(records)))
    changed = [{**records[0], "question": "Changed"}, records[1]]
    assert digest != records_manifest_sha256(changed)
