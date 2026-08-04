"""Paired checkpoint/control comparisons use identical sample IDs."""

from moonvit_glue.trajectory_analysis import paired_gap_stats


def test_paired_gap_retains_wins_losses_and_denominator():
    vision = [
        {"id": "a", "score": 1.0},
        {"id": "b", "score": 1.0},
        {"id": "c", "score": 0.0},
        {"id": "d", "score": 0.5},
    ]
    blind = [
        {"id": "a", "score": 0.0},
        {"id": "b", "score": 1.0},
        {"id": "c", "score": 1.0},
        {"id": "d", "score": 0.0},
    ]
    stats = paired_gap_stats(vision, blind, bootstrap_samples=200, seed=7)
    assert stats["denominator"] == 4
    assert stats["sum_a"] == 2.5
    assert stats["sum_b"] == 2.0
    assert stats["mean_gap"] == 0.125
    assert stats["a_only_better"] == 2
    assert stats["b_only_better"] == 1
    assert stats["equal"] == 1
    assert stats["bootstrap_samples"] == 200
    assert stats["ci95_low"] <= stats["mean_gap"] <= stats["ci95_high"]
