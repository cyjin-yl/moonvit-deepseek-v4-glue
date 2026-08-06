import pytest
import torch

from moonvit_glue.training_health import (
    DEFAULT_PROBE_STEPS,
    evaluate_guards,
    probe_due,
    summarize_batch_embeddings,
    summarize_probe,
    validate_health_contract,
)


def _contract():
    return {
        "format_version": "projector-health-contract-v1",
        "canonical_projector_width": 4096,
        "probe_schedule": {
            "initial_steps": list(DEFAULT_PROBE_STEPS),
            "every_after_step": 50,
        },
        "guards": {
            "hard": {
                "relative_spread_ratio_min": 0.25,
                "effective_rank_ratio_min": 0.50,
            },
            "warning": {
                "top1_variance_fraction": 0.80,
                "output_rms_ratio": 10.0,
            },
            "critical": {
                "top1_variance_fraction": 0.90,
                "output_rms_ratio": 50.0,
                "causal_consecutive_probe_points": 2,
                "consecutive_hard_failures": 2,
                "gradient_norm_max": 1e6,
            },
        },
    }


def test_probe_schedule_is_high_frequency_then_periodic():
    assert [step for step in range(0, 151) if probe_due(step)] == [
        0,
        1,
        2,
        5,
        10,
        20,
        30,
        50,
        75,
        100,
        150,
    ]
    assert probe_due(200, max_step=100) is False


def test_batch_summary_separates_between_image_and_within_token_spread():
    first = [torch.tensor([[1.0, 0.0], [1.0, 0.0]]), torch.tensor([[3.0, 0.0]])]
    second = [torch.tensor([[2.0, 0.0], [2.0, 0.0]]), torch.tensor([[4.0, 0.0]])]
    summary = summarize_batch_embeddings(first, second)
    assert summary["projector_output_rms"] > 0
    assert summary["between_image_rms"] > 0
    assert summary["within_image_token_rms"] == pytest.approx(0.0)
    assert summary["projector_effective_rank"] == pytest.approx(1.0)


def test_mean_direction_fraction_is_one_for_identical_images():
    sequences = [torch.tensor([[2.0, -1.0]]) for _ in range(3)]
    summary = summarize_batch_embeddings(sequences, sequences)
    assert summary["mean_direction_fraction"] == pytest.approx(1.0)


def test_probe_ratios_and_geometry_are_one_at_step0():
    sequences = [
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[-1.0, 0.0], [0.0, -1.0]]),
        torch.tensor([[2.0, 1.0], [1.0, 2.0]]),
    ]
    probe = summarize_probe(
        sequences,
        sequences,
        step0_projector_sequences=sequences,
        step0_receiver_sequences=sequences,
        step=0,
    )
    for role in ("projector", "receiver"):
        assert probe[role]["relative_spread_ratio"] == pytest.approx(1.0)
        assert probe[role]["effective_rank_ratio"] == pytest.approx(1.0)
        assert probe[role]["sample_rms_ratio"] == pytest.approx(1.0)
        assert probe[role]["pairwise_distance_correlation"] == pytest.approx(1.0)
        assert probe[role]["centered_gram_similarity"] == pytest.approx(1.0)


def test_critical_top1_and_rms_stop_training():
    contract = _contract()
    state = {}
    current = {
        "step": 1,
        "projector": {
            "relative_spread_ratio": 0.20,
            "effective_rank_ratio": 0.40,
            "top1_variance_fraction": 0.95,
            "sample_rms_ratio": 60.0,
        },
        "receiver": {
            "relative_spread_ratio": 0.20,
            "effective_rank_ratio": 0.40,
            "top1_variance_fraction": 0.95,
            "sample_rms_ratio": 60.0,
        },
        "has_nan_or_inf": False,
    }
    result = evaluate_guards(current, previous=None, state=state, contract=contract)
    assert result["stop"] is True
    assert "projector_top1_variance_critical" in result["critical"]
    assert "receiver_output_rms_critical" in result["critical"]


def test_two_causal_probe_failures_stop():
    contract = _contract()
    state = {}
    current = {
        "step": 1,
        "projector": {
            "relative_spread_ratio": 1.0,
            "effective_rank_ratio": 1.0,
            "top1_variance_fraction": 0.1,
            "sample_rms_ratio": 1.0,
        },
        "receiver": {
            "relative_spread_ratio": 1.0,
            "effective_rank_ratio": 1.0,
            "top1_variance_fraction": 0.1,
            "sample_rms_ratio": 1.0,
        },
        "causal": {
            "correct_preference": 0.2,
            "shuffled_preference": 0.3,
            "vision_minus_shuffle_correct_logp": -0.1,
        },
    }
    first = evaluate_guards(current, previous=None, state=state, contract=contract)
    assert first["stop"] is False
    second_current = {**current, "step": 2}
    second = evaluate_guards(
        second_current, previous=current, state=state, contract=contract
    )
    assert second["stop"] is True
    assert "causal_preference_critical" in second["critical"]


def test_health_contract_rejects_changed_thresholds():
    contract = _contract()
    validate_health_contract(contract)
    contract["guards"]["hard"]["relative_spread_ratio_min"] = 0.2
    with pytest.raises(ValueError):
        validate_health_contract(contract)
