import torch

from moonvit_glue.projector import ProjectorConfig, seeded_projector


def _state(projector):
    return {name: tensor.detach().clone() for name, tensor in projector.state_dict().items()}


def test_seeded_projector_is_exact_different_across_seeds_and_rng_isolated():
    config = ProjectorConfig(
        vision_width=3,
        language_width=8,
        merge_factor=2,
        projector_width=7,
    )
    torch.manual_seed(91)
    expected_next = torch.rand(4)
    torch.manual_seed(91)

    first = _state(seeded_projector(config, seed=20260805))
    actual_next = torch.rand(4)
    second = _state(seeded_projector(config, seed=20260805))
    random_control = _state(seeded_projector(config, seed=20260806))

    assert torch.equal(actual_next, expected_next)
    assert first.keys() == second.keys() == random_control.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert any(not torch.equal(first[name], random_control[name]) for name in first)
