import torch

from moonvit_glue.proxy_receiver import FixedPairwiseReceiverAdapter


def test_fixed_receiver_adapter_is_parameter_free_and_preserves_gradient_coverage():
    adapter = FixedPairwiseReceiverAdapter(
        canonical_width=8,
        receiver_width=4,
        seed=20260805,
    )
    inputs = torch.arange(16, dtype=torch.float32).reshape(2, 8).requires_grad_(True)
    outputs = adapter(inputs)
    outputs.sum().backward()

    assert outputs.shape == (2, 4)
    assert list(adapter.parameters()) == []
    assert torch.count_nonzero(inputs.grad).item() == inputs.numel()


def test_fixed_receiver_adapter_is_seeded_and_validates_two_to_one_width():
    inputs = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    first = FixedPairwiseReceiverAdapter(8, 4, seed=7)
    same = FixedPairwiseReceiverAdapter(8, 4, seed=7)
    different = FixedPairwiseReceiverAdapter(8, 4, seed=8)

    assert torch.equal(first.permutation, same.permutation)
    assert torch.equal(first.signs, same.signs)
    assert torch.equal(first(inputs), same(inputs))
    assert not torch.equal(first(inputs), different(inputs))

    try:
        FixedPairwiseReceiverAdapter(8, 3, seed=7)
    except ValueError as error:
        assert "twice" in str(error)
    else:
        raise AssertionError("expected a 2:1 width validation error")


def test_fixed_receiver_adapter_save_restore_freezes_exact_buffers(tmp_path):
    adapter = FixedPairwiseReceiverAdapter(8, 4, seed=20260805)
    adapter.save_pretrained(tmp_path)
    restored = FixedPairwiseReceiverAdapter.from_pretrained(tmp_path)

    assert restored.seed == 20260805
    assert torch.equal(restored.permutation, adapter.permutation)
    assert torch.equal(restored.signs, adapter.signs)
    assert torch.equal(
        restored(torch.arange(8, dtype=torch.float32)),
        adapter(torch.arange(8, dtype=torch.float32)),
    )
