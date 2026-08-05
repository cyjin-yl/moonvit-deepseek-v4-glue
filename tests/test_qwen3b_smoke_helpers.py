from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from smoke_qwen3b_proxy import _optimizer_states_equal, _verify_frozen_files


class FakeOptimizer:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cross-device regression")
def test_optimizer_state_verifier_compares_equal_cpu_and_cuda_scalars():
    common_group = [{"params": [0], "lr": 0.1}]
    cpu = FakeOptimizer(
        {
            "param_groups": common_group,
            "state": {0: {"step": torch.tensor(1.0), "exp_avg": torch.tensor([2.0])}},
        }
    )
    cuda = FakeOptimizer(
        {
            "param_groups": common_group,
            "state": {
                0: {
                    "step": torch.tensor(1.0, device="cuda"),
                    "exp_avg": torch.tensor([2.0], device="cuda"),
                }
            },
        }
    )
    changed = FakeOptimizer(
        {
            "param_groups": common_group,
            "state": {
                0: {
                    "step": torch.tensor(2.0, device="cuda"),
                    "exp_avg": torch.tensor([2.0], device="cuda"),
                }
            },
        }
    )

    assert _optimizer_states_equal(cpu, cuda)
    assert not _optimizer_states_equal(cpu, changed)


def test_frozen_file_verifier_requires_exact_bytes_and_sha256(tmp_path):
    payload = tmp_path / "weights.bin"
    payload.write_bytes(b"frozen-weights")
    contract = [
        {
            "path": payload.name,
            "bytes": len(b"frozen-weights"),
            "sha256": "a5eaf19bc006998f1b8b8901b0e420801199060ff436087d96e949d7c297d16c",
        }
    ]

    verified = _verify_frozen_files(tmp_path, contract, label="test weights")
    assert verified == contract

    payload.write_bytes(b"changed-weights")
    with pytest.raises(ValueError, match="test weights.*weights.bin"):
        _verify_frozen_files(tmp_path, contract, label="test weights")
