import random

import torch

from moonvit_glue.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from moonvit_glue.projector import PatchMergerProjector, ProjectorConfig


def _toy_projector() -> PatchMergerProjector:
    return PatchMergerProjector(
        ProjectorConfig(vision_width=3, language_width=5, merge_factor=2, projector_width=6)
    )


def test_resume_reproduces_training_trajectory(tmp_path):
    torch.manual_seed(0)
    inputs = [torch.randn(4, 2, 3) for _ in range(4)]

    def step_loss(model, optimizer):
        optimizer.zero_grad()
        loss = sum(out.pow(2).mean() for out in model(inputs))
        loss.backward()
        optimizer.step()
        return float(loss)

    projector = _toy_projector()
    optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3)
    rng = random.Random(0)
    history = [
        {"step": 1, "loss": step_loss(projector, optimizer)},
        {"step": 2, "loss": step_loss(projector, optimizer)},
    ]
    save_training_checkpoint(
        directory=tmp_path / "ckpt",
        projector=projector,
        optimizer=optimizer,
        step=2,
        history=history,
        rng=rng,
    )

    fresh = _toy_projector()
    fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    step, restored_history, restored_rng, directory = load_training_checkpoint(
        source=tmp_path / "ckpt", projector=fresh, optimizer=fresh_optimizer, device="cpu"
    )

    assert directory == tmp_path / "ckpt"
    assert step == 2
    assert restored_history == history
    assert restored_rng.getstate() == rng.getstate()
    assert (tmp_path / "ckpt" / "projector_bf16.safetensors").exists()
    assert (tmp_path / "ckpt" / "history.json").exists()

    # One more identical step on both must give identical loss and weights —
    # this only holds if optimizer moments were restored exactly.
    original_loss = step_loss(projector, optimizer)
    restored_loss = step_loss(fresh, fresh_optimizer)
    assert original_loss == restored_loss
    for original, copy in zip(projector.parameters(), fresh.parameters()):
        assert torch.equal(original, copy)


def test_resume_from_hf_repo_picks_latest_checkpoint(tmp_path, monkeypatch):
    """HF-repo resume scans checkpoints/step-* and takes the latest."""

    for step in (100, 300):
        projector = _toy_projector()
        optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3)
        save_training_checkpoint(
            directory=tmp_path / "repo" / "checkpoints" / f"step-{step:06d}",
            projector=projector,
            optimizer=optimizer,
            step=step,
            history=[{"step": step, "loss": 1.0 / step}],
            rng=random.Random(step),
        )

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda repo_id: tmp_path / "repo"
    )
    fresh = _toy_projector()
    fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    step, history, _, directory = load_training_checkpoint(
        source="user/fake-repo", projector=fresh, optimizer=fresh_optimizer, device="cpu"
    )
    assert step == 300
    assert history == [{"step": 300, "loss": 1.0 / 300}]
    assert directory.name == "step-000300"
