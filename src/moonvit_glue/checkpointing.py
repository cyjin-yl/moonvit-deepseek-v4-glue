"""Periodic training checkpoints with optional Hugging Face upload.

Every N steps the trainer writes a self-contained checkpoint — projector in
fp32 and bf16, optimizer state, RNG states, step counter, loss history — and,
when a repo id is configured, pushes it in a background thread. Checkpoints
double as the open-source trail (the community can watch the run form) and as
crash insurance: any checkpoint restores the exact training trajectory via
``load_training_checkpoint`` (``--resume`` in ``tools/train_overfit.py``).
Upload failures never stop training; the next checkpoint retries.
"""

from __future__ import annotations

import json
import random
import threading
from pathlib import Path

import torch
from safetensors.torch import save_file

from .projector import PatchMergerProjector

STATE_FILENAME = "training_state.pt"
BF16_FILENAME = "projector_bf16.safetensors"
HISTORY_FILENAME = "history.json"


def save_training_checkpoint(
    *,
    directory: str | Path,
    projector: PatchMergerProjector,
    optimizer: torch.optim.Optimizer,
    step: int,
    history: list[dict],
    rng: random.Random,
) -> Path:
    """Write a resumable checkpoint (weights fp32+bf16, optimizer, RNG, step)."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    projector.save_pretrained(directory)
    bf16_state = {
        key: value.detach().to(torch.bfloat16).contiguous().cpu()
        for key, value in projector.state_dict().items()
    }
    save_file(bf16_state, str(directory / BF16_FILENAME))
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "step": step,
            "history": history,
            "python_rng": rng.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": cuda_rng,
        },
        directory / STATE_FILENAME,
    )
    (directory / HISTORY_FILENAME).write_text(
        json.dumps({"step": step, "history": history}, indent=2) + "\n",
        encoding="utf-8",
    )
    return directory


def load_training_checkpoint(
    *,
    source: str | Path,
    projector: PatchMergerProjector,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
) -> tuple[int, list[dict], random.Random, Path]:
    """Restore projector, optimizer, RNG and step from a local dir or HF repo.

    ``source`` may be a checkpoint directory, or an HF repo id whose
    ``checkpoints/step-*`` folders are scanned for the latest step.
    """

    directory = Path(source)
    if not directory.exists():
        from huggingface_hub import snapshot_download

        repo_dir = Path(snapshot_download(str(source)))
        candidates = sorted((repo_dir / "checkpoints").glob("step-*"))
        if not candidates:
            raise FileNotFoundError(f"no checkpoints/step-* found in HF repo {source}")
        directory = candidates[-1]

    state = torch.load(
        directory / STATE_FILENAME, map_location="cpu", weights_only=False
    )
    restored = PatchMergerProjector.from_pretrained(directory, device=device)
    projector.load_state_dict(restored.state_dict())
    optimizer.load_state_dict(state["optimizer"])
    for param_state in optimizer.state.values():
        for key, value in param_state.items():
            if torch.is_tensor(value):
                param_state[key] = value.to(device)

    rng = random.Random()
    rng.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state.get("cuda_rng") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["step"], state["history"], rng, directory


class CheckpointUploader:
    """Serialize background uploads of checkpoint folders to one HF repo.

    Uploads run one at a time in a daemon thread; training never blocks except
    to serialize behind the previous upload. Errors are recorded, not raised.
    """

    def __init__(self, repo_id: str):
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.api = HfApi()
        self.api.create_repo(repo_id, repo_type="model", exist_ok=True)
        self._thread: threading.Thread | None = None
        self.errors: list[str] = []

    def upload_async(self, directory: str | Path, path_in_repo: str) -> None:
        if self._thread is not None:
            self._thread.join()  # one at a time; checkpoints are minutes apart

        def _run() -> None:
            try:
                self.api.upload_folder(
                    repo_id=self.repo_id,
                    repo_type="model",
                    folder_path=str(directory),
                    path_in_repo=path_in_repo,
                )
                print(f"[uploader] uploaded {path_in_repo or '.'}", flush=True)
            except Exception as exc:  # never kill training on upload failure
                self.errors.append(f"{path_in_repo}: {exc}")
                print(
                    f"[uploader] WARNING upload failed for {path_in_repo or '.'}: {exc}",
                    flush=True,
                )

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None
