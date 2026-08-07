"""Qwen 代理主干的固定 4096→2048 接收读出。

该模块没有可训练参数。主 projector 仍输出 DeepSeek 合同要求的 4096 维，
Qwen 代理只通过一个固定、行正交的 signed-pair projection 接收其信号。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn


class FixedPairwiseReceiverAdapter(nn.Module):
    """用确定性 signed pairs 把 2N 维读成 N 维，不引入可训练参数。"""

    config_filename = "proxy_receiver_config.json"
    weights_filename = "proxy_receiver.safetensors"

    def __init__(
        self,
        canonical_width: int,
        receiver_width: int,
        *,
        seed: int,
    ) -> None:
        super().__init__()
        if canonical_width != 2 * receiver_width:
            raise ValueError("canonical_width must be exactly twice receiver_width")
        if receiver_width <= 0:
            raise ValueError("receiver_width must be positive")
        self.canonical_width = int(canonical_width)
        self.receiver_width = int(receiver_width)
        self.seed = int(seed)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        permutation = torch.randperm(self.canonical_width, generator=generator)
        signs = torch.randint(
            0,
            2,
            (self.canonical_width,),
            generator=generator,
            dtype=torch.int8,
        ).mul_(2).sub_(1)
        self.register_buffer("permutation", permutation, persistent=True)
        self.register_buffer("signs", signs, persistent=True)

    def extra_repr(self) -> str:
        return (
            f"canonical_width={self.canonical_width}, "
            f"receiver_width={self.receiver_width}, seed={self.seed}"
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.canonical_width:
            raise ValueError(
                f"expected last dimension {self.canonical_width}, got {inputs.shape[-1]}"
            )
        ordered = inputs.index_select(-1, self.permutation)
        signed = ordered * self.signs.to(dtype=inputs.dtype)
        paired = signed.reshape(*inputs.shape[:-1], self.receiver_width, 2)
        return paired.sum(dim=-1) / math.sqrt(2.0)

    def save_pretrained(self, directory: str | Path) -> None:
        """保存精确 permutation/sign buffers，规避跨 torch 版本 RNG 漂移。"""

        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / self.config_filename).write_text(
            json.dumps(
                {
                    "format_version": "fixed-pairwise-receiver-v1",
                    "canonical_width": self.canonical_width,
                    "receiver_width": self.receiver_width,
                    "seed": self.seed,
                    "trainable_parameter_count": 0,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        save_file(
            {
                "permutation": self.permutation.detach().contiguous().cpu(),
                "signs": self.signs.detach().contiguous().cpu(),
            },
            str(destination / self.weights_filename),
            metadata={"format": "fixed-pairwise-receiver-v1"},
        )

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "FixedPairwiseReceiverAdapter":
        source = Path(directory)
        config = json.loads((source / cls.config_filename).read_text(encoding="utf-8"))
        adapter = cls(
            int(config["canonical_width"]),
            int(config["receiver_width"]),
            seed=int(config["seed"]),
        )
        state = load_file(str(source / cls.weights_filename), device=str(device))
        adapter.load_state_dict(state, strict=True)
        if sorted(adapter.permutation.tolist()) != list(range(adapter.canonical_width)):
            raise ValueError("saved receiver permutation is invalid")
        if set(adapter.signs.tolist()) - {-1, 1}:
            raise ValueError("saved receiver signs must be -1 or +1")
        return adapter.to(device=device)


class FixedGroupedReceiverAdapter(nn.Module):
    """Deterministic parameter-free 4096->receiver-width readout.

    The canonical projector stays 4096-dimensional for DeepSeek transfer. A
    proxy receiver with a different hidden width gets a fixed signed grouping;
    no receiver-specific trainable cross-attention or visual layer is added.
    """

    def __init__(self, canonical_width: int, receiver_width: int, *, seed: int) -> None:
        super().__init__()
        if not 0 < receiver_width <= canonical_width:
            raise ValueError("receiver width must be in (0, canonical width]")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        permutation = torch.randperm(canonical_width, generator=generator)
        signs = torch.randint(0, 2, (canonical_width,), generator=generator, dtype=torch.int8)
        signs = signs.mul_(2).sub_(1)
        pair_count = canonical_width - receiver_width
        group_sizes = [2] * pair_count + [1] * (2 * receiver_width - canonical_width)
        if len(group_sizes) != receiver_width or sum(group_sizes) != canonical_width:
            raise AssertionError("invalid grouped receiver layout")
        self.register_buffer("permutation", permutation, persistent=True)
        self.register_buffer("signs", signs, persistent=True)
        self.register_buffer("group_sizes", torch.tensor(group_sizes, dtype=torch.long), persistent=True)
        self.canonical_width = int(canonical_width)
        self.receiver_width = int(receiver_width)
        self.seed = int(seed)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.canonical_width:
            raise ValueError(f"expected last dimension {self.canonical_width}, got {inputs.shape[-1]}")
        values = inputs.index_select(-1, self.permutation) * self.signs.to(inputs.dtype)
        outputs = []
        offset = 0
        for size in self.group_sizes.tolist():
            outputs.append(values[..., offset:offset + size].sum(dim=-1) / math.sqrt(float(size)))
            offset += size
        return torch.stack(outputs, dim=-1)
