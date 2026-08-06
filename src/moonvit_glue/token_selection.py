"""确定性的视觉 token 选择与压缩工具。

这些操作只作用于 MoonViT 已缓存的 token 序列，不改变图像预处理。
``prefix`` 保留旧实验语义，``uniform`` 覆盖整个序列，``mean_pool``
按连续区间做平均，便于在固定 token 预算下检查空间覆盖是否是瓶颈。
"""

from __future__ import annotations

import torch


def select_visual_tokens(
    features: torch.Tensor,
    max_tokens: int,
    mode: str = "prefix",
) -> torch.Tensor:
    """返回最多 ``max_tokens`` 个视觉 token，保持尾部维度不变。

    ``features`` 的第一维是无 batch 的 token 轴，形状为 ``[T, ...]``。对 ``T <= max_tokens``
    的输入三种模式都原样返回（连续副本），避免无意义的重采样。
    """
    if features.ndim < 2:
        raise ValueError(f"features must have a token axis and at least one value axis, got {tuple(features.shape)}")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if mode not in {"prefix", "uniform", "mean_pool"}:
        raise ValueError(f"unknown token selection mode: {mode}")
    token_count = int(features.shape[0])
    if token_count == 0:
        raise ValueError("features must contain at least one token")
    target = min(token_count, int(max_tokens))
    if target == token_count:
        return features.contiguous()
    if mode == "prefix":
        return features[:target].contiguous()
    if mode == "uniform":
        # 整数 nearest-endpoint 公式，跨设备和 dtype 完全确定，并保留首尾。
        if target == 1:
            positions = torch.zeros(1, dtype=torch.long, device=features.device)
        else:
            numer = 2 * torch.arange(target, dtype=torch.long, device=features.device) * (token_count - 1)
            positions = (numer + (target - 1)) // (2 * (target - 1))
        return features.index_select(0, positions).contiguous()

    # 将完整序列切成 target 个连续区间后求均值，保留全图信息。
    pooled = []
    for index in range(target):
        start = (index * token_count) // target
        end = ((index + 1) * token_count) // target
        pooled.append(features[start:end].mean(dim=0))
    return torch.stack(pooled, dim=0).contiguous()
