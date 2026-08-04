"""固定 ID 的轨迹与控制输出统计比较。"""

from __future__ import annotations

import random


def paired_gap_stats(
    rows_a: list[dict],
    rows_b: list[dict],
    *,
    value_key: str = "score",
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict:
    """计算成对均值差与确定性的百分位 bootstrap 区间。"""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    by_a = {str(row["id"]): float(row[value_key]) for row in rows_a}
    by_b = {str(row["id"]): float(row[value_key]) for row in rows_b}
    if set(by_a) != set(by_b) or not by_a:
        raise ValueError("paired comparison requires identical non-empty sample IDs")
    identifiers = sorted(by_a)
    differences = [by_a[sample_id] - by_b[sample_id] for sample_id in identifiers]
    denominator = len(differences)
    mean_gap = sum(differences) / denominator
    try:
        import numpy as np

        values = np.asarray(differences, dtype=np.float64)
        generator = np.random.default_rng(seed)
        bootstraps = []
        # 分块向量化 pair bootstrap，避免完整矩阵分析退化为十亿级 Python 循环。
        for start in range(0, bootstrap_samples, 128):
            count = min(128, bootstrap_samples - start)
            indices = generator.integers(0, denominator, size=(count, denominator))
            bootstraps.extend(values[indices].mean(axis=1).tolist())
        bootstraps.sort()
    except ImportError:
        rng = random.Random(seed)
        bootstraps = sorted(
            sum(
                differences[rng.randrange(denominator)] for _ in range(denominator)
            )
            / denominator
            for _ in range(bootstrap_samples)
        )
    low_index = max(0, int(0.025 * bootstrap_samples) - 1)
    high_index = min(bootstrap_samples - 1, int(0.975 * bootstrap_samples))
    return {
        "denominator": denominator,
        "sum_a": sum(by_a.values()),
        "sum_b": sum(by_b.values()),
        "mean_a": sum(by_a.values()) / denominator,
        "mean_b": sum(by_b.values()) / denominator,
        "mean_gap": mean_gap,
        "a_only_better": sum(difference > 0 for difference in differences),
        "b_only_better": sum(difference < 0 for difference in differences),
        "equal": sum(difference == 0 for difference in differences),
        "ci95_low": bootstraps[low_index],
        "ci95_high": bootstraps[high_index],
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }
