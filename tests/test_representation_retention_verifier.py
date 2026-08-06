import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from verify_qwen3b_representation_retention import pairwise_row_key


def test_pairwise_recomputation_ignores_safetensors_key_enumeration_order():
    rows = [
        {"representation": "step0", "left_index": 1, "right_index": 2},
        {"representation": "current", "left_index": 0, "right_index": 2},
        {"representation": "step0", "left_index": 0, "right_index": 1},
    ]

    assert sorted(rows, key=pairwise_row_key) == [
        {"representation": "current", "left_index": 0, "right_index": 2},
        {"representation": "step0", "left_index": 0, "right_index": 1},
        {"representation": "step0", "left_index": 1, "right_index": 2},
    ]
