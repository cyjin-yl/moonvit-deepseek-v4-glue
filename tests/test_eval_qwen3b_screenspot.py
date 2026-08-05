import json
from pathlib import Path
import sys

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from eval_qwen3b_screenspot import decode_continuation, read_partial


class _Tokenizer:
    def decode(self, values, skip_special_tokens=True):
        assert skip_special_tokens is True
        return " ".join(str(value) for value in values)


def test_decode_continuation_uses_expanded_visual_prefix():
    token_ids, text = decode_continuation(
        _Tokenizer(), torch.tensor([[1, 2, 3, 40, 41]]), 3
    )
    assert token_ids == [40, 41]
    assert text == "40 41"


def test_partial_predictions_must_be_an_exact_manifest_prefix(tmp_path):
    path = tmp_path / "vision.partial.jsonl"
    path.write_text(
        json.dumps({"sample_id": "a", "prediction": "x"}) + "\n",
        encoding="utf-8",
    )
    assert read_partial(path, ["a", "b"])[0]["sample_id"] == "a"
    with pytest.raises(ValueError, match="exact manifest prefix"):
        read_partial(path, ["b", "a"])
