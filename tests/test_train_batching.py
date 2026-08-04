"""Batching for real alignment runs: samples in a batch must be distinct."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from types import SimpleNamespace

import pytest

from tools_common import next_batch, validate_text_only_backbone_config


def test_next_batch_returns_distinct_consecutive_records_and_advances_cursor():
    records = [{"id": i} for i in range(10)]
    batch, cursor = next_batch(records, cursor=0, batch_size=4)
    assert [record["id"] for record in batch] == [0, 1, 2, 3]
    assert cursor == 4
    batch, cursor = next_batch(records, cursor=cursor, batch_size=4)
    assert [record["id"] for record in batch] == [4, 5, 6, 7]


def test_next_batch_wraps_around_epoch_boundary_and_keeps_cursor_monotonic():
    records = [{"id": i} for i in range(10)]
    batch, cursor = next_batch(records, cursor=8, batch_size=4)
    assert [record["id"] for record in batch] == [8, 9, 0, 1]
    # Cursor must stay monotonic: resume reconstructs it as start_step * batch_size.
    assert cursor == 12


def test_text_only_backbone_guard_accepts_plain_causal_lm():
    config = SimpleNamespace(
        architectures=["Qwen2ForCausalLM"],
        model_type="qwen2",
        vision_config=None,
    )

    validate_text_only_backbone_config(config)


def test_text_only_backbone_guard_rejects_native_vlm():
    config = SimpleNamespace(
        architectures=["Qwen3_5ForConditionalGeneration"],
        model_type="qwen3_5",
        vision_config=SimpleNamespace(hidden_size=1024),
    )

    with pytest.raises(ValueError, match="native multimodal"):
        validate_text_only_backbone_config(config)
