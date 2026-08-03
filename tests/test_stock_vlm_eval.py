"""Stock-VLM eval adapter: pure helpers keep the same contract as eval_vlm."""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from eval_stock_vlm import build_messages, load_record_image, strip_thinking


def test_strip_thinking_removes_closed_block():
    assert strip_thinking("<think>let me reason</think>\n cat") == "cat"


def test_strip_thinking_multiple_blocks_and_whitespace():
    assert strip_thinking("<think>a</think>  <think>b</think>   dog ") == "dog"


def test_strip_thinking_unclosed_block_leaves_no_answer():
    assert strip_thinking("<think>still reasoning...") == ""


def test_strip_thinking_passthrough_plain_text():
    assert strip_thinking("42") == "42"


def test_build_messages_image_first():
    messages = build_messages("What brand?", with_image=True)
    assert messages == [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What brand?"}]}
    ]


def test_build_messages_blind_is_text_only():
    messages = build_messages("What brand?", with_image=False)
    assert messages == [{"role": "user", "content": [{"type": "text", "text": "What brand?"}]}]


def test_load_record_image_from_packed_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(buffer, format="PNG")
    image = load_record_image({"image_bytes": buffer.getvalue()}, base_dir=Path("."))
    assert image.size == (4, 4) and image.mode == "RGB"


def test_load_record_image_from_jsonl_path(tmp_path):
    Image.new("RGB", (3, 5)).save(tmp_path / "im.png")
    image = load_record_image({"image": "im.png"}, base_dir=tmp_path)
    assert image.size == (3, 5)


def test_stock_row_scores_through_shared_contract():
    # make_scored_row is imported from eval_vlm, so stock rows aggregate identically
    from eval_stock_vlm import make_scored_row

    row = make_scored_row(
        {"id": "s1", "question": "Brand?", "answers": ["nike"], "metric": "exact_match"},
        "Nike",
        0,
    )
    assert row["exact_match"] == 1.0
    assert row["prediction"] == "Nike"


@pytest.mark.parametrize("raw,expected", [("<think>x</think>(512, 341)", "(512, 341)")])
def test_grounding_prediction_survives_think_strip(raw, expected):
    from moonvit_glue.metrics import extract_point

    assert extract_point(strip_thinking(raw)) == (512.0, 341.0)
