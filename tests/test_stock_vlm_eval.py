"""Stock-VLM eval adapter: pure helpers keep the same contract as eval_vlm."""

import hashlib
import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from eval_stock_vlm import (
    build_messages,
    load_record_image,
    strip_thinking,
    verify_weight_manifest,
)


def test_strip_thinking_removes_closed_block():
    assert strip_thinking("<think>let me reason</think>\n cat") == "cat"


def test_strip_thinking_multiple_blocks_and_whitespace():
    assert strip_thinking("<think>a</think>  <think>b</think>   dog ") == "dog"


def test_strip_thinking_unclosed_block_leaves_no_answer():
    assert strip_thinking("<think>still reasoning...") == ""


def test_strip_thinking_passthrough_plain_text():
    assert strip_thinking("42") == "42"


def test_build_messages_image_first():
    messages = build_messages(
        {"question": "What brand?", "metric": "soft_vqa", "answers": ["Dakota"]},
        with_image=True,
    )
    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "What brand?\nRespond with only the short answer. Do not explain.",
                },
            ],
        }
    ]


def test_build_messages_blind_is_text_only():
    messages = build_messages(
        {"question": "What brand?", "metric": "soft_vqa", "answers": ["Dakota"]},
        with_image=False,
    )
    assert messages == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What brand?\nRespond with only the short answer. Do not explain.",
                }
            ],
        }
    ]


def test_build_messages_requests_normalized_coordinates_for_grounding():
    messages = build_messages(
        {"question": "Click the settings icon", "metric": "grounding", "gt_box": [1, 2, 3, 4]},
        with_image=True,
    )

    assert messages[0]["content"][-1]["text"] == (
        "Click the settings icon\n"
        "Respond with only the target point as (x, y), where x and y are integers "
        "normalized to 0–999. Do not explain."
    )


def test_build_messages_requests_only_the_option_letter_for_multiple_choice():
    messages = build_messages(
        {
            "question": "Which is correct?\nOptions:\nA. one\nB. two",
            "metric": "exact_match",
            "answers": ["B"],
        },
        with_image=True,
    )

    assert messages[0]["content"][-1]["text"].endswith(
        "Respond with only the option letter. Do not explain."
    )


def test_load_record_image_from_packed_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(buffer, format="PNG")
    image = load_record_image({"image_bytes": buffer.getvalue()}, base_dir=Path("."))
    assert image.size == (4, 4) and image.mode == "RGB"


def test_load_record_image_from_jsonl_path(tmp_path):
    Image.new("RGB", (3, 5)).save(tmp_path / "im.png")
    image = load_record_image({"image": "im.png"}, base_dir=tmp_path)
    assert image.size == (3, 5)


def test_load_record_image_limits_long_side_without_changing_aspect_ratio():
    buffer = io.BytesIO()
    Image.new("RGB", (1024, 512), color=(20, 40, 60)).save(buffer, format="PNG")

    image = load_record_image(
        {"image_bytes": buffer.getvalue()},
        base_dir=Path("."),
        max_image_side=448,
    )

    assert image.size == (448, 224)


def test_verify_weight_manifest_rejects_a_same_size_corrupted_shard(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shard = model_dir / "model-00001.safetensors"
    shard.write_bytes(b"correct tensor bytes")
    manifest = tmp_path / "weights.sha256.json"
    manifest.write_text(
        json.dumps({shard.name: hashlib.sha256(shard.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    verify_weight_manifest(model_dir, manifest)

    shard.write_bytes(b"corrupt tensor byte!")  # same byte length, different content

    with pytest.raises(ValueError, match="sha256 mismatch.*model-00001"):
        verify_weight_manifest(model_dir, manifest)


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
