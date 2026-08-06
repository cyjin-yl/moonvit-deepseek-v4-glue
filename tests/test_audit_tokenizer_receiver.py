from __future__ import annotations

import json

from tools.audit_tokenizer_receiver import build_report


def test_qwen25_visual_tokens_do_not_imply_native_vision(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 2048,
                "vocab_size": 151936,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "added_tokens_decoder": {
                    "151652": {"content": "<|vision_start|>", "special": True},
                    "151655": {"content": "<|image_pad|>", "special": True},
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_report(tmp_path)
    assert report["family"] == "qwen2"
    assert report["has_vision_config"] is False
    assert report["receiver_assessment"]["receiver_candidate"] is True
    assert report["image_tokens"]["ids"]["image_token_id_from_tokenizer"] == 151655


def test_qwen35_language_receiver_assessment(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "image_token_id": 248056,
                "vision_start_token_id": 248053,
                "vision_end_token_id": 248054,
                "vision_config": {"hidden_size": 1024},
                "text_config": {"hidden_size": 2560, "vocab_size": 248320},
            }
        ),
        encoding="utf-8",
    )
    report = build_report(tmp_path)
    assessment = report["receiver_assessment"]
    assert report["family"] == "qwen3.5"
    assert assessment["receiver_candidate"] is True
    assert assessment["receiver_hidden_size"] == 2560
    assert "language_model" in assessment["receiver_path"]
