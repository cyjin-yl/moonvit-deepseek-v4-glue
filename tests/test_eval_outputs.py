"""Eval outputs must be self-contained and aggregatable for HF publication."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from aggregate_eval import aggregate_reports
from eval_vlm import build_metadata, make_scored_row


def test_make_scored_row_is_self_contained():
    record = {"id": "x", "question": "Brand?", "answers": ["nike"], "metric": "exact_match"}
    row = make_scored_row(record, "Nike", 0)
    assert row["question"] == "Brand?"
    assert row["answers"] == ["nike"]
    assert row["prediction"] == "Nike"
    assert row["exact_match"] == 1.0


def test_make_scored_row_keeps_grounding_truth():
    record = {
        "id": "g", "question": "click save", "metric": "grounding",
        "gt_box": [1.0, 2.0, 3.0, 4.0],
    }
    row = make_scored_row(record, "click(start_box=[2,3])", 0)
    assert row["gt_box"] == [1.0, 2.0, 3.0, 4.0]
    assert "answers" not in row  # grounding records carry boxes, not text answers


def test_build_metadata_captures_run_context():
    args = SimpleNamespace(
        text_model="deepseek", vision_tower="v2", moonvit_v2_weights="/w/v2.safetensors",
        moonvit_model="moonshotai/MoonViT-SO-400M", projector="/ckpt/step-002100",
        data=Path("textvqa.jsonl"), limit=500, max_new_tokens=32, dtype="bfloat16", seed=0,
    )
    meta = build_metadata(args, git_sha="abc123")
    assert meta["text_model"] == "deepseek"
    assert meta["vision_tower"] == "v2"
    assert meta["vision_weights"] == "/w/v2.safetensors"
    assert meta["projector"] == "/ckpt/step-002100"
    assert meta["data"] == "textvqa.jsonl"
    assert meta["git"] == "abc123"
    assert meta["timestamp"]


def _report(name, vision, blind=None, mode="generation"):
    report = {"mode": mode, "summary": vision, "metadata": {"data": f"{name}.jsonl"}}
    if blind is not None:
        report["blind_summary"] = blind
    return report


def test_aggregate_builds_benchmark_matrix_with_blind_gap():
    summary = aggregate_reports([
        _report("textvqa", {"count": 3, "soft_vqa": 0.4}, {"count": 3, "soft_vqa": 0.25}),
        _report("ocrbench", {"count": 2, "exact_match": 0.5}),
    ])
    assert summary["benchmarks"]["textvqa"]["gap"]["soft_vqa"] == 0.15
    assert "gap" not in summary["benchmarks"]["ocrbench"]
    assert summary["skipped_non_generation"] == []


def test_aggregate_marks_in_domain_benchmarks_and_skips_shuffle_loss():
    summary = aggregate_reports([
        _report("screenspot", {"count": 2, "accuracy": 0.1, "parse_rate": 0.9},
                {"count": 2, "accuracy": 0.0, "parse_rate": 0.0}),
        _report("flickr8k", {"mean_delta": 0.0}, mode="shuffle_loss"),
    ])
    assert summary["benchmarks"]["screenspot"]["in_domain"] is True
    assert "flickr8k" not in summary["benchmarks"]
    assert summary["skipped_non_generation"] == ["flickr8k"]
