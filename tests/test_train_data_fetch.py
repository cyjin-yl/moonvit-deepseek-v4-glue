"""Train-split dataset specs: short-answer enforcement and image format control."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from fetch_eval_data import DATASETS, image_name_for, keep_record


def test_train_specs_target_train_splits_of_the_same_repos_as_eval():
    assert DATASETS["textvqa_train"].repo == "lmms-lab/textvqa"
    assert DATASETS["textvqa_train"].split == "train"
    assert DATASETS["docvqa_train"].repo == "lmms-lab/DocVQA"
    assert DATASETS["docvqa_train"].split == "train"
    assert DATASETS["flickr8k_train"].repo == "jxie/flickr8k"
    assert DATASETS["flickr8k_train"].split == "train"


def test_train_specs_enforce_short_answers_per_the_baseten_recipe():
    for name in ("textvqa_train", "docvqa_train", "flickr8k_train"):
        spec = DATASETS[name]
        assert spec.max_answer_words is not None, f"{name} must cap answer length"
        assert spec.max_answer_words <= 25, f"{name} cap too lax for grokking"


def test_photo_datasets_save_jpeg_to_control_archive_size():
    assert image_name_for("textvqa_train_000001", "jpeg") == "textvqa_train_000001.jpg"
    assert image_name_for("docvqa_train_000001", "png") == "docvqa_train_000001.png"
    assert DATASETS["textvqa_train"].image_format == "jpeg"
    assert DATASETS["flickr8k_train"].image_format == "jpeg"
    assert DATASETS["docvqa_train"].image_format == "png"  # scans compress well as PNG


def test_keep_record_drops_rows_without_any_short_answer():
    assert keep_record(["yes"], max_words=8)
    assert keep_record(["a quite long answer here", "ok"], max_words=8)
    assert not keep_record(["a quite long answer that just keeps going on"], max_words=4)
    assert keep_record(["short"], max_words=None)
