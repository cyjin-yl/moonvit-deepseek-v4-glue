"""Trajectory data preparation materializes only selection parity and held-out rows."""

import json
from collections import Counter

import pytest
from PIL import Image

from moonvit_glue.trajectory_data import (
    configured_conditions,
    make_shape_matched_blank_controls,
    make_pair_stratified_subset,
    make_stratified_control_records,
    prepare_trajectory_data,
)


def test_pair_stratified_subset_is_complete_balanced_and_reproducible():
    records = []
    for task in ("color", "shape"):
        for pair_index in range(4):
            for variant in ("a", "b"):
                records.append(
                    {
                        "id": f"{task}-{pair_index}-{variant}",
                        "task": task,
                        "pair_id": f"{task}-{pair_index}",
                        "pair_variant": variant,
                    }
                )

    first = make_pair_stratified_subset(records, pairs_per_task=2, seed=17)
    second = make_pair_stratified_subset(records, pairs_per_task=2, seed=17)

    assert first == second
    assert len(first) == 8
    assert Counter(row["task"] for row in first) == {"color": 4, "shape": 4}
    selected = Counter(row["pair_id"] for row in first)
    assert set(selected.values()) == {2}


def test_pair_stratified_subset_rejects_incomplete_pairs():
    records = [
        {"id": "x-a", "task": "color", "pair_id": "x", "pair_variant": "a"}
    ]
    with pytest.raises(ValueError, match="invalid minimal pair"):
        make_pair_stratified_subset(records, pairs_per_task=1, seed=1)


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_preparation_keeps_even_selection_and_historical_tail(tmp_path):
    eval_file = tmp_path / "textvqa.jsonl"
    train_file = tmp_path / "train.jsonl"
    _write(eval_file, [
        {"id": f"e{i}", "image": f"e{i}.png", "question": "q", "answers": ["a"], "metric": "exact_match"}
        for i in range(6)
    ])
    _write(train_file, [
        {"id": f"t{i}", "image": f"t{i}.png", "question": "q", "answers": ["a"], "source": "s"}
        for i in range(7)
    ])

    out = tmp_path / "out"
    manifest = prepare_trajectory_data(
        eval_files=[eval_file], train_file=train_file, output_dir=out, heldout_count=3
    )
    selection = [json.loads(line) for line in (out / "benchmark_selection.jsonl").read_text().splitlines()]
    heldout = [json.loads(line) for line in (out / "historical_heldout.jsonl").read_text().splitlines()]
    assert [row["id"] for row in selection] == ["e0", "e2", "e4"]
    assert all(row["_selection_parity"] == "even" for row in selection)
    assert [row["id"] for row in heldout] == ["t4", "t5", "t6"]
    assert manifest["discipline"]["final_half_materialized"] is False
    assert manifest["counts"]["benchmark_selection"] == 3
    assert manifest["counts"]["historical_heldout"] == 3


def test_benchmark_controls_are_stratified_derangements_and_reproducible():
    records = [
        {"id": f"a{i}", "benchmark": "a"} for i in range(4)
    ] + [
        {"id": f"b{i}", "benchmark": "b"} for i in range(3)
    ]

    first = make_stratified_control_records(records, seed=17, split="selection")
    second = make_stratified_control_records(records, seed=17, split="selection")

    assert first == second
    benchmark_by_id = {row["id"]: row["benchmark"] for row in records}
    assert {row["id"] for row in first} == set(benchmark_by_id)
    assert all(row["shuffled_image_id"] != row["id"] for row in first)
    assert all(
        benchmark_by_id[row["shuffled_image_id"]] == benchmark_by_id[row["id"]]
        for row in first
    )
    assert all(row["split"] == "selection" for row in first)
    assert all(isinstance(row["patch_permutation"]["seed"], int) for row in first)


def test_checkpoint_specific_conditions_extend_the_required_base_matrix():
    dataset = {
        "conditions": ["vision", "blind"],
        "checkpoint_condition_extensions": [
            {
                "checkpoint_ids": ["c0", "c2"],
                "conditions": ["blank", "shuffled_image"],
            }
        ],
    }

    assert configured_conditions(dataset, "c0") == [
        "vision",
        "blind",
        "blank",
        "shuffled_image",
    ]
    assert configured_conditions(dataset, "c1") == ["vision", "blind"]


def test_blank_controls_match_representative_image_shape_per_cached_grid(tmp_path):
    first_image = tmp_path / "a.png"
    second_image = tmp_path / "b.png"
    Image.new("RGB", (80, 40), (1, 2, 3)).save(first_image)
    Image.new("RGB", (40, 80), (4, 5, 6)).save(second_image)
    records = [
        {"id": "a", "image": str(first_image), "benchmark": "x"},
        {"id": "b", "image": str(second_image), "benchmark": "x"},
    ]
    controls = make_stratified_control_records(records, seed=2, split="selection")
    cache_records = [
        {"id": "a", "status": "ok", "feature_shape": [10, 4, 1024]},
        {"id": "b", "status": "ok", "feature_shape": [20, 4, 1024]},
    ]

    updated, blank_records = make_shape_matched_blank_controls(
        records,
        controls,
        cache_records,
        output_dir=tmp_path / "blank",
    )

    assert len(blank_records) == 2
    blank_by_id = {row["id"]: row for row in blank_records}
    assert {row["blank_image_id"] for row in updated} == set(blank_by_id)
    for row in updated:
        blank = blank_by_id[row["blank_image_id"]]
        expected_size = Image.open(first_image if row["id"] == "a" else second_image).size
        with Image.open(blank["image"]) as image:
            assert image.size == expected_size
            assert image.getpixel((0, 0)) == (255, 255, 255)
