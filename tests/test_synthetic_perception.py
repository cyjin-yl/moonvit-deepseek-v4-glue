"""Synthetic diagnostic data are paired, reproducible, and leakage-resistant."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from moonvit_glue.synthetic_perception import (
    SuiteConfig,
    generate_background_matched_aux,
    generate_suite,
    verify_suite,
)


TASKS = {"color", "shape", "count", "spatial", "ocr", "coordinate"}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_minimal_pairs_keep_question_and_flip_answer(tmp_path):
    output = tmp_path / "suite"
    generate_suite(output, SuiteConfig(samples_per_task=2, image_size=128, seed=17))

    rows = _rows(output / "train.jsonl")
    by_pair: dict[str, list[dict]] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], []).append(row)

    assert set(row["task"] for row in rows) == TASKS
    assert len(by_pair) == 2 * len(TASKS)
    for pair in by_pair.values():
        assert len(pair) == 2
        assert pair[0]["question"] == pair[1]["question"]
        assert pair[0]["answers"] != pair[1]["answers"]
        assert pair[0]["changed_attribute"] == pair[1]["changed_attribute"]
        assert pair[0]["image_sha256"] != pair[1]["image_sha256"]
        for row in pair:
            assert (output / row["image"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_splits_are_deterministic_and_templates_and_ocr_do_not_overlap(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    config = SuiteConfig(samples_per_task=3, image_size=96, seed=41)
    generate_suite(first, config)
    generate_suite(second, config)

    manifest_a = json.loads((first / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((second / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest_a["logical_dataset_sha256"] == manifest_b["logical_dataset_sha256"]
    assert manifest_a["counts"]["base_samples_by_split_task"] == {
        split: {task: 3 for task in sorted(TASKS)} for split in ("selection", "train")
    }
    assert manifest_a["leakage_checks"] == {
        "image_hash_overlap": 0,
        "ocr_string_overlap": 0,
        "pair_id_overlap": 0,
        "template_id_overlap": 0,
    }


def test_control_assignments_are_explicit_and_shuffle_is_a_derangement(tmp_path):
    output = tmp_path / "suite"
    generate_suite(output, SuiteConfig(samples_per_task=3, image_size=96, seed=5))

    controls = _rows(output / "controls.jsonl")
    assert controls
    for row in controls:
        assert row["blind_image"] is None
        assert row["blank_image"].endswith("blank.png")
        assert row["same_image_id"] != row["id"] or row["condition_notes"]["same_image_fixed_per_task"]
        assert row["shuffled_image_id"] != row["id"]
        assert row["patch_permutation"]["algorithm"] == "torch.randperm"
        assert isinstance(row["patch_permutation"]["seed"], int)


def test_verifier_checks_all_images_and_rejects_pair_question_tampering(tmp_path):
    output = tmp_path / "suite"
    generate_suite(output, SuiteConfig(samples_per_task=1, image_size=96, seed=13))
    verification = verify_suite(output)
    assert verification["status"] == "valid"
    assert verification["image_hashes_verified"] == 24

    rows = _rows(output / "selection.jsonl")
    rows[1]["question"] = "tampered question"
    with (output / "selection.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    try:
        verify_suite(output)
    except ValueError as error:
        assert "file hash mismatch" in str(error) or "pair question mismatch" in str(error)
    else:
        raise AssertionError("tampered synthetic suite was accepted")


def test_background_aux_changes_only_selection_background(tmp_path):
    authoritative = tmp_path / "authoritative"
    auxiliary = tmp_path / "background-matched"
    config = SuiteConfig(samples_per_task=1, image_size=96, seed=23)
    generate_suite(authoritative, config)

    manifest = generate_background_matched_aux(
        authoritative / "selection.jsonl",
        auxiliary,
        config,
    )

    source_rows = _rows(authoritative / "selection.jsonl")
    aux_rows = _rows(auxiliary / "selection_background_matched_aux.jsonl")
    assert len(aux_rows) == len(source_rows)
    for source, aux in zip(source_rows, aux_rows):
        assert aux["id"] == source["id"]
        assert aux["question"] == source["question"]
        assert aux["answers"] == source["answers"]
        assert aux["generation"] == source["generation"]
        assert aux["authoritative_image_sha256"] == source["image_sha256"]
    source_image = Image.open(authoritative / source_rows[0]["image"])
    aux_image = Image.open(auxiliary / aux_rows[0]["image"])
    assert source_image.getpixel((0, 0)) != aux_image.getpixel((0, 0))
    assert aux_image.getpixel((0, 0)) == (237, 243, 248)
    assert manifest["diagnostic_only"] is True
    assert manifest["records"] == len(source_rows)
