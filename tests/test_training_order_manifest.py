import hashlib
import json

import pytest
from PIL import Image

from moonvit_glue.training_order import (
    GROUNDING_ENRICHED_SELECTION_RULE,
    build_training_order_manifest,
    grounding_enriched_source_indices,
    load_ordered_records,
    verify_training_order_manifest,
)


def _write_fixture(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    records = []
    for index, source in enumerate(("textvqa_train", "showui_desktop", "docvqa_train")):
        image_path = images / f"sample-{index}.png"
        Image.new("RGB", (8 + index, 6 + index), (index * 40, 20, 30)).save(image_path)
        records.append(
            {
                "id": f"sample-{index}",
                "image": f"images/{image_path.name}",
                "question": f"question {index}",
                "answers": (
                    ["click(start_box=[12,34])"]
                    if source == "showui_desktop"
                    else [f"answer {index}"]
                ),
                "source": source,
            }
        )
    data_path = tmp_path / "train_mix.jsonl"
    data_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    contract = {
        "datasets": {
            "training_pack": {
                "records": 3,
                "sha256": data_sha256,
                "order_is_frozen": True,
            }
        },
        "training_budget": {
            "examples_seen_checkpoints": [2],
            "optimizer_steps_checkpoints": [1],
            "micro_batch_size": 1,
            "gradient_accumulation": 2,
            "real_global_batch": 2,
        },
        "image_preprocessing": {
            "train_max_image_side": 448,
            "train_max_visual_tokens": 256,
        },
        "vision_tower": {
            "name": "MoonViT-V2",
            "extracted_weights_sha256": "a" * 64,
        },
    }
    return data_path, records, contract


def test_training_order_manifest_preserves_prefix_and_binds_images(tmp_path):
    data_path, records, contract = _write_fixture(tmp_path)
    manifest = build_training_order_manifest(
        data_path=data_path,
        contract=contract,
        contract_sha256="b" * 64,
        examples_seen=2,
    )

    assert verify_training_order_manifest(manifest)
    assert [row["id"] for row in manifest["records"]] == ["sample-0", "sample-1"]
    assert manifest["selection"]["rule"] == "first_n_rows_preserve_source_order"
    assert manifest["selection"]["optimizer_steps"] == 1
    assert manifest["selection"]["subset_passes"] == 1.0
    assert manifest["selection"]["effective_epochs"] == pytest.approx(2 / 3)
    assert manifest["source_counts"] == {"showui_desktop": 1, "textvqa_train": 1}
    assert manifest["prompt_route_counts"] == {"grounding": 1, "short_answer": 1}
    assert manifest["records"][1]["target_answer"] == "click(start_box=[12, 34])"
    assert all(len(row["image_sha256"]) == 64 for row in manifest["records"])

    loaded = load_ordered_records(data_path=data_path, manifest=manifest)
    assert loaded == records[:2]


def test_training_order_manifest_rejects_mutation_and_non_budget_count(tmp_path):
    data_path, _, contract = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="preregistered examples-seen checkpoint"):
        build_training_order_manifest(
            data_path=data_path,
            contract=contract,
            contract_sha256="b" * 64,
            examples_seen=3,
        )

    manifest = build_training_order_manifest(
        data_path=data_path,
        contract=contract,
        contract_sha256="b" * 64,
        examples_seen=2,
    )
    manifest["records"][0]["id"] = "changed"
    assert not verify_training_order_manifest(manifest)


def test_training_order_manifest_rejects_ambiguous_grounding_target(tmp_path):
    data_path, records, contract = _write_fixture(tmp_path)
    records[1]["answers"] = ["try [12, 34] or [56, 78]"]
    data_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    contract["datasets"]["training_pack"]["sha256"] = hashlib.sha256(
        data_path.read_bytes()
    ).hexdigest()

    with pytest.raises(ValueError, match="strict click action grammar"):
        build_training_order_manifest(
            data_path=data_path,
            contract=contract,
            contract_sha256="b" * 64,
            examples_seen=2,
        )


def test_training_order_manifest_preserves_punctuation_only_vqa_target(tmp_path):
    data_path, records, contract = _write_fixture(tmp_path)
    records[0]["answers"] = ["(", "(", "parentheses"]
    data_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    contract["datasets"]["training_pack"]["sha256"] = hashlib.sha256(
        data_path.read_bytes()
    ).hexdigest()

    manifest = build_training_order_manifest(
        data_path=data_path,
        contract=contract,
        contract_sha256="b" * 64,
        examples_seen=2,
    )

    assert manifest["records"][0]["target_answer"] == "("
    assert (
        manifest["records"][0]["target_transform"]
        == "vqa_raw_majority_empty_normalization_fallback"
    )
    assert verify_training_order_manifest(manifest)


def test_grounding_enriched_order_selects_first_rows_per_route_and_alternates(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    sources = (
        "textvqa_train",
        "showui_desktop",
        "docvqa_train",
        "showui_desktop",
        "train",
        "showui_desktop",
    )
    records = []
    for index, source in enumerate(sources):
        image_path = images / f"enriched-{index}.png"
        Image.new("RGB", (9 + index, 7), (index * 20, 30, 40)).save(image_path)
        records.append(
            {
                "id": f"enriched-{index}",
                "image": f"images/{image_path.name}",
                "question": f"question {index}",
                "answers": (
                    [f"click(start_box=[{index},{index + 1}])"]
                    if source == "showui_desktop"
                    else [f"answer {index}"]
                ),
                "source": source,
            }
        )
    data_path = tmp_path / "train_mix.jsonl"
    data_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    contract = {
        "datasets": {
            "training_pack": {
                "records": len(records),
                "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "order_is_frozen": True,
            }
        },
        "training_budget": {
            "examples_seen_checkpoints": [4],
            "optimizer_steps_checkpoints": [2],
            "micro_batch_size": 1,
            "gradient_accumulation": 2,
            "real_global_batch": 2,
        },
        "image_preprocessing": {
            "train_max_image_side": 448,
            "train_max_visual_tokens": 256,
        },
        "vision_tower": {
            "name": "MoonViT-V2",
            "extracted_weights_sha256": "a" * 64,
        },
    }

    source_indices = grounding_enriched_source_indices(
        records,
        grounding_examples=2,
        short_answer_examples=2,
    )
    assert source_indices == [1, 0, 3, 2]
    manifest = build_training_order_manifest(
        data_path=data_path,
        contract=contract,
        contract_sha256="b" * 64,
        examples_seen=4,
        source_indices=source_indices,
        selection_rule=GROUNDING_ENRICHED_SELECTION_RULE,
        selection_metadata={
            "grounding_examples": 2,
            "short_answer_examples": 2,
            "within_route_order": "frozen_source_order",
            "merge_rule": "alternate_grounding_then_short_answer",
        },
    )

    assert verify_training_order_manifest(manifest)
    assert manifest["prompt_route_counts"] == {"grounding": 2, "short_answer": 2}
    assert [row["source_row_index"] for row in manifest["records"]] == [1, 0, 3, 2]
    assert load_ordered_records(data_path=data_path, manifest=manifest) == [
        records[index] for index in source_indices
    ]

    with pytest.raises(ValueError, match="registered selection"):
        build_training_order_manifest(
            data_path=data_path,
            contract=contract,
            contract_sha256="b" * 64,
            examples_seen=4,
            source_indices=[1, 2, 3, 0],
            selection_rule=GROUNDING_ENRICHED_SELECTION_RULE,
            selection_metadata={
                "grounding_examples": 2,
                "short_answer_examples": 2,
                "within_route_order": "frozen_source_order",
                "merge_rule": "alternate_grounding_then_short_answer",
            },
        )

    manifest["records"][0]["source_row_index"] = 3
    assert not verify_training_order_manifest(manifest)
