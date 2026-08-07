import json
import hashlib
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest
import torch

from moonvit_glue.fixed_budget import (
    feature_cache_contract,
    fixed_batch_record_indices,
    route_training_example,
    validate_fixed_budget_contract,
    validate_resume_history,
)
from moonvit_glue.projector import PatchMergerProjector, ProjectorConfig


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from train_qwen3b_proxy import (
    canonical_sha256,
    load_architecture_overlay,
    save_bound_checkpoint,
    supervision_provenance,
    _zero_gradient_allowed_for_projector,
    verify_bound_checkpoint,
)


def test_gated_residual_zero_branch_gradient_is_explicitly_allowed_at_zero_gate():
    torch.manual_seed(20260807)
    projector = PatchMergerProjector(
        ProjectorConfig(
            vision_width=3,
            language_width=5,
            merge_factor=4,
            projector_width=6,
            residual_mode="gated",
        )
    )
    assert _zero_gradient_allowed_for_projector(projector, "residual.weight") == (
        True,
        "gated_residual_branch_at_zero_gate",
    )
    assert _zero_gradient_allowed_for_projector(projector, "residual_gate") == (
        False,
        None,
    )
    with torch.no_grad():
        projector.residual_gate.fill_(0.1)
    assert _zero_gradient_allowed_for_projector(projector, "residual.weight") == (
        False,
        None,
    )

    # 真实反向传播回归：gate=0 时分支权重梯度为零，但 gate 本身必须
    # 通过链式法则收到非零梯度；否则 gated residual 会变成永久死分支。
    with torch.no_grad():
        projector.residual_gate.zero_()
    features = [torch.randn(2, 4, 3)]
    loss = projector(features)[0].square().mean()
    loss.backward()
    assert projector.residual.weight.grad is not None
    assert int(torch.count_nonzero(projector.residual.weight.grad).item()) == 0
    assert projector.residual_gate.grad is not None
    assert int(torch.count_nonzero(projector.residual_gate.grad).item()) > 0


def test_supervision_provenance_binds_order_and_raw_answer_hashes():
    answers = ["click(start_box=[10,20])"]
    question = "Open the menu."
    entry = {
        "id": "showui-1",
        "source": "showui_desktop",
        "source_row_index": 7,
        "image": "images/showui-1.jpg",
        "image_sha256": "a" * 64,
        "image_width": 1920,
        "image_height": 1080,
        "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
        "answers_sha256": canonical_sha256(answers),
        "target_transform": "legacy_click_spacing_to_canonical",
    }
    record = {"id": "showui-1", "question": question, "answers": answers}
    provenance = supervision_provenance(
        entry=entry,
        record=record,
        routed=SimpleNamespace(record_id="showui-1", prompt_route="grounding"),
    )
    assert provenance == {
        "source": "showui_desktop",
        "source_row_index": 7,
        "image": "images/showui-1.jpg",
        "image_sha256": "a" * 64,
        "image_width": 1920,
        "image_height": 1080,
        "question_sha256": entry["question_sha256"],
        "answers_sha256": entry["answers_sha256"],
        "target_transform": "legacy_click_spacing_to_canonical",
        "target_provenance": "showui_point_encoded_in_answer",
    }
    broken = dict(record)
    broken["answers"] = ["click(start_box=[11,20])"]
    with pytest.raises(ValueError, match="answers SHA differs"):
        supervision_provenance(
            entry=entry,
            record=broken,
            routed=SimpleNamespace(record_id="showui-1", prompt_route="grounding"),
        )


def _contract() -> dict:
    return {
        "schema_version": "qwen25-3b-community-eval-contract-v1",
        "proxy_model": {
            "architecture": "Qwen2ForCausalLM",
            "hidden_size": 2048,
            "frozen": True,
        },
        "vision_tower": {
            "extracted_weights_sha256": "v" * 64,
            "frozen": True,
        },
        "canonical_projector": {
            "output_width": 4096,
            "parameter_count": 33_564_672,
        },
        "qwen_proxy_receiver": {
            "input_width": 4096,
            "output_width": 2048,
            "trainable_parameter_count": 0,
        },
        "image_preprocessing": {
            "train_max_image_side": 448,
            "train_max_visual_tokens": 256,
        },
        "prompt_and_generation": {
            "system_prompt": "ground {rule}",
            "user_prompt": "Target: {instruction}",
            "short_answer_system_prompt": "short",
            "short_answer_user_prompt": "Question: {question}",
        },
        "training_budget": {
            "examples_seen_checkpoints": [16],
            "optimizer_steps_checkpoints": [2],
            "micro_batch_size": 1,
            "gradient_accumulation": 8,
            "real_global_batch": 8,
            "language_dtype": "float16",
            "projector_dtype": "float32",
            "activation_checkpointing": True,
            "feature_cache_allowed": True,
            "effective_epochs_denominator": 100,
        },
    }


def _order() -> dict:
    return {
        "schema_version": "qwen3b-training-order-v1",
        "manifest_sha256": "m" * 64,
        "records_sha256": "r" * 64,
        "unique_image_sha256": 15,
        "feature_cache": {
            "max_image_side": 448,
            "max_visual_tokens": 256,
            "moonvit_weights_sha256": "v" * 64,
            "storage_dtype": "float32",
        },
        "selection": {
            "examples_seen": 16,
            "optimizer_steps": 2,
            "micro_batch_size": 1,
            "gradient_accumulation": 8,
            "real_global_batch": 8,
            "effective_epochs_denominator": 100,
        },
        "records": [{"id": f"row-{index}"} for index in range(16)],
    }


def _cache() -> dict:
    records = []
    for index in range(16):
        row = {
            "id": f"row-{index}",
            "dtype": "float32",
            "feature_shape": [256 if index == 0 else 8, 4, 1024],
        }
        if index == 15:
            row["alias_of"] = "row-0"
        records.append(row)
    return {
        "count": 16,
        "unique_feature_spans": 15,
        "aliased_records": 1,
        "training_order_manifest_sha256": "m" * 64,
        "training_order_records_sha256": "r" * 64,
        "moonvit_weights_sha256": "v" * 64,
        "max_image_side": 448,
        "max_visual_tokens": 256,
        "vision_width": 1024,
        "merge_factor": 4,
        "records": records,
    }


def test_fixed_budget_contract_binds_model_order_cache_and_budget():
    result = validate_fixed_budget_contract(_contract(), _order(), _cache())
    assert result == {
        "examples_seen": 16,
        "optimizer_steps": 2,
        "micro_batch_size": 1,
        "gradient_accumulation": 8,
        "real_global_batch": 8,
        "unique_feature_spans": 15,
        "aliased_records": 1,
        "maximum_visual_tokens": 256,
        "projector_output_width": 4096,
        "receiver_output_width": 2048,
    }

    broken = _cache()
    broken["records"][0]["feature_shape"] = [257, 4, 1024]
    with pytest.raises(ValueError, match="visual-token budget"):
        validate_fixed_budget_contract(_contract(), _order(), broken)


def test_feature_cache_contract_resolves_explicit_v1_architecture_binding():
    contract = _contract()
    contract["vision_tower"] = {"frozen": True}
    contract["feature_cache_binding"] = {
        "vision_tower": "v1",
        "moonvit_model": "moonshotai/MoonViT-SO-400M",
        "moonvit_revision": "a" * 40,
        "vision_width": 1152,
        "merge_factor": 4,
        "require_tower_identity": True,
    }
    resolved = feature_cache_contract(contract)
    assert resolved == {
        "vision_tower": "v1",
        "moonvit_model": "moonshotai/MoonViT-SO-400M",
        "moonvit_revision": "a" * 40,
        "moonvit_weights_sha256": None,
        "vision_width": 1152,
        "merge_factor": 4,
        "require_tower_identity": True,
    }


def test_architecture_overlay_keeps_qwen_budget_and_rebinds_v1_interface(tmp_path: Path):
    core = _contract()
    core_path = tmp_path / "core.json"
    core_path.write_text(json.dumps(core), encoding="utf-8")
    projector_config_path = tmp_path / "configs" / "v1.json"
    projector_config_path.parent.mkdir()
    projector_config_path.write_text(
        json.dumps({"vision_width": 1152, "language_width": 4096}),
        encoding="utf-8",
    )
    source_config_sha = __import__("hashlib").sha256(
        projector_config_path.read_bytes()
    ).hexdigest()
    sidecar = {
        "base_contract": {
            "path": "core.json",
            "sha256": __import__("hashlib").sha256(core_path.read_bytes()).hexdigest(),
        },
        "arms": {
            "v1": {
                "vision_tower": {
                    "name": "MoonViT-SO-400M",
                    "cache_tower_id": "v1",
                    "model": "moonshotai/MoonViT-SO-400M",
                    "revision": "a" * 40,
                    "vision_width": 1152,
                    "merge_factor": 4,
                    "weights_sha256": "b" * 64,
                    "require_tower_identity": True,
                },
                "projector": {
                    "config_path": "configs/v1.json",
                    "config_sha256": "c" * 64,
                    "source_config_sha256": source_config_sha,
                    "variant": "legacy_pre_norm",
                    "output_width": 4096,
                    "parameter_count": 40_000_000,
                    "initialization": {
                        "step0": {"seed": 20260805, "weights_sha256": "d" * 64},
                        "random_projector": {"seed": 20260806, "weights_sha256": "e" * 64},
                    },
                },
            }
        },
    }
    sidecar_path = tmp_path / "architecture.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    effective, metadata = load_architecture_overlay(
        core_contract_path=core_path,
        core_contract=core,
        architecture_control_path=sidecar_path,
        architecture_arm="v1",
    )
    assert metadata is not None
    assert effective["vision_tower"]["vision_width"] == 1152
    assert effective["feature_cache_binding"]["require_tower_identity"] is True
    assert effective["canonical_projector"]["output_width"] == 4096
    assert effective["canonical_projector"]["initialization_seed"] == 20260805
    assert effective["training_budget"] == core["training_budget"]


def test_fixed_batch_indices_and_resume_history_never_wrap_or_skip():
    assert fixed_batch_record_indices(
        optimizer_step=0, total_examples=16, gradient_accumulation=8
    ) == list(range(8))
    assert fixed_batch_record_indices(
        optimizer_step=1, total_examples=16, gradient_accumulation=8
    ) == list(range(8, 16))
    with pytest.raises(ValueError, match="outside the frozen order"):
        fixed_batch_record_indices(
            optimizer_step=2, total_examples=16, gradient_accumulation=8
        )

    history = [
        {"step": 1, "examples_seen": 8, "answer_tokens_seen": 11},
        {"step": 2, "examples_seen": 16, "answer_tokens_seen": 23},
    ]
    assert validate_resume_history(
        start_step=2,
        history=history,
        total_examples=16,
        gradient_accumulation=8,
    ) == {
        "next_example_index": 16,
        "examples_seen": 16,
        "answer_tokens_seen": 23,
    }
    history[1]["examples_seen"] = 15
    with pytest.raises(ValueError, match="examples_seen"):
        validate_resume_history(
            start_step=2,
            history=history,
            total_examples=16,
            gradient_accumulation=8,
        )


def test_route_training_example_uses_frozen_route_and_target():
    contract = _contract()
    grounding = route_training_example(
        contract,
        {"id": "g", "prompt_route": "grounding", "target_answer": "click"},
        {"id": "g", "question": "Save button"},
    )
    assert grounding.system_prompt == "ground {rule}"
    assert grounding.user_prompt == "Target: Save button"
    assert grounding.target_answer == "click"

    short = route_training_example(
        contract,
        {"id": "q", "prompt_route": "short_answer", "target_answer": "nine"},
        {"id": "q", "question": "What number?"},
    )
    assert short.system_prompt == "short"
    assert short.user_prompt == "Question: What number?"
    assert short.target_answer == "nine"

    with pytest.raises(ValueError, match="unknown prompt route"):
        route_training_example(
            contract,
            {"id": "x", "prompt_route": "caption", "target_answer": "x"},
            {"id": "x", "question": "x"},
        )


def test_bound_checkpoint_rehashes_every_resume_file(tmp_path: Path):
    projector = PatchMergerProjector(
        ProjectorConfig(
            vision_width=3,
            language_width=5,
            merge_factor=2,
            projector_width=6,
        )
    )
    optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3)
    history = [
        {
            "step": 1,
            "optimizer_steps": 1,
            "examples_seen": 8,
            "answer_tokens_seen": 13,
            "effective_epochs": 0.08,
            "subset_passes": 0.5,
        }
    ]
    binding = {
        "runner_git_sha": "a" * 40,
        "contract_file_sha256": "b" * 64,
    }
    directory = tmp_path / "step-000001"
    manifest = save_bound_checkpoint(
        directory=directory,
        projector=projector,
        optimizer=optimizer,
        step=1,
        history=history,
        rng=random.Random(7),
        binding=binding,
    )
    assert manifest["step"] == 1
    assert manifest["progress"]["examples_seen"] == 8
    assert verify_bound_checkpoint(directory, expected_binding=binding) == manifest

    (directory / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="file inventory"):
        verify_bound_checkpoint(directory, expected_binding=binding)
    (directory / "unexpected.bin").unlink()

    (directory / "history.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file inventory"):
        verify_bound_checkpoint(directory, expected_binding=binding)
