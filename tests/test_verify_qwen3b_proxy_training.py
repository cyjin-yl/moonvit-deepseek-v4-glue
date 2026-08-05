import hashlib
import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from verify_qwen3b_proxy_training import (
    build_checkpoint_binding,
    canonical_sha256,
    validate_training_history,
    verify_checkpoint_inventory,
)


def test_checkpoint_binding_is_rebuilt_from_identity_manifests(tmp_path):
    contract_path = tmp_path / "contract.json"
    order_path = tmp_path / "order.json"
    cache_path = tmp_path / "cache.json"
    for path in (contract_path, order_path, cache_path):
        path.write_text(path.name, encoding="utf-8")
    binding = build_checkpoint_binding(
        contract={
            "canonical_projector": {
                "initialization_contract": {"step0": {"weights_sha256": "p" * 64}}
            },
            "qwen_proxy_receiver": {"buffer_sha256": "q" * 64},
        },
        order={"manifest_sha256": "m" * 64, "records_sha256": "r" * 64},
        cache={"records_sha256": "c" * 64, "git_sha": "g" * 40},
        contract_path=contract_path,
        order_path=order_path,
        cache_manifest_path=cache_path,
        runner_git_sha="x" * 40,
    )
    assert binding["training_order_manifest_sha256"] == "m" * 64
    assert binding["feature_cache_records_sha256"] == "c" * 64
    assert binding["runner_git_sha"] == "x" * 40


def test_training_history_rebuilds_exact_fixed_batches_and_tokens():
    order = [{"id": f"row-{index}"} for index in range(4)]
    supervision = [
        {"id": f"row-{index}", "answer_tokens": index + 1} for index in range(4)
    ]
    history = [
        {
            "step": 1,
            "optimizer_steps": 1,
            "batch_start_index": 0,
            "batch_end_index": 1,
            "examples_seen": 2,
            "answer_tokens_seen": 3,
            "batch_record_ids_sha256": canonical_sha256(["row-0", "row-1"]),
            "loss": 2.0,
        },
        {
            "step": 2,
            "optimizer_steps": 2,
            "batch_start_index": 2,
            "batch_end_index": 3,
            "examples_seen": 4,
            "answer_tokens_seen": 10,
            "batch_record_ids_sha256": canonical_sha256(["row-2", "row-3"]),
            "loss": 1.0,
        },
    ]
    result = validate_training_history(
        history, supervision, order, gradient_accumulation=2
    )
    assert result["examples_seen"] == 4
    assert result["answer_tokens_seen"] == 10
    assert result["loss_mean"] == 1.5


def test_checkpoint_inventory_rehashes_exact_five_files(tmp_path):
    rows = []
    for name in (
        "history.json",
        "projector.safetensors",
        "projector_bf16.safetensors",
        "projector_config.json",
        "training_state.pt",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode("utf-8"))
        rows.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "files": sorted(rows, key=lambda row: row["path"]),
        "file_count": 5,
        "total_bytes": sum(row["bytes"] for row in rows),
    }
    (tmp_path / "CHECKPOINT_MANIFEST.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert len(verify_checkpoint_inventory(tmp_path)["files"]) == 5
    (tmp_path / "unexpected.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="inventory"):
        verify_checkpoint_inventory(tmp_path)
