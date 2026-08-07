#!/usr/bin/env python3
"""Verify the frozen runtime entrypoint audit against repository sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = (
    ROOT
    / "experiments"
    / "qwen3b_community_eval_20260805"
    / "runtime_entrypoint_audit_20260808"
    / "runtime_entrypoint_audit_v1.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["schema_version"] == "runtime-entrypoint-audit-v1"
    for item in audit["source_files"]:
        source = ROOT / item["path"]
        assert source.is_file(), item["path"]
        assert sha256(source) == item["sha256"], item["path"]

    answers = audit["answers"]
    assert answers["production_unified_trainer_exists"] is False
    assert answers["real_0731_end_to_end_verified"] is False
    assert answers["gate_d_status"] == "NO-GO"

    stripped = (ROOT / answers["qwen25_7b_actual_recent_trainer"]).read_text(
        encoding="utf-8"
    )
    assert '"status": "diagnostic_only"' in stripped
    assert '"capability_claim_allowed": False' in stripped

    model = (ROOT / "src/moonvit_glue/model.py").read_text(encoding="utf-8")
    assert 'if self.backbone_kind == "deepseek_v4"' in model
    assert "input_ids=merged.routing_input_ids" in model
    assert "_override_embedding_lookup" in model

    merge = (ROOT / "src/moonvit_glue/merge.py").read_text(encoding="utf-8")
    assert "placeholder_token_id" in merge
    assert "position_ids" in merge
    assert "ignore_index" in merge

    required = set(audit["hard_blockers"])
    assert {
        "finite_real_quantized_input_dgrad",
        "full_hash_moe_image_forward_backward_and_routing",
        "real_0731_prefill_decode_kv_cache_generate",
        "fixed_four_condition_real_0731_benchmark",
    } <= required
    assert audit["next_local_gate"]["checkpoint_steps"] == [
        0,
        1,
        2,
        5,
        10,
        20,
        30,
        50,
        75,
        100,
    ]
    print(json.dumps({"verified": True, "gate_d": "NO-GO", "sources": len(audit["source_files"])}))


if __name__ == "__main__":
    main()
