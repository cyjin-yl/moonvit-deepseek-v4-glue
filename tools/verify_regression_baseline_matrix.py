import json
from pathlib import Path


MATRIX = Path("regression_baseline_matrix_v1.json")


def main() -> int:
    data = json.loads(MATRIX.read_text(encoding="utf-8"))
    assert data["schema_version"] == "qwen3b-regression-baseline-matrix-v1"
    fixed = data["fixed_evaluation"]
    assert fixed["conditions"] == ["vision", "blind", "shuffled", "random_projector"]
    assert fixed["generation"]["do_sample"] is False
    assert fixed["generation"]["temperature"] == 0
    assert fixed["generation"]["bootstrap_samples"] >= 2000
    ids = {row["id"] for row in data["rows"]}
    required = {
        "qwen25_3b_legacy_v2_full_public_previous_best",
        "qwen25_7b_exact_v2_full_public_margin05",
        "qwen25_7b_v1_teacher_forced_only",
        "qwen35_4b_external_v1_margin05_glm50",
        "qwen35_4b_external_v1_scale003_ceonly_glm50",
        "qwen35_4b_external_v2_margin05_glm50",
        "qwen35_4b_native_vlm_positive_control",
        "qwen35_9b_stripped_native_short_probe",
    }
    assert required <= ids
    for row in data["rows"]:
        assert row["evidence"]
        assert row["capability_claim_allowed"] is False
        assert row["deepseek_transfer_label"] in {
            "directly_transferable",
            "transferable_with_runtime_validation",
            "qwen_specific_not_transferable",
        }
        for key in ("vision_minus_blind_ci95", "vision_minus_shuffled_ci95"):
            if key in row:
                ci = row[key]
                assert len(ci) == 2 and ci[0] <= ci[1]
        if row["id"] == "qwen35_4b_native_vlm_positive_control":
            assert row["not_comparable_to_projector_rank"] is True
            assert row["scope"] == "positive_control_separate"
        if row["id"] == "qwen25_3b_legacy_v2_full_public_previous_best":
            assert "legacy" in row["tower"] and "not exact K3" in row["tower"]
        if row["id"] == "qwen35_4b_external_v1_margin05_glm50":
            assert row["training_rows_per_optimizer_step"] == 32
            assert "full32-repair" in row["evidence"]
        if row["id"] == "qwen35_4b_external_v1_scale003_ceonly_glm50":
            assert row["training_rows_per_optimizer_step"] == 8
    assert data["gate_d"]["status"] == "NO-GO"
    print(json.dumps({"verified": True, "rows": len(data["rows"]), "gate_d": "NO-GO"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
