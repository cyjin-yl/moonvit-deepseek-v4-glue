#!/usr/bin/env python3
"""独立验证 Package 12 的 raw rows、匹配约束、hash 与统计分母。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def count_csv(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def verify_declared_files(directory: Path, files: dict) -> None:
    for name, metadata in files.items():
        path = directory / name
        if not path.is_file():
            raise ValueError(f"declared file is absent: {path}")
        if path.stat().st_size != int(metadata["bytes"]):
            raise ValueError(f"declared byte count drifted: {path}")
        if sha256(path) != metadata["sha256"]:
            raise ValueError(f"declared SHA-256 drifted: {path}")


def assert_final_half_false(value, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            child = f"{path}.{key}"
            if key == "final_half_scored" and nested is not False:
                raise ValueError(f"final-half guard failed at {child}")
            assert_final_half_false(nested, path=child)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            assert_final_half_false(nested, path=f"{path}[{index}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / "raw" / "adaptation"
    stratified = raw / "balanced_compare_projector_v1"
    global_random = raw / "global_random_projector_v1"
    verification_dir = raw / "batch_stratification_verification_v1"
    evaluation = raw / "batch_stratification_eval_v1"
    analysis = raw / "batch_stratification_analysis_v1"
    gradients = raw / "batch_stratification_gradients_v1"
    charts = root / "charts"

    stratified_summary = load_json(stratified / "SUMMARY.json")
    global_summary = load_json(global_random / "SUMMARY.json")
    verification = load_json(verification_dir / "VERIFICATION.json")
    evaluation_summary = load_json(evaluation / "SUMMARY.json")
    decision = load_json(analysis / "DECISIONS.json")
    gradient_summary = load_json(gradients / "SUMMARY.json")
    chart_summary = load_json(charts / "CHARTS.json")

    for name, payload in {
        "stratified": stratified_summary,
        "global": global_summary,
        "verification": verification,
        "evaluation": evaluation_summary,
        "analysis": decision,
        "gradients": gradient_summary,
        "charts": chart_summary,
    }.items():
        if payload.get("status") != "valid":
            raise ValueError(f"{name} payload is not valid")
        assert_final_half_false(payload, path=name)

    if not all(verification["checks"].values()):
        raise ValueError("one or more matched-order invariants failed")
    if stratified_summary["base_projector_sha256"] != global_summary["base_projector_sha256"]:
        raise ValueError("base projector file hash differs")
    if (
        stratified_summary["checkpoints"]["step-000000"]["weights_tensor_sha256"]
        != global_summary["checkpoints"]["step-000000"]["weights_tensor_sha256"]
    ):
        raise ValueError("step-0 projector tensors differ")
    if stratified_summary["optimizer_resume"]["source_sha256"] != global_summary["optimizer_resume"]["source_sha256"]:
        raise ValueError("optimizer resume source differs")
    if stratified_summary.get("order_strategy") not in {None, "stratified_balanced"}:
        raise ValueError("stratified arm order strategy drifted")
    if global_summary.get("order_strategy") != "global_random":
        raise ValueError("global arm order strategy drifted")

    checkpoint_manifests = 0
    for directory, summary in ((stratified, stratified_summary), (global_random, global_summary)):
        verify_declared_files(directory, summary["files"])
        for checkpoint_id, expected in summary["checkpoints"].items():
            checkpoint = directory / "checkpoints" / checkpoint_id
            manifest = load_json(checkpoint / "MANIFEST.json")
            if manifest != expected:
                raise ValueError(f"checkpoint manifest differs from summary: {checkpoint}")
            config_meta = manifest["files"]["projector_config.json"]
            config_path = checkpoint / "projector_config.json"
            if config_path.stat().st_size != int(config_meta["bytes"]) or sha256(config_path) != config_meta["sha256"]:
                raise ValueError(f"checkpoint projector config failed hash verification: {checkpoint}")
            checkpoint_manifests += 1
    forbidden_large = [path for path in root.rglob("*") if path.suffix in {".safetensors", ".pt", ".bin"}]
    if forbidden_large:
        raise ValueError(f"package unexpectedly contains checkpoint weights: {forbidden_large[:3]}")

    verify_declared_files(evaluation, evaluation_summary["files"])
    preference_rows = count_jsonl(evaluation / "preference_records.jsonl")
    generation_rows = count_jsonl(evaluation / "generation_records.jsonl")
    if preference_rows != int(evaluation_summary["preference_rows"]) or preference_rows != 50400:
        raise ValueError("preference row denominator drifted")
    if generation_rows != int(evaluation_summary["generation_rows"]) or generation_rows != 8400:
        raise ValueError("generation row denominator drifted")
    if int(evaluation_summary["states"]) != 7:
        raise ValueError("evaluation state count drifted")

    lora_summary = raw / "balanced_compare_lora_v1" / "SUMMARY.json"
    index_summary = raw / "batch_stratification_index_v1" / "SUMMARY.json"
    if sha256(lora_summary) != evaluation_summary["sources"]["lora_summary_sha256"]:
        raise ValueError("evaluation LoRA source hash drifted")
    if sha256(index_summary) != evaluation_summary["sources"]["projector_summary_sha256"]:
        raise ValueError("evaluation projector index hash drifted")
    if sha256(evaluation / "preference_records.jsonl") != decision["sources"]["preference_records_sha256"]:
        raise ValueError("analysis preference source hash drifted")
    if sha256(evaluation / "generation_records.jsonl") != decision["sources"]["generation_records_sha256"]:
        raise ValueError("analysis generation source hash drifted")
    if sha256(verification_dir / "VERIFICATION.json") != decision["sources"]["verification_sha256"]:
        raise ValueError("analysis verification source hash drifted")

    metric_rows = count_csv(analysis / "metrics.csv")
    contrast_rows = count_csv(analysis / "contrasts.csv")
    if (metric_rows, contrast_rows) != (735, 525):
        raise ValueError("analysis CSV row counts drifted")
    if decision["batch_effect"] != "mixed_or_underpowered":
        raise ValueError("pre-registered batch-effect verdict drifted")
    endpoint = next(row for row in decision["endpoint_paired_preference_contrasts"] if row["task"] == "overall")
    if int(endpoint["denominator"]) != 1200 or float(endpoint["ci95_low"]) >= 0 or float(endpoint["ci95_high"]) <= 0:
        raise ValueError("endpoint overall CI no longer straddles zero")

    verify_declared_files(gradients, gradient_summary["files"])
    gradient_norm_rows = count_csv(gradients / "gradient_norms.csv")
    gradient_cosine_rows = count_csv(gradients / "gradient_cosines.csv")
    if (gradient_norm_rows, gradient_cosine_rows) != (42, 105):
        raise ValueError("gradient diagnostic row counts drifted")
    if len(gradient_summary["state_summaries"]) != 7:
        raise ValueError("gradient state count drifted")
    if gradient_summary["state_summaries"]["global-step100"]["negative_cosine_pairs"] != 0:
        raise ValueError("global endpoint conflict count drifted")
    if gradient_summary["state_summaries"]["stratified-step100"]["negative_cosine_pairs"] != 6:
        raise ValueError("stratified endpoint conflict count drifted")

    verify_declared_files(charts, chart_summary["charts"])
    output = {
        "status": "valid",
        "format_version": "batch-stratification-package-verification-v1",
        "matched_training": {
            "records_per_arm": 2400,
            "steps_per_arm": 100,
            "checkpoint_manifests": checkpoint_manifests,
            "all_verification_checks": True,
            "stratified_balanced_batches": verification["stratified_order"]["balanced_batches"],
            "global_balanced_batches": verification["global_order"]["balanced_batches"],
            "global_max_task_count": verification["global_order"]["max_task_count_in_batch"],
        },
        "evaluation": {
            "states": 7,
            "preference_rows": preference_rows,
            "generation_rows": generation_rows,
            "analysis_metric_rows": metric_rows,
            "analysis_contrast_rows": contrast_rows,
            "batch_effect": decision["batch_effect"],
            "endpoint_overall_gap": endpoint["mean_gap"],
            "endpoint_overall_ci95": [endpoint["ci95_low"], endpoint["ci95_high"]],
        },
        "gradients": {
            "norm_rows": gradient_norm_rows,
            "cosine_rows": gradient_cosine_rows,
            "stratified_endpoint_negative_pairs": 6,
            "global_endpoint_negative_pairs": 0,
        },
        "large_checkpoint_weights_in_git": False,
        "final_half_scored": False,
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    output_path = args.out or root / "PACKAGE_VERIFICATION.json"
    output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
