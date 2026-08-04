#!/usr/bin/env python3
"""从冻结表示独立拟合逐层 ridge probe，并保存逐样本预测与 pair-bootstrap。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from moonvit_glue.mechanism_probe import (
    LinearProbe,
    apply_linear_probe,
    fit_linear_probe,
    pair_bootstrap_accuracy_delta,
    pair_label_permutation_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty analysis CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def accuracy(prediction: torch.Tensor, labels: torch.Tensor) -> float:
    return float(prediction.eq(labels).float().mean())


def balanced_accuracy(prediction: torch.Tensor, labels: torch.Tensor, classes: int) -> float:
    values = []
    for class_index in range(classes):
        selected = labels.eq(class_index)
        if not bool(selected.any()):
            raise ValueError("balanced accuracy cell is missing a class")
        values.append(prediction[selected].eq(labels[selected]).float().mean())
    return float(torch.stack(values).mean())


def probe_tensor_payload(prefix: str, probe: LinearProbe) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}__mean": probe.mean.contiguous(),
        f"{prefix}__scale": probe.scale.contiguous(),
        f"{prefix}__coefficients": probe.coefficients.contiguous(),
        f"{prefix}__metadata": torch.tensor(
            [probe.class_count, probe.alpha], dtype=torch.float64
        ),
    }


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite layerwise probe analysis: {args.out}")
    args.out.mkdir(parents=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_summary = json.loads((args.run / "SUMMARY.json").read_text(encoding="utf-8"))
    if run_summary.get("status") != "valid" or run_summary["metadata"]["final_half_scored"]:
        raise ValueError("probe source run is not a valid selection-only extraction")
    checkpoints = [row["id"] for row in config["checkpoints"]]
    available = set(run_summary["checkpoints"])
    checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint in available]
    if not checkpoints:
        raise ValueError("no configured checkpoints exist in representation run")
    conditions = [str(value) for value in config["extraction"]["conditions"]]
    classes = [str(value) for value in config["dataset"]["classes"]]
    class_count = len(classes)
    alpha = float(config["probe"]["alpha"])
    device = torch.device(args.device)
    random_generator = torch.Generator(device="cpu").manual_seed(
        int(config["probe"]["random_label_seed"])
    )

    metrics: list[dict] = []
    native_metrics: list[dict] = []
    predictions: list[dict] = []
    intervals: list[dict] = []
    probes: dict[str, torch.Tensor] = {}
    prediction_index: dict[tuple[str, str, str], dict[str, bool]] = {}

    for checkpoint_index, checkpoint in enumerate(checkpoints):
        train_path = args.run / f"{checkpoint}__train__vision.safetensors"
        train = load_file(str(train_path), device="cpu")
        train_meta = read_jsonl(args.run / f"{checkpoint}__train__vision.jsonl")
        train_labels = train["labels"].long()
        if len(train_meta) != train_labels.numel():
            raise ValueError("train representation metadata denominator mismatch")
        permutation = torch.randperm(train_labels.numel(), generator=random_generator)
        random_labels = train_labels[permutation]
        sites = sorted(
            key
            for key in train
            if key.startswith(("tower_", "projector_", "layer_"))
        )
        condition_payloads = {
            condition: load_file(
                str(args.run / f"{checkpoint}__selection__{condition}.safetensors"),
                device="cpu",
            )
            for condition in conditions
        }
        condition_meta = {
            condition: read_jsonl(
                args.run / f"{checkpoint}__selection__{condition}.jsonl"
            )
            for condition in conditions
        }
        for condition in conditions:
            if len(condition_meta[condition]) != int(
                condition_payloads[condition]["labels"].numel()
            ):
                raise ValueError("selection representation metadata denominator mismatch")

        # 现有 LM head 的逐 checkpoint 四类读出单独保存，避免和训练 probe 混淆。
        for condition in conditions:
            payload = condition_payloads[condition]
            target = payload["labels"].long()
            source = payload["source_labels"].long()
            native_prediction = payload["shape_logits"].argmax(dim=1)
            native_metrics.append(
                {
                    "checkpoint": checkpoint,
                    "condition": condition,
                    "records": target.numel(),
                    "target_accuracy": accuracy(native_prediction, target),
                    "target_balanced_accuracy": balanced_accuracy(
                        native_prediction, target, class_count
                    ),
                    "source_accuracy": accuracy(native_prediction, source),
                }
            )

        for site_index, site in enumerate(sites):
            train_features = train[site].to(device=device, dtype=torch.float32)
            probe = fit_linear_probe(
                train_features,
                train_labels.to(device),
                class_count=class_count,
                alpha=alpha,
            )
            random_probe = fit_linear_probe(
                train_features,
                random_labels.to(device),
                class_count=class_count,
                alpha=alpha,
            )
            prefix = f"{checkpoint}__{site}"
            probes.update(probe_tensor_payload(prefix, probe))
            probes.update(probe_tensor_payload(prefix + "__random_labels", random_probe))
            condition_predictions: dict[str, torch.Tensor] = {}
            for condition in conditions:
                payload = condition_payloads[condition]
                features = payload[site].to(device=device, dtype=torch.float32)
                target = payload["labels"].long().to(device)
                source = payload["source_labels"].long().to(device)
                predicted, _ = apply_linear_probe(probe, features)
                random_predicted, _ = apply_linear_probe(random_probe, features)
                condition_predictions[condition] = predicted.cpu()
                row = {
                    "checkpoint": checkpoint,
                    "site": site,
                    "condition": condition,
                    "records": target.numel(),
                    "target_accuracy": accuracy(predicted, target),
                    "target_balanced_accuracy": balanced_accuracy(
                        predicted, target, class_count
                    ),
                    "source_accuracy": accuracy(predicted, source),
                    "random_label_target_accuracy": accuracy(random_predicted, target),
                    "alpha": alpha,
                }
                if condition == "vision":
                    permutation = pair_label_permutation_test(
                        predictions=predicted.detach().cpu().tolist(),
                        labels=target.detach().cpu().tolist(),
                        pair_ids=[
                            str(meta["pair_id"]) for meta in condition_meta[condition]
                        ],
                        seed=int(config["probe"]["pair_label_permutation_seed"])
                        + checkpoint_index * 10000
                        + site_index,
                        samples=int(config["probe"]["pair_label_permutation_samples"]),
                    )
                    row.update(
                        {
                            "pair_null_mean": permutation["null_mean"],
                            "pair_null_ci95_low": permutation["null_ci95_low"],
                            "pair_null_ci95_high": permutation["null_ci95_high"],
                            "pair_permutation_p": permutation["p_value"],
                            "pair_permutation_samples": permutation[
                                "permutation_samples"
                            ],
                        }
                    )
                else:
                    row.update(
                        {
                            "pair_null_mean": None,
                            "pair_null_ci95_low": None,
                            "pair_null_ci95_high": None,
                            "pair_permutation_p": None,
                            "pair_permutation_samples": 0,
                        }
                    )
                metrics.append(row)
                per_id = {}
                for index, meta in enumerate(condition_meta[condition]):
                    item = {
                        "checkpoint": checkpoint,
                        "site": site,
                        "condition": condition,
                        "id": str(meta["id"]),
                        "pair_id": str(meta["pair_id"]),
                        "target_label": classes[int(target[index])],
                        "source_id": str(meta["source_id"]),
                        "source_label": classes[int(source[index])],
                        "prediction": classes[int(predicted[index])],
                        "random_label_prediction": classes[int(random_predicted[index])],
                        "target_correct": bool(predicted[index] == target[index]),
                        "source_correct": bool(predicted[index] == source[index]),
                        "random_label_target_correct": bool(
                            random_predicted[index] == target[index]
                        ),
                    }
                    predictions.append(item)
                    per_id[item["id"]] = item["target_correct"]
                prediction_index[(checkpoint, site, condition)] = per_id
                if condition == "vision":
                    boot = pair_bootstrap_accuracy_delta(
                        correct_a=[row["target_correct"] for row in predictions[-len(target) :]],
                        correct_b=[
                            row["random_label_target_correct"]
                            for row in predictions[-len(target) :]
                        ],
                        pair_ids=[meta["pair_id"] for meta in condition_meta[condition]],
                        seed=int(config["probe"]["bootstrap_seed"])
                        + checkpoint_index * 1000
                        + site_index,
                        samples=int(config["probe"]["bootstrap_samples"]),
                    )
                    intervals.append(
                        {
                            "comparison": "vision_vs_random_label_probe",
                            "checkpoint": checkpoint,
                            "site": site,
                            **boot,
                        }
                    )
            vision_ids = condition_meta["vision"]
            for condition_offset, condition in enumerate(conditions):
                if condition == "vision":
                    continue
                vision_map = prediction_index[(checkpoint, site, "vision")]
                control_map = prediction_index[(checkpoint, site, condition)]
                ordered_ids = [str(row["id"]) for row in vision_ids]
                boot = pair_bootstrap_accuracy_delta(
                    correct_a=[vision_map[sample_id] for sample_id in ordered_ids],
                    correct_b=[control_map[sample_id] for sample_id in ordered_ids],
                    pair_ids=[str(row["pair_id"]) for row in vision_ids],
                    seed=int(config["probe"]["bootstrap_seed"])
                    + checkpoint_index * 1000
                    + site_index * 10
                    + condition_offset,
                    samples=int(config["probe"]["bootstrap_samples"]),
                )
                intervals.append(
                    {
                        "comparison": f"vision_vs_{condition}",
                        "checkpoint": checkpoint,
                        "site": site,
                        **boot,
                    }
                )

    # step 1500→2000 使用同一 selection ID 做逐 pair 差值，直接定位坍缩层。
    if {"step-001500", "step-002000"}.issubset(checkpoints):
        sites = sorted(
            {
                site
                for checkpoint, site, condition in prediction_index
                if checkpoint == "step-001500" and condition == "vision"
            }
        )
        metadata = read_jsonl(args.run / "step-001500__selection__vision.jsonl")
        ordered_ids = [str(row["id"]) for row in metadata]
        for site_index, site in enumerate(sites):
            earlier = prediction_index[("step-001500", site, "vision")]
            later = prediction_index[("step-002000", site, "vision")]
            boot = pair_bootstrap_accuracy_delta(
                correct_a=[earlier[sample_id] for sample_id in ordered_ids],
                correct_b=[later[sample_id] for sample_id in ordered_ids],
                pair_ids=[str(row["pair_id"]) for row in metadata],
                seed=int(config["probe"]["bootstrap_seed"]) + 9000 + site_index,
                samples=int(config["probe"]["bootstrap_samples"]),
            )
            intervals.append(
                {
                    "comparison": "step1500_vs_step2000_vision",
                    "checkpoint": "step-001500_minus_step-002000",
                    "site": site,
                    **boot,
                }
            )

    metrics_path = args.out / "probe_metrics.csv"
    native_path = args.out / "native_logit_lens.csv"
    interval_path = args.out / "probe_intervals.csv"
    predictions_path = args.out / "probe_predictions.jsonl"
    probes_path = args.out / "PROBES.safetensors"
    write_csv(metrics_path, metrics)
    write_csv(native_path, native_metrics)
    write_csv(interval_path, intervals)
    with predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    save_file(probes, str(probes_path), metadata={"format": "layerwise-linear-probes-v1"})

    vision_metrics = [row for row in metrics if row["condition"] == "vision"]
    decisions = {"checkpoints": {}, "status": "valid"}
    for checkpoint in checkpoints:
        rows = [row for row in vision_metrics if row["checkpoint"] == checkpoint]
        assistant = [row for row in rows if row["site"].endswith("_assistant")]
        projector = [row for row in rows if row["site"].startswith("projector_")]
        tower = [row for row in rows if row["site"].startswith("tower_")]
        decisions["checkpoints"][checkpoint] = {
            "best_assistant": max(assistant, key=lambda row: row["target_balanced_accuracy"]),
            "final_assistant": next(
                row for row in assistant if row["site"] == "layer_24_assistant"
            ),
            "best_projector": max(projector, key=lambda row: row["target_balanced_accuracy"]),
            "best_tower": max(tower, key=lambda row: row["target_balanced_accuracy"]),
        }
    decisions["interpretation_limits"] = [
        "probe alpha is fixed before selection scoring",
        "all probe fitting uses the disjoint synthetic train split",
        "selection bootstrap resamples complete a/b pairs",
        "linear recoverability does not by itself establish causal use by the language head",
        "the single random-training-label probe is an overfit diagnostic; pair-label permutation is the primary association null",
        "final odd halves remain unscored",
    ]
    decisions_path = args.out / "DECISIONS.json"
    decisions_path.write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "valid",
        "format_version": "layerwise-probe-analysis-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_summary_sha256": sha256(args.run / "SUMMARY.json"),
        "checkpoints": checkpoints,
        "classes": classes,
        "metric_rows": len(metrics),
        "native_rows": len(native_metrics),
        "interval_rows": len(intervals),
        "prediction_rows": len(predictions),
        "probe_tensor_count": len(probes),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (
                metrics_path,
                native_path,
                interval_path,
                predictions_path,
                probes_path,
                decisions_path,
            )
        },
        "final_half_scored": False,
    }
    (args.out / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
