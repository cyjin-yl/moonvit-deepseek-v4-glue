#!/usr/bin/env python3
"""Run the fixed multi-task generation contract for one checkpoint.

This runner is intentionally separate from the per-step health probe.  It
keeps raw reports for vision/blind/shuffled/random-projector and writes a
small CSV row for every task metric so several checkpoint runs can be joined
into growth curves without reinterpreting the raw predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


METRIC_KEYS = ("soft_vqa", "anls", "exact_match", "token_f1", "accuracy", "parse_rate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--projector", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dataset", action="append", required=True, help="name=JSONL path")
    parser.add_argument("--vision-tower", choices=("v1", "v2"), default="v1")
    parser.add_argument("--moonvit-v2-weights", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-image-side", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--random-projector-seed", type=int, default=20260806)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--optimizer-step", type=int, required=True)
    parser.add_argument("--examples-seen", type=int, required=True)
    parser.add_argument("--canonical-projector", action="store_true")
    parser.add_argument("--receiver-adapter-seed", type=int, default=20260806)
    return parser.parse_args()


def parse_dataset_specs(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"dataset must be NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError(f"dataset must be NAME=PATH, got {value!r}")
        result.append((name, Path(path)))
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def common_args(args: argparse.Namespace, data: Path, report: Path) -> list[str]:
    command = [
        args.python,
        str(args.repo_root / "tools" / "eval_vlm.py"),
        "--text-model", args.text_model,
        "--projector", str(args.projector),
        "--data", str(data),
        "--out", str(report),
        "--vision-tower", args.vision_tower,
        "--dtype", args.dtype,
        "--max-image-side", str(args.max_image_side),
        "--max-new-tokens", str(args.max_new_tokens),
        "--seed", str(args.seed),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.vision_tower == "v2":
        if args.moonvit_v2_weights is None:
            raise ValueError("V2 evaluation requires --moonvit-v2-weights")
        command.extend(["--moonvit-v2-weights", str(args.moonvit_v2_weights)])
    if args.canonical_projector:
        command.extend(["--canonical-projector", "--receiver-adapter-seed", str(args.receiver_adapter_seed)])
    return command


def summary_rows(
    payload: dict,
    *,
    dataset: str,
    checkpoint_id: str,
    optimizer_step: int,
    examples_seen: int,
    condition: str,
    source_report: str,
) -> list[dict]:
    summary = payload.get("summary", {})
    rows = []
    for metric in METRIC_KEYS:
        if metric in summary:
            rows.append({
                "checkpoint": checkpoint_id,
                "optimizer_step": optimizer_step,
                "examples_seen": examples_seen,
                "dataset": dataset,
                "condition": condition,
                "metric": metric,
                "score": float(summary[metric]),
                "count": int(summary.get("count", 0)),
                "source_report": source_report,
            })
    return rows


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    datasets = parse_dataset_specs(args.dataset)
    commands = []
    rows: list[dict] = []
    failures: list[dict] = []
    for dataset, data in datasets:
        if not data.exists():
            raise FileNotFoundError(data)
        dataset_out = args.out / dataset
        dataset_out.mkdir()
        # One vision invocation also produces the matched blind pass.
        vision_report = dataset_out / "vision_and_blind.json"
        vision_command = common_args(args, data, vision_report) + ["--blind"]
        commands.append({"condition": "vision+blind", "dataset": dataset, "argv": vision_command})
        result = subprocess.run(vision_command, capture_output=True, text=True, check=False)
        (dataset_out / "vision_and_blind.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            failures.append({"dataset": dataset, "condition": "vision+blind", "returncode": result.returncode})
        elif vision_report.exists():
            payload = json.loads(vision_report.read_text(encoding="utf-8"))
            rows.extend(summary_rows(payload, dataset=dataset, checkpoint_id=args.checkpoint_id,
                                      optimizer_step=args.optimizer_step, examples_seen=args.examples_seen,
                                      condition="vision", source_report=str(vision_report)))
            blind_payload = {**payload, "condition": "blind", "summary": payload.get("blind_summary", {}),
                             "records": payload.get("blind_records", [])}
            blind_report = dataset_out / "blind.json"
            blind_report.write_text(json.dumps(blind_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            rows.extend(summary_rows(blind_payload, dataset=dataset, checkpoint_id=args.checkpoint_id,
                                      optimizer_step=args.optimizer_step, examples_seen=args.examples_seen,
                                      condition="blind", source_report=str(blind_report)))

        shuffled_report = dataset_out / "shuffled.json"
        shuffled_command = common_args(args, data, shuffled_report) + ["--shuffled"]
        commands.append({"condition": "shuffled", "dataset": dataset, "argv": shuffled_command})
        result = subprocess.run(shuffled_command, capture_output=True, text=True, check=False)
        (dataset_out / "shuffled.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            failures.append({"dataset": dataset, "condition": "shuffled", "returncode": result.returncode})
        elif shuffled_report.exists():
            payload = json.loads(shuffled_report.read_text(encoding="utf-8"))
            rows.extend(summary_rows(payload, dataset=dataset, checkpoint_id=args.checkpoint_id,
                                      optimizer_step=args.optimizer_step, examples_seen=args.examples_seen,
                                      condition="shuffled", source_report=str(shuffled_report)))

        random_report = dataset_out / "random_projector.json"
        random_command = common_args(args, data, random_report) + [
            "--random-projector", "--random-projector-seed", str(args.random_projector_seed),
        ]
        commands.append({"condition": "random_projector", "dataset": dataset, "argv": random_command})
        result = subprocess.run(random_command, capture_output=True, text=True, check=False)
        (dataset_out / "random_projector.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            failures.append({"dataset": dataset, "condition": "random_projector", "returncode": result.returncode})
        elif random_report.exists():
            payload = json.loads(random_report.read_text(encoding="utf-8"))
            rows.extend(summary_rows(payload, dataset=dataset, checkpoint_id=args.checkpoint_id,
                                      optimizer_step=args.optimizer_step, examples_seen=args.examples_seen,
                                      condition="random_projector", source_report=str(random_report)))

    with (args.out / "commands.jsonl").open("w", encoding="utf-8") as stream:
        for command in commands:
            stream.write(json.dumps(command, ensure_ascii=False) + "\n")
    with (args.out / "CURVE.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["checkpoint", "optimizer_step", "examples_seen", "dataset", "condition", "metric", "score", "count", "source_report"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "community-multitask-eval-v1",
        "status": "valid" if not failures else "failed",
        "checkpoint": {"id": args.checkpoint_id, "optimizer_step": args.optimizer_step, "examples_seen": args.examples_seen},
        "datasets": [{"name": name, "path": str(path), "sha256": sha256(path)} for name, path in datasets],
        "conditions": ["vision", "blind", "shuffled", "random_projector"],
        "rows": len(rows),
        "failures": failures,
        "raw_artifacts": "per-dataset report JSON, logs, commands.jsonl, CURVE.csv",
    }
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
