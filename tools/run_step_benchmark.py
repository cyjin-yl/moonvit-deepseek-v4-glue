"""Measure real serial-accumulation step time on the V100 training path."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


STEP_ROW_FIELDS = [
    "micro_batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "actual_batched_forward",
    "optimizer_step",
    "step_wall_seconds",
    "examples_per_second",
    "peak_gpu_memory_bytes",
    "included_after_warmup",
]


def write_step_rows(path: str | Path, rows: list[dict]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=STEP_ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize_step_reports(
    payloads: list[dict], *, warmup_steps: int
) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    runs: list[dict] = []
    for payload in payloads:
        report = payload["report"]
        history = [row for row in payload["history"] if "step_wall_seconds" in row]
        for index, row in enumerate(history):
            rows.append({
                "micro_batch_size": report["micro_batch_size"],
                "gradient_accumulation_steps": report["gradient_accumulation_steps"],
                "effective_batch_size": report["effective_batch_size"],
                "actual_batched_forward": report["actual_batched_forward"],
                "optimizer_step": row["step"],
                "step_wall_seconds": row["step_wall_seconds"],
                "examples_per_second": row["examples_per_second"],
                "peak_gpu_memory_bytes": row.get("peak_gpu_memory_bytes"),
                "included_after_warmup": index >= warmup_steps,
            })
        measured = history[warmup_steps:]
        if not measured:
            raise ValueError("warmup excludes every measured optimizer step")
        timings = [float(row["step_wall_seconds"]) for row in measured]
        mean_time = sum(timings) / len(timings)
        effective_batch_size = int(report["effective_batch_size"])
        runs.append({
            "micro_batch_size": int(report["micro_batch_size"]),
            "gradient_accumulation_steps": int(
                report["gradient_accumulation_steps"]
            ),
            "effective_batch_size": effective_batch_size,
            "actual_batched_forward": bool(report["actual_batched_forward"]),
            "measured_steps": len(measured),
            "mean_step_wall_seconds": mean_time,
            "min_step_wall_seconds": min(timings),
            "max_step_wall_seconds": max(timings),
            "examples_per_second_from_mean": effective_batch_size / mean_time,
            "peak_gpu_memory_bytes": int(report["peak_gpu_memory_bytes"]),
        })
    return {"warmup_steps_excluded": warmup_steps, "runs": runs}, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--moonvit-v2-weights", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--results-out", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--gradient-accumulation-steps", type=int, nargs="+",
                        default=[1, 4, 8])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--shuffle-repeats", type=int, default=1)
    parser.add_argument("--max-image-side", type=int, default=448)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, label in ((args.results_out, "results"), (args.work_root, "work")):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {label} directory: {path}")
        path.mkdir(parents=True)
    cache_manifest = json.loads(
        (args.feature_cache / "MANIFEST.json").read_text(encoding="utf-8")
    )
    if cache_manifest["max_image_side"] != args.max_image_side:
        raise ValueError("feature cache and benchmark max-image-side differ")
    validation_manifest = args.results_out / "validation_manifest.json"
    payloads: list[dict] = []
    failures: list[dict] = []
    child_env = os.environ.copy()
    child_env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    with (args.results_out / "commands.jsonl").open("w", encoding="utf-8") as commands:
        for accumulation in args.gradient_accumulation_steps:
            label = f"serial-accum-{accumulation}"
            work_out = args.work_root / label
            command = [
                args.python,
                "tools/train_overfit.py",
                "--text-model", args.text_model,
                "--vision-tower", "v2",
                "--moonvit-v2-weights", str(args.moonvit_v2_weights),
                "--moonvit-v2-attn", "eager",
                "--data", str(args.data),
                "--feature-cache", str(args.feature_cache),
                "--limit", str(args.limit),
                "--steps", str(args.steps),
                "--micro-batch-size", "1",
                "--gradient-accumulation-steps", str(accumulation),
                "--eval-samples", str(args.eval_samples),
                "--validation-manifest", str(validation_manifest),
                "--shuffle-repeats", str(args.shuffle_repeats),
                "--max-image-side", str(args.max_image_side),
                "--dtype", "float32",
                "--seed", str(args.seed),
                "--checkpoint-every", "0",
                "--out", str(work_out),
            ]
            commands.write(json.dumps({"label": label, "argv": command}) + "\n")
            commands.flush()
            log_path = args.results_out / f"{label}.log"
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=child_env,
                )
            if result.returncode:
                failures.append({
                    "label": label,
                    "returncode": result.returncode,
                    "log": log_path.name,
                })
                continue
            report_path = work_out / "overfit_report.json"
            published_report = args.results_out / f"{label}.overfit_report.json"
            shutil.copy2(report_path, published_report)
            payloads.append(json.loads(report_path.read_text(encoding="utf-8")))
    summary, rows = summarize_step_reports(payloads, warmup_steps=args.warmup_steps)
    summary.update({
        "run_id": args.run_id,
        "feature_cache_manifest": str(args.feature_cache / "MANIFEST.json"),
        "feature_cache_records_sha256": cache_manifest["records_sha256"],
        "failures": failures,
    })
    (args.results_out / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_step_rows(args.results_out / "step_times.csv", rows)
    (args.results_out / "failures.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
