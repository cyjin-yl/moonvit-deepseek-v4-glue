#!/usr/bin/env python3
"""按任务保存确定性的成功、失败与因果翻转审计案例。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


CASE_TYPES = (
    "vision_success_blind_failure",
    "vision_and_blind_failure",
    "teacher_forced_positive_generation_failure",
    "patch_permutation_flip",
    "background_prediction_flip",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_sample(generation: dict[str, dict], preference: dict | None) -> set[str]:
    vision = generation["vision"]
    blind = generation["blind"]
    patch = generation["patch_permutation"]
    background = generation["background_matched_aux"]
    labels = set()
    if bool(vision["correct"]) and not bool(blind["correct"]):
        labels.add("vision_success_blind_failure")
    if not bool(vision["correct"]) and not bool(blind["correct"]):
        labels.add("vision_and_blind_failure")
    if (
        preference is not None
        and preference.get("failure") is None
        and float(preference["correct_margin"]) > 0
        and not bool(vision["correct"])
    ):
        labels.add("teacher_forced_positive_generation_failure")
    if (
        bool(vision["correct"])
        and not bool(patch["correct"])
        and vision["normalized_prediction"] != patch["normalized_prediction"]
    ):
        labels.add("patch_permutation_flip")
    if (
        vision["normalized_prediction"] != background["normalized_prediction"]
        and bool(vision["correct"]) != bool(background["correct"])
    ):
        labels.add("background_prediction_flip")
    return labels


def require_checkpoint_rows(
    checkpoint: str,
    generation_rows: list[dict],
    preference_rows: list[dict],
) -> None:
    """拒绝把拼错 checkpoint 的零行结果登记为有效审计。"""
    if not generation_rows:
        raise ValueError(f"checkpoint {checkpoint!r} has no synthetic generation rows")
    if not preference_rows:
        raise ValueError(f"checkpoint {checkpoint!r} has no vision preference rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-run", required=True, type=Path)
    parser.add_argument("--preference-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--checkpoint", default="step-002000")
    parser.add_argument("--per-task", type=int, default=10)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite failure audit: {args.out}")
    args.out.mkdir(parents=True)

    generation_rows = [
        row
        for row in read_jsonl(args.generation_run / "records.jsonl")
        if row["checkpoint"] == args.checkpoint and row["dataset"] == "synthetic"
    ]
    preference_rows = [
        row
        for row in read_jsonl(args.preference_run / "preference_records.jsonl")
        if row["checkpoint"] == args.checkpoint and row["condition"] == "vision"
    ]
    require_checkpoint_rows(args.checkpoint, generation_rows, preference_rows)
    by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in generation_rows:
        by_sample[str(row["id"])][str(row["condition"])] = row
    preference = {str(row["id"]): row for row in preference_rows}
    required_conditions = {
        "vision",
        "blind",
        "patch_permutation",
        "background_matched_aux",
    }
    incomplete = [
        sample_id
        for sample_id, conditions in by_sample.items()
        if not required_conditions.issubset(conditions)
    ]
    if incomplete:
        raise ValueError(f"failure audit generation cells are incomplete: {incomplete[:3]}")

    candidates: dict[tuple[str, str], list[str]] = defaultdict(list)
    for sample_id in sorted(by_sample):
        rows = by_sample[sample_id]
        task = str(rows["vision"]["task"])
        for case_type in classify_sample(rows, preference.get(sample_id)):
            candidates[(task, case_type)].append(sample_id)

    output_rows = []
    tasks = sorted({str(rows["vision"]["task"]) for rows in by_sample.values()})
    for task in tasks:
        for case_type in CASE_TYPES:
            for rank, sample_id in enumerate(candidates[(task, case_type)][: args.per_task], 1):
                rows = by_sample[sample_id]
                vision = rows["vision"]
                pref = preference.get(sample_id)
                output_rows.append(
                    {
                        "case_type": case_type,
                        "task": task,
                        "rank": rank,
                        "checkpoint": args.checkpoint,
                        "id": sample_id,
                        "pair_id": vision.get("pair_id"),
                        "image": vision.get("image"),
                        "image_sha256": vision.get("image_sha256"),
                        "question": vision["question"],
                        "ground_truth": vision["answers"],
                        "vision_prediction": vision.get("prediction"),
                        "blind_prediction": rows["blind"].get("prediction"),
                        "patch_prediction": rows["patch_permutation"].get("prediction"),
                        "background_prediction": rows["background_matched_aux"].get(
                            "prediction"
                        ),
                        "vision_correct": bool(vision["correct"]),
                        "blind_correct": bool(rows["blind"]["correct"]),
                        "patch_correct": bool(rows["patch_permutation"]["correct"]),
                        "background_correct": bool(
                            rows["background_matched_aux"]["correct"]
                        ),
                        "correct_logp_mean": pref.get("correct_logp_mean") if pref else None,
                        "counterfactual_logp_mean": (
                            pref.get("counterfactual_logp_mean") if pref else None
                        ),
                        "correct_margin": pref.get("correct_margin") if pref else None,
                        "visual_source_ids": {
                            condition: rows[condition].get("visual_source_id")
                            for condition in sorted(required_conditions)
                        },
                    }
                )

    output_path = args.out / "FAILURE_AUDIT.jsonl"
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    candidate_counts = {
        task: {
            case_type: len(candidates[(task, case_type)]) for case_type in CASE_TYPES
        }
        for task in tasks
    }
    selected_counts = Counter((row["task"], row["case_type"]) for row in output_rows)
    shortfalls = [
        {
            "task": task,
            "case_type": case_type,
            "available": candidate_counts[task][case_type],
            "requested": args.per_task,
        }
        for task in tasks
        for case_type in CASE_TYPES
        if selected_counts[(task, case_type)] < args.per_task
    ]
    summary = {
        "format_version": "perception-failure-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "lexicographically first sample IDs satisfying each fixed predicate",
        "checkpoint": args.checkpoint,
        "per_task_requested": args.per_task,
        "candidate_counts": candidate_counts,
        "selected_records": len(output_rows),
        "shortfalls": shortfalls,
        "final_half_scored": False,
        "sources": {
            "generation_records_sha256": sha256(args.generation_run / "records.jsonl"),
            "preference_records_sha256": sha256(
                args.preference_run / "preference_records.jsonl"
            ),
        },
        "output": {
            "bytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
        },
    }
    (args.out / "AUDIT_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
