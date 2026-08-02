"""Fetch pinned benchmark datasets and convert them to the eval JSONL schema.

Requires the ``datasets`` package (not a runtime dependency of the glue).

Each dataset lands under ``--out-dir`` as ``<name>.jsonl`` plus shared
``images/`` PNG files, and is registered in ``MANIFEST.json`` with the
resolved hub revision and a sha256 of the JSONL — the same
trust-the-manifest discipline as the 0xSero graft.

Example::

    python tools/fetch_eval_data.py --dataset screenspot --limit 200
    python tools/fetch_eval_data.py --dataset textvqa --limit 500 --out-dir data/eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCALE = 999.0


@dataclass(frozen=True)
class FetchSpec:
    repo: str
    split: str
    metric: str
    config: str | None = None
    question_field: str | None = "question"
    answers_field: str = "answers"


DATASETS = {
    "textvqa": FetchSpec(
        repo="lmms-lab/textvqa", split="validation", metric="soft_vqa"
    ),
    "docvqa": FetchSpec(
        repo="lmms-lab/DocVQA", config="DocVQA", split="validation", metric="anls"
    ),
    "ocrbench": FetchSpec(
        repo="lmms-lab/OCRBench", split="test", metric="exact_match",
        answers_field="answer",
    ),
    "screenspot": FetchSpec(
        repo="rootsautomation/ScreenSpot", split="test", metric="grounding",
        question_field="instruction",
    ),
    # Caption data for projector overfit/alignment runs; question is a constant.
    "flickr8k": FetchSpec(
        repo="nlphuji/flickr8k", split="train", metric="token_f1",
        question_field=None, answers_field="caption",
    ),
}


def normalize_box(box, width: int, height: int) -> list[float]:
    """Pixel [x1, y1, x2, y2] -> clamped [0, 999] normalized box.

    rootsautomation/ScreenSpot stores absolute pixel corners; rows whose
    values stay within [0, 1] are treated as fractions instead. The sanity
    printout in ``fetch`` makes a format drift obvious.
    """

    values = [float(v) for v in box]
    if max(values) <= 1.0:
        values = [values[0] * width, values[1] * height, values[2] * width, values[3] * height]
    x1, y1, x2, y2 = values
    normalized = [x1 / width * SCALE, y1 / height * SCALE, x2 / width * SCALE, y2 / height * SCALE]
    return [min(max(v, 0.0), SCALE) for v in normalized]


def fetch(name: str, spec: FetchSpec, limit: int | None, out_dir: Path, revision: str | None) -> dict:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    dataset = load_dataset(spec.repo, spec.config, split=spec.split, revision=revision)
    resolved_revision = HfApi().dataset_info(spec.repo, revision=revision).sha

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    records = []
    box_stats = {"max_seen": 0.0}
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        record_id = f"{name}_{index:06d}"
        image = row["image"].convert("RGB")
        image_name = f"{record_id}.png"
        image.save(images_dir / image_name)

        record = {
            "id": record_id,
            "image": f"images/{image_name}",
            "question": row[spec.question_field] if spec.question_field else "Describe the image.",
            "metric": spec.metric,
        }
        if spec.metric == "grounding":
            box_stats["max_seen"] = max(box_stats["max_seen"], *(float(v) for v in row["bbox"]))
            record["gt_box"] = normalize_box(row["bbox"], image.width, image.height)
            record["gt_box_pixels"] = [float(v) for v in row["bbox"]]
            record["image_size"] = [image.width, image.height]
        else:
            answers = row[spec.answers_field]
            record["answers"] = [answers] if isinstance(answers, str) else list(answers)
        records.append(record)

    jsonl_path = out_dir / f"{name}.jsonl"
    body = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    jsonl_path.write_text(body, encoding="utf-8")
    if spec.metric == "grounding":
        print(f"bbox sanity: max raw value seen = {box_stats['max_seen']} (pixels expected > 1)")

    return {
        "dataset": name,
        "repo": spec.repo,
        "config": spec.config,
        "split": spec.split,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "rows": len(records),
        "jsonl": jsonl_path.name,
        "jsonl_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--revision", default=None, help="Pin a hub revision; resolved sha is always recorded")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    entry = fetch(args.dataset, DATASETS[args.dataset], args.limit, args.out_dir, args.revision)

    manifest_path = args.out_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    manifest = [item for item in manifest if not (item["dataset"] == entry["dataset"] and item["rows"] == entry["rows"])]
    manifest.append(entry)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(entry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
