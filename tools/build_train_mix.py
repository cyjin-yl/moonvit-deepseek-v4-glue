"""Assemble the ~66k short-QA train mix with benchmark decontamination.

Merges per-source JSONL files (our fetch schema {image, question, answers} or
the 0xSero conversations schema), caps each source, drops any train image that
collides with an eval benchmark image (average-hash hamming <= threshold),
copies surviving images into a self-contained output directory, and writes
``train_mix.jsonl`` plus ``decontamination_report.json``.

Leakage policy: we train on *train* splits of benchmarked datasets (TextVQA,
DocVQA) and evaluate on their official validation splits; the 0xSero GUI
subsets (screenshots/multistep) are excluded up front because we benchmark on
ScreenSpot; the hash pass below is the mechanical guarantee on top.

Three independent mechanisms, each counted separately in the report:
perceptual average-hash (hamming <= threshold) catches resized/recompressed
duplicates; exact RGB-pixel sha256 catches byte-identical image content with
different containers; normalized question text (with ``--eval-jsonl``) catches
cross-split text leakage even when images differ.

Example::

    python tools/build_train_mix.py \
        --source data/train_raw/textvqa_train.jsonl:all:ours \
        --source data/train_raw/docvqa_train.jsonl:25000:ours \
        --source data/sft_art/train.jsonl:10000:0xsero:data/sft_art/imgs \
        --eval-images data/eval/images --eval-jsonl data/eval/textvqa.jsonl \
        --out data/train_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

from PIL import Image

from fetch_eval_data import keep_record

MAX_ANSWER_WORDS = 20


def average_hash(image: Image.Image, size: int = 8) -> int:
    gray = image.convert("L").resize((size, size), Image.LANCZOS)
    pixels = list(gray.getdata())
    mean = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= mean)
    return value


def pixel_sha256(image: Image.Image) -> str:
    """Exact content hash: catches byte-identical pixels under different containers."""

    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def normalize_text(text: str) -> str:
    """Case/punctuation-insensitive question text for cross-split near-dup checks."""

    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def is_contaminated(candidate: int, eval_hashes: list[int], threshold: int = 6) -> bool:
    return any(hamming(candidate, eval_hash) <= threshold for eval_hash in eval_hashes)


def select_subset(records: list, cap: int | None, rng: random.Random) -> list:
    if cap is None or len(records) <= cap:
        return list(records)
    return rng.sample(records, cap)


def strip_image_span(text: str) -> str:
    marker = "|end_of_image|"
    if marker in text:
        text = text.split(marker, 1)[1]
    return text.strip()


def normalize_0xsero_row(row: dict) -> dict | None:
    turns = row.get("conversations") or []
    if not row.get("image"):
        return None
    user = next((turn for turn in turns if turn.get("role") == "user"), None)
    assistant = next((turn for turn in turns if turn.get("role") == "assistant"), None)
    if not user or not assistant:
        return None
    question = strip_image_span(str(user.get("content", "")))
    answer = str(assistant.get("content", "")).strip()
    if not question or not answer:
        return None
    return {"image": row["image"], "question": question, "answers": [answer]}


def parse_source(spec: str) -> tuple[Path, int | None, str, Path | None]:
    parts = spec.split(":")
    if len(parts) < 3 or parts[2] not in ("ours", "0xsero"):
        raise ValueError(f"--source must be path:cap:ours|0xsero[:imageroot], got {spec!r}")
    cap = None if parts[1] == "all" else int(parts[1])
    imageroot = Path(parts[3]) if len(parts) > 3 else None
    return Path(parts[0]), cap, parts[2], imageroot


def hash_eval_images(eval_dirs: list[Path]) -> tuple[list[int], set[str]]:
    hashes, pixel_hashes = [], set()
    for directory in eval_dirs:
        for path in sorted(directory.glob("**/*")):
            if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                with Image.open(path) as image:
                    hashes.append(average_hash(image))
                    pixel_hashes.add(pixel_sha256(image))
    return hashes, pixel_hashes


def load_eval_questions(eval_jsonls: list[Path]) -> set[str]:
    questions = set()
    for jsonl_path in eval_jsonls:
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("question"):
                    questions.add(normalize_text(record["question"]))
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", required=True,
                        help="jsonl:cap:ours|0xsero[:imageroot]; repeatable")
    parser.add_argument("--eval-images", action="append", type=Path, required=True)
    parser.add_argument("--eval-jsonl", action="append", type=Path, default=[],
                        help="eval records for question-text near-dup checks; repeatable")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    images_out = args.out / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    eval_hashes, eval_pixel_hashes = hash_eval_images(args.eval_images)
    eval_questions = load_eval_questions(args.eval_jsonl)
    print(f"eval images hashed: {len(eval_hashes)} aHash + {len(eval_pixel_hashes)} pixel "
          f"(threshold {args.threshold}); eval questions: {len(eval_questions)}")

    merged, report_sources = [], []
    for spec in args.source:
        jsonl_path, cap, kind, imageroot = parse_source(spec)
        name = jsonl_path.stem
        root = imageroot or jsonl_path.parent
        rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        normalized = []
        for row in rows:
            record = normalize_0xsero_row(row) if kind == "0xsero" else {
                "image": row["image"], "question": row["question"], "answers": row["answers"],
            }
            if record and keep_record(record["answers"], MAX_ANSWER_WORDS):
                normalized.append(record)
        selected = select_subset(normalized, cap, rng)
        kept = dropped = 0
        drop_reasons = {"missing": 0, "ahash": 0, "pixel": 0, "text": 0}
        for record in selected:
            source_image = root / record["image"]
            if not source_image.exists():
                dropped += 1
                drop_reasons["missing"] += 1
                continue
            with Image.open(source_image) as image:
                candidate = average_hash(image)
                candidate_pixel = pixel_sha256(image)
            if is_contaminated(candidate, eval_hashes, args.threshold):
                dropped += 1
                drop_reasons["ahash"] += 1
                continue
            if candidate_pixel in eval_pixel_hashes:
                dropped += 1
                drop_reasons["pixel"] += 1
                continue
            if normalize_text(record["question"]) in eval_questions:
                dropped += 1
                drop_reasons["text"] += 1
                continue
            record_id = f"{name}_{kept:06d}"
            image_name = f"{record_id}{source_image.suffix.lower()}"
            shutil.copyfile(source_image, images_out / image_name)
            merged.append({
                "id": record_id,
                "image": f"images/{image_name}",
                "question": record["question"],
                "answers": record["answers"],
                "source": name,
            })
            kept += 1
        report_sources.append({
            "source": name, "jsonl": str(jsonl_path), "kind": kind,
            "rows": len(rows), "after_answer_filter": len(normalized),
            "cap": cap, "selected": len(selected), "kept": kept,
            "dropped_missing_or_contaminated": dropped,
            "drop_reasons": drop_reasons,
        })
        print(f"{name}: {len(rows)} rows -> kept {kept} (dropped {dropped}: {drop_reasons})", flush=True)

    rng.shuffle(merged)
    body = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in merged)
    (args.out / "train_mix.jsonl").write_text(body, encoding="utf-8")
    report = {
        "seed": args.seed, "threshold": args.threshold, "eval_images": len(eval_hashes),
        "eval_pixel_hashes": len(eval_pixel_hashes), "eval_questions": len(eval_questions),
        "total_kept": len(merged), "sources": report_sources,
    }
    (args.out / "decontamination_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"train_mix: {len(merged)} records -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
