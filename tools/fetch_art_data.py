"""Build the 0xSero art SFT subset (WikiArt + fashion) from local parquet shards.

Offline port of 0xSero/fable-glm-vision ``build_art_dataset.py``: same QA
templates, same label filters, same <=2 QA-per-image cap, same image resize
(<= max-pixels, dimensions rounded to 28px, JPEG q90), and the same 0xSero
conversations schema — so ``tools/build_train_mix.py`` consumes the output
through ``normalize_0xsero_row`` unchanged. Hub downloads are replaced by
aria2-prefetched parquet (``tools/prefetch_parquet.py``); rows are read with
raw pyarrow (the workstation's datasets+dill stack cannot pickle pyarrow's
MonthDayNano).

Two passes per dataset: pass 1 collects label metadata and decides validity,
pass 2 decodes only the sampled images — WikiArt parquets decode to several
GB of pixels, so nothing image-shaped is held across the sampling shuffle.

Example::

    python tools/fetch_art_data.py \
        --wikiart-files $PQ/wikiart/*.parquet --fashion-files $PQ/fashion/*.parquet \
        --out-dir data/sft_art --img-dir data/sft_art/imgs
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image

from fetch_eval_data import decode_image_value, iter_parquet_rows

PATCH = 14
MERGE = 2
ROUND = PATCH * MERGE  # 28px, MoonViT patch*merge — kept from the 0xSero script

# 0xSero/GLM training-time convention; build_train_mix.strip_image_span strips it.
IMAGE_SPAN = "|begin_of_image|" + "|image|" * 128 + "|end_of_image|"

WIKIART_LABELS = ("artist", "style", "genre", "date")
FASHION_LABELS = ("baseColour", "articleType", "season", "usage", "masterCategory")


def preprocess_size(w: int, h: int, max_pixels: int) -> tuple[int, int]:
    """Aspect-preserving resize to <= max_pixels, dims rounded to 28px (0xSero)."""

    scale = (max_pixels / (w * h)) ** 0.5
    if scale < 1.0:
        w, h = int(w * scale), int(h * scale)
    w = max(ROUND, (w // ROUND) * ROUND)
    h = max(ROUND, (h // ROUND) * ROUND)
    return w, h


def save_image(img: Image.Image, img_dir: Path, prefix: str, idx: int, max_pixels: int) -> str:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = preprocess_size(*img.size, max_pixels)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.LANCZOS)
    path = img_dir / f"{prefix}_{idx:06d}.jpg"
    img.save(path, "JPEG", quality=90)
    return str(path)


def make_example(img_path: str, question: str, answer: str) -> dict:
    return {
        "image": img_path,
        "conversations": [
            {"role": "user", "content": f"{IMAGE_SPAN}\n{question}"},
            {"role": "assistant", "content": answer},
        ],
    }


def wikiart_qa(labels: dict, rng: random.Random) -> list[tuple[str, str]]:
    artist = labels.get("artist") or ""
    style = labels.get("style") or ""
    genre = labels.get("genre") or ""
    date = labels.get("date") or ""
    qa = []
    if artist and artist.lower() not in ("unknown", "unknown artist"):
        qa.append((rng.choice([
            "Who painted this?",
            "Who is the artist of this painting?",
            "Name the artist who created this work.",
        ]), artist))
    if style:
        qa.append((rng.choice([
            "What art style is this painting?",
            "What artistic movement does this painting belong to?",
        ]), style))
    if genre:
        qa.append((rng.choice([
            "What genre is this painting?",
            "What type of painting is this?",
        ]), genre))
    if date and date.isdigit() and len(date) == 4:
        qa.append(("When was this painted?", date))
    rng.shuffle(qa)
    return qa[:2]


def fashion_qa(labels: dict, rng: random.Random) -> list[tuple[str, str]]:
    qa = []
    if labels.get("baseColour"):
        qa.append((rng.choice([
            "What is the main color of this item?",
            "What color is this product?",
        ]), labels["baseColour"]))
    if labels.get("articleType"):
        qa.append((rng.choice([
            "What type of item is this?",
            "What product is shown in this image?",
        ]), labels["articleType"]))
    if labels.get("season"):
        qa.append(("What season is this item designed for?", labels["season"]))
    if labels.get("usage"):
        qa.append(("What occasion is this item for?", labels["usage"]))
    if labels.get("masterCategory"):
        qa.append(("What product category does this belong to?", labels["masterCategory"]))
    rng.shuffle(qa)
    return qa[:2]


def collect_metadata(data_files: list[Path], label_fields: tuple[str, ...]) -> list[dict]:
    """Pass 1: labels only, image bytes dropped on the floor."""

    metadata = []
    for row in iter_parquet_rows(data_files):
        labels = {}
        for field in label_fields:
            value = row.get(field)
            labels[field] = value.strip() if isinstance(value, str) else ""
        labels["has_image"] = row.get("image") is not None
        metadata.append(labels)
    return metadata


def build_subset(data_files: list[Path], label_fields: tuple[str, ...], qa_fn,
                 limit: int, img_dir: Path, prefix: str, max_pixels: int,
                 rng: random.Random) -> list[dict]:
    img_dir.mkdir(parents=True, exist_ok=True)  # save_image failures are swallowed below
    metadata = collect_metadata(data_files, label_fields)
    qa_per_index = [
        (index, qa_fn(labels, rng))
        for index, labels in enumerate(metadata)
        if labels["has_image"]
    ]
    order = [index for index, qa in qa_per_index if qa]
    rng.shuffle(order)
    qa_of = dict(qa_per_index)
    chosen = set(order[:limit])

    rows, used = [], 0
    for index, row in enumerate(iter_parquet_rows(data_files)):  # pass 2: decode sampled only
        if used >= limit:
            break
        if index not in chosen:
            continue
        image = decode_image_value(row.get("image"))
        if image is None:
            continue
        try:
            path = save_image(image, img_dir, prefix, index, max_pixels)
        except Exception:
            continue
        for question, answer in qa_of[index]:
            rows.append(make_example(path, question, answer))
        used += 1
        if used % 5000 == 0:
            print(f"  {used:,} {prefix} images", flush=True)
    print(f"  {prefix}: {len(rows):,} examples from {used:,} images", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wikiart-files", type=Path, nargs="+", required=True)
    parser.add_argument("--fashion-files", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/sft_art"))
    parser.add_argument("--img-dir", type=Path, default=None, help="default: <out-dir>/imgs")
    parser.add_argument("--n-wikiart", type=int, default=25_000)
    parser.add_argument("--n-fashion", type=int, default=12_000)
    parser.add_argument("--max-pixels", type=int, default=300_000)
    parser.add_argument("--train-frac", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    img_dir = args.img_dir or (args.out_dir / "imgs")
    img_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = (
        build_subset(args.wikiart_files, WIKIART_LABELS, wikiart_qa, args.n_wikiart,
                     img_dir, "wikiart", args.max_pixels, rng)
        + build_subset(args.fashion_files, FASHION_LABELS, fashion_qa, args.n_fashion,
                       img_dir, "fashion", args.max_pixels, rng)
    )
    rng.shuffle(rows)
    n_train = int(len(rows) * args.train_frac)
    with open(args.out_dir / "train.jsonl", "w") as handle:
        for row in rows[:n_train]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.out_dir / "val.jsonl", "w") as handle:
        for row in rows[n_train:]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"DONE: {n_train:,} train / {len(rows) - n_train:,} val -> {args.out_dir}")


if __name__ == "__main__":
    main()
