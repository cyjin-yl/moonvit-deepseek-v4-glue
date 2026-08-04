"""Deterministic minimal-pair visual perception diagnostics.

The generator deliberately keeps the language prompt constant inside each pair.
Only the named visual attribute changes, and both variants have explicit answer
regions so later masking and activation-patching experiments can reuse them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import PIL
from PIL import Image, ImageDraw, ImageFont


FORMAT_VERSION = "synthetic-perception-v1"
TASKS = ("color", "shape", "count", "spatial", "ocr", "coordinate")
SPLITS = ("train", "selection")

COLORS = {
    "red": "#d62728",
    "blue": "#1f77b4",
    "green": "#2ca02c",
    "yellow": "#f2c94c",
    "black": "#111111",
    "white": "#f7f7f7",
}
SHAPES = ("circle", "square", "triangle", "star")
GRID_LABELS = (
    "top-left", "top-center", "top-right",
    "middle-left", "center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
)


@dataclass(frozen=True)
class SuiteConfig:
    """Generation parameters; 200 means 200 *base pairs* per task and split."""

    samples_per_task: int = 200
    image_size: int = 256
    seed: int = 20260804
    background_train: str = "#edf3f8"
    background_selection: str = "#fff5e6"

    def validate(self) -> None:
        if self.samples_per_task < 1:
            raise ValueError("samples_per_task must be positive")
        if self.image_size < 64:
            raise ValueError("image_size must be at least 64")


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Pillow ships this font, which avoids an operating-system font dependency.
    return ImageFont.load_default(size=max(8, size))


def _regular_polygon(cx: float, cy: float, radius: float, vertices: int, rotation: float) -> list[tuple[float, float]]:
    return [
        (
            cx + radius * math.cos(rotation + 2 * math.pi * index / vertices),
            cy + radius * math.sin(rotation + 2 * math.pi * index / vertices),
        )
        for index in range(vertices)
    ]


def _shape_points(shape: str, box: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    left, top, right, bottom = box
    cx, cy = (left + right) / 2, (top + bottom) / 2
    radius = min(right - left, bottom - top) / 2
    if shape == "triangle":
        return _regular_polygon(cx, cy, radius, 3, -math.pi / 2)
    if shape == "star":
        points = []
        for index in range(10):
            angle = -math.pi / 2 + index * math.pi / 5
            r = radius if index % 2 == 0 else radius * 0.43
            points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        return points
    raise ValueError(f"{shape} has no polygon points")


def _draw_shape(
    draw: ImageDraw.ImageDraw,
    shape: str,
    box: tuple[int, int, int, int],
    fill: str,
    *,
    outline: str = "#273142",
    width: int = 3,
) -> None:
    if shape == "circle":
        draw.ellipse(box, fill=fill, outline=outline, width=width)
    elif shape == "square":
        draw.rectangle(box, fill=fill, outline=outline, width=width)
    elif shape in {"triangle", "star"}:
        draw.polygon(_shape_points(shape, box), fill=fill, outline=outline)
    else:
        raise ValueError(f"unknown shape: {shape}")


def _canvas(config: SuiteConfig, split: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    background = config.background_train if split == "train" else config.background_selection
    image = Image.new("RGB", (config.image_size, config.image_size), background)
    draw = ImageDraw.Draw(image)
    margin = max(4, config.image_size // 32)
    if split == "train":
        draw.rounded_rectangle(
            (margin, margin, config.image_size - margin, config.image_size - margin),
            radius=margin * 2,
            outline="#9aabc0",
            width=max(1, config.image_size // 128),
        )
    else:
        draw.rectangle(
            (margin, margin, config.image_size - margin, config.image_size - margin),
            outline="#c39b64",
            width=max(1, config.image_size // 128),
        )
    return image, draw


def _box_around(cx: int, cy: int, radius: int) -> tuple[int, int, int, int]:
    return (cx - radius, cy - radius, cx + radius, cy + radius)


def _pair_spec(task: str, split: str, index: int, config: SuiteConfig, ocr_values: tuple[str, str] | None) -> dict:
    rng = random.Random(_stable_seed(config.seed, split, task, index))
    size = config.image_size
    center = size // 2
    radius = max(12, size // 6)
    question_templates = {
        "train": {
            "color": "What color is the object?",
            "shape": "What shape is shown?",
            "count": "How many objects are shown?",
            "spatial": "{spatial_question}",
            "ocr": "Read the code in the image.",
            "coordinate": "Which grid cell contains the target?",
        },
        "selection": {
            "color": "Name the object's color.",
            "shape": "Identify the geometric shape.",
            "count": "Count the visible objects.",
            "spatial": "{spatial_question}",
            "ocr": "Transcribe the displayed code exactly.",
            "coordinate": "Give the target's grid position.",
        },
    }
    template_id = f"{split}-{task}-layout-v1"

    if task == "color":
        first, second = rng.sample(sorted(COLORS), 2)
        shape = rng.choice(SHAPES)
        common = {"shape": shape, "box": _box_around(center, center, radius)}
        variants = [
            {"render": {**common, "color": color}, "answer": color, "answer_region": common["box"]}
            for color in (first, second)
        ]
        changed = "fill_color"
        question = question_templates[split][task]
    elif task == "shape":
        first, second = rng.sample(list(SHAPES), 2)
        color = rng.choice(["red", "blue", "green", "yellow"])
        box = _box_around(center, center, radius)
        variants = [
            {"render": {"shape": shape, "color": color, "box": box}, "answer": shape, "answer_region": box}
            for shape in (first, second)
        ]
        changed = "geometry"
        question = question_templates[split][task]
    elif task == "count":
        first, second = rng.sample(range(1, 10), 2)
        color = rng.choice(["red", "blue", "green", "yellow"])
        shape = rng.choice(["circle", "square", "triangle"])
        variants = [
            {
                "render": {"count": count, "shape": shape, "color": color},
                "answer": str(count),
                "answer_region": (size // 8, size // 8, size * 7 // 8, size * 7 // 8),
            }
            for count in (first, second)
        ]
        changed = "object_count"
        question = question_templates[split][task]
    elif task == "spatial":
        subtype = ("horizontal", "vertical", "containment", "nearest")[index % 4]
        offset = size // 4
        object_radius = max(8, size // 12)
        if subtype == "horizontal":
            fixed = (center, center)
            positions = ((center - offset, center), (center + offset, center))
            answers = ("left", "right")
            spatial_question = "Is the red circle left or right of the blue square?"
            render = lambda position: {"subtype": subtype, "moving": position, "fixed": fixed, "radius": object_radius}
        elif subtype == "vertical":
            fixed = (center, center)
            positions = ((center, center - offset), (center, center + offset))
            answers = ("above", "below")
            spatial_question = "Is the red circle above or below the blue square?"
            render = lambda position: {"subtype": subtype, "moving": position, "fixed": fixed, "radius": object_radius}
        elif subtype == "containment":
            fixed = (center, center)
            positions = ((center, center), (center + offset + object_radius, center))
            answers = ("inside", "outside")
            spatial_question = "Is the red circle inside or outside the blue square?"
            render = lambda position: {"subtype": subtype, "moving": position, "fixed": fixed, "radius": object_radius}
        else:
            positions = ((center - offset, center), (center + offset, center))
            answers = ("red circle", "blue square")
            spatial_question = "Which object is nearest to the black star?"
            render = lambda position: {"subtype": subtype, "target": position, "radius": object_radius}
        variants = [
            {
                "render": render(position),
                "answer": answer,
                "answer_region": _box_around(position[0], position[1], object_radius * 2),
            }
            for position, answer in zip(positions, answers)
        ]
        changed = "target_position"
        question = question_templates[split][task].format(spatial_question=spatial_question)
    elif task == "ocr":
        if ocr_values is None:
            raise ValueError("OCR task requires reserved strings")
        text_box = (size // 8, size * 3 // 8, size * 7 // 8, size * 5 // 8)
        variants = [
            {"render": {"text": text}, "answer": text, "answer_region": text_box}
            for text in ocr_values
        ]
        changed = "glyph_sequence"
        question = question_templates[split][task]
    elif task == "coordinate":
        first, second = rng.sample(range(9), 2)
        variants = [
            {
                "render": {"cell": cell},
                "answer": GRID_LABELS[cell],
                "answer_region": _grid_cell_box(size, cell),
            }
            for cell in (first, second)
        ]
        changed = "target_location"
        question = question_templates[split][task]
    else:
        raise ValueError(f"unknown task: {task}")

    return {
        "question": question,
        "template_id": template_id,
        "changed_attribute": changed,
        "variants": variants,
    }


def _grid_cell_box(size: int, cell: int) -> tuple[int, int, int, int]:
    margin = size // 8
    span = size - 2 * margin
    cell_size = span / 3
    row, column = divmod(cell, 3)
    return (
        round(margin + column * cell_size),
        round(margin + row * cell_size),
        round(margin + (column + 1) * cell_size),
        round(margin + (row + 1) * cell_size),
    )


def _render(spec: dict, task: str, split: str, config: SuiteConfig) -> Image.Image:
    image, draw = _canvas(config, split)
    size = config.image_size
    if task in {"color", "shape"}:
        _draw_shape(draw, spec["shape"], tuple(spec["box"]), COLORS[spec["color"]])
    elif task == "count":
        positions = [
            (column, row)
            for row in range(3)
            for column in range(3)
        ]
        radius = max(5, size // 22)
        start = size // 4
        spacing = size // 4
        for column, row in positions[: int(spec["count"])]:
            cx, cy = start + column * spacing, start + row * spacing
            _draw_shape(
                draw,
                spec["shape"],
                _box_around(cx, cy, radius),
                COLORS[spec["color"]],
                width=max(1, size // 128),
            )
    elif task == "spatial":
        radius = int(spec["radius"])
        subtype = spec["subtype"]
        if subtype == "nearest":
            circle = (size // 4, size // 2)
            square = (size * 3 // 4, size // 2)
            _draw_shape(draw, "circle", _box_around(*circle, radius), COLORS["red"])
            _draw_shape(draw, "square", _box_around(*square, radius), COLORS["blue"])
            _draw_shape(draw, "star", _box_around(*spec["target"], max(4, radius // 2)), COLORS["black"])
        elif subtype == "containment":
            fixed = tuple(spec["fixed"])
            draw.rectangle(_box_around(*fixed, radius * 2), outline=COLORS["blue"], width=max(2, size // 64))
            _draw_shape(draw, "circle", _box_around(*spec["moving"], radius), COLORS["red"])
        else:
            _draw_shape(draw, "square", _box_around(*spec["fixed"], radius), COLORS["blue"])
            _draw_shape(draw, "circle", _box_around(*spec["moving"], radius), COLORS["red"])
    elif task == "ocr":
        text = str(spec["text"])
        font = _font(max(14, size // (len(text) + 2)))
        box = draw.textbbox((0, 0), text, font=font, stroke_width=1)
        width, height = box[2] - box[0], box[3] - box[1]
        draw.text(
            ((size - width) / 2, (size - height) / 2),
            text,
            font=font,
            fill="#101820",
            stroke_width=1,
            stroke_fill="#101820",
        )
    elif task == "coordinate":
        margin = size // 8
        span = size - 2 * margin
        for index in range(4):
            value = round(margin + index * span / 3)
            draw.line((margin, value, size - margin, value), fill="#65758b", width=max(1, size // 128))
            draw.line((value, margin, value, size - margin), fill="#65758b", width=max(1, size // 128))
        box = _grid_cell_box(size, int(spec["cell"]))
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        _draw_shape(draw, "star", _box_around(cx, cy, max(6, size // 20)), COLORS["red"])
    else:
        raise ValueError(task)
    return image


def generate_background_matched_aux(
    authoritative_selection: Path | str,
    output_dir: Path | str,
    config: SuiteConfig = SuiteConfig(),
) -> dict:
    """Re-render fixed selection scenes with the train background for diagnosis only."""

    config.validate()
    source_path = Path(authoritative_selection).resolve()
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite background auxiliary: {output}")
    output.mkdir(parents=True)
    rows = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows:
        raise ValueError("authoritative selection is empty")

    matched_config = SuiteConfig(
        samples_per_task=config.samples_per_task,
        image_size=config.image_size,
        seed=config.seed,
        background_train=config.background_train,
        background_selection=config.background_train,
    )
    auxiliary_rows: list[dict] = []
    image_hashes: dict[str, str] = {}
    for source in rows:
        if source.get("split") != "selection":
            raise ValueError(f"background auxiliary accepts selection rows only: {source['id']}")
        source_image = source_path.parent / str(source["image"])
        if _sha256(source_image) != str(source["image_sha256"]):
            raise ValueError(f"authoritative image hash mismatch: {source['id']}")
        relative_image = Path(str(source["image"]))
        destination = output / relative_image
        destination.parent.mkdir(parents=True, exist_ok=True)
        rendered = _render(
            source["generation"]["render"],
            str(source["task"]),
            "selection",
            matched_config,
        )
        rendered.save(destination, format="PNG", optimize=False, compress_level=9)
        digest = _sha256(destination)
        image_hashes[str(source["id"])] = digest
        auxiliary = dict(source)
        auxiliary["authoritative_image_sha256"] = str(source["image_sha256"])
        auxiliary["image_sha256"] = digest
        auxiliary["auxiliary"] = {
            "name": "selection_background_matched_aux",
            "diagnostic_only": True,
            "background_from": config.background_selection,
            "background_to": config.background_train,
        }
        auxiliary_rows.append(auxiliary)

    records_path = output / "selection_background_matched_aux.jsonl"
    _write_jsonl(records_path, auxiliary_rows)
    manifest = {
        "format_version": "synthetic-background-matched-aux-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "training_allowed": False,
        "final_evaluation_half_used": False,
        "authoritative_selection": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
        },
        "background_from": config.background_selection,
        "background_to": config.background_train,
        "scene_source": "authoritative rows generation.render; no scene resampling",
        "records": len(auxiliary_rows),
        "tasks": dict(
            sorted(
                {
                    task: sum(str(row["task"]) == task for row in auxiliary_rows)
                    for task in {str(row["task"]) for row in auxiliary_rows}
                }.items()
            )
        ),
        "image_hashes_sha256": _json_hash(image_hashes),
        "files": {
            records_path.name: {
                "bytes": records_path.stat().st_size,
                "sha256": _sha256(records_path),
            }
        },
    }
    _write_json(output / "MANIFEST.json", manifest)
    return manifest


def _ocr_pairs(config: SuiteConfig) -> dict[str, list[tuple[str, str]]]:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    used: set[str] = set()
    result: dict[str, list[tuple[str, str]]] = {}
    for split in SPLITS:
        rng = random.Random(_stable_seed(config.seed, split, "ocr-pool"))
        pairs = []
        while len(pairs) < config.samples_per_task:
            values = []
            while len(values) < 2:
                length = rng.randint(2, 6)
                value = "".join(rng.choice(alphabet) for _ in range(length))
                if value not in used:
                    used.add(value)
                    values.append(value)
            pairs.append((values[0], values[1]))
        result[split] = pairs
    return result


def _control_image(path: Path, config: SuiteConfig, split: str, kind: str) -> None:
    if kind == "blank":
        color = config.background_train if split == "train" else config.background_selection
        image = Image.new("RGB", (config.image_size, config.image_size), color)
    else:
        image = Image.new("RGB", (config.image_size, config.image_size), "#b8c0cc")
        draw = ImageDraw.Draw(image)
        step = max(8, config.image_size // 8)
        for offset in range(0, config.image_size, step):
            draw.line((offset, 0, 0, offset), fill="#7f8a99", width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)


def _leakage_checks(rows_by_split: dict[str, list[dict]]) -> dict[str, int]:
    train = rows_by_split["train"]
    selection = rows_by_split["selection"]
    def values(rows: list[dict], key: str) -> set[str]:
        return {str(row[key]) for row in rows}
    train_ocr = {row["answers"][0] for row in train if row["task"] == "ocr"}
    selection_ocr = {row["answers"][0] for row in selection if row["task"] == "ocr"}
    return {
        "image_hash_overlap": len(values(train, "image_sha256") & values(selection, "image_sha256")),
        "ocr_string_overlap": len(train_ocr & selection_ocr),
        "pair_id_overlap": len(values(train, "pair_id") & values(selection, "pair_id")),
        "template_id_overlap": len(values(train, "template_id") & values(selection, "template_id")),
    }


def generate_suite(output_dir: Path | str, config: SuiteConfig = SuiteConfig()) -> dict:
    """Generate both splits and their causal-control assignments.

    The destination must not exist, preventing accidental replacement of an old
    experimental data package.
    """

    config.validate()
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing synthetic suite: {output}")
    output.mkdir(parents=True)
    ocr_pairs = _ocr_pairs(config)
    rows_by_split: dict[str, list[dict]] = {}
    controls: list[dict] = []
    counts: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        split_rows = []
        counts[split] = {}
        for task in TASKS:
            for index in range(config.samples_per_task):
                pair_id = f"{split}-{task}-{index:05d}"
                pair = _pair_spec(
                    task,
                    split,
                    index,
                    config,
                    ocr_pairs[split][index] if task == "ocr" else None,
                )
                for variant_index, variant_name in enumerate(("a", "b")):
                    sample_id = f"{pair_id}-{variant_name}"
                    relative = Path("images") / split / task / f"{sample_id}.png"
                    image_path = output / relative
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image = _render(pair["variants"][variant_index]["render"], task, split, config)
                    image.save(image_path, format="PNG", optimize=False, compress_level=9)
                    variant = pair["variants"][variant_index]
                    row = {
                        "id": sample_id,
                        "pair_id": pair_id,
                        "pair_variant": variant_name,
                        "split": split,
                        "task": task,
                        "source": f"synthetic_{task}",
                        "image": relative.as_posix(),
                        "image_sha256": _sha256(image_path),
                        "image_size": [config.image_size, config.image_size],
                        "question": pair["question"],
                        "answers": [variant["answer"]],
                        "metric": "exact_match",
                        "template_id": pair["template_id"],
                        "changed_attribute": pair["changed_attribute"],
                        "answer_region_pixels": list(variant["answer_region"]),
                        "generation": {
                            "base_index": index,
                            "seed": _stable_seed(config.seed, split, task, index),
                            "render": variant["render"],
                        },
                    }
                    split_rows.append(row)
            counts[split][task] = config.samples_per_task
        split_rows.sort(key=lambda row: row["id"])
        rows_by_split[split] = split_rows
        _write_jsonl(output / f"{split}.jsonl", split_rows)

        blank_path = Path("controls") / split / "blank.png"
        same_path = Path("controls") / split / "same.png"
        _control_image(output / blank_path, config, split, "blank")
        _control_image(output / same_path, config, split, "same")
        for task in TASKS:
            task_rows = [row for row in split_rows if row["task"] == task]
            rng = random.Random(_stable_seed(config.seed, split, task, "shuffle"))
            offset = rng.randrange(1, len(task_rows))
            shuffled = task_rows[offset:] + task_rows[:offset]
            for row, shuffled_row in zip(task_rows, shuffled):
                controls.append({
                    "id": row["id"],
                    "split": split,
                    "task": task,
                    "true_image": row["image"],
                    "blind_image": None,
                    "blank_image": blank_path.as_posix(),
                    "same_image": same_path.as_posix(),
                    "same_image_id": f"control:{split}:same",
                    "shuffled_image": shuffled_row["image"],
                    "shuffled_image_id": shuffled_row["id"],
                    "patch_permutation": {
                        "algorithm": "torch.randperm",
                        "seed": _stable_seed(config.seed, split, row["id"], "patch-permutation"),
                        "scope": "merged MoonViT spatial token axis; values and token count preserved",
                    },
                    "condition_notes": {
                        "same_image_fixed_per_task": False,
                        "same_image_fixed_per_split": True,
                        "shuffle_within_task": True,
                    },
                })

    controls.sort(key=lambda row: row["id"])
    _write_jsonl(output / "controls.jsonl", controls)
    leakage = _leakage_checks(rows_by_split)
    if any(leakage.values()):
        raise RuntimeError(f"synthetic split leakage detected: {leakage}")

    logical_rows = [
        {key: value for key, value in row.items() if key != "image"}
        for split in SPLITS for row in rows_by_split[split]
    ]
    manifest = {
        "format_version": FORMAT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "moonvit_glue.synthetic_perception.generate_suite",
        "pillow_version": PIL.__version__,
        "config": asdict(config),
        "tasks": list(TASKS),
        "splits": list(SPLITS),
        "counts": {
            "base_samples_by_split_task": {
                split: {task: counts[split][task] for task in sorted(TASKS)}
                for split in sorted(SPLITS)
            },
            "rendered_samples_by_split": {
                split: len(rows_by_split[split]) for split in sorted(SPLITS)
            },
            "control_assignments": len(controls),
        },
        "leakage_checks": leakage,
        "logical_dataset_sha256": _json_hash(logical_rows),
        "files": {},
        "notes": {
            "minimal_pair_rule": "question is byte-identical within each pair; one named visual attribute changes",
            "ocr_text": "OCR glyphs are the task stimulus itself; no extra answer label or textual hint is rendered",
            "final_evaluation_half_used": False,
            "images_committed_to_git": False,
        },
    }

    with (output / "counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "task", "base_pairs", "rendered_samples"])
        writer.writeheader()
        for split in sorted(SPLITS):
            for task in sorted(TASKS):
                writer.writerow({
                    "split": split,
                    "task": task,
                    "base_pairs": counts[split][task],
                    "rendered_samples": counts[split][task] * 2,
                })

    for filename in ("train.jsonl", "selection.jsonl", "controls.jsonl", "counts.csv"):
        path = output / filename
        manifest["files"][filename] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    _write_json(output / "MANIFEST.json", manifest)
    summary = {
        "status": "valid",
        "base_samples": sum(sum(task_counts.values()) for task_counts in counts.values()),
        "rendered_samples": sum(len(rows) for rows in rows_by_split.values()),
        "control_assignments": len(controls),
        "failures": 0,
        "leakage_checks": leakage,
        "logical_dataset_sha256": manifest["logical_dataset_sha256"],
    }
    _write_json(output / "SUMMARY.json", summary)
    (output / "failures.jsonl").write_text("", encoding="utf-8")
    return manifest


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_suite(output_dir: Path | str) -> dict:
    """Independently verify manifests, images, pairs, controls, and split leakage."""

    output = Path(output_dir)
    manifest = json.loads((output / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported synthetic format: {manifest.get('format_version')}")
    for filename, expected in manifest["files"].items():
        path = output / filename
        if not path.is_file() or path.stat().st_size != int(expected["bytes"]) or _sha256(path) != expected["sha256"]:
            raise ValueError(f"file hash mismatch: {filename}")

    rows_by_split = {split: _read_jsonl(output / f"{split}.jsonl") for split in SPLITS}
    all_rows = [row for split in SPLITS for row in rows_by_split[split]]
    identifiers = [str(row["id"]) for row in all_rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate sample ids")

    image_hashes_verified = 0
    pairs: dict[str, list[dict]] = {}
    for row in all_rows:
        image_path = output / row["image"]
        if not image_path.is_file() or _sha256(image_path) != row["image_sha256"]:
            raise ValueError(f"image hash mismatch: {row['id']}")
        image_hashes_verified += 1
        pairs.setdefault(str(row["pair_id"]), []).append(row)
    for pair_id, pair in pairs.items():
        if len(pair) != 2:
            raise ValueError(f"pair cardinality mismatch: {pair_id}")
        if pair[0]["question"] != pair[1]["question"]:
            raise ValueError(f"pair question mismatch: {pair_id}")
        if pair[0]["answers"] == pair[1]["answers"]:
            raise ValueError(f"pair answer did not flip: {pair_id}")
        if pair[0]["changed_attribute"] != pair[1]["changed_attribute"]:
            raise ValueError(f"pair attribute mismatch: {pair_id}")

    leakage = _leakage_checks(rows_by_split)
    if leakage != manifest["leakage_checks"] or any(leakage.values()):
        raise ValueError(f"split leakage detected: {leakage}")

    controls = _read_jsonl(output / "controls.jsonl")
    if {row["id"] for row in controls} != set(identifiers):
        raise ValueError("control assignments do not exactly cover dataset ids")
    same_images: dict[str, set[str]] = {split: set() for split in SPLITS}
    for row in controls:
        if row["blind_image"] is not None:
            raise ValueError(f"blind control carries an image: {row['id']}")
        if row["shuffled_image_id"] == row["id"]:
            raise ValueError(f"shuffle fixed point: {row['id']}")
        if not (output / row["blank_image"]).is_file():
            raise ValueError(f"missing blank control: {row['id']}")
        if not (output / row["same_image"]).is_file():
            raise ValueError(f"missing same-image control: {row['id']}")
        same_images[row["split"]].add(row["same_image"])
        patch = row["patch_permutation"]
        if patch.get("algorithm") != "torch.randperm" or not isinstance(patch.get("seed"), int):
            raise ValueError(f"invalid patch permutation: {row['id']}")
    if any(len(paths) != 1 for paths in same_images.values()):
        raise ValueError("same-image control is not fixed within a split")

    logical_rows = [
        {key: value for key, value in row.items() if key != "image"}
        for row in all_rows
    ]
    logical_hash = _json_hash(logical_rows)
    if logical_hash != manifest["logical_dataset_sha256"]:
        raise ValueError("logical dataset hash mismatch")
    return {
        "status": "valid",
        "format_version": FORMAT_VERSION,
        "image_hashes_verified": image_hashes_verified,
        "pairs_verified": len(pairs),
        "controls_verified": len(controls),
        "file_hashes_verified": len(manifest["files"]),
        "leakage_checks": leakage,
        "logical_dataset_sha256": logical_hash,
    }
