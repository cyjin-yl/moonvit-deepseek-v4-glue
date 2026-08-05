"""冻结并验证训练记录顺序、监督来源与原图身份。"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .grounding_contract import format_click_action, parse_click_action
from .metrics import normalize_answer


SCHEMA_VERSION = "qwen3b-training-order-v1"

# 训练包早于评测合同，部分 click 监督只缺少逗号后的 canonical 空格。
# 这里仅接受单一、完整、整数动作；自然语言、多个坐标和浮点数仍会被拒绝。
_LEGACY_CLICK_ACTION = re.compile(
    r"\A[ \t\r\n]*click\(start_box=\[\s*(?P<x>[0-9]{1,4})\s*,\s*"
    r"(?P<y>[0-9]{1,4})\s*\]\)[ \t\r\n]*\Z"
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _logical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_sha256(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return _logical_sha256(payload)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"training row {line_number} is not an object")
            records.append(value)
    return records


def _prompt_route(record: dict[str, Any]) -> str:
    source = str(record.get("source") or "")
    return "grounding" if source.startswith("showui") else "short_answer"


def canonical_training_target(
    record: dict[str, Any],
) -> tuple[str, str]:
    """把原始答案确定性转换为实际送入 teacher forcing 的目标。"""

    answers = record.get("answers")
    if not isinstance(answers, list):
        raise ValueError("training record answers must be a list")
    raw_answers = [str(answer) for answer in answers if str(answer).strip()]
    if not raw_answers:
        raise ValueError("training record has no usable answer")

    if _prompt_route(record) == "grounding":
        raw_target = raw_answers[0].strip()
        point = parse_click_action(raw_target)
        if point is None:
            match = _LEGACY_CLICK_ACTION.fullmatch(raw_target)
            if match is None:
                raise ValueError("grounding answer violates the strict click action grammar")
            point = (int(match.group("x")), int(match.group("y")))
        target = format_click_action(point)
        transform = (
            "identity"
            if raw_target == target
            else "legacy_click_spacing_to_canonical"
        )
        return target, transform

    if len(raw_answers) == 1:
        target = raw_answers[0].strip()
        return target, "single_answer_passthrough"

    normalized = [normalize_answer(answer) for answer in raw_answers]
    counts = Counter(normalized)
    first_index = {answer: normalized.index(answer) for answer in counts}
    target = max(counts, key=lambda answer: (counts[answer], -first_index[answer]))
    if not target:
        # TextVQA 含 ``("` 这类语义完全由标点承载的答案；VQA 评分规范化会
        # 把它清空。训练监督回退到原始字符串多数票，避免制造空 target。
        raw_stripped = [answer.strip() for answer in raw_answers]
        raw_counts = Counter(raw_stripped)
        raw_first_index = {answer: raw_stripped.index(answer) for answer in raw_counts}
        target = max(
            raw_counts,
            key=lambda answer: (raw_counts[answer], -raw_first_index[answer]),
        )
        return target, "vqa_raw_majority_empty_normalization_fallback"
    return target, "vqa_normalized_majority"


def build_training_order_manifest(
    *,
    data_path: str | Path,
    contract: dict[str, Any],
    contract_sha256: str,
    examples_seen: int,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """取冻结训练包前 N 行，并绑定每一行与原始图像的 SHA-256。"""

    data_path = Path(data_path).resolve()
    training_pack = contract["datasets"]["training_pack"]
    budget = contract["training_budget"]
    if training_pack.get("order_is_frozen") is not True:
        raise ValueError("training pack does not attest a frozen source order")
    if examples_seen not in [int(value) for value in budget["examples_seen_checkpoints"]]:
        raise ValueError("examples_seen must be a preregistered examples-seen checkpoint")

    actual_data_sha256 = _file_sha256(data_path)
    if actual_data_sha256 != str(training_pack["sha256"]):
        raise ValueError("training pack SHA-256 differs from the frozen contract")
    records = _load_jsonl(data_path)
    expected_total = int(training_pack["records"])
    if len(records) != expected_total:
        raise ValueError(f"training pack row count differs: {len(records)} != {expected_total}")
    if examples_seen > len(records):
        raise ValueError("examples_seen exceeds the frozen training pack")

    micro_batch = int(budget["micro_batch_size"])
    accumulation = int(budget["gradient_accumulation"])
    global_batch = int(budget["real_global_batch"])
    if micro_batch * accumulation != global_batch:
        raise ValueError("training budget batch fields are inconsistent")
    if examples_seen % global_batch:
        raise ValueError("examples_seen must be divisible by the real global batch")
    optimizer_steps = examples_seen // global_batch
    checkpoint_index = [int(value) for value in budget["examples_seen_checkpoints"]].index(
        examples_seen
    )
    if optimizer_steps != int(budget["optimizer_steps_checkpoints"][checkpoint_index]):
        raise ValueError("examples-seen and optimizer-step checkpoints disagree")

    selected = records[:examples_seen]
    seen_ids: set[str] = set()
    image_root = data_path.parent.resolve()
    manifest_records: list[dict[str, Any]] = []
    for index, record in enumerate(selected):
        record_id = str(record.get("id") or "")
        if not record_id or record_id in seen_ids:
            raise ValueError(f"training record id is empty or duplicated: {record_id!r}")
        seen_ids.add(record_id)
        answers = record.get("answers")
        try:
            target_answer, target_transform = canonical_training_target(record)
        except ValueError as error:
            raise ValueError(f"invalid training target for {record_id}: {error}") from error
        relative_image = Path(str(record.get("image") or ""))
        image_path = (image_root / relative_image).resolve()
        try:
            image_path.relative_to(image_root)
        except ValueError as error:
            raise ValueError(f"training image escapes its data root: {record_id}") from error
        if not image_path.is_file():
            raise FileNotFoundError(f"training image is missing: {image_path}")
        encoded_image = image_path.read_bytes()
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        route = _prompt_route(record)
        manifest_records.append(
            {
                "index": index,
                "source_row_index": index,
                "id": record_id,
                "source": str(record.get("source") or "unknown"),
                "image": relative_image.as_posix(),
                "image_bytes": len(encoded_image),
                "image_sha256": hashlib.sha256(encoded_image).hexdigest(),
                "image_width": int(image_width),
                "image_height": int(image_height),
                "prompt_route": route,
                "target_answer": target_answer,
                "target_answer_sha256": hashlib.sha256(
                    target_answer.encode("utf-8")
                ).hexdigest(),
                "target_transform": target_transform,
                "record_sha256": _logical_sha256(record),
                "question_sha256": hashlib.sha256(
                    str(record.get("question") or "").encode("utf-8")
                ).hexdigest(),
                "answers_sha256": _logical_sha256(answers),
            }
        )
        if progress is not None:
            progress(index + 1, examples_seen)

    source_counts = Counter(row["source"] for row in manifest_records)
    route_counts = Counter(row["prompt_route"] for row in manifest_records)
    target_transform_counts = Counter(
        row["target_transform"] for row in manifest_records
    )
    image_hashes = [row["image_sha256"] for row in manifest_records]
    effective_denominator = int(budget.get("effective_epochs_denominator", expected_total))
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": str(contract_sha256),
        "data": {
            "path": str(data_path),
            "bytes": data_path.stat().st_size,
            "sha256": actual_data_sha256,
            "total_records": len(records),
        },
        "selection": {
            "rule": "first_n_rows_preserve_source_order",
            "shuffle": False,
            "holdout_removed": False,
            "examples_seen": examples_seen,
            "optimizer_steps": optimizer_steps,
            "micro_batch_size": micro_batch,
            "gradient_accumulation": accumulation,
            "real_global_batch": global_batch,
            "subset_passes": 1.0,
            "effective_epochs_denominator": effective_denominator,
            "effective_epochs": examples_seen / effective_denominator,
        },
        "feature_cache": {
            "vision_tower": contract["vision_tower"]["name"],
            "moonvit_weights_sha256": contract["vision_tower"][
                "extracted_weights_sha256"
            ],
            "max_image_side": int(
                contract["image_preprocessing"]["train_max_image_side"]
            ),
            "max_visual_tokens": int(
                contract["image_preprocessing"]["train_max_visual_tokens"]
            ),
            "storage_dtype": "float32",
            "content_address_field": "image_sha256",
        },
        "source_counts": dict(sorted(source_counts.items())),
        "prompt_route_counts": dict(sorted(route_counts.items())),
        "target_transform_counts": dict(sorted(target_transform_counts.items())),
        "unique_record_ids": len(seen_ids),
        "unique_image_paths": len({row["image"] for row in manifest_records}),
        "unique_image_sha256": len(set(image_hashes)),
        "records_sha256": _logical_sha256(manifest_records),
        "records": manifest_records,
        "training_results_exist": False,
        "final_half_scored": False,
    }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    return manifest


def verify_training_order_manifest(manifest: dict[str, Any]) -> bool:
    """验证 manifest 自哈希、顺序哈希、计数与 batch 算术。"""

    try:
        if manifest.get("schema_version") != SCHEMA_VERSION:
            return False
        if manifest.get("manifest_sha256") != _manifest_sha256(manifest):
            return False
        records = manifest["records"]
        if manifest.get("records_sha256") != _logical_sha256(records):
            return False
        if [row["index"] for row in records] != list(range(len(records))):
            return False
        if [row["source_row_index"] for row in records] != list(range(len(records))):
            return False
        ids = [str(row["id"]) for row in records]
        if len(ids) != len(set(ids)) or len(records) != int(
            manifest["selection"]["examples_seen"]
        ):
            return False
        source_counts = dict(sorted(Counter(row["source"] for row in records).items()))
        route_counts = dict(
            sorted(Counter(row["prompt_route"] for row in records).items())
        )
        target_transform_counts = dict(
            sorted(Counter(row["target_transform"] for row in records).items())
        )
        if source_counts != manifest["source_counts"]:
            return False
        if route_counts != manifest["prompt_route_counts"]:
            return False
        if target_transform_counts != manifest["target_transform_counts"]:
            return False
        for row in records:
            target = str(row["target_answer"])
            if not target or row["target_answer_sha256"] != hashlib.sha256(
                target.encode("utf-8")
            ).hexdigest():
                return False
            if row["prompt_route"] == "grounding" and parse_click_action(target) is None:
                return False
        selection = manifest["selection"]
        if (
            int(selection["micro_batch_size"])
            * int(selection["gradient_accumulation"])
            != int(selection["real_global_batch"])
        ):
            return False
        if (
            int(selection["optimizer_steps"]) * int(selection["real_global_batch"])
            != int(selection["examples_seen"])
        ):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def load_ordered_records(
    *, data_path: str | Path, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    """从已冻结数据文件按 manifest 顺序重建记录，并验证每行逻辑 SHA。"""

    if not verify_training_order_manifest(manifest):
        raise ValueError("training order manifest is invalid")
    data_path = Path(data_path).resolve()
    if _file_sha256(data_path) != manifest["data"]["sha256"]:
        raise ValueError("training data SHA-256 differs from the order manifest")
    records = _load_jsonl(data_path)
    if len(records) != int(manifest["data"]["total_records"]):
        raise ValueError("training data row count differs from the order manifest")
    selected = []
    for entry in manifest["records"]:
        record = records[int(entry["source_row_index"])]
        if str(record.get("id")) != str(entry["id"]):
            raise ValueError(f"training record id differs at index {entry['index']}")
        if _logical_sha256(record) != str(entry["record_sha256"]):
            raise ValueError(f"training record SHA-256 differs: {entry['id']}")
        selected.append(record)
    return selected
