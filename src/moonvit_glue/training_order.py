"""冻结并验证训练记录顺序、监督来源与原图身份。"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image

from .grounding_contract import format_click_action, parse_click_action
from .metrics import normalize_answer


SCHEMA_VERSION = "qwen3b-training-order-v1"
PREFIX_SELECTION_RULE = "first_n_rows_preserve_source_order"
GROUNDING_ENRICHED_SELECTION_RULE = (
    "first_n_per_route_alternate_grounding_then_short_answer"
)

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


def grounding_enriched_source_indices(
    records: Sequence[dict[str, Any]],
    *,
    grounding_examples: int,
    short_answer_examples: int,
) -> list[int]:
    """按冻结源顺序取两类首批记录，再以 grounding-first 交替合并。"""

    if grounding_examples <= 0 or short_answer_examples <= 0:
        raise ValueError("grounding-enriched route counts must be positive")
    by_route = {"grounding": [], "short_answer": []}
    for source_index, record in enumerate(records):
        by_route[_prompt_route(record)].append(source_index)
    requested = {
        "grounding": int(grounding_examples),
        "short_answer": int(short_answer_examples),
    }
    for route, count in requested.items():
        if count > len(by_route[route]):
            raise ValueError(
                f"grounding-enriched selection exceeds {route} records: "
                f"{count} > {len(by_route[route])}"
            )
    selected: list[int] = []
    for offset in range(max(requested.values())):
        if offset < requested["grounding"]:
            selected.append(by_route["grounding"][offset])
        if offset < requested["short_answer"]:
            selected.append(by_route["short_answer"][offset])
    return selected


def _validated_selection_metadata(
    *,
    records: Sequence[dict[str, Any]],
    selected_indices: Sequence[int],
    selection_rule: str,
    selection_metadata: dict[str, Any] | None,
    reserved_fields: set[str],
) -> dict[str, Any]:
    """在读取图片前验证预注册选样规则与其元数据。"""

    metadata = dict(selection_metadata or {})
    reserved = reserved_fields & set(metadata)
    if reserved:
        raise ValueError(f"selection metadata overrides reserved fields: {sorted(reserved)}")

    if selection_rule == PREFIX_SELECTION_RULE:
        if list(selected_indices) != list(range(len(selected_indices))):
            raise ValueError("prefix selection requires the exact leading source rows")
        return metadata

    if selection_rule != GROUNDING_ENRICHED_SELECTION_RULE:
        raise ValueError(f"unknown training selection rule: {selection_rule}")
    required_metadata = {
        "grounding_examples",
        "short_answer_examples",
        "within_route_order",
        "merge_rule",
    }
    missing = required_metadata - set(metadata)
    if missing:
        raise ValueError(
            f"grounding-enriched selection metadata is missing: {sorted(missing)}"
        )
    if metadata["within_route_order"] != "frozen_source_order":
        raise ValueError("grounding-enriched selection must preserve route source order")
    if metadata["merge_rule"] != "alternate_grounding_then_short_answer":
        raise ValueError("grounding-enriched selection must use the registered merge rule")
    expected_indices = grounding_enriched_source_indices(
        records,
        grounding_examples=int(metadata["grounding_examples"]),
        short_answer_examples=int(metadata["short_answer_examples"]),
    )
    if list(selected_indices) != expected_indices:
        raise ValueError(
            "grounding-enriched source indices differ from the registered selection"
        )
    return metadata


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
    source_indices: Sequence[int] | None = None,
    selection_rule: str = PREFIX_SELECTION_RULE,
    selection_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """按预注册索引取记录，并绑定每一行与原始图像的 SHA-256。"""

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

    if source_indices is None:
        if selection_rule != PREFIX_SELECTION_RULE:
            raise ValueError("non-prefix selection requires explicit source indices")
        selected_indices = list(range(examples_seen))
    else:
        selected_indices = [int(value) for value in source_indices]
    if len(selected_indices) != examples_seen:
        raise ValueError("source index count differs from examples_seen")
    if len(selected_indices) != len(set(selected_indices)):
        raise ValueError("training selection contains duplicate source indices")
    if any(index < 0 or index >= len(records) for index in selected_indices):
        raise ValueError("training selection source index is out of range")

    reserved_selection_fields = {
        "rule",
        "shuffle",
        "holdout_removed",
        "examples_seen",
        "optimizer_steps",
        "micro_batch_size",
        "gradient_accumulation",
        "real_global_batch",
        "subset_passes",
        "effective_epochs_denominator",
        "effective_epochs",
    }
    metadata = _validated_selection_metadata(
        records=records,
        selected_indices=selected_indices,
        selection_rule=selection_rule,
        selection_metadata=selection_metadata,
        reserved_fields=reserved_selection_fields,
    )

    selected = [(source_index, records[source_index]) for source_index in selected_indices]
    seen_ids: set[str] = set()
    image_root = data_path.parent.resolve()
    manifest_records: list[dict[str, Any]] = []
    for index, (source_index, record) in enumerate(selected):
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
                "source_row_index": source_index,
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
    selection: dict[str, Any] = {
        "rule": selection_rule,
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
    }
    selection.update(metadata)

    cache_binding = contract.get("feature_cache_binding", {})
    if not isinstance(cache_binding, dict):
        raise ValueError("feature_cache_binding must be an object when provided")
    feature_cache = {
        "vision_tower": contract["vision_tower"].get("name"),
        "moonvit_weights_sha256": contract["vision_tower"].get(
            "extracted_weights_sha256"
        ),
        "max_image_side": int(contract["image_preprocessing"]["train_max_image_side"]),
        "max_visual_tokens": int(contract["image_preprocessing"]["train_max_visual_tokens"]),
        "storage_dtype": "float32",
        "content_address_field": "image_sha256",
    }
    for key in ("moonvit_model", "moonvit_revision", "vision_width", "merge_factor"):
        if key in cache_binding and cache_binding[key] is not None:
            feature_cache[key] = cache_binding[key]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": str(contract_sha256),
        "data": {
            "path": str(data_path),
            "bytes": data_path.stat().st_size,
            "sha256": actual_data_sha256,
            "total_records": len(records),
        },
        "selection": selection,
        "feature_cache": feature_cache,
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
        selection = manifest["selection"]
        if selection.get("shuffle") is not False:
            return False
        if selection.get("holdout_removed") is not False:
            return False
        source_indices = [int(row["source_row_index"]) for row in records]
        if len(source_indices) != len(set(source_indices)):
            return False
        if any(
            index < 0 or index >= int(manifest["data"]["total_records"])
            for index in source_indices
        ):
            return False
        rule = str(selection["rule"])
        if rule == PREFIX_SELECTION_RULE:
            if source_indices != list(range(len(records))):
                return False
        elif rule == GROUNDING_ENRICHED_SELECTION_RULE:
            if (
                selection.get("within_route_order") != "frozen_source_order"
                or selection.get("merge_rule")
                != "alternate_grounding_then_short_answer"
            ):
                return False
            grounding = int(selection["grounding_examples"])
            short_answer = int(selection["short_answer_examples"])
            if grounding + short_answer != len(records):
                return False
            expected_routes = []
            for offset in range(max(grounding, short_answer)):
                if offset < grounding:
                    expected_routes.append("grounding")
                if offset < short_answer:
                    expected_routes.append("short_answer")
            actual_routes = [str(row["prompt_route"]) for row in records]
            if actual_routes != expected_routes:
                return False
            for route in ("grounding", "short_answer"):
                route_indices = [
                    source_index
                    for source_index, actual_route in zip(source_indices, actual_routes)
                    if actual_route == route
                ]
                if route_indices != sorted(route_indices):
                    return False
        else:
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
