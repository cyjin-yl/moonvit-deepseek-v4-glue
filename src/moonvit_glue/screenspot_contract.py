"""ScreenSpot 公共测试集的冻结、分层与因果控制工具。"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

SCALE_MAX = 999.0

_PLATFORM_ALIASES = {
    "android": "Android",
    "ios": "iOS",
    "windows": "Windows",
    "macos": "macOS",
    "mac": "macOS",
    "osx": "macOS",
    "web": "Web",
    "forum": "Web",
    "gitlab": "Web",
    "shop": "Web",
    "tool": "Web",
}
_TYPE_ALIASES = {
    "text": "text",
    "icon": "icon/widget",
    "widget": "icon/widget",
    "icon-widget": "icon/widget",
    "icon/widget": "icon/widget",
}


def canonical_platform(value: str) -> str:
    key = str(value).strip().lower()
    try:
        return _PLATFORM_ALIASES[key]
    except KeyError as error:
        raise ValueError(f"unknown ScreenSpot platform: {value!r}") from error


def canonical_target_type(value: str) -> str:
    key = str(value).strip().lower()
    try:
        return _TYPE_ALIASES[key]
    except KeyError as error:
        raise ValueError(f"unknown ScreenSpot target type: {value!r}") from error


def normalize_screenspot_bbox(
    raw_bbox: Sequence[float],
    *,
    width: int,
    height: int,
    source_format: str = "fractional_xyxy",
) -> list[float]:
    """把明确声明格式的 ScreenSpot bbox 转成 999-scale xyxy。

    固定的 ``bevaya/ScreenSpot`` revision 使用 ``fractional_xyxy``；SeeClick
    最初发布的 JSON 使用 ``pixel_xywh``。调用者必须显式保留来源格式。
    """

    if len(raw_bbox) != 4:
        raise ValueError("ScreenSpot bbox must contain four values")
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    values = [float(value) for value in raw_bbox]
    if source_format == "fractional_xyxy":
        x1, y1, x2, y2 = values
        normalized = [value * SCALE_MAX for value in (x1, y1, x2, y2)]
    elif source_format == "pixel_xyxy":
        x1, y1, x2, y2 = values
        normalized = [
            x1 / width * SCALE_MAX,
            y1 / height * SCALE_MAX,
            x2 / width * SCALE_MAX,
            y2 / height * SCALE_MAX,
        ]
    elif source_format == "pixel_xywh":
        x, y, box_width, box_height = values
        if box_width < 0 or box_height < 0:
            raise ValueError("ScreenSpot bbox width and height must be non-negative")
        normalized = [
            x / width * SCALE_MAX,
            y / height * SCALE_MAX,
            (x + box_width) / width * SCALE_MAX,
            (y + box_height) / height * SCALE_MAX,
        ]
    else:
        raise ValueError(f"unknown ScreenSpot bbox source format: {source_format!r}")
    if normalized[0] > normalized[2] or normalized[1] > normalized[3]:
        raise ValueError("ScreenSpot xyxy bbox must be ordered")
    return [min(max(value, 0.0), SCALE_MAX) for value in normalized]


def _stable_rank(seed: str, namespace: str, value: str) -> str:
    payload = f"{seed}\0{namespace}\0{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_stratified_subset(
    records: Iterable[Mapping[str, Any]], *, size: int, seed: str
) -> list[dict[str, Any]]:
    """按 ``platform × target_type`` 尽量均衡选样，且与输入顺序无关。"""

    rows = [dict(record) for record in records]
    if size < 1 or size > len(rows):
        raise ValueError("subset size must be in [1, len(records)]")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("sample IDs must be unique")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["platform"]), str(row["target_type"]))].append(row)
    strata = sorted(groups)
    if not strata:
        raise ValueError("at least one stratum is required")

    target_counts = {stratum: size // len(strata) for stratum in strata}
    remainder_order = sorted(
        strata,
        key=lambda stratum: _stable_rank(seed, "stratum-remainder", repr(stratum)),
    )
    for stratum in remainder_order[: size % len(strata)]:
        target_counts[stratum] += 1

    # 小 strata 不够时，把缺额稳定地分配给仍有容量的 strata。
    deficit = 0
    for stratum in strata:
        available = len(groups[stratum])
        if target_counts[stratum] > available:
            deficit += target_counts[stratum] - available
            target_counts[stratum] = available
    while deficit:
        candidates = [
            stratum
            for stratum in remainder_order
            if target_counts[stratum] < len(groups[stratum])
        ]
        if not candidates:
            raise ValueError("unable to allocate requested stratified subset")
        for stratum in candidates:
            if not deficit:
                break
            target_counts[stratum] += 1
            deficit -= 1

    selected: list[dict[str, Any]] = []
    for stratum in strata:
        ordered = sorted(
            groups[stratum],
            key=lambda row: _stable_rank(seed, "within-stratum", str(row["sample_id"])),
        )
        selected.extend(ordered[: target_counts[stratum]])
    return sorted(
        selected,
        key=lambda row: _stable_rank(seed, "evaluation-order", str(row["sample_id"])),
    )


def deterministic_image_derangement(
    records: Iterable[Mapping[str, Any]], *, seed: str
) -> dict[str, str]:
    """生成无 fixed point、也不复用同一图像哈希的确定性错图映射。"""

    rows = [dict(record) for record in records]
    if len(rows) < 2:
        raise ValueError("derangement requires at least two records")
    ordered = sorted(
        rows,
        key=lambda row: _stable_rank(seed, "derangement-order", str(row["sample_id"])),
    )
    if len({row["sample_id"] for row in ordered}) != len(ordered):
        raise ValueError("sample IDs must be unique")

    for offset in range(1, len(ordered)):
        pairs = [
            (row, ordered[(index + offset) % len(ordered)])
            for index, row in enumerate(ordered)
        ]
        if all(
            source["sample_id"] != target["sample_id"]
            and source["image_sha256"] != target["image_sha256"]
            for source, target in pairs
        ):
            return {
                str(source["sample_id"]): str(target["sample_id"])
                for source, target in sorted(
                    pairs, key=lambda pair: str(pair[0]["sample_id"])
                )
            }
    raise ValueError("no image-level derangement exists for these records")


def _canonical_manifest_payload(manifest: Mapping[str, Any]) -> bytes:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def seal_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """写入覆盖全部其余字段的 canonical SHA-256。"""

    sealed = dict(manifest)
    sealed.pop("manifest_sha256", None)
    sealed["manifest_sha256"] = hashlib.sha256(
        _canonical_manifest_payload(sealed)
    ).hexdigest()
    return sealed


def verify_manifest(manifest: Mapping[str, Any]) -> bool:
    expected = manifest.get("manifest_sha256")
    if not isinstance(expected, str):
        return False
    return hashlib.sha256(_canonical_manifest_payload(manifest)).hexdigest() == expected
