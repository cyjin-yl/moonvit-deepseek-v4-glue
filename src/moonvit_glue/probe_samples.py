"""冻结 receiver 诊断样本与特征缓存的显式绑定。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def receiver_probe_supervision(sample: Mapping[str, Any]) -> tuple[str, str]:
    """返回真实问题和固定目标答案；缺字段时硬失败，禁止伪监督回退。"""
    question = sample.get("question") or sample.get("instruction")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"receiver probe sample {sample.get('id')!r} has no question/instruction")
    answer = sample.get("target_answer")
    if answer is None:
        answers = sample.get("answers")
        if not isinstance(answers, Sequence) or isinstance(answers, (str, bytes)) or not answers:
            raise ValueError(f"receiver probe sample {sample.get('id')!r} has no target_answer/answers")
        answer = answers[0]
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"receiver probe sample {sample.get('id')!r} has an empty target answer")
    return question.strip(), answer.strip()


def load_receiver_probe_records(
    manifest_path: Path,
    cache_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """按 ID 和图片 SHA 将冻结样本 manifest 绑定到 feature cache。"""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("receiver probe manifest must contain non-empty records")
    cache_by_id = {str(record.get("id")): record for record in cache_records}
    if len(cache_by_id) != len(cache_records):
        raise ValueError("feature cache contains duplicate sample ids")
    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    bound: list[dict[str, Any]] = []
    for raw in records:
        sample = dict(raw)
        sample_id = str(sample.get("id") or "")
        image_sha = str(sample.get("image_sha256") or "")
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"missing or duplicate receiver probe sample id: {sample_id!r}")
        if not image_sha or image_sha in seen_images:
            raise ValueError(f"missing or duplicate receiver probe image SHA for {sample_id!r}")
        cached = cache_by_id.get(sample_id)
        if cached is None:
            raise ValueError(f"receiver probe sample is absent from feature cache: {sample_id}")
        if str(cached.get("image_sha256") or "") != image_sha:
            raise ValueError(f"receiver probe image SHA mismatch for {sample_id}")
        receiver_probe_supervision(sample)
        seen_ids.add(sample_id)
        seen_images.add(image_sha)
        bound.append(sample)
    return manifest, bound
