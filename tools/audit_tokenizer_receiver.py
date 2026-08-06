"""Audit tokenizer/config image-token seams without downloading model weights.

This is deliberately a metadata-first audit.  It reads ``config.json`` and the
tokenizer files from a local snapshot, then optionally inspects an embedding
tensor when a local safetensors file is supplied and the optional dependency is
installed.  It never calls ``from_pretrained`` and never downloads weights.

Examples::

    python tools/audit_tokenizer_receiver.py --model-dir /models/Qwen2.5-3B-Instruct
    python tools/audit_tokenizer_receiver.py --model-dir /models/Qwen3.5-4B --out audit.json

The report distinguishes a tokenizer's reserved image IDs from a trained
multimodal receiver.  A pure-text model may contain visual-looking tokens in
its tokenizer while having no vision module at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def token_entries(tokenizer_json: dict[str, Any] | None, tokenizer_config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return a normalized list from either tokenizer JSON representation."""

    entries: list[dict[str, Any]] = []
    if tokenizer_json:
        raw = tokenizer_json.get("added_tokens", [])
        if isinstance(raw, list):
            entries.extend(item for item in raw if isinstance(item, dict) and "id" in item)
    if not entries and tokenizer_config:
        raw = tokenizer_config.get("added_tokens_decoder", {})
        if isinstance(raw, dict):
            for token_id, item in raw.items():
                if isinstance(item, dict):
                    entries.append({"id": int(token_id), **item})
    # A tokenizer JSON and tokenizer_config can both contain the same entries.
    unique: dict[int, dict[str, Any]] = {}
    for item in entries:
        unique[int(item["id"])] = {
            "id": int(item["id"]),
            "content": item.get("content"),
            "special": bool(item.get("special", False)),
        }
    return [unique[key] for key in sorted(unique)]


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def identify_image_tokens(config: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Collect image/vision IDs from config fields and tokenizer entries."""

    ids: dict[str, int] = {}
    for key in (
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
        "vision_pad_token_id",
        "image_pad_token_id",
    ):
        value = config.get(key)
        if isinstance(value, int):
            ids[key] = value
    for key, value in (config.get("text_config") or {}).items():
        if key.endswith("token_id") and isinstance(value, int) and _contains_any(key, ("image", "vision", "video")):
            ids.setdefault(key, value)

    matching: list[dict[str, Any]] = []
    for item in entries:
        content = str(item.get("content") or "")
        if _contains_any(content, ("image", "vision", "video", "mm_span", "place_holder")):
            matching.append(item)
            if content in ("<|image_pad|>", "<｜image｜>", "<｜image2｜>"):
                ids.setdefault("image_token_id_from_tokenizer", int(item["id"]))
            elif "vision_start" in content:
                ids.setdefault("vision_start_token_id_from_tokenizer", int(item["id"]))
            elif "vision_end" in content:
                ids.setdefault("vision_end_token_id_from_tokenizer", int(item["id"]))
    return {"ids": ids, "matching_entries": matching}


def infer_family(config: dict[str, Any]) -> str:
    model_type = str(config.get("model_type", "")).lower()
    architectures = " ".join(map(str, config.get("architectures", []))).lower()
    if "qwen3_5" in model_type or "qwen3_5" in architectures:
        return "qwen3.5"
    if model_type == "qwen2" or "qwen2forcausallm" in architectures:
        return "qwen2"
    if "deepseek_v4" in model_type or "deepseekv4" in architectures:
        return "deepseek-v4"
    return "unknown"


def receiver_assessment(config: dict[str, Any], family: str) -> dict[str, Any]:
    has_vision = isinstance(config.get("vision_config"), dict)
    if family == "qwen2":
        return {
            "pure_text_config": not has_vision,
            "receiver_candidate": not has_vision,
            "receiver_path": "Qwen2ForCausalLM(inputs_embeds)",
            "native_vision_module": has_vision,
            "notes": "Tokenizer may still contain visual-looking additions; config has no vision path.",
        }
    if family == "qwen3.5":
        text = config.get("text_config") or {}
        return {
            "pure_text_config": False,
            "receiver_candidate": True,
            "receiver_path": "Qwen3_5TextModel(inputs_embeds) via model.language_model",
            "native_vision_module": has_vision,
            "receiver_hidden_size": text.get("hidden_size"),
            "notes": "The language_model submodule is a valid text receiver, but the checkpoint is an aligned native VLM.",
        }
    if family == "deepseek-v4":
        return {
            "pure_text_config": True,
            "receiver_candidate": True,
            "receiver_path": "DeepseekV4ForCausalLM(inputs_embeds + legal input_ids)",
            "native_vision_module": has_vision,
            "requires_input_ids_for_routing": True,
            "notes": "Keep placeholder IDs in input_ids for hash-MoE routing even when image positions use inputs_embeds.",
        }
    return {
        "pure_text_config": not has_vision,
        "receiver_candidate": not has_vision,
        "receiver_path": "unknown",
        "native_vision_module": has_vision,
    }


def _bf16_stats(raw: bytes) -> dict[str, float]:
    values = [struct.unpack("<f", struct.pack("<I", struct.unpack("<H", raw[i : i + 2])[0] << 16))[0] for i in range(0, len(raw), 2)]
    if not values:
        return {}
    mean = sum(values) / len(values)
    return {
        "norm": math.sqrt(sum(x * x for x in values)),
        "mean": mean,
        "std": math.sqrt(sum((x - mean) ** 2 for x in values) / max(1, len(values) - 1)),
        "min": min(values),
        "max": max(values),
    }


def inspect_local_embedding(model_dir: Path, token_ids: list[int]) -> dict[str, Any]:
    """Inspect embedding metadata/rows when local safetensors is available.

    ``safe_open.get_slice`` is used when present, so requesting a few rows does
    not require materializing the complete embedding matrix.  The report marks
    this section unavailable when no local weight or optional dependency exists.
    """

    index = read_json(model_dir / "model.safetensors.index.json")
    weight_file: Path | None = None
    embedding_key: str | None = None
    if index:
        weight_map = index.get("weight_map", {})
        candidates = [
            key for key in weight_map if key.endswith("embed_tokens.weight") or key in {"embed.weight", "model.embed_tokens.weight"}
        ]
        if candidates:
            embedding_key = candidates[0]
            weight_file = model_dir / str(weight_map[embedding_key])
    if weight_file is None:
        for candidate in sorted(model_dir.glob("*.safetensors")):
            weight_file = candidate
            break
    if weight_file is None or not weight_file.is_file():
        return {"status": "unavailable", "reason": "no local safetensors embedding shard"}
    try:
        from safetensors import safe_open  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional install
        return {"status": "unavailable", "reason": f"safetensors import failed: {exc}"}
    try:
        with safe_open(str(weight_file), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if embedding_key is None:
                embedding_key = next((key for key in keys if key.endswith("embed_tokens.weight") or key == "embed.weight"), None)
            if embedding_key is None:
                return {"status": "unavailable", "reason": "embedding tensor key not found", "weight_file": str(weight_file)}
            tensor = handle.get_slice(embedding_key) if hasattr(handle, "get_slice") else handle.get_tensor(embedding_key)
            shape = list(tensor.get_shape() if hasattr(tensor, "get_shape") else tensor.shape)
            rows: dict[str, Any] = {}
            for token_id in sorted(set(token_ids)):
                if token_id < 0 or token_id >= shape[0]:
                    continue
                row = tensor[token_id] if hasattr(tensor, "__getitem__") else tensor[token_id : token_id + 1][0]
                if hasattr(row, "detach"):
                    row = row.detach().cpu()
                    # Convert through bytes to keep this helper independent of dtype.
                    if str(row.dtype).endswith("bfloat16"):
                        raw = row.contiguous().view(-1).numpy().tobytes()
                        rows[str(token_id)] = _bf16_stats(raw)
                    else:
                        values = row.float().view(-1).tolist()
                        mean = sum(values) / len(values)
                        rows[str(token_id)] = {
                            "norm": math.sqrt(sum(x * x for x in values)),
                            "mean": mean,
                            "std": math.sqrt(sum((x - mean) ** 2 for x in values) / max(1, len(values) - 1)),
                        }
            return {
                "status": "ok",
                "weight_file": str(weight_file),
                "weight_file_sha256": sha256_file(weight_file),
                "tensor_key": embedding_key,
                "shape": shape,
                "rows": rows,
            }
    except Exception as exc:  # pragma: no cover - defensive for varied safetensors versions
        return {"status": "unavailable", "reason": f"embedding inspection failed: {exc}", "weight_file": str(weight_file)}


def build_report(model_dir: Path, inspect_rows: bool = False) -> dict[str, Any]:
    config = read_json(model_dir / "config.json") or {}
    tokenizer_config = read_json(model_dir / "tokenizer_config.json") or {}
    tokenizer_json = read_json(model_dir / "tokenizer.json")
    entries = token_entries(tokenizer_json, tokenizer_config)
    family = infer_family(config)
    image = identify_image_tokens(config, entries)
    files = {}
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json"):
        path = model_dir / name
        if path.is_file():
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    token_ids = sorted({int(x) for x in image["ids"].values() if isinstance(x, int)})
    report: dict[str, Any] = {
        "schema_version": "tokenizer-receiver-audit-v1",
        "model_dir": str(model_dir.resolve()),
        "family": family,
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "hidden_size": config.get("hidden_size") or (config.get("text_config") or {}).get("hidden_size"),
        "vocab_size": config.get("vocab_size") or (config.get("text_config") or {}).get("vocab_size"),
        "has_vision_config": isinstance(config.get("vision_config"), dict),
        "config": config,
        "files": files,
        "image_tokens": image,
        "tokenizer_entry_count": len(entries),
        "receiver_assessment": receiver_assessment(config, family),
    }
    if inspect_rows:
        report["embedding_audit"] = inspect_local_embedding(model_dir, token_ids)
    else:
        report["embedding_audit"] = {"status": "skipped", "reason": "pass --inspect-rows to inspect local weights"}
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Local HF snapshot; no network access is performed")
    parser.add_argument("--out", type=Path, help="Write JSON report here; stdout is always emitted")
    parser.add_argument("--inspect-rows", action="store_true", help="Inspect image-token embedding rows from local safetensors")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_dir.is_dir():
        raise SystemExit(f"model directory does not exist: {args.model_dir}")
    report = build_report(args.model_dir, inspect_rows=args.inspect_rows)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
