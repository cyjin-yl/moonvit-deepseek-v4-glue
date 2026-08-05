"""验证 Qwen2.5-3B 固定 revision 的文件、tokenizer 与纯文本模型合同。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_bundle_sha256(files: list[dict[str, Any]]) -> str:
    tokenizer_names = {"merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"}
    payload = [
        {
            "path": item["path"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
        for item in files
        if item["path"] in tokenizer_names
    ]
    payload.sort(key=lambda item: item["path"])
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    expected = contract["proxy_model"]
    actual_files = []
    for item in expected["files"]:
        path = args.model_dir / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = {
            "path": item["path"],
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if actual["bytes"] != item["bytes"] or actual["sha256"] != item["sha256"]:
            raise ValueError(f"model file contract mismatch: {actual}")
        actual_files.append(actual)

    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["Qwen2ForCausalLM"]:
        raise ValueError(f"unexpected architectures: {config.get('architectures')}")
    if config.get("model_type") != "qwen2" or "vision_config" in config:
        raise ValueError("proxy must be a qwen2 pure-text config without vision_config")
    if config.get("hidden_size") != 2048:
        raise ValueError(f"unexpected hidden_size: {config.get('hidden_size')}")

    tokenizer_config = json.loads(
        (args.model_dir / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    chat_template_sha256 = hashlib.sha256(
        tokenizer_config["chat_template"].encode("utf-8")
    ).hexdigest()
    bundle_sha256 = tokenizer_bundle_sha256(actual_files)
    if chat_template_sha256 != expected["chat_template_sha256"]:
        raise ValueError("chat template SHA-256 mismatch")
    if bundle_sha256 != expected["tokenizer_bundle_sha256"]:
        raise ValueError("tokenizer bundle SHA-256 mismatch")

    from transformers import AutoConfig, AutoTokenizer

    loaded_config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False
    )
    if loaded_config.__class__.__name__ != "Qwen2Config":
        raise ValueError(f"unexpected AutoConfig class: {loaded_config.__class__.__name__}")
    if tokenizer.convert_tokens_to_ids("<|image_pad|>") != 151655:
        raise ValueError("pinned <|image_pad|> token ID is not 151655")
    if tokenizer.convert_tokens_to_ids("<|im_end|>") != 151645:
        raise ValueError("pinned <|im_end|> token ID is not 151645")
    original_vocab_size = len(tokenizer)
    tokenizer.add_tokens([])
    if len(tokenizer) != original_vocab_size:
        raise ValueError("tokenizer vocabulary changed during verification")

    index = json.loads(
        (args.model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    report = {
        "schema_version": "qwen25-3b-model-contract-verification-v1",
        "status": "verified",
        "repo": expected["repo"],
        "resolved_revision": expected["resolved_revision"],
        "model_dir": str(args.model_dir.resolve()),
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "has_vision_config": "vision_config" in config,
        "hidden_size": config["hidden_size"],
        "parameter_count_bf16": expected["parameter_count_bf16"],
        "files": actual_files,
        "tokenizer_bundle_sha256": bundle_sha256,
        "chat_template_sha256": chat_template_sha256,
        "tokenizer_length": len(tokenizer),
        "image_placeholder_token_id": tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        "eos_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "weight_map_tensor_count": len(index["weight_map"]),
        "weight_index_total_size": index.get("metadata", {}).get("total_size"),
        "contract_config_sha256": sha256_file(args.contract),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
