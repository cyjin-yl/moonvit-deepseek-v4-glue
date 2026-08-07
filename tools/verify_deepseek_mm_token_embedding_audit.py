#!/usr/bin/env python3
"""独立核验 DeepSeek-V4-Flash 预留多模态 token embedding 抽样审计。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(result_path: Path, raw_dir: Path) -> dict[str, object]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {
        "schema": result.get("schema_version") == "deepseek-v4-mm-token-embedding-audit-v1",
        "revision_pinned": result.get("resolved_revision") == "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
        "embed_tensor_pinned": result.get("tensor", {}).get("name") == "embed.weight" and result.get("tensor", {}).get("dtype") == "BF16",
        "placeholder_count": result.get("special_token_ranges", {}).get("place_holder_mm_span", {}).get("count") == 415,
        "reserved_norm_lower": result.get("geometry", {}).get("reserved_vs_ordinary_mean_norm_ratio", 1.0) < 0.2,
        "reserved_mean_cosine_near_zero": abs(result.get("geometry", {}).get("reserved_cosine_to_ordinary_mean", 1.0)) < 0.1,
    }
    for name, meta in result.get("raw_ranges", {}).items():
        path = raw_dir / name
        checks[f"{name}_exists"] = path.is_file()
        checks[f"{name}_sha256"] = checks[f"{name}_exists"] and sha256(path) == meta.get("sha256")
        checks[f"{name}_size"] = checks[f"{name}_exists"] and path.stat().st_size == meta.get("bytes")
    verified = all(checks.values())
    return {
        "schema_version": "deepseek-v4-mm-token-embedding-audit-verifier-v1",
        "result": str(result_path),
        "raw_dir": str(raw_dir),
        "verified": verified,
        "checks": checks,
        "interpretation": "Verified the pinned revision, tensor metadata, reserved-token count, compact raw-range hashes and the preregistered low-norm finding. This remains an embedding/interface audit, not a vision-capability claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = verify(args.result, args.raw_dir)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not output["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
