#!/usr/bin/env python3
"""在任何 4k 训练结果产生前冻结记录顺序、监督路由与原图身份。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from moonvit_glue.training_order import build_training_order_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--examples-seen", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite training order: {args.out}")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))

    def progress(done: int, total: int) -> None:
        if done == 1 or done == total or done % 100 == 0:
            print(f"[{done}/{total}] froze record and image identity", flush=True)

    manifest = build_training_order_manifest(
        data_path=args.data,
        contract=contract,
        contract_sha256=_sha256(args.contract),
        examples_seen=args.examples_seen,
        progress=progress,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    print(
        json.dumps(
            {
                "status": "frozen",
                "out": str(args.out),
                "manifest_sha256": manifest["manifest_sha256"],
                "records_sha256": manifest["records_sha256"],
                "examples_seen": manifest["selection"]["examples_seen"],
                "optimizer_steps": manifest["selection"]["optimizer_steps"],
                "source_counts": manifest["source_counts"],
                "prompt_route_counts": manifest["prompt_route_counts"],
                "target_transform_counts": manifest["target_transform_counts"],
                "unique_image_sha256": manifest["unique_image_sha256"],
                "training_results_exist": False,
                "final_half_scored": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
