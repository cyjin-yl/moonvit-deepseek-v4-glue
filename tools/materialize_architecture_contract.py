#!/usr/bin/env python3
"""把一个架构 sidecar 展开成可供 order/cache 工具使用的有效合同。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_qwen3b_proxy import load_architecture_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--architecture-control", type=Path, required=True)
    parser.add_argument("--architecture-arm", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    core = json.loads(args.contract.read_text(encoding="utf-8"))
    effective, metadata = load_architecture_overlay(
        core_contract_path=args.contract,
        core_contract=core,
        architecture_control_path=args.architecture_control,
        architecture_arm=args.architecture_arm,
    )
    if metadata is None:
        raise RuntimeError("architecture overlay did not produce metadata")
    effective["architecture_control"] = metadata
    effective["architecture_control"]["sidecar_path"] = str(
        args.architecture_control.resolve()
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(effective, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "materialized",
                "out": str(args.out),
                "architecture_arm": args.architecture_arm,
                "effective_contract_sha256": metadata["effective_contract_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
