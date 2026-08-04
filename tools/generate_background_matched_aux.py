#!/usr/bin/env python3
"""生成仅用于诊断、采用 train 背景色的 selection 辅助集。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moonvit_glue.synthetic_perception import (
    SuiteConfig,
    generate_background_matched_aux,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    synthetic = payload.get("synthetic", payload)
    config = SuiteConfig(
        samples_per_task=int(synthetic["samples_per_task"]),
        image_size=int(synthetic["image_size"]),
        seed=int(synthetic["seed"]),
        background_train=str(synthetic["background_train"]),
        background_selection=str(synthetic["background_selection"]),
    )
    manifest = generate_background_matched_aux(args.selection, args.out, config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
