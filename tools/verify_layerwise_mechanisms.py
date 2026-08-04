#!/usr/bin/env python3
"""独立验证 package 4 的表示、probe 与 activation-patching 原始产物。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moonvit_glue.mechanism_verification import (
    verify_activation_patching,
    verify_probe_analysis,
    verify_representations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--patching", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = {
        "status": "valid",
        "representations": verify_representations(args.representations),
        "analysis": verify_probe_analysis(args.analysis, args.representations),
        "patching": verify_activation_patching(args.patching),
        "final_half_scored": False,
    }
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
