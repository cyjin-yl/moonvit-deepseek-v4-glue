#!/usr/bin/env python3
"""Verify every file, image, pair, split, and control in a synthetic suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moonvit_glue.synthetic_perception import verify_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify_suite(args.data)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        if args.out.exists():
            raise FileExistsError(f"refusing to overwrite verification: {args.out}")
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
