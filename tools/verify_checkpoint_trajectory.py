#!/usr/bin/env python3
"""校验原始精确分母、文件哈希与轨迹输出有限性。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moonvit_glue.trajectory_verification import verify_trajectory_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify_trajectory_run(args.run)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        if args.out.exists():
            raise FileExistsError(f"refusing to overwrite verification: {args.out}")
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
