#!/usr/bin/env python3
"""独立验证单条 projector/LoRA adaptation 训练轨迹。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moonvit_glue.adaptation_verification import verify_training_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify_training_run(args.run)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
