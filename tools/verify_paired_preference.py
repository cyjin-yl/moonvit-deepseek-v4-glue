#!/usr/bin/env python3
"""校验已完成的 paired-preference 轨迹 run。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moonvit_glue.preference_verification import verify_preference_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_preference_run(args.run), indent=2))


if __name__ == "__main__":
    main()
