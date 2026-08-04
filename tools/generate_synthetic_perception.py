#!/usr/bin/env python3
"""Generate the reproducible six-task synthetic perception diagnostic suite."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from moonvit_glue.synthetic_perception import SuiteConfig, generate_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--config", type=Path, help="Optional JSON config; CLI values override only when explicitly supplied")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        payload = json.loads(args.config.read_text(encoding="utf-8"))
        synthetic = payload.get("synthetic", payload)
        config = SuiteConfig(
            samples_per_task=int(synthetic.get("samples_per_task", args.samples_per_task)),
            image_size=int(synthetic.get("image_size", args.image_size)),
            seed=int(synthetic.get("seed", args.seed)),
            background_train=str(synthetic.get("background_train", "#edf3f8")),
            background_selection=str(synthetic.get("background_selection", "#fff5e6")),
        )
    else:
        config = SuiteConfig(
            samples_per_task=args.samples_per_task,
            image_size=args.image_size,
            seed=args.seed,
        )
    manifest = generate_suite(args.out, config)
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        git_sha = ""
    run = {
        "status": "valid",
        "output": str(args.out.resolve()),
        "git_sha_at_generation": git_sha or None,
        "logical_dataset_sha256": manifest["logical_dataset_sha256"],
        "counts": manifest["counts"],
        "leakage_checks": manifest["leakage_checks"],
    }
    (args.out / "RUN.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
