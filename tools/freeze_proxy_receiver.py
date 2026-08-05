"""冻结 Qwen 3B 代理使用的无参数 4096→2048 receiver buffers。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from moonvit_glue.proxy_receiver import FixedPairwiseReceiverAdapter


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--canonical-width", type=int, default=4096)
    parser.add_argument("--receiver-width", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.out_dir}")
    adapter = FixedPairwiseReceiverAdapter(
        args.canonical_width,
        args.receiver_width,
        seed=args.seed,
    )
    adapter.save_pretrained(args.out_dir)
    restored = FixedPairwiseReceiverAdapter.from_pretrained(args.out_dir)
    probe = torch.arange(args.canonical_width, dtype=torch.float32)
    if not torch.equal(adapter(probe), restored(probe)):
        raise RuntimeError("receiver save/restore probe differs")

    files = []
    for path in sorted(args.out_dir.iterdir()):
        if path.is_file() and path.name != "MANIFEST.json":
            files.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = {
        "schema_version": "fixed-pairwise-receiver-artifact-v1",
        "canonical_width": args.canonical_width,
        "receiver_width": args.receiver_width,
        "seed": args.seed,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in adapter.parameters()
        ),
        "permutation_is_valid": sorted(adapter.permutation.tolist())
        == list(range(args.canonical_width)),
        "sign_values": sorted(set(adapter.signs.tolist())),
        "files": files,
    }
    (args.out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
