"""Extract the MoonViT-V2 vision tower from a Kimi-K3 checkpoint shard.

Reads only the shard file(s) containing ``vision_tower.*`` keys (for K3 this
is ``model-00096-of-000096.safetensors``, ~800 MB out of a ~1.56 TB repo),
strict-loads them into the vendored MoonViT3d tower to prove completeness,
and writes a self-contained artifact directory:

    moonvit_v2.safetensors   bare (unprefixed) vision-tower weights
    vision_config.json       KimiK3VisionConfig parameters
    preprocessor_config.json K3 image-processor config
    code/                    vendored inference code (LICENSE included)

Usage:
    python tools/extract_moonvit_v2.py <shard.safetensors> <out_dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from moonvit_glue.moonvit_v2 import (
    load_moonvit_v2_encoder,
    load_vision_tower_state_dict,
)
from moonvit_glue.vendor.kimi_k3.configuration_kimi_k3 import KimiK3VisionConfig

_VENDOR_DIR = Path(__file__).parent.parent / "src" / "moonvit_glue" / "vendor" / "kimi_k3"


def _key_prefix_report(weights_path: Path) -> dict[str, int]:
    prefixes: Counter[str] = Counter()
    with safe_open(str(weights_path), framework="pt") as handle:
        for key in handle.keys():
            prefixes[key.split(".")[0]] += 1
    return dict(prefixes)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path, help="K3 shard with vision_tower.* keys")
    parser.add_argument("out_dir", type=Path, help="artifact output directory")
    args = parser.parse_args()

    print(f"Key prefixes in {args.shard.name}: {_key_prefix_report(args.shard)}")

    state = load_vision_tower_state_dict(args.shard)
    total = sum(t.numel() for t in state.values())
    print(f"Extracted {len(state)} tensors, {total / 1e6:.1f} M parameters")

    # Strict-load into the real tower before writing anything: the artifact
    # is only useful if it reconstructs the full vision tower.
    encoder = load_moonvit_v2_encoder(args.shard, attn_implementation="eager")
    del encoder
    print("Strict load into MoonViT3d (1024-wide, 27-layer): OK")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Sort keys for a deterministic byte layout so the artifact sha256 is
    # reproducible across machines.
    state = {key: state[key] for key in sorted(state)}
    weights_path = args.out_dir / "moonvit_v2.safetensors"
    save_file(
        state,
        str(weights_path),
        metadata={"format": "pt", "source": "moonshotai/Kimi-K3 vision_tower"},
    )
    vision_config = KimiK3VisionConfig()
    (args.out_dir / "vision_config.json").write_text(
        json.dumps(vision_config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy(
        _VENDOR_DIR / "preprocessor_config.json",
        args.out_dir / "preprocessor_config.json",
    )
    code_dir = args.out_dir / "code"
    if code_dir.exists():
        shutil.rmtree(code_dir)
    shutil.copytree(_VENDOR_DIR, code_dir, ignore=shutil.ignore_patterns("__pycache__"))

    manifest = {
        "weights_file": weights_path.name,
        "weights_sha256": _sha256(weights_path),
        "tensor_count": len(state),
        "parameter_count": total,
        "source_repo": "moonshotai/Kimi-K3",
        "source_shard": args.shard.name,
        "source_shard_sha256": _sha256(args.shard),
        "vision_width": 1024,
        "merge_factor": 4,
    }
    (args.out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Artifact written to {args.out_dir}")
    print(f"MANIFEST: weights sha256 {manifest['weights_sha256'][:16]}..., "
          f"shard sha256 {manifest['source_shard_sha256'][:16]}...")


if __name__ == "__main__":
    main()
