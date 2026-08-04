#!/usr/bin/env python3
"""构建并交叉校验 package 3 projector checkpoint manifest。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preference-summary", required=True, type=Path)
    parser.add_argument("--generation-summary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    preference = json.loads(args.preference_summary.read_text(encoding="utf-8"))
    generation = json.loads(args.generation_summary.read_text(encoding="utf-8"))
    output = []
    for checkpoint in config["checkpoints"]:
        checkpoint_id = str(checkpoint["id"])
        preference_state = preference["checkpoints"][checkpoint_id]["state_sha256"]
        generation_state = generation["checkpoints"][checkpoint_id]["state_sha256"]
        if preference_state != generation_state:
            raise ValueError(f"checkpoint state drift between evaluation paths: {checkpoint_id}")
        row = {
            "id": checkpoint_id,
            "kind": checkpoint["kind"],
            "optimizer_steps": int(checkpoint["optimizer_steps"]),
            "examples_seen": int(checkpoint["examples_seen"]),
            "effective_epochs": float(checkpoint["effective_epochs"]),
            "source_path": checkpoint.get("path"),
            "state_sha256": preference_state,
            "initialization_claim": checkpoint.get("initialization_claim"),
            "random_seed": checkpoint.get("random_seed"),
            "files": {},
        }
        if checkpoint["kind"] == "trained":
            source = Path(checkpoint["path"])
            for name in (
                "projector.safetensors",
                "projector_bf16.safetensors",
                "projector_config.json",
                "history.json",
                "training_state.pt",
            ):
                path = source / name
                if path.exists():
                    row["files"][name] = {
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
            if row["files"]["projector.safetensors"]["sha256"] != preference_state:
                raise ValueError(f"projector file hash disagrees with run: {checkpoint_id}")
        output.append(row)

    aliases = []
    by_id = {row["id"]: row for row in output}
    for alias in config.get("aliases", []):
        source = str(alias["source"])
        alias_id = str(alias["id"])
        if preference["checkpoints"][alias_id].get("alias_of") != source:
            raise ValueError(f"preference alias mismatch: {alias_id}")
        if generation["checkpoints"][alias_id].get("alias_of") != source:
            raise ValueError(f"generation alias mismatch: {alias_id}")
        aliases.append(
            {
                "id": alias_id,
                "source": source,
                "state_sha256": by_id[source]["state_sha256"],
                "reason": alias.get("reason"),
                "rerun": False,
            }
        )
    manifest = {
        "format_version": "checkpoint-perception-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoints": output,
        "aliases": aliases,
        "cross_path_state_hashes_equal": True,
        "preference_summary_sha256": sha256(args.preference_summary),
        "generation_summary_sha256": sha256(args.generation_summary),
        "final_half_scored": False,
    }
    args.out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
