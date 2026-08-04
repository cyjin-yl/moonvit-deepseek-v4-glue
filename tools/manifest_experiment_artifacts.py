#!/usr/bin/env python3
"""把实验包内全部文件哈希进确定性 artifact manifest。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", default="ARTIFACT_MANIFEST.json")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / args.out
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != output:
            relative = path.relative_to(root).as_posix()
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    payload = {
        "format_version": "experiment-artifact-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": root.name,
        "git_base_sha": git or None,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(value["bytes"] for value in files.values()),
        "final_half_scored": False,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in ("file_count", "total_bytes")}, indent=2))


if __name__ == "__main__":
    main()
