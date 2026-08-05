"""Publish frozen projector controls and verify the immutable HF commit."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_files(root: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected = {
        item["path"]: {"bytes": item["bytes"], "sha256": item["sha256"]}
        for role in manifest["roles"]
        for item in role["files"]
    }
    manifest_path = root / "MANIFEST.json"
    expected["MANIFEST.json"] = {
        "bytes": manifest_path.stat().st_size,
        "sha256": sha256_file(manifest_path),
    }
    return expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--path-in-repo", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    root = args.artifact_dir.resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    expected = _expected_files(root, manifest)
    api = HfApi()
    commit = api.upload_folder(
        folder_path=str(root),
        repo_id=args.repo_id,
        repo_type="model",
        path_in_repo=args.path_in_repo,
        commit_message="Freeze Qwen 3B projector initialization controls",
    )
    tree = list(
        api.list_repo_tree(
            args.repo_id,
            path_in_repo=args.path_in_repo,
            recursive=True,
            expand=True,
            revision=commit.oid,
            repo_type="model",
        )
    )
    by_relative = {}
    prefix = args.path_in_repo.rstrip("/") + "/"
    for entry in tree:
        if not getattr(entry, "path", "").startswith(prefix):
            continue
        relative = entry.path[len(prefix) :]
        if relative:
            by_relative[relative] = entry

    verified_files = []
    for relative, contract in sorted(expected.items()):
        entry = by_relative.get(relative)
        if entry is None:
            raise RuntimeError(f"published file missing from immutable commit: {relative}")
        remote_size = int(entry.size)
        lfs = getattr(entry, "lfs", None)
        lfs_sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
        if lfs_sha256 is not None:
            remote_sha256 = lfs_sha256
            method = "HF LFS SHA-256"
        else:
            downloaded = Path(
                hf_hub_download(
                    args.repo_id,
                    filename=f"{args.path_in_repo.rstrip('/')}/{relative}",
                    revision=commit.oid,
                    repo_type="model",
                )
            )
            remote_sha256 = sha256_file(downloaded)
            method = "downloaded immutable small file SHA-256"
        matches = (
            remote_size == int(contract["bytes"])
            and remote_sha256 == contract["sha256"]
        )
        if not matches:
            raise RuntimeError(f"published file verification failed: {relative}")
        verified_files.append(
            {
                "path": relative,
                "bytes": remote_size,
                "sha256": remote_sha256,
                "verification_method": method,
                "matches_local_contract": True,
            }
        )

    result = {
        "schema_version": "projector-initializations-publication-v1",
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "repo_type": "model",
        "path_in_repo": args.path_in_repo,
        "commit_oid": commit.oid,
        "commit_url": commit.commit_url,
        "source_manifest_sha256": manifest["manifest_sha256"],
        "all_files_verified": True,
        "file_count": len(verified_files),
        "files": verified_files,
        "paid_resource_used": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
