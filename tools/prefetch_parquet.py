"""Prefetch hub parquet shards with aria2 for fully offline dataset builds.

The workstation proxy passes the HF API quickly but stalls both single-stream
curl and hf_transfer on LFS payloads; aria2 with ``-x8`` segments sustains
multi-MB/s through it. This tool lists the parquet shards of a
(config, split) via the hub API, downloads missing/incomplete ones with
aria2 (``-c`` resume), then sha256-verifies every shard against the repo's
LFS oid (a ``.sha256ok`` marker caches the pass; corrupt shards are purged
and re-downloaded — segmented resume can silently deliver bad pieces), and
emits the ``--data-files`` line for ``tools/fetch_eval_data.py``.

Example::

    python tools/prefetch_parquet.py --repo jxie/flickr8k --split train \
        --out-dir $HDD/staging/parquet/flickr8k
    python tools/prefetch_parquet.py --repo MMMU/MMMU_Pro --split test \
        --path-prefix "standard (10 options)/" --out-dir $HDD/staging/parquet/mmmu_pro
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"


def matches_shard(path: str, split: str, path_prefix: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name.endswith(".parquet") and path.startswith(path_prefix) and f"{split}-" in name


def list_parquet_shards(repo: str, split: str, revision: str, path_prefix: str) -> list[tuple[str, int, str | None]]:
    from huggingface_hub import HfApi

    # The proxy drops API TLS connections at random (SSL EOF); retry briefly.
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            api = HfApi()
            entries = api.list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True)
            shards = sorted(
                (entry.path, entry.size)
                for entry in entries
                if matches_shard(entry.path, split, path_prefix)
            )
            # list_repo_tree(expand=True) leaves .lfs empty on hub 1.21, so fetch
            # LFS digests separately; the attribute is .oid on some versions and
            # .sha256 on others.
            hashes: dict[str, str] = {}
            paths = [path for path, _ in shards]
            for i in range(0, len(paths), 100):
                for info in api.get_paths_info(repo, paths[i:i + 100], repo_type="dataset",
                                               revision=revision, expand=True):
                    lfs = getattr(info, "lfs", None)
                    digest = (getattr(lfs, "oid", None) or getattr(lfs, "sha256", None)) if lfs else None
                    if digest:
                        hashes[info.path] = digest
            break
        except Exception as exc:  # noqa: BLE001 — any transient hub/SSL failure
            last_error = exc
            print(f"list_repo_tree attempt {attempt + 1}/5 failed: {type(exc).__name__}: {exc}", flush=True)
            import time

            time.sleep(5)
    else:
        raise SystemExit(f"list_repo_tree failed after 5 attempts: {last_error}")
    if not shards:
        raise SystemExit(f"no parquet shards match split={split!r} prefix={path_prefix!r} in {repo}")
    missing = [path for path, _ in shards if path not in hashes]
    if missing:
        print(f"WARNING: no LFS digest for {len(missing)}/{len(shards)} shard(s); "
              f"those fall back to size-only verification", flush=True)
    return [(path, size, hashes.get(path)) for path, size in shards]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _marker(out_path: Path) -> Path:
    return Path(str(out_path) + ".sha256ok")


def _purge(out_path: Path) -> None:
    # Corrupt bytes can sit anywhere in a segmented download; only a fresh
    # fetch is safe (aria2 -c would happily resume onto bad pieces).
    for p in (out_path, Path(str(out_path) + ".aria2"), _marker(out_path)):
        p.unlink(missing_ok=True)


def hash_verified(out_path: Path, size: int, expected: str | None) -> bool:
    """True when the shard is usable; manages the .sha256ok marker cache."""
    if not out_path.exists() or out_path.stat().st_size != size:
        return False
    if expected is None:  # non-LFS file: size is all the API gives us
        return True
    marker = _marker(out_path)
    if marker.exists() and marker.read_text().strip() == expected:
        return True
    actual = sha256_of(out_path)
    if actual == expected:
        marker.write_text(actual)
        return True
    print(f"sha256 MISMATCH {out_path.name}: {actual} != {expected}; purging for re-download", flush=True)
    _purge(out_path)
    return False


def aria2_fetch(url: str, size: int, expected: str | None, out_path: Path) -> None:
    if hash_verified(out_path, size, expected):
        print(f"skip (complete): {out_path.name}", flush=True)
        return
    cmd = [
        "aria2c", "-x8", "-s8", "-k", "4M", "-c", "--file-allocation=none",
        # The xet-bridge CDN behind the proxy drops TLS handshakes and 403s
        # range-signed URLs at random; unlimited retries are the working strategy.
        "--max-tries=0", "--retry-wait=5", "--timeout=30", "--connect-timeout=20",
        "--summary-interval=15",
    ]
    # aria2 does NOT honor the (uppercase) HTTPS_PROXY env var and silently goes
    # direct otherwise — from the workstation, direct HF is GFW-throttled to
    # ~250 KB/s aggregate while the Clash node does ~460 KB/s per target.
    proxy = (os.environ.get("PREFETCH_ARIA2_PROXY") or os.environ.get("https_proxy")
             or os.environ.get("HTTPS_PROXY"))
    if proxy:
        cmd += [f"--all-proxy={proxy}"]
    cmd += ["-d", str(out_path.parent), "-o", out_path.name, url]
    # aria2 treats TLS handshake failures as fatal per invocation regardless of
    # --max-tries, so re-invoke the whole process; -c resumes each time.
    for attempt in range(10):
        proc = subprocess.run(cmd, check=False)
        if hash_verified(out_path, size, expected):
            return
        current = out_path.stat().st_size if out_path.exists() else 0
        print(f"aria2 attempt {attempt + 1}/10 exited {proc.returncode}; "
              f"at {current / 1e6:.1f}/{size / 1e6:.1f} MB; resuming in 5s", flush=True)
        time.sleep(5)
    raise SystemExit(f"aria2 could not complete {out_path.name} after 10 attempts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--path-prefix", default="", help="restrict to paths under this prefix (config dir)")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shards = list_parquet_shards(args.repo, args.split, args.revision, args.path_prefix)
    total = sum(size for _, size, _ in shards)
    print(f"{len(shards)} shard(s), {total / 1e9:.2f} GB -> {args.out_dir}", flush=True)
    files = []
    for path, size, expected in shards:
        url = RESOLVE.format(repo=args.repo, revision=args.revision, path=quote(path))
        out_path = args.out_dir / path.rsplit("/", 1)[-1]
        aria2_fetch(url, size, expected, out_path)
        print(f"sha256 {sha256_of(out_path)}  {out_path.name}", flush=True)
        files.append(out_path)
    print("\n--data-files " + " ".join(str(f) for f in files))


if __name__ == "__main__":
    main()
