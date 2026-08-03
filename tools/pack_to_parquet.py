"""Pack a JSONL dataset (our ``{image, question, answers}`` schema) into
ordered parquet shards with embedded image bytes.

Rental-server economics: uploading/loading tens of thousands of small image
files is slow in both directions; a few big parquet shards download as
resumable blobs and read sequentially. The original JSONL protocol is
preserved exactly — every top-level field keeps its name and value, rows stay
in JSONL order, and the relative ``image`` path is kept as a field. The only
addition is an ``image_bytes`` column (raw file bytes, ``null`` when the
record has no image). ``tools_common.load_records`` reads both formats
transparently, and the JSONL + MANIFEST stay uploaded as the protocol
reference.

Example::

    python tools/pack_to_parquet.py --jsonl data/train_v1/train_mix.jsonl \
        --out data/train_v1/packed/train_mix.parquet --shard-rows 20000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pack_jsonl(jsonl_path: Path, out_path: Path, shard_rows: int = 20000,
               base_dir: Path | None = None) -> list[Path]:
    """Write ``jsonl_path`` records to parquet shard(s); returns shard paths."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    base = base_dir or jsonl_path.parent
    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    # from_pylist infers the schema from the first row, so normalize every row
    # to the union of keys (missing -> null) or later fields vanish silently.
    keys: list[str] = []
    for record in records:
        for key in record:
            if key not in keys:
                keys.append(key)
    n_shards = max(1, -(-len(records) // shard_rows))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for shard_index in range(n_shards):
        chunk = records[shard_index * shard_rows:(shard_index + 1) * shard_rows]
        rows = []
        for record in chunk:
            row = {key: record.get(key) for key in keys}
            image_rel = record.get("image")
            row["image_bytes"] = (base / image_rel).read_bytes() if image_rel else None
            rows.append(row)
        if n_shards == 1:
            shard_path = out_path
        else:
            shard_path = out_path.with_name(
                f"{out_path.stem}-{shard_index:05d}-of-{n_shards:05d}{out_path.suffix}"
            )
        pq.write_table(pa.Table.from_pylist(rows), shard_path, compression="snappy")
        written.append(shard_path)
        print(f"wrote {shard_path.name}: {len(chunk)} rows, "
              f"{shard_path.stat().st_size / 1e6:.1f} MB", flush=True)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="single-shard output path, or stem for -0000X-of-0000Y shards")
    parser.add_argument("--shard-rows", type=int, default=20000)
    parser.add_argument("--base-dir", type=Path, default=None,
                        help="image root for relative paths (default: the JSONL's directory)")
    args = parser.parse_args()
    written = pack_jsonl(args.jsonl, args.out, args.shard_rows, args.base_dir)
    total = sum(p.stat().st_size for p in written)
    print(f"PACK_DONE {len(written)} shard(s), {total / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
