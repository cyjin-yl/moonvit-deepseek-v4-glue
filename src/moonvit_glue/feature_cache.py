"""Auditable sharded caches for frozen MoonViT feature groups."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor


def _records_sha256(records: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class FeatureCacheWriter:
    """Write variable-length ``[tokens, merge, width]`` tensors in shards."""

    def __init__(
        self,
        root: str | Path,
        *,
        cache_metadata: dict[str, Any],
        shard_size: int = 64,
    ) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_metadata = dict(cache_metadata)
        self.shard_size = shard_size
        self.records: list[dict[str, Any]] = []
        self._pending: list[tuple[dict[str, Any], Tensor]] = []
        self._seen_ids: set[str] = set()
        self._shard_paths: list[Path] = []
        self._closed = False

    def add(
        self,
        *,
        sample_id: str,
        feature: Tensor,
        image_sha256: str,
        image_size: tuple[int, int],
    ) -> None:
        if self._closed:
            raise RuntimeError("feature cache writer is already closed")
        sample_id = str(sample_id)
        if not sample_id or sample_id in self._seen_ids:
            raise ValueError(f"sample id must be non-empty and unique: {sample_id!r}")
        if feature.ndim != 3 or feature.shape[0] == 0:
            raise ValueError("feature must have shape [tokens, merge, width]")
        width, height = (int(value) for value in image_size)
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        entry = {
            "id": sample_id,
            "image_sha256": str(image_sha256),
            "image_width": width,
            "image_height": height,
            "feature_shape": list(feature.shape),
            "dtype": str(feature.dtype).removeprefix("torch."),
        }
        self._seen_ids.add(sample_id)
        self._pending.append((entry, feature.detach().cpu().contiguous()))
        if len(self._pending) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        shard_name = f"features-{len(self.records) // self.shard_size:05d}.safetensors"
        trailing_shape = tuple(self._pending[0][1].shape[1:])
        dtype = self._pending[0][1].dtype
        if any(
            tuple(feature.shape[1:]) != trailing_shape or feature.dtype != dtype
            for _, feature in self._pending
        ):
            raise ValueError("all features in one cache shard need matching merge/width/dtype")
        combined = torch.cat([feature for _, feature in self._pending], dim=0)
        shard_path = self.root / shard_name
        save_file(
            {"features": combined},
            str(shard_path),
            metadata={"format": "moonvit-feature-cache-v1"},
        )
        self._shard_paths.append(shard_path)
        offset = 0
        for entry, feature in self._pending:
            token_count = int(feature.shape[0])
            entry.update(
                {
                    "shard": shard_name,
                    "start": offset,
                    "end": offset + token_count,
                }
            )
            self.records.append(entry)
            offset += token_count
        self._pending.clear()

    def close(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("feature cache writer is already closed")
        self._flush()
        manifest = {
            **self.cache_metadata,
            "count": len(self.records),
            "records_sha256": _records_sha256(self.records),
            "shards": [
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
                for path in self._shard_paths
            ],
            "records": self.records,
        }
        (self.root / "MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._closed = True
        return manifest


class FeatureCache:
    """Read feature groups by stable sample ID from a cache manifest."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        manifest_path = self.root / "MANIFEST.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("cache_format_version") != 1:
            raise ValueError("unsupported feature cache format version")
        if _records_sha256(self.manifest["records"]) != self.manifest.get(
            "records_sha256"
        ):
            raise ValueError("feature cache records SHA-256 mismatch")
        self.metadata = {
            key: value for key, value in self.manifest.items() if key != "records"
        }
        self._records = {str(record["id"]): record for record in self.manifest["records"]}
        if len(self._records) != len(self.manifest["records"]):
            raise ValueError("feature cache manifest contains duplicate sample IDs")
        self._loaded_shard: str | None = None
        self._loaded_features: Tensor | None = None

    def __len__(self) -> int:
        return len(self._records)

    def get(
        self,
        sample_id: str,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> list[Tensor]:
        record = self._records.get(str(sample_id))
        if record is None:
            raise KeyError(f"sample id is absent from feature cache: {sample_id}")
        shard = record["shard"]
        if shard != self._loaded_shard:
            self._loaded_features = load_file(
                str(self.root / shard), device="cpu"
            )["features"]
            self._loaded_shard = shard
        assert self._loaded_features is not None
        feature = self._loaded_features[record["start"] : record["end"]]
        return [feature.to(device=device, dtype=dtype)]
