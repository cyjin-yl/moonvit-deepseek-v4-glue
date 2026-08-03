"""Parquet packing: order, field, and image-byte fidelity between JSONL and parquet."""

import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

pq = pytest.importorskip("pyarrow.parquet")

from pack_to_parquet import pack_jsonl
from tools_common import load_records


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _make_dataset(tmp_path: Path) -> Path:
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.png").write_bytes(_png_bytes((255, 0, 0)))
    (tmp_path / "images" / "b.png").write_bytes(_png_bytes((0, 255, 0)))
    records = [
        {"id": "r0", "image": "images/a.png", "question": "q0", "answers": ["a0", "a0b"]},
        {"id": "r1", "image": None, "question": "q1", "answers": ["a1"]},
        {"id": "r2", "image": "images/b.png", "question": "q2", "answers": ["a2"],
         "gt_box": [1, 2, 3, 4], "gt_point": [5, 6]},
    ]
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return jsonl


def test_single_shard_roundtrip_preserves_everything(tmp_path):
    jsonl = _make_dataset(tmp_path)
    written = pack_jsonl(jsonl, tmp_path / "packed" / "data.parquet")
    assert len(written) == 1
    records = load_records(written[0])
    assert [r["id"] for r in records] == ["r0", "r1", "r2"]
    assert records[0]["answers"] == ["a0", "a0b"]
    assert records[2]["gt_box"] == [1, 2, 3, 4]
    assert records[0]["image_bytes"] == (tmp_path / "images" / "a.png").read_bytes()
    assert records[2]["image_bytes"] == (tmp_path / "images" / "b.png").read_bytes()
    assert records[1]["image_bytes"] is None


def test_multi_shard_directory_loads_in_order(tmp_path):
    jsonl = _make_dataset(tmp_path)
    written = pack_jsonl(jsonl, tmp_path / "packed" / "data.parquet", shard_rows=2)
    assert len(written) == 2
    assert written[0].name == "data-00000-of-00002.parquet"
    records = load_records(written[0].parent)
    assert [r["id"] for r in records] == ["r0", "r1", "r2"]
    assert records[2]["gt_point"] == [5, 6]


def test_jsonl_loading_still_works(tmp_path):
    jsonl = _make_dataset(tmp_path)
    records = load_records(jsonl)
    assert [r["id"] for r in records] == ["r0", "r1", "r2"]
    assert "image_bytes" not in records[0]
