"""Art SFT subset builder: 0xSero schema compatibility and label filtering."""

import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from fetch_art_data import IMAGE_SPAN, build_subset, fashion_qa, wikiart_qa

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _png_bytes(color) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_parquet(path: Path, rows: list[dict]) -> None:
    image_struct = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    columns = {key: [row.get(key) for row in rows] for key in rows[0]}
    columns["image"] = pa.array(columns["image"], type=image_struct)
    pq.write_table(pa.table(columns), str(path))


def test_wikiart_qa_filters_unknown_artist_and_bad_date():
    import random

    rng = random.Random(0)
    assert wikiart_qa({"artist": "Unknown Artist", "style": "", "genre": "", "date": ""}, rng) == []
    qa = wikiart_qa({"artist": "Claude Monet", "style": "Impressionism", "genre": "landscape", "date": "1872"}, rng)
    answers = {answer for _, answer in qa}
    assert answers <= {"Claude Monet", "Impressionism", "landscape", "1872"}
    assert len(qa) <= 2  # <=2 QA per image cap
    # All answers are short — the mixer's hard red line:
    assert all(len(answer.split()) <= 4 for _, answer in qa)
    bad_date = wikiart_qa({"artist": "", "style": "", "genre": "", "date": "c. 1872"}, rng)
    assert all("1872" not in answer for _, answer in bad_date)


def test_fashion_qa_uses_label_fields():
    import random

    qa = fashion_qa({"baseColour": "Navy Blue", "articleType": "Shirts", "season": "Fall",
                     "usage": "Casual", "masterCategory": "Apparel"}, random.Random(1))
    assert 1 <= len(qa) <= 2
    assert all(answer in {"Navy Blue", "Shirts", "Fall", "Casual", "Apparel"} for _, answer in qa)


def test_build_subset_output_feeds_build_train_mix(tmp_path):
    from build_train_mix import normalize_0xsero_row

    rows = [
        {"image": {"bytes": _png_bytes((i, 0, 0)), "path": f"{i}.png"},
         "artist": f"Artist {i}", "style": "Cubism", "genre": "portrait", "date": "1907"}
        for i in range(4)
    ]
    parquet_path = tmp_path / "train-00000-of-00001-x.parquet"
    _write_parquet(parquet_path, rows)

    import random

    out = build_subset([parquet_path], ("artist", "style", "genre", "date"), wikiart_qa,
                       3, tmp_path / "imgs", "wikiart", 300_000, random.Random(42))

    assert 0 < len(out) <= 6  # <=2 examples per image, 3-image cap
    for example in out:
        normalized = normalize_0xsero_row(example)
        assert normalized is not None, "0xSero schema must survive build_train_mix normalization"
        assert normalized["question"] and normalized["answers"][0]
        assert len(normalized["answers"][0].split()) <= 4
        assert (tmp_path / "imgs" / Path(example["image"]).name).exists()
        # The GLM image span is required by the mixer's strip_image_span:
        assert example["conversations"][0]["content"].startswith(IMAGE_SPAN)
