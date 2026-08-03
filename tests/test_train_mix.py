"""Train-mix assembly: source caps, 0xSero row normalization, decontamination."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from build_train_mix import (
    average_hash,
    hamming,
    is_contaminated,
    normalize_0xsero_row,
    select_subset,
    strip_image_span,
)


def _noise_image(seed: int):
    import random

    from PIL import Image

    rng = random.Random(seed)
    data = bytes(rng.randrange(256) for _ in range(64 * 64))
    return Image.frombytes("L", (64, 64), data)


def test_average_hash_deterministic_and_sensitive_to_content():
    base = _noise_image(1)
    near = _noise_image(1)
    near.putpixel((5, 5), 255)
    far = _noise_image(2)
    assert average_hash(base) == average_hash(base.copy())
    assert hamming(average_hash(base), average_hash(near)) <= 8
    assert hamming(average_hash(base), average_hash(far)) > 12


def test_select_subset_caps_and_is_seeded():
    records = [{"id": i} for i in range(10)]
    import random

    first = select_subset(records, 4, random.Random(7))
    second = select_subset(records, 4, random.Random(7))
    assert len(first) == 4
    assert first == second
    assert select_subset(records[:3], 4, random.Random(7)) == records[:3]


def test_strip_image_span_removes_placeholder_prefix():
    text = "|begin_of_image||image||image||end_of_image|What style is this?"
    assert strip_image_span(text) == "What style is this?"
    assert strip_image_span("plain question") == "plain question"


def test_normalize_0xsero_row_extracts_short_qa():
    row = {
        "image": "wikiart_014834.jpg",
        "conversations": [
            {"role": "user", "content": "|begin_of_image||image||end_of_image|What style?"},
            {"role": "assistant", "content": "Realism"},
        ],
        "source": "art",
    }
    assert normalize_0xsero_row(row) == {
        "image": "wikiart_014834.jpg",
        "question": "What style?",
        "answers": ["Realism"],
    }
    assert normalize_0xsero_row({"image": "x.jpg", "conversations": []}) is None
    assert normalize_0xsero_row({"conversations": row["conversations"]}) is None


def test_is_contaminated_flags_exact_and_near_duplicate_hashes():
    eval_hashes = [0b1111000011110000]
    assert is_contaminated(0b1111000011110000, eval_hashes, threshold=6)
    assert is_contaminated(0b1111000011110011, eval_hashes, threshold=6)  # 2 bits off
    assert not is_contaminated(0b0000111100001111, eval_hashes, threshold=6)
