from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from materialize_screenspot_images import materialized_record


def test_materialized_record_keeps_fixed_scoring_and_image_identity():
    sample = {
        "sample_id": "screenspot-x",
        "instruction": "click save",
        "bbox_999_xyxy": [10, 20, 30, 40],
        "platform": "Web",
        "target_type": "text",
        "image_sha256": "a" * 64,
        "image_width": 100,
        "image_height": 200,
        "source_parquet": "data/test.parquet",
        "source_row_index": 7,
    }
    assert materialized_record(sample, "images/a.bin") == {
        "id": "screenspot-x",
        "image": "images/a.bin",
        "question": "click save",
        "gt_box": [10.0, 20.0, 30.0, 40.0],
        "platform": "Web",
        "target_type": "text",
        "image_sha256": "a" * 64,
        "image_width": 100,
        "image_height": 200,
        "source_parquet": "data/test.parquet",
        "source_row_index": 7,
    }
