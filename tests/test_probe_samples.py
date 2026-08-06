import json

import pytest

from moonvit_glue.probe_samples import load_receiver_probe_records, receiver_probe_supervision


def _write_manifest(path, records):
    path.write_text(json.dumps({"records": records}), encoding="utf-8")


def test_probe_manifest_binds_real_answers_by_id_and_image_sha(tmp_path):
    records = [
        {"id": "a", "image_sha256": "1" * 64, "question": "Where?", "answers": ["left"]},
        {"id": "b", "image_sha256": "2" * 64, "instruction": "Click it", "target_answer": "click(start_box=[1,2])"},
    ]
    path = tmp_path / "probe.json"
    _write_manifest(path, records)
    _, bound = load_receiver_probe_records(path, [
        {"id": "a", "image_sha256": "1" * 64},
        {"id": "b", "image_sha256": "2" * 64},
    ])
    assert receiver_probe_supervision(bound[0]) == ("Where?", "left")
    assert receiver_probe_supervision(bound[1]) == ("Click it", "click(start_box=[1,2])")


@pytest.mark.parametrize("record", [
    {"id": "a", "image_sha256": "1" * 64, "answers": ["x"]},
    {"id": "a", "image_sha256": "1" * 64, "question": "q"},
])
def test_probe_manifest_rejects_missing_supervision(tmp_path, record):
    path = tmp_path / "probe.json"
    _write_manifest(path, [record])
    with pytest.raises(ValueError):
        load_receiver_probe_records(path, [{"id": "a", "image_sha256": "1" * 64}])


def test_probe_manifest_rejects_cache_identity_mismatch(tmp_path):
    path = tmp_path / "probe.json"
    _write_manifest(path, [{"id": "a", "image_sha256": "1" * 64, "question": "q", "answers": ["a"]}])
    with pytest.raises(ValueError, match="SHA mismatch"):
        load_receiver_probe_records(path, [{"id": "a", "image_sha256": "2" * 64}])
