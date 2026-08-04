"""Preference verification binds raw rows, pairs, conditions, and config hashes."""

import hashlib
import json

from moonvit_glue.preference_verification import verify_preference_run


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _valid_run(tmp_path):
    config = {
        "checkpoints": [{"id": "c0"}],
        "aliases": [{"id": "current", "source": "c0"}],
        "synthetic": {"expected_records": 2, "conditions": ["vision", "blind"]},
    }
    config_path = tmp_path / "CONFIG.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    rows = [
        {
            "checkpoint": "c0",
            "condition": condition,
            "id": sample_id,
            "pair_id": "pair-1",
            "pair_variant": variant,
            "task": "color",
            "correct_answer_tokens": 1,
            "counterfactual_answer_tokens": 1,
            "correct_logp_mean": -0.4,
            "counterfactual_logp_mean": -0.7,
            "correct_token_nll": 0.4,
            "counterfactual_token_nll": 0.7,
            "correct_margin": 0.3,
            "preference_correct": True,
            "failure": None,
        }
        for condition in ("vision", "blind")
        for sample_id, variant in (("a", "a"), ("b", "b"))
    ]
    raw_path = tmp_path / "preference_records.jsonl"
    failure_path = tmp_path / "failures.jsonl"
    _jsonl(raw_path, rows)
    failure_path.write_text("", encoding="utf-8")
    summary = {
        "status": "valid",
        "metadata": {
            "config_sha256": _hash(config_path),
            "final_half_scored": False,
        },
        "checkpoints": {"c0": {}, "current": {"alias_of": "c0"}},
        "raw_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _hash(path)}
            for path in (raw_path, failure_path)
        },
    }
    (tmp_path / "SUMMARY.json").write_text(json.dumps(summary), encoding="utf-8")
    return rows


def test_preference_verifier_accepts_complete_pair_matrix(tmp_path):
    _valid_run(tmp_path)

    verified = verify_preference_run(tmp_path)

    assert verified["status"] == "valid"
    assert verified["rows_verified"] == 4
    assert verified["pairs_per_cell"] == 1


def test_preference_verifier_rejects_condition_id_drift(tmp_path):
    rows = _valid_run(tmp_path)
    rows[-1]["id"] = "c"
    raw_path = tmp_path / "preference_records.jsonl"
    _jsonl(raw_path, rows)
    summary_path = tmp_path / "SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["raw_files"][raw_path.name] = {
        "bytes": raw_path.stat().st_size,
        "sha256": _hash(raw_path),
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    try:
        verify_preference_run(tmp_path)
    except ValueError as error:
        assert "sample IDs mismatch" in str(error)
    else:
        raise AssertionError("condition-specific preference ID drift was accepted")


def test_preference_verifier_rejects_non_finite_margin(tmp_path):
    rows = _valid_run(tmp_path)
    rows[0]["correct_margin"] = float("nan")
    raw_path = tmp_path / "preference_records.jsonl"
    _jsonl(raw_path, rows)
    summary_path = tmp_path / "SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["raw_files"][raw_path.name] = {
        "bytes": raw_path.stat().st_size,
        "sha256": _hash(raw_path),
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    try:
        verify_preference_run(tmp_path)
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("non-finite preference margin was accepted")
