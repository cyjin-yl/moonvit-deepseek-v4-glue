import hashlib
import json
import sys
from pathlib import Path

from moonvit_glue.training_health import evaluate_guards


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import verify_qwen3b_training_health as verifier  # noqa: E402


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probe(step: int):
    return {
        "step": step,
        "projector": {
            "relative_spread_ratio": 1.0,
            "effective_rank_ratio": 1.0,
            "top1_variance_fraction": 0.2,
            "sample_rms_ratio": 1.0,
        },
        "receiver": {
            "relative_spread_ratio": 1.0,
            "effective_rank_ratio": 1.0,
            "top1_variance_fraction": 0.2,
            "sample_rms_ratio": 1.0,
        },
        "causal": {
            "correct_preference": 0.75,
            "shuffled_preference": 0.50,
            "vision_minus_shuffle_correct_logp": 0.1,
        },
        "has_nan_or_inf": False,
    }


def test_independent_health_verifier_recomputes_guard_trajectory(tmp_path, monkeypatch):
    contract_path = ROOT / "configs" / "qwen3b-projector-health-v1.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "RUN_CONFIG.json",
        {"health_setup": {"contract_file_sha256": _sha(contract_path)}},
    )
    health_row = {
        "optimizer_step": 1,
        "projector_output_rms": 1.0,
        "receiver_output_rms": 1.0,
        "between_image_rms": 0.5,
        "within_image_token_rms": 0.5,
        "relative_spread": 0.5,
        "mean_direction_fraction": 0.5,
        "projector_gradient_norm_before_clip": 1.0,
        "projector_gradient_norm_after_clip": 1.0,
        "ce_loss": 2.0,
        "geometry_loss": 0.0,
        "total_loss": 2.0,
        "learning_rate": 1e-4,
        "has_nan_or_inf": False,
    }
    _write_jsonl(
        run / "train_health.jsonl",
        [
            {"event": "run_start", "optimizer_step": 0, "examples_seen": 0},
            health_row,
        ],
    )
    state = {}
    step0 = _probe(0)
    step0["guards"] = evaluate_guards(
        step0, previous=None, state=state, contract=contract
    )
    step1 = _probe(1)
    step1["guards"] = evaluate_guards(
        step1, previous=step0, state=state, contract=contract
    )
    _write_jsonl(run / "probe_metrics.jsonl", [step0, step1])
    files = []
    for name in ("train_health.jsonl", "probe_metrics.jsonl"):
        path = run / name
        files.append(
            {"path": name, "bytes": path.stat().st_size, "sha256": _sha(path)}
        )
    _write_json(
        run / "HEALTH_ARTIFACT_MANIFEST.json",
        {
            "format_version": "projector-health-artifact-manifest-v1",
            "files": files,
            "file_count": len(files),
            "total_bytes": sum(row["bytes"] for row in files),
        },
    )
    out = tmp_path / "VERIFICATION.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_qwen3b_training_health.py",
            "--run",
            str(run),
            "--health-contract",
            str(contract_path),
            "--out",
            str(out),
        ],
    )

    verifier.main()

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["status"] == "verified"
    assert result["probe_steps"] == [0, 1]
    assert result["guards_recomputed"] == 2
