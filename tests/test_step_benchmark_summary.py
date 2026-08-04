"""Step-time probes report serial accumulation honestly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from run_step_benchmark import summarize_step_reports, write_step_rows


def test_summary_keeps_raw_step_times_and_excludes_warmup():
    payload = {
        "report": {
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 4,
            "actual_batched_forward": False,
            "peak_gpu_memory_bytes": 123,
        },
        "history": [
            {"step": 1, "step_wall_seconds": 9.0, "examples_per_second": 0.4},
            {"step": 2, "step_wall_seconds": 2.0, "examples_per_second": 2.0},
            {"step": 3, "step_wall_seconds": 4.0, "examples_per_second": 1.0},
        ],
    }

    summary, rows = summarize_step_reports([payload], warmup_steps=1)

    assert [row["included_after_warmup"] for row in rows] == [False, True, True]
    assert summary["runs"][0] == {
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 4,
        "actual_batched_forward": False,
        "measured_steps": 2,
        "mean_step_wall_seconds": 3.0,
        "min_step_wall_seconds": 2.0,
        "max_step_wall_seconds": 4.0,
        "examples_per_second_from_mean": 4 / 3,
        "peak_gpu_memory_bytes": 123,
    }


def test_empty_failed_run_still_writes_a_parseable_csv(tmp_path):
    path = tmp_path / "steps.csv"

    write_step_rows(path, [])

    assert path.read_text(encoding="utf-8").startswith(
        "micro_batch_size,gradient_accumulation_steps,effective_batch_size"
    )
