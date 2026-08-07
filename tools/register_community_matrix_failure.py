#!/usr/bin/env python3
"""Register an immutable failure row for the frozen community matrix.

The long-running matrix runners call this helper when an arm cannot produce a
valid result. It preserves the reason and raw logs, updates the single matrix
summary, and refuses to overwrite an existing attempt directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "experiments/community_scale_model_ablation_20260808/MATRIX_SUMMARY.json"
ARTIFACT_ROOT = ROOT / "experiments/community_scale_model_ablation_20260808/failure_artifacts"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--attempt", required=True, type=int)
    parser.add_argument("--failure-class", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--run-output", type=Path, default=None)
    parser.add_argument("--log", action="append", type=Path, default=[])
    parser.add_argument("--artifact", action="append", type=Path, default=[])
    parser.add_argument("--contract-examples-seen", type=int, default=57600)
    parser.add_argument("--optimizer-steps-requested", type=int, default=900)
    parser.add_argument("--stop-step", type=int, default=None)
    parser.add_argument("--examples-seen", type=int, default=None)
    parser.add_argument(
        "--status",
        default="failed_runtime",
        choices=["failed_runtime", "failed_resource", "failed_evaluation"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError(SUMMARY_PATH)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    rows = summary["rows"]
    if args.arm not in rows:
        raise KeyError(f"arm is not registered in matrix: {args.arm}")
    artifact_dir = ARTIFACT_ROOT / f"{args.arm}_attempt{args.attempt}"
    if artifact_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable failure artifact: {artifact_dir}"
        )
    artifact_dir.mkdir(parents=True)

    sources: list[Path] = []
    for source in [*args.log, *args.artifact]:
        source = source.resolve()
        if source.is_file():
            sources.append(source)
    copied: list[dict[str, str]] = []
    for source in sources:
        destination = artifact_dir / source.name
        shutil.copy2(source, destination)
        copied.append({"file": destination.name, "sha256": sha256_file(destination)})
    (artifact_dir / "SHA256SUMS").write_text(
        "".join(f"{entry['sha256']}  {entry['file']}\n" for entry in copied),
        encoding="utf-8",
    )

    captured_at = datetime.now(timezone.utc).astimezone().isoformat()
    failure = {
        "schema_version": "community-matrix-failure-v1",
        "captured_at": captured_at,
        "git_sha": git_sha(),
        "arm": args.arm,
        "attempt": args.attempt,
        "contract_examples_seen": args.contract_examples_seen,
        "optimizer_steps_requested": args.optimizer_steps_requested,
        "stop_optimizer_step": args.stop_step,
        "examples_seen": args.examples_seen,
        "run_output": str(args.run_output) if args.run_output else None,
        "failure_class": args.failure_class,
        "capability_claim_allowed": False,
        "reason": args.reason,
        "raw_artifacts": [entry["file"] for entry in copied],
    }
    failure_path = artifact_dir / "FAILURE.json"
    failure_path.write_text(
        json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    row = rows[args.arm]
    history = row.setdefault("failure_history", [])
    history.append(
        {
            "attempt": args.attempt,
            "class": args.failure_class,
            "artifact": str(failure_path.relative_to(ROOT).as_posix()),
            "reason": args.reason,
        }
    )
    row["attempts"] = max(int(row.get("attempts", 0)), args.attempt)
    row["result"] = None
    row["status"] = args.status
    row["current_attempt"] = {
        "attempt": args.attempt,
        "status": args.status,
        "contract_examples_seen": args.contract_examples_seen,
        "optimizer_steps_requested": args.optimizer_steps_requested,
        "optimizer_step_stopped": args.stop_step,
        "examples_seen_stopped": args.examples_seen,
        "train_output": str(args.run_output) if args.run_output else None,
        "failure_artifact": str(failure_path.relative_to(ROOT).as_posix()),
        "failure_class": args.failure_class,
        "capability_claim_allowed": False,
    }
    summary["updated_at"] = captured_at
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"registered {args.arm} attempt {args.attempt}: {failure_path}")


if __name__ == "__main__":
    main()
