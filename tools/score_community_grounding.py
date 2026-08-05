"""按冻结合同评分 ScreenSpot 生成 JSONL，并保存全部原始产物。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from moonvit_glue.grounding_evaluation import (
    REQUIRED_CONDITIONS,
    evaluate_conditions,
    read_prediction_jsonl,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _parse_prediction_args(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--prediction must be CONDITION=PATH")
        condition, raw_path = value.split("=", 1)
        if condition in result:
            raise ValueError(f"duplicate prediction condition: {condition}")
        result[condition] = Path(raw_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="CONDITION=PATH",
        help=f"repeat for: {', '.join(REQUIRED_CONDITIONS)}",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260805)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.out_dir}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_ids = [sample["sample_id"] for sample in manifest["samples"]]
    prediction_paths = _parse_prediction_args(args.prediction)
    predictions = {
        condition: read_prediction_jsonl(path, expected_ids)
        for condition, path in prediction_paths.items()
    }
    result = evaluate_conditions(
        manifest,
        predictions,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        allow_partial=args.allow_partial,
    )

    scores_dir = args.out_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    for condition in result["condition_order"]:
        rows = result["conditions"][condition].pop("scores")
        _atomic_jsonl(scores_dir / f"{condition}.jsonl", rows)

    result["inputs"] = {
        "manifest": {
            "path": str(args.manifest),
            "bytes": args.manifest.stat().st_size,
            "sha256": sha256_file(args.manifest),
        },
        "predictions": {
            condition: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for condition, path in sorted(prediction_paths.items())
        },
    }
    summary_path = args.out_dir / "SUMMARY.json"
    _atomic_json(summary_path, result)

    artifacts = []
    for path in sorted(args.out_dir.rglob("*")):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            artifacts.append(
                {
                    "path": path.relative_to(args.out_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    _atomic_json(
        args.out_dir / "ARTIFACT_MANIFEST.json",
        {
            "schema_version": "community-grounding-artifacts-v1",
            "files": artifacts,
            "file_count": len(artifacts),
            "total_bytes": sum(item["bytes"] for item in artifacts),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
