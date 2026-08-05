import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    images = tmp_path / "images"
    images.mkdir()
    records = []
    for index, source in enumerate(
        ("textvqa_train", "showui_desktop", "docvqa_train", "showui_desktop")
    ):
        image = images / f"sample-{index}.png"
        Image.new("RGB", (8 + index, 7), (index * 30, 20, 40)).save(image)
        records.append(
            {
                "id": f"sample-{index}",
                "image": f"images/{image.name}",
                "question": f"question {index}",
                "answers": (
                    [f"click(start_box=[{index},{index + 1}])"]
                    if source.startswith("showui")
                    else [f"answer {index}"]
                ),
                "source": source,
            }
        )
    data = tmp_path / "train.jsonl"
    data.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    contract = {
        "datasets": {
            "training_pack": {
                "records": len(records),
                "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "order_is_frozen": True,
            }
        },
        "training_budget": {
            "examples_seen_checkpoints": [4],
            "optimizer_steps_checkpoints": [2],
            "micro_batch_size": 1,
            "gradient_accumulation": 2,
            "real_global_batch": 2,
        },
        "image_preprocessing": {
            "train_max_image_side": 448,
            "train_max_visual_tokens": 256,
        },
        "vision_tower": {
            "name": "MoonViT-V2",
            "extracted_weights_sha256": "a" * 64,
        },
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return data, contract_path


def _run_tool(*args: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_grounding_enriched_freezer_and_independent_verifier(tmp_path):
    data, contract = _write_fixture(tmp_path)
    manifest = tmp_path / "MANIFEST.json"
    frozen = _run_tool(
        str(ROOT / "tools" / "freeze_qwen3b_grounding_enriched_order.py"),
        "--contract",
        str(contract),
        "--data",
        str(data),
        "--grounding-examples",
        "2",
        "--short-answer-examples",
        "2",
        "--out",
        str(manifest),
    )
    assert frozen.returncode == 0, frozen.stderr
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert [row["source_row_index"] for row in payload["records"]] == [1, 0, 3, 2]

    verification = tmp_path / "INDEPENDENT_VERIFICATION.json"
    verified = _run_tool(
        str(ROOT / "tools" / "verify_qwen3b_training_order.py"),
        "--manifest",
        str(manifest),
        "--contract",
        str(contract),
        "--data",
        str(data),
        "--out",
        str(verification),
    )
    assert verified.returncode == 0, verified.stderr
    checked = json.loads(verification.read_text(encoding="utf-8"))
    assert checked["status"] == "valid"
    assert checked["checks"]["selection_matches_registered_rule"] is True
    assert checked["matched_records"] == checked["matched_images"] == 4
