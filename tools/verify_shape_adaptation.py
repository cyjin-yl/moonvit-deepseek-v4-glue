#!/usr/bin/env python3
"""独立验证 package 5 的训练、评测与 bootstrap 产物。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moonvit_glue.adaptation_verification import (
    sha256,
    verify_analysis,
    verify_evaluation,
    verify_training_run,
)
from moonvit_glue.mechanism_verification import (
    verify_probe_analysis,
    verify_representations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora-training", required=True, type=Path)
    parser.add_argument("--projector-training", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--lora-representations", type=Path)
    parser.add_argument("--lora-probes", type=Path)
    parser.add_argument("--projector-representations", type=Path)
    parser.add_argument("--projector-probes", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    lora = verify_training_run(args.lora_training)
    projector = verify_training_run(args.projector_training)
    lora_order = args.lora_training / "training_order.jsonl"
    projector_order = args.projector_training / "training_order.jsonl"
    if sha256(lora_order) != sha256(projector_order):
        raise ValueError("LoRA and projector continuation did not share the exact data order")
    result = {
        "status": "valid",
        "lora_training": lora,
        "projector_training": projector,
        "equal_training_order_sha256": sha256(lora_order),
        "evaluation": verify_evaluation(args.evaluation),
        "analysis": verify_analysis(args.analysis, args.evaluation),
        "final_half_scored": False,
    }
    layerwise_paths = (
        args.lora_representations,
        args.lora_probes,
        args.projector_representations,
        args.projector_probes,
    )
    if any(layerwise_paths) and not all(layerwise_paths):
        raise ValueError("all four layerwise paths must be supplied together")
    if all(layerwise_paths):
        result["layerwise"] = {
            "lora": {
                "representations": verify_representations(args.lora_representations),
                "probes": verify_probe_analysis(
                    args.lora_probes, args.lora_representations
                ),
            },
            "projector": {
                "representations": verify_representations(
                    args.projector_representations
                ),
                "probes": verify_probe_analysis(
                    args.projector_probes, args.projector_representations
                ),
            },
        }
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
