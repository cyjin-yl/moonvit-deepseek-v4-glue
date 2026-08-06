import hashlib
import json
from pathlib import Path

import torch

from moonvit_glue.projector import PatchMergerProjector, ProjectorConfig
from moonvit_glue.projector_binding import validate_variant_binding


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registered_variant_binds_base_and_step0_output(tmp_path):
    base_dir = tmp_path / "base"
    variant_dir = tmp_path / "variant"
    base = PatchMergerProjector(
        ProjectorConfig(vision_width=3, language_width=8, merge_factor=4)
    )
    variant = PatchMergerProjector(
        ProjectorConfig(
            vision_width=3,
            language_width=8,
            merge_factor=4,
            residual_mode="zero_init",
        )
    )
    variant_state = variant.state_dict()
    for key, value in base.state_dict().items():
        variant_state[key].copy_(value)
    with torch.no_grad():
        variant.residual.weight.zero_()
    base.save_pretrained(base_dir)
    variant.save_pretrained(variant_dir)

    contract_path = tmp_path / "contract.json"
    variant_config_path = variant_dir / "projector_config.json"
    contract = {
        "projector_source_file": "src/moonvit_glue/projector.py",
        "projector_source_file_sha256": _sha(ROOT / "src/moonvit_glue/projector.py"),
        "training_runner_source_file": "tools/train_qwen3b_proxy.py",
        "training_runner_source_file_sha256": _sha(ROOT / "tools/train_qwen3b_proxy.py"),
        "projector_binding_source_file": "src/moonvit_glue/projector_binding.py",
        "projector_binding_source_file_sha256": _sha(
            ROOT / "src/moonvit_glue/projector_binding.py"
        ),
        "base_projector": {
            "config_sha256": _sha(base_dir / "projector_config.json"),
            "weights_sha256": _sha(base_dir / "projector.safetensors"),
        },
        "arms": [
            {
                "name": "zero",
                "residual_mode": "zero_init",
                "config_sha256": _sha(variant_config_path),
                "weights_sha256": _sha(variant_dir / "projector.safetensors"),
                "parameter_count": sum(p.numel() for p in variant.parameters()),
            }
        ],
        "boundary": {"canonical_output_width": 8},
    }
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    binding = validate_variant_binding(
        root=ROOT,
        contract_path=contract_path,
        arm_name="zero",
        projector_dir=variant_dir,
        base_dir=base_dir,
    )
    assert binding.kind == "registered_variant"
    assert binding.arm == "zero"
    assert binding.residual_mode == "zero_init"
    assert binding.parameter_count == sum(p.numel() for p in variant.parameters())
    assert binding.base_weights_sha256 == _sha(base_dir / "projector.safetensors")
