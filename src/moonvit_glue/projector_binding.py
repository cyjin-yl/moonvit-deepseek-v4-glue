"""冻结 projector 变体的文件、参数和 step0 输出绑定。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .projector import PatchMergerProjector


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProjectorBinding:
    """训练和 checkpoint 都要保存的 projector 身份。"""

    kind: str
    weights_sha256: str
    parameter_count: int
    variant_contract_path: str | None = None
    variant_contract_sha256: str | None = None
    arm: str | None = None
    residual_mode: str | None = None
    config_sha256: str | None = None
    base_weights_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_binding(*, weights_sha256: str, parameter_count: int) -> ProjectorBinding:
    return ProjectorBinding(
        kind="canonical_step0",
        weights_sha256=weights_sha256,
        parameter_count=int(parameter_count),
    )


def validate_variant_binding(
    *,
    root: Path,
    contract_path: Path,
    arm_name: str,
    projector_dir: Path,
    base_dir: Path,
) -> ProjectorBinding:
    """验证残差结构合同，并确认变体初始输出逐元素等于 base。

    这一步在加载语言主干前运行，避免错误的结构或权重占用 GPU 后才失败。
    """

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_sha = sha256_file(contract_path)
    arms = {str(row["name"]): row for row in contract.get("arms", [])}
    if arm_name not in arms:
        raise ValueError(f"projector variant arm is absent: {arm_name}")
    arm = arms[arm_name]
    source_path = root / str(contract["projector_source_file"])
    if sha256_file(source_path) != str(contract["projector_source_file_sha256"]):
        raise ValueError("projector source differs from the variant contract")
    runner_path = root / str(contract["training_runner_source_file"])
    if sha256_file(runner_path) != str(contract["training_runner_source_file_sha256"]):
        raise ValueError("training runner differs from the variant contract")
    binding_path = root / str(contract["projector_binding_source_file"])
    if sha256_file(binding_path) != str(contract["projector_binding_source_file_sha256"]):
        raise ValueError("projector binding module differs from the variant contract")

    base_config = base_dir / "projector_config.json"
    base_weights = base_dir / "projector.safetensors"
    variant_config = projector_dir / "projector_config.json"
    variant_weights = projector_dir / "projector.safetensors"
    for path in (base_config, base_weights, variant_config, variant_weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    base_contract = contract["base_projector"]
    base_config_sha = sha256_file(base_config)
    base_weights_sha = sha256_file(base_weights)
    if base_config_sha != str(base_contract["config_sha256"]):
        raise ValueError("base projector config differs from the variant contract")
    if base_weights_sha != str(base_contract["weights_sha256"]):
        raise ValueError("base projector weights differ from the variant contract")
    config_sha = sha256_file(variant_config)
    weights_sha = sha256_file(variant_weights)
    if config_sha != str(arm["config_sha256"]):
        raise ValueError("projector variant config differs from the frozen arm")
    if weights_sha != str(arm["weights_sha256"]):
        raise ValueError("projector variant weights differ from the frozen arm")

    config = json.loads(variant_config.read_text(encoding="utf-8"))
    if config.get("residual_mode") != arm["residual_mode"]:
        raise ValueError("projector variant residual mode differs from the arm")
    if int(config.get("language_width", -1)) != int(
        contract["boundary"]["canonical_output_width"]
    ):
        raise ValueError("projector variant output width differs from canonical 4096")
    if config.get("output_norm", "none") != "none":
        raise ValueError("residual variant cannot combine output normalization")

    base = PatchMergerProjector.from_pretrained(base_dir, device="cpu")
    variant = PatchMergerProjector.from_pretrained(projector_dir, device="cpu")
    parameter_count = sum(parameter.numel() for parameter in variant.parameters())
    if parameter_count != int(arm["parameter_count"]):
        raise ValueError("projector variant parameter count differs from the arm")
    base_state = base.state_dict()
    variant_state = variant.state_dict()
    for key, value in base_state.items():
        if key not in variant_state or not value.equal(variant_state[key]):
            raise ValueError(f"projector variant changed base initialization: {key}")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260805)
        features = [
            torch.randn(2, int(config["merge_factor"]), int(config["vision_width"]))
        ]
    with torch.no_grad():
        base_output = base(features)[0]
        variant_output = variant(features)[0]
    if not torch.equal(base_output, variant_output):
        raise ValueError("projector variant step0 output differs from base")

    return ProjectorBinding(
        kind="registered_variant",
        weights_sha256=weights_sha,
        parameter_count=int(parameter_count),
        variant_contract_path=str(contract_path.resolve()),
        variant_contract_sha256=contract_sha,
        arm=arm_name,
        residual_mode=str(arm["residual_mode"]),
        config_sha256=config_sha,
        base_weights_sha256=base_weights_sha,
    )
