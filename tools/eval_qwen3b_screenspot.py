#!/usr/bin/env python3
"""按冻结合同生成 Qwen2.5-3B ScreenSpot 七条件预测。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import torch

import moonvit_glue.chat_contract as chat_contract_module
import moonvit_glue.model as model_module
import moonvit_glue.projector as projector_module
import moonvit_glue.proxy_receiver as proxy_receiver_module
import moonvit_glue.screenspot_runtime as screenspot_runtime_module
from moonvit_glue import FeatureCache
from moonvit_glue.chat_contract import build_chat_prompt
from moonvit_glue.grounding_contract import parse_click_action
from moonvit_glue.model import VisionCausalLM
from moonvit_glue.projector import PatchMergerProjector
from moonvit_glue.proxy_receiver import FixedPairwiseReceiverAdapter
from moonvit_glue.screenspot_contract import verify_manifest
from moonvit_glue.screenspot_runtime import (
    shuffled_image_mapping,
    validate_screenspot_feature_cache,
)
from train_qwen3b_proxy import (
    checkpoint_files,
    sha256_file,
    verify_bound_checkpoint,
    verify_frozen_files,
)
from verify_feature_cache import verify_feature_cache


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = list(streams)

    def write(self, text: str) -> int:
        alive = []
        for stream in self.streams:
            try:
                stream.write(text)
                stream.flush()
                alive.append(stream)
            except (BrokenPipeError, OSError):
                continue
        self.streams = alive
        return len(text)

    def flush(self) -> None:
        for stream in list(self.streams):
            try:
                stream.flush()
            except (BrokenPipeError, OSError):
                self.streams.remove(stream)


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def set_stage(stage: dict[str, str], name: str) -> None:
    stage["name"] = name
    print(f"stage: {name}", flush=True)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def git_tracked_worktree_clean() -> bool:
    unstaged = subprocess.run(["git", "diff", "--quiet", "--"], check=False)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--"], check=False
    )
    return unstaged.returncode == 0 and staged.returncode == 0


def load_architecture_spec(
    *,
    contract: dict[str, Any],
    contract_path: Path,
    architecture_control_path: Path | None,
    architecture_arm: str | None,
) -> dict[str, Any]:
    """Resolve the optional V1/V2 sidecar while preserving the V2 default."""

    root = Path(__file__).resolve().parents[1]
    if architecture_control_path is None:
        vision = contract["vision_tower"]
        projector = contract["canonical_projector"]
        return {
            "source": None,
            "source_sha256": None,
            "arm": None,
            "vision": {
                "name": vision.get("name", "MoonViT-V2"),
                "cache_tower_id": "v2",
                "model": vision.get("source_repo"),
                "revision": vision.get("source_resolved_revision"),
                "vision_width": int(vision.get("vision_width", 1024)),
                "merge_factor": int(vision.get("merge_factor", 4)),
                "weights_sha256": vision.get("extracted_weights_sha256"),
                "require_tower_identity": False,
            },
            "projector": {
                "config_path": contract["canonical_projector"].get("config"),
                "config_sha256": projector.get("config_sha256"),
                "variant": projector.get("projector_variant", "legacy_pre_norm"),
                "output_width": int(projector["output_width"]),
                "parameter_count": int(projector["parameter_count"]),
                "initialization": projector.get("initialization_contract", {}),
            },
        }

    sidecar = json.loads(architecture_control_path.read_text(encoding="utf-8"))
    if architecture_arm is None:
        raise ValueError("--architecture-arm is required with --architecture-control")
    arms = sidecar.get("arms")
    if not isinstance(arms, dict) or architecture_arm not in arms:
        raise ValueError(f"architecture arm is absent: {architecture_arm}")
    base = sidecar.get("base_contract")
    if isinstance(base, dict):
        raw_base_path = Path(str(base.get("path", "")))
        base_path = raw_base_path if raw_base_path.is_absolute() else root / raw_base_path
        if not base_path.is_file():
            candidate = architecture_control_path.parent / raw_base_path
            if candidate.is_file():
                base_path = candidate
        if not base_path.is_file():
            raise FileNotFoundError(f"architecture base contract is absent: {base_path}")
        expected_sha = base.get("sha256")
        if expected_sha and sha256_file(base_path) != str(expected_sha):
            raise ValueError("architecture base contract SHA-256 differs")
        if base_path.resolve() != contract_path.resolve():
            raise ValueError("architecture sidecar is bound to a different core contract")
    arm = arms[architecture_arm]
    vision = arm.get("vision_tower")
    projector = arm.get("projector")
    if not isinstance(vision, dict) or not isinstance(projector, dict):
        raise ValueError("architecture arm requires vision_tower and projector objects")
    required_vision = ("name", "vision_width", "merge_factor")
    if any(key not in vision for key in required_vision):
        raise ValueError("architecture arm vision_tower is missing interface fields")
    required_projector = ("config_path", "config_sha256", "output_width", "parameter_count")
    if any(key not in projector for key in required_projector):
        raise ValueError("architecture arm projector is missing binding fields")
    return {
        "source": str(architecture_control_path.resolve()),
        "source_sha256": sha256_file(architecture_control_path),
        "arm": architecture_arm,
        "vision": {
            **vision,
            "cache_tower_id": vision.get(
                "cache_tower_id",
                "v1"
                if "MoonViT-SO-400M" in str(vision.get("model", ""))
                else "v2",
            ),
            "vision_width": int(vision["vision_width"]),
            "merge_factor": int(vision["merge_factor"]),
            "revision": vision.get("revision", vision.get("resolved_revision")),
            "require_tower_identity": bool(vision.get("require_tower_identity", True)),
        },
        "projector": {
            **projector,
            "output_width": int(projector["output_width"]),
            "parameter_count": int(projector["parameter_count"]),
            "initialization": projector.get("initialization", {}),
        },
    }


def runtime_source_files() -> list[dict[str, Any]]:
    paths = (
        Path(__file__).resolve(),
        Path(chat_contract_module.__file__).resolve(),
        Path(model_module.__file__).resolve(),
        Path(projector_module.__file__).resolve(),
        Path(proxy_receiver_module.__file__).resolve(),
        Path(screenspot_runtime_module.__file__).resolve(),
        Path(__file__).with_name("train_qwen3b_proxy.py").resolve(),
        Path(__file__).with_name("verify_feature_cache.py").resolve(),
    )
    return [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ]


def verify_projector(
    directory: Path,
    *,
    role: str,
    contract: dict[str, Any],
    expected_training_runner_git_sha: str,
    architecture_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = directory / "projector_config.json"
    weights_path = directory / "projector.safetensors"
    projector_spec = (
        architecture_spec["projector"]
        if architecture_spec is not None
        else contract["canonical_projector"]
    )
    expected_config_sha = projector_spec.get("config_sha256")
    if expected_config_sha and sha256_file(config_path) != str(expected_config_sha):
        raise ValueError(f"{role} projector config differs from the contract")
    weights_sha = sha256_file(weights_path)
    initialization = projector_spec.get(
        "initialization",
        projector_spec.get("initialization_contract", {}),
    )
    if role in ("step0", "random_projector"):
        expected = initialization.get(role, {}).get("weights_sha256")
        if expected is not None and weights_sha != str(expected):
            raise ValueError(f"{role} projector weights differ from the contract")

    checkpoint_manifest = None
    manifest_path = directory / "CHECKPOINT_MANIFEST.json"
    if manifest_path.is_file():
        checkpoint_manifest = verify_bound_checkpoint(directory, expected_binding={})
        if role == "current_candidate":
            if (
                int(checkpoint_manifest.get("step", -1)) != 500
                or int(checkpoint_manifest.get("progress", {}).get("examples_seen", -1))
                != 4000
                or checkpoint_manifest.get("runner_git_sha")
                != expected_training_runner_git_sha
            ):
                raise ValueError("current candidate is not the bound formal 4k checkpoint")
    elif role == "current_candidate":
        raise ValueError("current candidate checkpoint manifest is absent")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("language_width", -1)) != int(projector_spec["output_width"]):
        raise ValueError(
            f"{role} projector output width differs from the architecture contract"
        )
    from safetensors.torch import load_file

    state = load_file(str(weights_path), device="cpu")
    parameter_count = int(sum(int(value.numel()) for value in state.values()))
    del state
    if parameter_count != int(projector_spec["parameter_count"]):
        raise ValueError(f"{role} projector parameter count differs from the architecture contract")
    variant = str(config.get("projector_variant", "legacy_pre_norm"))
    expected_variant = projector_spec.get("variant")
    if expected_variant is not None and variant != str(expected_variant):
        raise ValueError(f"{role} projector variant differs from the architecture contract")
    return {
        "role": role,
        "directory": str(directory.resolve()),
        "config_sha256": sha256_file(config_path),
        "weights_bytes": weights_path.stat().st_size,
        "weights_sha256": weights_sha,
        "checkpoint_manifest_sha256": (
            sha256_file(manifest_path) if manifest_path.is_file() else None
        ),
        "checkpoint_step": (
            int(checkpoint_manifest["step"]) if checkpoint_manifest else 0
        ),
        "projector_variant": variant,
        "parameter_count": parameter_count,
    }


def decode_continuation(
    tokenizer: Any, generated: torch.Tensor, prefix_length: int
) -> tuple[list[int], str]:
    if generated.ndim != 2 or generated.shape[0] != 1:
        raise ValueError(f"unexpected generated shape: {tuple(generated.shape)}")
    if generated.shape[1] < prefix_length:
        raise ValueError("generation is shorter than its expanded prefix")
    token_ids = [int(value) for value in generated[0, prefix_length:].tolist()]
    return token_ids, tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def read_partial(path: Path, expected_ids: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    actual = [row.get("sample_id") for row in rows]
    if actual != expected_ids[: len(rows)]:
        raise ValueError(f"partial predictions are not an exact manifest prefix: {path}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--architecture-control",
        type=Path,
        help="Optional sidecar V1/V2 architecture-control contract",
    )
    parser.add_argument(
        "--architecture-arm",
        help="Arm name in --architecture-control (for example v1_community or v2_k3_exact)",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--expected-cache-runner-git-sha", required=True)
    parser.add_argument("--expected-training-runner-git-sha", required=True)
    parser.add_argument("--receiver-dir", type=Path, required=True)
    parser.add_argument("--current-projector", type=Path, required=True)
    parser.add_argument("--step0-projector", type=Path, required=True)
    parser.add_argument("--random-projector", type=Path, required=True)
    parser.add_argument("--previous-projector", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--development-limit", type=int)
    parser.add_argument("--allow-dirty-development-run", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run_generation_condition(
    *,
    name: str,
    samples: list[dict[str, Any]],
    shuffled: dict[str, str],
    prompts: dict[str, Any],
    tokenizer: Any,
    language_model: torch.nn.Module,
    model: VisionCausalLM,
    projector: PatchMergerProjector | None,
    projector_sha256: str | None,
    feature_cache: FeatureCache,
    device: torch.device,
    generation_kwargs: dict[str, Any],
    predictions_dir: Path,
) -> dict[str, Any]:
    final_path = predictions_dir / f"{name}.jsonl"
    partial_path = predictions_dir / f"{name}.partial.jsonl"
    expected_ids = [str(row["sample_id"]) for row in samples]
    if final_path.exists():
        rows = read_partial(final_path, expected_ids)
        if len(rows) != len(samples):
            raise ValueError(f"completed prediction count differs: {final_path}")
        return {"condition": name, "records": len(rows), "reused_complete": True}
    rows = read_partial(partial_path, expected_ids)
    parsed = sum(parse_click_action(str(row["prediction"])) is not None for row in rows)
    started = time.perf_counter()
    with partial_path.open("a", encoding="utf-8") as stream:
        for index in range(len(rows), len(samples)):
            sample = samples[index]
            sample_id = str(sample["sample_id"])
            item_started = time.perf_counter()
            if name == "blind":
                prompt = prompts[sample_id]["blind"]
                input_ids = torch.tensor(
                    [prompt.input_ids], dtype=torch.long, device=device
                )
                attention_mask = torch.ones_like(input_ids)
                generated = language_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )
                prefix_length = len(prompt.input_ids)
                input_image_id = None
                visual_tokens = 0
            else:
                if projector is None:
                    raise ValueError(f"visual condition has no projector: {name}")
                input_image_id = (
                    shuffled[sample_id] if name == "shuffled" else sample_id
                )
                feature_groups = feature_cache.get(
                    input_image_id, device=device, dtype=torch.float32
                )
                prompt = prompts[sample_id]["vision"]
                input_ids = torch.tensor(
                    [prompt.input_ids], dtype=torch.long, device=device
                )
                attention_mask = torch.ones_like(input_ids)
                canonical_embeddings = projector(feature_groups)
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_embeddings=canonical_embeddings,
                    **generation_kwargs,
                )
                visual_tokens = int(feature_groups[0].shape[0])
                prefix_length = len(prompt.input_ids) - 1 + visual_tokens
            token_ids, prediction = decode_continuation(
                tokenizer, generated, prefix_length
            )
            parse_result = parse_click_action(prediction)
            parsed += int(parse_result is not None)
            row = {
                "evaluation_order": index,
                "sample_id": sample_id,
                "prediction": prediction,
                "continuation_token_ids": token_ids,
                "parse_result": parse_result,
                "input_image_sample_id": input_image_id,
                "visual_tokens": visual_tokens,
                "projector_sha256": projector_sha256,
                "generation_wall_seconds": time.perf_counter() - item_started,
            }
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            del generated, input_ids, attention_mask
            if name != "blind":
                del feature_groups, canonical_embeddings
            if (index + 1) % 5 == 0 or index + 1 == len(samples):
                print(
                    f"{name} [{index + 1}/{len(samples)}] parsed={parsed}",
                    flush=True,
                )
    partial_path.replace(final_path)
    return {
        "condition": name,
        "records": len(samples),
        "parse_count": parsed,
        "wall_seconds": time.perf_counter() - started,
        "reused_complete": False,
        "file": str(final_path),
        "file_sha256": sha256_file(final_path),
    }


def _run(args: argparse.Namespace, stage: dict[str, str]) -> dict[str, Any]:
    started = time.perf_counter()
    tracked_clean = git_tracked_worktree_clean()
    if not tracked_clean and not args.allow_dirty_development_run:
        raise RuntimeError("tracked Git worktree is dirty; formal evaluation is refused")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    architecture_spec = load_architecture_spec(
        contract=contract,
        contract_path=args.contract,
        architecture_control_path=args.architecture_control,
        architecture_arm=args.architecture_arm,
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not verify_manifest(manifest):
        raise ValueError("ScreenSpot manifest self-hash verification failed")
    matching_datasets = [
        row
        for key, row in contract["datasets"].items()
        if key in ("screenspot_glm50", "screenspot_full")
        and row["name"] == manifest["name"]
        and row["manifest_sha256"] == manifest["manifest_sha256"]
    ]
    if len(matching_datasets) != 1:
        raise ValueError("ScreenSpot manifest is outside the frozen Qwen3B contract")
    samples = list(manifest["samples"])
    if args.development_limit is not None:
        if args.development_limit <= 0 or args.development_limit > len(samples):
            raise ValueError("development limit falls outside the manifest")
        samples = samples[: args.development_limit]
    formal_run = (
        tracked_clean
        and not args.allow_dirty_development_run
        and args.development_limit is None
    )

    set_stage(stage, "cache_and_frozen_file_verification")
    cache_verification = verify_feature_cache(
        args.feature_cache,
        expected_count=len(manifest["samples"]),
        expected_git_sha=args.expected_cache_runner_git_sha,
    )
    cache_manifest_path = args.feature_cache / "MANIFEST.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    preprocessing = contract["image_preprocessing"]
    cache_binding = validate_screenspot_feature_cache(
        manifest,
        cache_manifest,
        dataset_manifest_file_sha256=sha256_file(args.manifest),
        max_image_side=int(preprocessing["eval_max_image_side"]),
        max_visual_tokens=int(preprocessing["eval_max_visual_tokens"]),
        moonvit_weights_sha256=architecture_spec["vision"].get("weights_sha256"),
        vision_width=int(architecture_spec["vision"]["vision_width"]),
        merge_factor=int(architecture_spec["vision"]["merge_factor"]),
        vision_tower=architecture_spec["vision"].get("cache_tower_id"),
        moonvit_model=architecture_spec["vision"].get("model"),
        moonvit_revision=architecture_spec["vision"].get("revision"),
        require_tower_identity=bool(
            architecture_spec["vision"].get("require_tower_identity", False)
        ),
    )
    model_files = verify_frozen_files(
        args.model_dir, contract["proxy_model"]["files"], label="Qwen contract"
    )
    receiver_path = args.receiver_dir / "proxy_receiver.safetensors"
    if sha256_file(receiver_path) != contract["qwen_proxy_receiver"]["buffer_sha256"]:
        raise ValueError("proxy receiver weights differ from the contract")
    previous_dir = args.previous_projector or args.step0_projector
    projector_sources = {
        role: verify_projector(
            path,
            role=role,
            contract=contract,
            expected_training_runner_git_sha=args.expected_training_runner_git_sha,
            architecture_spec=architecture_spec,
        )
        for role, path in (
            ("current_candidate", args.current_projector),
            ("step0", args.step0_projector),
            ("random_projector", args.random_projector),
        )
    }
    previous_sha = sha256_file(previous_dir / "projector.safetensors")
    step0_sha = projector_sources["step0"]["weights_sha256"]
    if previous_sha != step0_sha:
        projector_sources["previous_best"] = verify_projector(
            previous_dir,
            role="previous_best",
            contract=contract,
            expected_training_runner_git_sha=args.expected_training_runner_git_sha,
            architecture_spec=architecture_spec,
        )
    else:
        projector_sources["previous_best"] = {
            **projector_sources["step0"],
            "role": "previous_best",
            "alias_of": "step0",
        }

    binding = {
        "format_version": "qwen3b-screenspot-generation-v1",
        "runner_git_sha": git_sha(),
        "git_tracked_worktree_clean": tracked_clean,
        "formal_run": formal_run,
        "contract_file_sha256": sha256_file(args.contract),
        "architecture_control": {
            "path": architecture_spec["source"],
            "sha256": architecture_spec["source_sha256"],
            "arm": architecture_spec["arm"],
        },
        "vision_tower_binding": architecture_spec["vision"],
        "projector_binding": architecture_spec["projector"],
        "dataset_name": manifest["name"],
        "dataset_manifest_file_sha256": sha256_file(args.manifest),
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "feature_cache_manifest_file_sha256": sha256_file(cache_manifest_path),
        "selected_records": len(samples),
        "projector_sources": projector_sources,
        "receiver_sha256": sha256_file(receiver_path),
        "model_files": model_files,
        "runtime_source_files": runtime_source_files(),
        "prompt_and_generation": contract["prompt_and_generation"],
        "cache_binding": cache_binding,
        "cache_verification": cache_verification,
        "paid_resources_used": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    config_path = args.out / "RUN_CONFIG.json"
    if args.resume:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("binding_sha256") != binding["binding_sha256"]:
            raise ValueError("resume evaluation binding differs")
    else:
        write_json(config_path, binding)

    set_stage(stage, "model_and_prompt_load")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import transformers

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Qwen3B ScreenSpot evaluation requires the V100")
    torch.manual_seed(20260805)
    torch.cuda.manual_seed_all(20260805)
    torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    model_config = AutoConfig.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False
    )
    if (
        model_config.architectures != [contract["proxy_model"]["architecture"]]
        or hasattr(model_config, "vision_config")
        or int(model_config.hidden_size) != int(contract["proxy_model"]["hidden_size"])
    ):
        raise ValueError("evaluation backbone is not the pinned pure-text Qwen model")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False
    )
    prompt_contract = contract["prompt_and_generation"]
    placeholder_id = int(prompt_contract["image_placeholder_token_id"])
    if tokenizer.convert_tokens_to_ids(prompt_contract["image_placeholder_token"]) != placeholder_id:
        raise ValueError("Qwen image placeholder differs from the contract")
    if int(tokenizer.eos_token_id) != int(prompt_contract["eos_token_id"]):
        raise ValueError("Qwen EOS token differs from the contract")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    chat_template_sha = hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()
    if chat_template_sha != contract["proxy_model"]["chat_template_sha256"]:
        raise ValueError("Qwen chat template differs from the contract")
    prompts = {}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        user_prompt = prompt_contract["user_prompt"].format(
            instruction=sample["instruction"]
        )
        prompts[sample_id] = {
            "vision": build_chat_prompt(
                tokenizer,
                system_prompt=prompt_contract["system_prompt"],
                user_prompt=user_prompt,
                placeholder_token_id=placeholder_id,
                include_image=True,
            ),
            "blind": build_chat_prompt(
                tokenizer,
                system_prompt=prompt_contract["system_prompt"],
                user_prompt=user_prompt,
                placeholder_token_id=placeholder_id,
                include_image=False,
            ),
        }

    language_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    language_model.requires_grad_(False).eval()
    language_model.config.use_cache = True
    qwen_parameter_count = sum(
        parameter.numel() for parameter in language_model.parameters()
    )
    qwen_dtypes = sorted(
        {str(parameter.dtype) for parameter in language_model.parameters()}
    )
    if qwen_parameter_count != int(contract["proxy_model"]["parameter_count_bf16"]):
        raise ValueError("loaded Qwen parameter count differs from the contract")
    if qwen_dtypes != ["torch.float16"]:
        raise ValueError(f"loaded Qwen runtime dtype differs: {qwen_dtypes}")
    projectors = {
        "current_candidate": PatchMergerProjector.from_pretrained(
            args.current_projector, device=device, dtype=torch.float32
        ).eval(),
        "step0": PatchMergerProjector.from_pretrained(
            args.step0_projector, device=device, dtype=torch.float32
        ).eval(),
        "random_projector": PatchMergerProjector.from_pretrained(
            args.random_projector, device=device, dtype=torch.float32
        ).eval(),
    }
    if previous_sha != step0_sha:
        projectors["previous_best"] = PatchMergerProjector.from_pretrained(
            previous_dir, device=device, dtype=torch.float32
        ).eval()
    receiver = FixedPairwiseReceiverAdapter.from_pretrained(
        args.receiver_dir, device=device
    ).eval()
    model = VisionCausalLM(
        language_model=language_model,
        projector=projectors["current_candidate"],
        receiver_adapter=receiver,
        placeholder_token_id=placeholder_id,
        backbone_kind="generic",
        freeze_language_model=True,
        pad_token_id=int(tokenizer.pad_token_id),
    ).to(device).eval()
    feature_cache = FeatureCache(args.feature_cache)
    generation_kwargs = {
        "do_sample": False,
        "max_new_tokens": int(prompt_contract["max_new_tokens"]),
        "eos_token_id": int(prompt_contract["eos_token_id"]),
        "pad_token_id": int(tokenizer.pad_token_id),
        "use_cache": True,
    }
    shuffled = shuffled_image_mapping(manifest)
    predictions_dir = args.out / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    set_stage(stage, "condition_generation")
    summaries = []
    condition_specs = [
        ("blind", None, None),
        (
            "current_candidate",
            projectors["current_candidate"],
            projector_sources["current_candidate"]["weights_sha256"],
        ),
        (
            "shuffled",
            projectors["current_candidate"],
            projector_sources["current_candidate"]["weights_sha256"],
        ),
        ("step0", projectors["step0"], step0_sha),
        (
            "random_projector",
            projectors["random_projector"],
            projector_sources["random_projector"]["weights_sha256"],
        ),
    ]
    if previous_sha != step0_sha:
        condition_specs.append(
            ("previous_best", projectors["previous_best"], previous_sha)
        )
    for name, projector, projector_sha in condition_specs:
        summaries.append(
            run_generation_condition(
                name=name,
                samples=samples,
                shuffled=shuffled,
                prompts=prompts,
                tokenizer=tokenizer,
                language_model=language_model,
                model=model,
                projector=projector,
                projector_sha256=projector_sha,
                feature_cache=feature_cache,
                device=device,
                generation_kwargs=generation_kwargs,
                predictions_dir=predictions_dir,
            )
        )

    aliases = {"vision": "current_candidate"}
    if previous_sha == step0_sha:
        aliases["previous_best"] = "step0"
    for destination, source in aliases.items():
        destination_path = predictions_dir / f"{destination}.jsonl"
        source_path = predictions_dir / f"{source}.jsonl"
        if destination_path.exists():
            if sha256_file(destination_path) != sha256_file(source_path):
                raise ValueError(f"condition alias differs: {destination}")
        else:
            shutil.copyfile(source_path, destination_path)

    set_stage(stage, "complete")
    prediction_files = {
        path.stem: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(predictions_dir.glob("*.jsonl"))
        if ".partial" not in path.name
    }
    summary = {
        "status": "valid" if formal_run else "development_only",
        "formal_generation_complete": formal_run,
        "capability_claim_allowed_before_scoring": False,
        "dataset_name": manifest["name"],
        "architecture_control": {
            "path": architecture_spec["source"],
            "sha256": architecture_spec["source_sha256"],
            "arm": architecture_spec["arm"],
            "vision": architecture_spec["vision"],
            "projector": architecture_spec["projector"],
        },
        "records": len(samples),
        "condition_summaries": summaries,
        "condition_aliases": aliases,
        "prediction_files": prediction_files,
        "transformers_version": transformers.__version__,
        "gpu_name": torch.cuda.get_device_name(device),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "qwen_parameter_count": qwen_parameter_count,
        "qwen_runtime_dtypes": qwen_dtypes,
        "qwen_trainable_parameter_count": sum(
            parameter.numel()
            for parameter in language_model.parameters()
            if parameter.requires_grad
        ),
        "total_wall_seconds": time.perf_counter() - started,
        "paid_resources_used": False,
        "transfer_label": "directly_transferable",
    }
    (args.out / "FAILURE.json").unlink(missing_ok=True)
    write_json(args.out / "SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    if args.out.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite ScreenSpot evaluation: {args.out}")
    if not args.out.exists():
        args.out.mkdir(parents=True)
    log_handle = (args.out / "run.log").open("a" if args.resume else "w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _Tee(original_stdout, log_handle)
    sys.stderr = _Tee(original_stderr, log_handle)
    stage = {"name": "initialization"}
    try:
        _run(args, stage)
    except BaseException as exc:
        failure = {
            "status": "failed",
            "stage": stage["name"],
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "capability_claim_allowed": False,
            "paid_resources_used": False,
        }
        write_json(args.out / "FAILURE.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    main()
