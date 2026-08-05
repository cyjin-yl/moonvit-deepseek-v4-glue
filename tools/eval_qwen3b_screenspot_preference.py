#!/usr/bin/env python3
"""用冻结 Qwen2.5-3B 合同评测 ScreenSpot teacher-forced 成对偏好。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import moonvit_glue.grounding_preference as grounding_preference_module
import moonvit_glue.paired_preference as paired_preference_module
from moonvit_glue import FeatureCache
from moonvit_glue.chat_contract import ChatSupervision, build_chat_supervision
from moonvit_glue.grounding_preference import (
    build_counterfactual_targets,
    make_preference_row,
    paired_preference_bootstrap,
    summarize_preference_rows,
)
from moonvit_glue.merge import expand_image_placeholders
from moonvit_glue.paired_preference import answer_logprob_stats
from moonvit_glue.projector import PatchMergerProjector
from moonvit_glue.proxy_receiver import FixedPairwiseReceiverAdapter
from moonvit_glue.screenspot_contract import verify_manifest
from moonvit_glue.screenspot_runtime import (
    shuffled_image_mapping,
    validate_screenspot_feature_cache,
)

from eval_qwen3b_screenspot import (
    _Tee,
    canonical_sha256,
    git_sha,
    git_tracked_worktree_clean,
    set_stage,
    verify_projector,
    write_json,
)
from train_qwen3b_proxy import sha256_file, verify_frozen_files
from verify_feature_cache import verify_feature_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
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
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260805)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def runtime_source_files() -> list[dict[str, Any]]:
    paths = (
        Path(__file__).resolve(),
        Path(grounding_preference_module.__file__).resolve(),
        Path(paired_preference_module.__file__).resolve(),
        Path(__file__).with_name("eval_qwen3b_screenspot.py").resolve(),
        Path(__file__).with_name("train_qwen3b_proxy.py").resolve(),
        Path(__file__).with_name("verify_feature_cache.py").resolve(),
    )
    return [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ]


def supervision_batch(
    supervisions: list[ChatSupervision],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 correct/counterfactual 两条 assistant supervision 组成真实 batch。"""

    if len(supervisions) != 2:
        raise ValueError("preference scoring requires exactly two candidates")
    max_length = max(len(row.input_ids) for row in supervisions)
    input_ids = torch.full(
        (2, max_length), int(pad_token_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for index, row in enumerate(supervisions):
        length = len(row.input_ids)
        input_ids[index, :length] = torch.tensor(
            row.input_ids, dtype=torch.long, device=device
        )
        attention_mask[index, :length] = 1
        labels[index, :length] = torch.tensor(
            row.labels, dtype=torch.long, device=device
        )
    return input_ids, attention_mask, labels


@torch.inference_mode()
def score_candidate_pair(
    *,
    language_model: torch.nn.Module,
    projector: PatchMergerProjector | None,
    receiver: FixedPairwiseReceiverAdapter | None,
    feature_groups: list[torch.Tensor] | None,
    supervisions: list[ChatSupervision],
    placeholder_token_id: int,
    pad_token_id: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """一次 batch=2 前向同时评分 correct 与 counterfactual 坐标答案。"""

    visual = feature_groups is not None
    if visual != (projector is not None and receiver is not None):
        raise ValueError("visual preference inputs and projector/receiver disagree")
    input_ids, attention_mask, labels = supervision_batch(
        supervisions, pad_token_id=pad_token_id, device=device
    )
    if not visual:
        outputs = language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return answer_logprob_stats(outputs.logits, labels)

    assert projector is not None and receiver is not None and feature_groups is not None
    canonical = projector(feature_groups)[0]
    receiver_embedding = receiver(canonical)
    image_embeddings = [receiver_embedding, receiver_embedding]
    text_embeddings = language_model.get_input_embeddings()(input_ids)
    merged = expand_image_placeholders(
        input_ids=input_ids,
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        placeholder_token_id=placeholder_token_id,
        attention_mask=attention_mask,
        labels=labels,
        pad_token_id=pad_token_id,
    )
    outputs = language_model(
        inputs_embeds=merged.inputs_embeds,
        attention_mask=merged.attention_mask,
        position_ids=merged.position_ids,
        use_cache=False,
    )
    if merged.labels is None:
        raise AssertionError("visual preference labels were not expanded")
    return answer_logprob_stats(outputs.logits, merged.labels)


def read_partial(
    path: Path, *, expected_ids: list[str], expected_condition: str
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    actual_ids = [str(row.get("sample_id")) for row in rows]
    if actual_ids != expected_ids[: len(rows)]:
        raise ValueError(f"partial preference rows are not an exact prefix: {path}")
    if any(str(row.get("condition")) != expected_condition for row in rows):
        raise ValueError(f"partial preference condition differs: {path}")
    return rows


def run_condition(
    *,
    name: str,
    samples: list[dict[str, Any]],
    shuffled: dict[str, str],
    targets: dict[str, dict[str, str]],
    supervisions: dict[str, dict[str, list[ChatSupervision]]],
    language_model: torch.nn.Module,
    projector: PatchMergerProjector | None,
    projector_sha256: str | None,
    receiver: FixedPairwiseReceiverAdapter,
    feature_cache: FeatureCache,
    placeholder_token_id: int,
    pad_token_id: int,
    device: torch.device,
    rows_dir: Path,
) -> dict[str, Any]:
    final_path = rows_dir / f"{name}.jsonl"
    partial_path = rows_dir / f"{name}.partial.jsonl"
    expected_ids = [str(sample["sample_id"]) for sample in samples]
    if final_path.exists():
        rows = read_partial(
            final_path, expected_ids=expected_ids, expected_condition=name
        )
        if len(rows) != len(samples):
            raise ValueError(f"completed preference count differs: {final_path}")
        return {"condition": name, "records": len(rows), "reused_complete": True}
    rows = read_partial(
        partial_path, expected_ids=expected_ids, expected_condition=name
    )
    started = time.perf_counter()
    with partial_path.open("a", encoding="utf-8") as stream:
        for index in range(len(rows), len(samples)):
            sample = samples[index]
            sample_id = str(sample["sample_id"])
            item_started = time.perf_counter()
            if name == "blind":
                input_image_id = None
                feature_groups = None
                selected_projector = None
                selected_receiver = None
                selected_supervisions = supervisions[sample_id]["blind"]
                visual_tokens = 0
            else:
                input_image_id = (
                    shuffled[sample_id] if name.endswith("_shuffled") else sample_id
                )
                feature_groups = feature_cache.get(
                    input_image_id, device=device, dtype=torch.float32
                )
                selected_projector = projector
                selected_receiver = receiver
                selected_supervisions = supervisions[sample_id]["vision"]
                visual_tokens = int(feature_groups[0].shape[0])
            correct_stats, counterfactual_stats = score_candidate_pair(
                language_model=language_model,
                projector=selected_projector,
                receiver=selected_receiver,
                feature_groups=feature_groups,
                supervisions=selected_supervisions,
                placeholder_token_id=placeholder_token_id,
                pad_token_id=pad_token_id,
                device=device,
            )
            target = targets[sample_id]
            row = make_preference_row(
                sample=sample,
                condition=name,
                input_image_sample_id=input_image_id,
                counterfactual_sample_id=target["counterfactual_sample_id"],
                correct_answer=target["correct_answer"],
                counterfactual_answer=target["counterfactual_answer"],
                correct_stats=correct_stats,
                counterfactual_stats=counterfactual_stats,
            )
            row.update(
                {
                    "evaluation_order": index,
                    "visual_tokens": visual_tokens,
                    "projector_sha256": projector_sha256,
                    "scoring_wall_seconds": time.perf_counter() - item_started,
                }
            )
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            rows.append(row)
            if (index + 1) % 5 == 0 or index + 1 == len(samples):
                preferred = sum(bool(value["correct_preferred"]) for value in rows)
                print(
                    f"{name} [{index + 1}/{len(samples)}] preferred={preferred}",
                    flush=True,
                )
    partial_path.replace(final_path)
    return {
        "condition": name,
        "records": len(rows),
        "preference_count": sum(bool(row["correct_preferred"]) for row in rows),
        "wall_seconds": time.perf_counter() - started,
        "reused_complete": False,
        "file": str(final_path),
        "file_sha256": sha256_file(final_path),
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run(args: argparse.Namespace, stage: dict[str, str]) -> dict[str, Any]:
    started = time.perf_counter()
    tracked_clean = git_tracked_worktree_clean()
    if not tracked_clean and not args.allow_dirty_development_run:
        raise RuntimeError("tracked Git worktree is dirty; formal preference run refused")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not verify_manifest(manifest):
        raise ValueError("ScreenSpot manifest self-hash verification failed")
    matching = [
        row
        for key, row in contract["datasets"].items()
        if key in ("screenspot_glm50", "screenspot_full")
        and row["name"] == manifest["name"]
        and row["manifest_sha256"] == manifest["manifest_sha256"]
    ]
    if len(matching) != 1:
        raise ValueError("ScreenSpot manifest is outside the fixed Qwen3B contract")
    all_samples = list(manifest["samples"])
    samples = all_samples
    if args.development_limit is not None:
        if not 0 < args.development_limit <= len(samples):
            raise ValueError("development limit falls outside the manifest")
        samples = samples[: args.development_limit]
    formal_run = (
        tracked_clean
        and not args.allow_dirty_development_run
        and args.development_limit is None
        and args.bootstrap_samples == 2_000
        and args.bootstrap_seed == 20260805
    )

    set_stage(stage, "cache_and_frozen_file_verification")
    cache_verification = verify_feature_cache(
        args.feature_cache,
        expected_count=len(all_samples),
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
        moonvit_weights_sha256=contract["vision_tower"]["extracted_weights_sha256"],
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
            directory,
            role=role,
            contract=contract,
            expected_training_runner_git_sha=args.expected_training_runner_git_sha,
        )
        for role, directory in (
            ("current_candidate", args.current_projector),
            ("step0", args.step0_projector),
            ("random_projector", args.random_projector),
        )
    }
    previous_sha = sha256_file(previous_dir / "projector.safetensors")
    step0_sha = projector_sources["step0"]["weights_sha256"]
    if previous_sha == step0_sha:
        projector_sources["previous_best"] = {
            **projector_sources["step0"],
            "role": "previous_best",
            "alias_of": "step0",
        }
    else:
        projector_sources["previous_best"] = verify_projector(
            previous_dir,
            role="previous_best",
            contract=contract,
            expected_training_runner_git_sha=args.expected_training_runner_git_sha,
        )
    targets = build_counterfactual_targets(manifest)
    target_binding_sha256 = canonical_sha256(targets)
    binding = {
        "format_version": "qwen3b-screenspot-teacher-forced-preference-v1",
        "runner_git_sha": git_sha(),
        "git_tracked_worktree_clean": tracked_clean,
        "formal_run": formal_run,
        "contract_file_sha256": sha256_file(args.contract),
        "dataset_name": manifest["name"],
        "dataset_manifest_file_sha256": sha256_file(args.manifest),
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "feature_cache_manifest_file_sha256": sha256_file(cache_manifest_path),
        "selected_records": len(samples),
        "projector_sources": projector_sources,
        "receiver_sha256": sha256_file(receiver_path),
        "model_files": model_files,
        "runtime_source_files": runtime_source_files(),
        "prompt_contract": contract["prompt_and_generation"],
        "target_rule": "rounded bbox center versus frozen shuffled-image sample rounded bbox center",
        "target_binding_sha256": target_binding_sha256,
        "score_rule": "token-normalized assistant answer plus im_end log probability; strict greater-than",
        "candidate_batch_size": 2,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "cache_binding": cache_binding,
        "cache_verification": cache_verification,
        "paid_resources_used": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    config_path = args.out / "RUN_CONFIG.json"
    if args.resume:
        if not config_path.is_file():
            raise FileNotFoundError("resume requires an existing RUN_CONFIG.json")
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("binding_sha256") != binding["binding_sha256"]:
            raise ValueError("resume preference binding differs")
    else:
        write_json(config_path, binding)

    set_stage(stage, "model_and_supervision_load")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import transformers

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Qwen3B preference evaluation requires the V100")
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
        raise ValueError("preference backbone is not the pinned pure-text Qwen model")
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
    supervisions: dict[str, dict[str, list[ChatSupervision]]] = {}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        target = targets[sample_id]
        user_prompt = prompt_contract["user_prompt"].format(
            instruction=sample["instruction"]
        )
        supervisions[sample_id] = {}
        for mode, include_image in (("vision", True), ("blind", False)):
            supervisions[sample_id][mode] = [
                build_chat_supervision(
                    tokenizer,
                    system_prompt=prompt_contract["system_prompt"],
                    user_prompt=user_prompt,
                    answer=answer,
                    placeholder_token_id=placeholder_id,
                    include_image=include_image,
                )
                for answer in (
                    target["correct_answer"],
                    target["counterfactual_answer"],
                )
            ]

    language_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    language_model.requires_grad_(False).eval()
    language_model.config.use_cache = False
    qwen_parameter_count = sum(parameter.numel() for parameter in language_model.parameters())
    qwen_dtypes = sorted({str(parameter.dtype) for parameter in language_model.parameters()})
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
    feature_cache = FeatureCache(args.feature_cache)
    shuffled = shuffled_image_mapping(manifest)
    rows_dir = args.out / "preferences"
    rows_dir.mkdir(exist_ok=True)

    set_stage(stage, "teacher_forced_scoring")
    condition_specs: list[tuple[str, PatchMergerProjector | None, str | None]] = [
        ("blind", None, None),
        (
            "current_candidate",
            projectors["current_candidate"],
            projector_sources["current_candidate"]["weights_sha256"],
        ),
        (
            "current_candidate_shuffled",
            projectors["current_candidate"],
            projector_sources["current_candidate"]["weights_sha256"],
        ),
        ("step0", projectors["step0"], step0_sha),
        ("step0_shuffled", projectors["step0"], step0_sha),
        (
            "random_projector",
            projectors["random_projector"],
            projector_sources["random_projector"]["weights_sha256"],
        ),
        (
            "random_projector_shuffled",
            projectors["random_projector"],
            projector_sources["random_projector"]["weights_sha256"],
        ),
    ]
    if previous_sha != step0_sha:
        condition_specs.extend(
            [
                ("previous_best", projectors["previous_best"], previous_sha),
                (
                    "previous_best_shuffled",
                    projectors["previous_best"],
                    previous_sha,
                ),
            ]
        )
    condition_summaries = []
    for name, projector, projector_sha in condition_specs:
        condition_summaries.append(
            run_condition(
                name=name,
                samples=samples,
                shuffled=shuffled,
                targets=targets,
                supervisions=supervisions,
                language_model=language_model,
                projector=projector,
                projector_sha256=projector_sha,
                receiver=receiver,
                feature_cache=feature_cache,
                placeholder_token_id=placeholder_id,
                pad_token_id=int(tokenizer.pad_token_id),
                device=device,
                rows_dir=rows_dir,
            )
        )

    aliases = {
        "vision": "current_candidate",
        "shuffled": "current_candidate_shuffled",
    }
    if previous_sha == step0_sha:
        aliases.update(
            {
                "previous_best": "step0",
                "previous_best_shuffled": "step0_shuffled",
            }
        )
    for destination, source in aliases.items():
        destination_path = rows_dir / f"{destination}.jsonl"
        source_path = rows_dir / f"{source}.jsonl"
        if destination_path.exists():
            if sha256_file(destination_path) != sha256_file(source_path):
                raise ValueError(f"preference alias differs: {destination}")
        else:
            shutil.copyfile(source_path, destination_path)

    set_stage(stage, "paired_bootstrap")
    rows_by_condition = {
        path.stem: load_rows(path)
        for path in sorted(rows_dir.glob("*.jsonl"))
        if ".partial" not in path.name
    }
    flat_rows = [row for rows in rows_by_condition.values() for row in rows]
    preference_summary = summarize_preference_rows(flat_rows)
    comparison_specs = {
        "vision-minus-blind": ("vision", "blind"),
        "vision-minus-shuffled": ("vision", "shuffled"),
        "trained-minus-random-projector": ("vision", "random_projector"),
        "current-candidate-minus-previous-best": ("current_candidate", "previous_best"),
        "step0-minus-blind": ("step0", "blind"),
        "step0-minus-step0-shuffled": ("step0", "step0_shuffled"),
    }
    comparisons = {
        name: paired_preference_bootstrap(
            rows_by_condition[first],
            rows_by_condition[second],
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        for name, (first, second) in comparison_specs.items()
    }

    set_stage(stage, "complete")
    files = {
        name: {
            "path": str(rows_dir / f"{name}.jsonl"),
            "bytes": (rows_dir / f"{name}.jsonl").stat().st_size,
            "sha256": sha256_file(rows_dir / f"{name}.jsonl"),
        }
        for name in rows_by_condition
    }
    summary = {
        "status": "valid" if formal_run else "development_only",
        "formal_preference_complete": formal_run,
        "capability_claim_allowed": False,
        "dataset_name": manifest["name"],
        "records": len(samples),
        "condition_summaries": condition_summaries,
        "condition_aliases": aliases,
        "preference_files": files,
        "preference_summary": preference_summary,
        "comparisons": comparisons,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
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
        raise FileExistsError(f"refusing to overwrite preference output: {args.out}")
    if not args.out.exists():
        args.out.mkdir(parents=True)
    log_handle = (args.out / "run.log").open(
        "a" if args.resume else "w", encoding="utf-8"
    )
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
