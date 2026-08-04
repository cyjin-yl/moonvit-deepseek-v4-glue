#!/usr/bin/env python3
"""在 paired-counterfactual 运行中逐层替换正确图激活并测量答案 margin 恢复。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file

from extract_layerwise_representations import (
    answer_token_ids,
    load_checkpoint_state,
    matched_random_state,
    pair_index,
    visual_prompt_batch,
)
from moonvit_glue import FeatureCache, PatchMergerProjector, ProjectorConfig
from moonvit_glue.mechanism_probe import (
    last_active_indices,
    pair_bootstrap_mean,
    patch_hidden_output,
    square_grid_region_mask,
)
from moonvit_glue.merge import expand_image_placeholders
from tools_common import load_records, validate_text_only_backbone_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--checkpoint-ids", nargs="*", default=None)
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--scan-layers", nargs="*", type=int, default=None)
    return parser.parse_args()


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def read_ids(path: Path, all_records: list[dict]) -> list[dict]:
    requested = [str(row["id"]) for row in json.loads(path.read_text(encoding="utf-8"))["records"]]
    by_id = {str(record["id"]): record for record in all_records}
    missing = [sample_id for sample_id in requested if sample_id not in by_id]
    if missing:
        raise ValueError(f"patching selection IDs are missing: {missing[:3]}")
    return [by_id[sample_id] for sample_id in requested]


def wrong_label_donors(records: list[dict], seed: int) -> dict[str, str]:
    """为每个目标选一个不同 pair、不同答案的确定性 donor。"""
    result = {}
    for target in records:
        candidates = [
            source
            for source in records
            if source["answers"][0] != target["answers"][0]
            and source["pair_id"] != target["pair_id"]
        ]
        if not candidates:
            raise ValueError("wrong-label activation control has no donor")
        donor = min(
            candidates,
            key=lambda source: hashlib.sha256(
                f"{seed}:{target['id']}:{source['id']}".encode()
            ).hexdigest(),
        )
        result[str(target["id"])] = str(donor["id"])
    return result


def projected_groups(
    records: list[dict], source_ids: list[str], cache: FeatureCache, projector, *, device, dtype
) -> list[torch.Tensor]:
    if len(records) != len(source_ids):
        raise ValueError("patching records/source IDs disagree")
    groups = [cache.get(source_id, device=device, dtype=dtype)[0] for source_id in source_ids]
    return projector(groups)


def prepare_inputs(
    *, language_model, tokenizer, records, image_embeddings, placeholder, prompt_template, device
):
    ids, mask = visual_prompt_batch(
        tokenizer,
        prompt_template,
        [str(record["question"]) for record in records],
        placeholder,
        device,
    )
    text = language_model.get_input_embeddings()(ids)
    return expand_image_placeholders(
        input_ids=ids,
        text_embeddings=text,
        image_embeddings=image_embeddings,
        placeholder_token_id=placeholder,
        attention_mask=mask,
        pad_token_id=int(tokenizer.pad_token_id),
    )


def language_forward(language_model, merged, *, hidden_states: bool):
    return language_model(
        inputs_embeds=merged.inputs_embeds,
        attention_mask=merged.attention_mask,
        position_ids=merged.position_ids,
        use_cache=False,
        output_hidden_states=hidden_states,
        return_dict=True,
    )


def captured_layer_forward(language_model, layers, merged):
    """直接捕获每个 decoder layer 的 hook 输出，避免末层 final-norm 空间错位。"""
    captured: list[torch.Tensor | None] = [None] * len(layers)
    handles = []
    for layer_index, layer in enumerate(layers):
        def hook(_module, _inputs, output, *, index=layer_index):
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                raise ValueError("decoder layer hook did not return a hidden tensor")
            captured[index] = hidden.detach()

        handles.append(layer.register_forward_hook(hook))
    try:
        outputs = language_forward(language_model, merged, hidden_states=False)
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in captured):
        raise ValueError("decoder layer hook missed an activation")
    return outputs, [value for value in captured if value is not None]


def margins(outputs, merged, class_tokens, target_labels, counter_labels) -> torch.Tensor:
    last = last_active_indices(merged.attention_mask)
    batch = torch.arange(last.numel(), device=last.device)
    logits = outputs.logits[batch, last][:, class_tokens]
    return logits.gather(1, target_labels[:, None]).squeeze(1) - logits.gather(
        1, counter_labels[:, None]
    ).squeeze(1)


def patched_forward(
    *, language_model, layers, merged, layer_index: int, donor, token_mask, class_tokens,
    target_labels, counter_labels
):
    def hook(_module, _inputs, output):
        return patch_hidden_output(output, donor, token_mask)

    handle = layers[layer_index].register_forward_hook(hook)
    try:
        outputs = language_forward(language_model, merged, hidden_states=False)
    finally:
        handle.remove()
    return margins(outputs, merged, class_tokens, target_labels, counter_labels)


def blend_regions(clean: list[torch.Tensor], counter: list[torch.Tensor], region: str) -> list[torch.Tensor]:
    blended = []
    for clean_row, counter_row in zip(clean, counter, strict=True):
        if clean_row.shape != counter_row.shape:
            raise ValueError("clean/counter projector token shapes disagree")
        mask = square_grid_region_mask(clean_row.shape[0], region).to(clean_row.device)
        value = counter_row.clone()
        value[mask] = clean_row[mask]
        blended.append(value)
    return blended


def append_rows(
    stream,
    *,
    checkpoint: str,
    records: list[dict],
    mate: dict[str, str],
    wrong_sources: list[str],
    intervention: str,
    layer_index: int,
    clean_margin: torch.Tensor,
    counter_margin: torch.Tensor,
    wrong_margin: torch.Tensor,
    patched_margin: torch.Tensor,
) -> list[dict]:
    rows = []
    for index, record in enumerate(records):
        clean = float(clean_margin[index])
        counter = float(counter_margin[index])
        wrong = float(wrong_margin[index])
        patched = float(patched_margin[index])
        denominator = clean - counter
        row = {
            "checkpoint": checkpoint,
            "intervention": intervention,
            "layer_index": layer_index,
            "id": str(record["id"]),
            "pair_id": str(record["pair_id"]),
            "pair_variant": str(record["pair_variant"]),
            "target_answer": str(record["answers"][0]),
            "counterfactual_source_id": mate[str(record["id"])],
            "wrong_label_source_id": wrong_sources[index],
            "clean_margin": clean,
            "counterfactual_margin": counter,
            "wrong_label_margin": wrong,
            "patched_margin": patched,
            "effect_vs_counterfactual": patched - counter,
            "normalized_recovery": (
                (patched - counter) / denominator if abs(denominator) > 1e-9 else None
            ),
            "clean_preference": clean > 0,
            "counterfactual_preference": counter > 0,
            "patched_preference": patched > 0,
        }
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)
    stream.flush()
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite activation patch run: {args.out}")
    args.out.mkdir(parents=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.checkpoint_ids:
        requested = set(args.checkpoint_ids)
        config["checkpoints"] = [row for row in config["checkpoints"] if row["id"] in requested]
        if len(config["checkpoints"]) != len(requested):
            raise ValueError("unknown patching checkpoint override")
    patch_config = config["activation_patching"]
    scan_layers = args.scan_layers or [int(value) for value in patch_config["scan_layer_indices"]]
    negative_layers = [
        int(value) for value in patch_config["negative_control_layer_indices"] if int(value) in scan_layers
    ]
    config["screening_override"] = {
        "checkpoint_ids": args.checkpoint_ids,
        "limit_pairs": args.limit_pairs,
        "scan_layers": args.scan_layers,
    }
    (args.out / "CONFIG.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    started = time.time()
    device = torch.device(config["device"])
    dtype = getattr(torch, config["dtype"])
    torch.manual_seed(int(config["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["seed"]))
        torch.cuda.reset_peak_memory_stats(device)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_path = str(config["text_model"])
    model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    validate_text_only_backbone_config(model_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    language_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, local_files_only=True
    ).to(device).eval()
    language_model.requires_grad_(False)
    layers = language_model.model.layers
    if max(scan_layers + negative_layers) >= len(layers):
        raise ValueError("patching layer index exceeds language-model depth")
    projector_source = Path(config["projector_config_source"])
    projector_config = ProjectorConfig(
        **json.loads(
            (projector_source / "projector_config.json").read_text(encoding="utf-8")
        )
    )
    projector = PatchMergerProjector(projector_config).to(device=device, dtype=dtype).eval()
    random_row = next(
        (row for row in config["checkpoints"] if row["kind"] == "random"), None
    )
    random_state = matched_random_state(
        projector_config, int(random_row["random_seed"]) if random_row else 0
    )

    dataset = config["dataset"]
    all_selection = load_records(Path(dataset["selection_data"]))
    records = read_ids(args.selection / "patching_selection_ids.json", all_selection)
    if args.limit_pairs is not None:
        keep_pairs = sorted({str(row["pair_id"]) for row in records})[: args.limit_pairs]
        keep = set(keep_pairs)
        records = [row for row in records if str(row["pair_id"]) in keep]
    expected = 2 * (args.limit_pairs or int(patch_config["pair_count"]))
    if len(records) != expected:
        raise ValueError("activation patching denominator mismatch")
    mate = pair_index(records)
    all_task_records = [row for row in all_selection if row["task"] == dataset["task"]]
    donor_by_id = wrong_label_donors(all_task_records, int(patch_config["pair_selection_seed"]) + 1)
    by_id = {str(row["id"]): row for row in all_task_records}
    class_names = [str(value) for value in dataset["classes"]]
    class_index = {name: index for index, name in enumerate(class_names)}
    labels = {str(row["id"]): class_index[str(row["answers"][0])] for row in all_task_records}
    cache = FeatureCache(dataset["selection_feature_cache"])
    class_tokens = answer_token_ids(tokenizer, class_names, device)
    placeholder = int(config["placeholder_token_id"])
    batch_size = int(patch_config["batch_size"])

    raw_path = args.out / "patching_records.jsonl"
    all_rows: list[dict] = []
    checkpoint_summaries = {}
    with raw_path.open("w", encoding="utf-8", newline="\n") as raw_stream, torch.inference_mode():
        for checkpoint in config["checkpoints"]:
            state_hash = load_checkpoint_state(projector, checkpoint, random_state)
            clean_by_id: dict[str, float] = {}
            counter_by_id: dict[str, float] = {}
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                clean_ids = [str(row["id"]) for row in batch]
                counter_ids = [mate[sample_id] for sample_id in clean_ids]
                wrong_ids = [donor_by_id[sample_id] for sample_id in clean_ids]
                clean_embeddings = projected_groups(
                    batch, clean_ids, cache, projector, device=device, dtype=dtype
                )
                counter_embeddings = projected_groups(
                    batch, counter_ids, cache, projector, device=device, dtype=dtype
                )
                wrong_embeddings = projected_groups(
                    batch, wrong_ids, cache, projector, device=device, dtype=dtype
                )
                clean_merged = prepare_inputs(
                    language_model=language_model,
                    tokenizer=tokenizer,
                    records=batch,
                    image_embeddings=clean_embeddings,
                    placeholder=placeholder,
                    prompt_template=str(config["prompt_template"]),
                    device=device,
                )
                counter_merged = prepare_inputs(
                    language_model=language_model,
                    tokenizer=tokenizer,
                    records=batch,
                    image_embeddings=counter_embeddings,
                    placeholder=placeholder,
                    prompt_template=str(config["prompt_template"]),
                    device=device,
                )
                wrong_merged = prepare_inputs(
                    language_model=language_model,
                    tokenizer=tokenizer,
                    records=batch,
                    image_embeddings=wrong_embeddings,
                    placeholder=placeholder,
                    prompt_template=str(config["prompt_template"]),
                    device=device,
                )
                target_labels = torch.tensor(
                    [labels[sample_id] for sample_id in clean_ids], device=device
                )
                counter_labels = torch.tensor(
                    [labels[sample_id] for sample_id in counter_ids], device=device
                )
                clean_outputs, clean_layer_outputs = captured_layer_forward(
                    language_model, layers, clean_merged
                )
                counter_outputs = language_forward(
                    language_model, counter_merged, hidden_states=False
                )
                wrong_outputs, wrong_layer_outputs = captured_layer_forward(
                    language_model, layers, wrong_merged
                )
                clean_margin = margins(
                    clean_outputs, clean_merged, class_tokens, target_labels, counter_labels
                )
                counter_margin = margins(
                    counter_outputs,
                    counter_merged,
                    class_tokens,
                    target_labels,
                    counter_labels,
                )
                wrong_margin = margins(
                    wrong_outputs, wrong_merged, class_tokens, target_labels, counter_labels
                )
                for index, sample_id in enumerate(clean_ids):
                    clean_by_id[sample_id] = float(clean_margin[index])
                    counter_by_id[sample_id] = float(counter_margin[index])
                image_mask = (
                    counter_merged.routing_input_ids.eq(placeholder)
                    & counter_merged.attention_mask.bool()
                )
                assistant_mask = torch.zeros_like(image_mask)
                assistant_mask.scatter_(
                    1, last_active_indices(counter_merged.attention_mask)[:, None], True
                )
                for layer_index in scan_layers:
                    clean_donor = clean_layer_outputs[layer_index]
                    for intervention, token_mask in (
                        ("correct_image_span", image_mask),
                        ("correct_assistant", assistant_mask),
                    ):
                        patched = patched_forward(
                            language_model=language_model,
                            layers=layers,
                            merged=counter_merged,
                            layer_index=layer_index,
                            donor=clean_donor,
                            token_mask=token_mask,
                            class_tokens=class_tokens,
                            target_labels=target_labels,
                            counter_labels=counter_labels,
                        )
                        if (
                            intervention == "correct_assistant"
                            and layer_index == len(layers) - 1
                            and float((patched - clean_margin).abs().max()) > 1e-6
                        ):
                            raise ValueError(
                                "final-layer assistant patch failed to reproduce clean margin"
                            )
                        all_rows.extend(
                            append_rows(
                                raw_stream,
                                checkpoint=checkpoint["id"],
                                records=batch,
                                mate=mate,
                                wrong_sources=wrong_ids,
                                intervention=intervention,
                                layer_index=layer_index,
                                clean_margin=clean_margin,
                                counter_margin=counter_margin,
                                wrong_margin=wrong_margin,
                                patched_margin=patched,
                            )
                        )
                    if layer_index in negative_layers:
                        for intervention, donor in (
                            (
                                "wrong_label_donor_image_span",
                                wrong_layer_outputs[layer_index],
                            ),
                            (
                                "zero_image_span",
                                torch.zeros_like(clean_donor),
                            ),
                        ):
                            patched = patched_forward(
                                language_model=language_model,
                                layers=layers,
                                merged=counter_merged,
                                layer_index=layer_index,
                                donor=donor,
                                token_mask=image_mask,
                                class_tokens=class_tokens,
                                target_labels=target_labels,
                                counter_labels=counter_labels,
                            )
                            all_rows.extend(
                                append_rows(
                                    raw_stream,
                                    checkpoint=checkpoint["id"],
                                    records=batch,
                                    mate=mate,
                                    wrong_sources=wrong_ids,
                                    intervention=intervention,
                                    layer_index=layer_index,
                                    clean_margin=clean_margin,
                                    counter_margin=counter_margin,
                                    wrong_margin=wrong_margin,
                                    patched_margin=patched,
                                )
                            )
                for intervention, region in (
                    ("input_clean_center", "center"),
                    ("input_clean_outer", "outer"),
                    ("input_clean_full", "full"),
                ):
                    blended = blend_regions(clean_embeddings, counter_embeddings, region)
                    merged = prepare_inputs(
                        language_model=language_model,
                        tokenizer=tokenizer,
                        records=batch,
                        image_embeddings=blended,
                        placeholder=placeholder,
                        prompt_template=str(config["prompt_template"]),
                        device=device,
                    )
                    patched = margins(
                        language_forward(language_model, merged, hidden_states=False),
                        merged,
                        class_tokens,
                        target_labels,
                        counter_labels,
                    )
                    all_rows.extend(
                        append_rows(
                            raw_stream,
                            checkpoint=checkpoint["id"],
                            records=batch,
                            mate=mate,
                            wrong_sources=wrong_ids,
                            intervention=intervention,
                            layer_index=-1,
                            clean_margin=clean_margin,
                            counter_margin=counter_margin,
                            wrong_margin=wrong_margin,
                            patched_margin=patched,
                        )
                    )
            relation_error = max(
                abs(counter_by_id[sample_id] + clean_by_id[mate[sample_id]])
                for sample_id in clean_by_id
            )
            if relation_error > 5e-3:
                raise ValueError(
                    f"paired clean/counter margin antisymmetry failed: {relation_error}"
                )
            pair_preferences = []
            for pair_id in sorted({str(row["pair_id"]) for row in records}):
                ids = [str(row["id"]) for row in records if str(row["pair_id"]) == pair_id]
                pair_preferences.append(all(clean_by_id[sample_id] > 0 for sample_id in ids))
            checkpoint_summaries[checkpoint["id"]] = {
                "state_sha256": state_hash,
                "records": len(clean_by_id),
                "pairs": len(pair_preferences),
                "mean_clean_margin": sum(clean_by_id.values()) / len(clean_by_id),
                "mean_counterfactual_margin": sum(counter_by_id.values()) / len(counter_by_id),
                "strict_paired_preference": sum(pair_preferences) / len(pair_preferences),
                "paired_margin_antisymmetry_max_abs": relation_error,
                "layer_output_capture": str(patch_config["layer_output_capture"]),
            }

    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in all_rows:
        grouped[(row["checkpoint"], row["intervention"], row["layer_index"])].append(row)
    curves = []
    for group_index, ((checkpoint, intervention, layer_index), rows) in enumerate(
        sorted(grouped.items())
    ):
        bootstrap = pair_bootstrap_mean(
            values=[row["effect_vs_counterfactual"] for row in rows],
            pair_ids=[row["pair_id"] for row in rows],
            seed=int(config["probe"]["bootstrap_seed"]) + 20000 + group_index,
            samples=int(config["probe"]["bootstrap_samples"]),
        )
        curves.append(
            {
                "checkpoint": checkpoint,
                "intervention": intervention,
                "layer_index": layer_index,
                "records": len(rows),
                "pairs": bootstrap["pairs"],
                "mean_clean_margin": sum(row["clean_margin"] for row in rows) / len(rows),
                "mean_counterfactual_margin": sum(
                    row["counterfactual_margin"] for row in rows
                )
                / len(rows),
                "mean_patched_margin": sum(row["patched_margin"] for row in rows)
                / len(rows),
                "mean_effect": bootstrap["mean"],
                "effect_ci95_low": bootstrap["ci95_low"],
                "effect_ci95_high": bootstrap["ci95_high"],
                "patched_preference_rate": sum(row["patched_preference"] for row in rows)
                / len(rows),
                "bootstrap_samples": bootstrap["bootstrap_samples"],
                "bootstrap_seed": bootstrap["seed"],
            }
        )
    curve_path = args.out / "patching_curve.csv"
    write_csv(curve_path, curves)
    decisions = {"status": "valid", "checkpoints": {}}
    for checkpoint in checkpoint_summaries:
        rows = [row for row in curves if row["checkpoint"] == checkpoint]
        decisions["checkpoints"][checkpoint] = {
            "baseline": checkpoint_summaries[checkpoint],
            "best_correct_image_span": max(
                (row for row in rows if row["intervention"] == "correct_image_span"),
                key=lambda row: row["mean_effect"],
            ),
            "best_correct_assistant": max(
                (row for row in rows if row["intervention"] == "correct_assistant"),
                key=lambda row: row["mean_effect"],
            ),
            "input_regions": [row for row in rows if row["layer_index"] == -1],
        }
    decisions["interpretation_limits"] = [
        "primary effects average every preregistered patching pair and both directions",
        "continuous confidence intervals resample complete a/b pairs",
        "assistant-state replacement is a localization positive control",
        "image-span replacement tests downstream causal transmission",
        "wrong-label donor and zero replacement are negative controls at preregistered layers",
        "final odd halves remain unscored",
    ]
    decisions_path = args.out / "DECISIONS.json"
    decisions_path.write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "valid",
        "format_version": "activation-patching-v1",
        "metadata": {
            "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha(),
            "host": platform.node(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "dtype": str(dtype).removeprefix("torch."),
            "wall_seconds": time.time() - started,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "final_half_scored": False,
        },
        "selection_manifest_sha256": sha256(args.selection / "MANIFEST.json"),
        "records": len(records),
        "pairs": len(records) // 2,
        "scan_layers": scan_layers,
        "language_layers": len(layers),
        "negative_control_layers": negative_layers,
        "checkpoints": checkpoint_summaries,
        "raw_rows": len(all_rows),
        "curve_rows": len(curves),
        "files": {
            raw_path.name: {"bytes": raw_path.stat().st_size, "sha256": sha256(raw_path)},
            curve_path.name: {"bytes": curve_path.stat().st_size, "sha256": sha256(curve_path)},
            decisions_path.name: {
                "bytes": decisions_path.stat().st_size,
                "sha256": sha256(decisions_path),
            },
        },
    }
    (args.out / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
