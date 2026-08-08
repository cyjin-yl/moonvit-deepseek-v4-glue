"""Score a glue-built VLM on JSONL benchmark files.

JSONL schema (produced by ``tools/fetch_eval_data.py``)::

    {
      "id": "textvqa_val_000123",
      "image": "images/textvqa_val_000123.png",  # relative to the JSONL file
      "question": "What brand is shown?",
      "metric": "exact_match" | "soft_vqa" | "anls" | "token_f1" | "grounding",
      "answers": ["..."],                        # text metrics
      "gt_point": [x, y],                        # grounding, normalized 0..999
      "gt_box": [x1, y1, x2, y2]                 # grounding, normalized 0..999
    }

Two modes:

* default: generate an answer per record and score it. ``--blind`` repeats
  every record without the image so language priors are reported separately.
* ``--shuffle-loss``: teacher-forced loss on the first reference answer with
  the true image versus shuffled images. This is the cheapest alignment
  signal before any training has happened.

Examples::

    python tools/eval_vlm.py --text-model Qwen/Qwen2.5-0.5B --random-projector \
        --placeholder-token-id 151643 --data data/eval/textvqa_val.jsonl --limit 20

    python tools/eval_vlm.py --text-model deepseek-ai/DeepSeek-V4-Flash-0731 \
        --projector checkpoints/projector --data data/eval/screenspot.jsonl
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

from moonvit_glue import (
    FeatureCache,
    MoonViTEncoder,
    PatchMergerProjector,
    ProjectorConfig,
    VisionCausalLM,
    load_moonvit_v2_encoder,
    resolve_placeholder_token_id,
    seeded_projector,
)
from moonvit_glue.screenspot_runtime import validate_feature_cache_interface
from moonvit_glue.metrics import score_record, summarize
from tools_common import build_prompt_ids, encode_image, load_records
from training_protocol import make_derangements, records_manifest_sha256
from moonvit_glue.proxy_receiver import FixedGroupedReceiverAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text-model", required=True, help="HF id or local path of the text backbone")
    parser.add_argument("--moonvit-model", default="moonshotai/MoonViT-SO-400M")
    parser.add_argument(
        "--moonvit-revision",
        default=None,
        help="Pinned Hugging Face revision for the standalone V1 tower",
    )
    parser.add_argument("--vision-tower", choices=["v1", "v2"], default="v1",
                        help="v1 = MoonViT-SO-400M from HF; v2 = Kimi K3 MoonViT-V2 from extracted weights")
    parser.add_argument("--moonvit-v2-weights", default=None,
                        help="Path to extracted moonvit_v2.safetensors (required with --vision-tower v2)")
    parser.add_argument("--moonvit-v2-attn", default="eager",
                        choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--projector", help="Directory with projector_config.json + projector.safetensors")
    parser.add_argument("--random-projector", action="store_true", help="Use a freshly initialized projector (smoke runs)")
    parser.add_argument("--canonical-projector", action="store_true", help="Load a canonical 4096-wide projector and apply the fixed receiver adapter.")
    parser.add_argument("--receiver-adapter-seed", type=int, default=20260806)
    parser.add_argument(
        "--projector-variant",
        choices=["legacy_pre_norm", "kimi_k3_v2"],
        default="legacy_pre_norm",
        help="Projector structure for --random-projector; loaded checkpoints carry their own variant",
    )
    parser.add_argument(
        "--random-projector-seed",
        type=int,
        default=None,
        help="Optional isolated seed for a reproducible random-projector control",
    )
    parser.add_argument("--data", required=True, type=Path, help="Benchmark JSONL file")
    parser.add_argument("--feature-cache", type=Path, default=None,
                        help="Frozen MoonViT feature cache for image-bearing passes")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--record-slice", choices=["even", "odd"], default=None,
                        help="Deterministic half-split: even=checkpoint-selection, odd=final (run once)")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--image-token", default=None,
                        help="Placeholder token; default auto-detects DeepSeek/Qwen candidates")
    parser.add_argument("--placeholder-token-id", type=int, default=None,
                        help="Explicit placeholder id for tokenizers without the image token")
    parser.add_argument("--prompt-template", default="User: {image}\n{question}\nAssistant:")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-image-side", type=int, default=None,
                        help="Downscale images before MoonViT; required when the text model's "
                        "context is smaller than the merged token count")
    parser.add_argument("--out", type=Path, default=None, help="Where to write the JSON report")
    parser.add_argument("--upload-repo", default=None,
                        help="HF repo id; the report JSON is uploaded to eval/<tag>/")
    parser.add_argument("--run-tag", default=None, help="HF path segment; default UTC timestamp")
    parser.add_argument("--blind", action="store_true", help="Also score every record without its image")
    parser.add_argument("--shuffled", action="store_true", help="Generate with a seeded derangement of the images while scoring original answers.")
    parser.add_argument("--shuffle-loss", action="store_true", help="Teacher-forced true-vs-shuffled image loss")
    parser.add_argument("--shuffle-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.random_projector and not args.projector:
        parser.error("Pass --projector DIR, or --random-projector for smoke runs")
    return args


def build_model(args: argparse.Namespace, feature_cache=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    if feature_cache is not None:
        if feature_cache.manifest.get("max_image_side") != args.max_image_side:
            raise ValueError("feature cache and evaluation max-image-side differ")
        moonvit = None
        vision_width = int(feature_cache.manifest["vision_width"])
        merge_factor = int(feature_cache.manifest["merge_factor"])
        # A cache is a frozen tower boundary.  Check the explicit identity when
        # it is available; legacy V2 caches may omit the remote-code fields.
        validate_feature_cache_interface(
            feature_cache.manifest,
            vision_width=vision_width,
            merge_factor=merge_factor,
            max_image_side=args.max_image_side,
            vision_tower=args.vision_tower,
            moonvit_model=(args.moonvit_model if args.vision_tower == "v1" else None),
            moonvit_revision=args.moonvit_revision,
            require_tower_identity=args.moonvit_revision is not None,
        )
    else:
        if args.vision_tower == "v2":
            if not args.moonvit_v2_weights:
                raise ValueError("--vision-tower v2 requires --moonvit-v2-weights")
            moonvit = load_moonvit_v2_encoder(
                args.moonvit_v2_weights,
                attn_implementation=args.moonvit_v2_attn,
                torch_dtype=dtype,
            )
        else:
            moonvit = MoonViTEncoder.from_pretrained(
                args.moonvit_model,
                revision=args.moonvit_revision,
                torch_dtype=dtype,
            )
        moonvit.to(device)
        vision_width = moonvit.vision_width
        merge_factor = moonvit.merge_factor
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    language_model = AutoModelForCausalLM.from_pretrained(args.text_model, dtype=dtype)
    language_model.to(device)

    if args.placeholder_token_id is not None:
        placeholder_token_id = args.placeholder_token_id
    else:
        placeholder_token_id = resolve_placeholder_token_id(tokenizer, args.image_token)

    if args.random_projector:
        projector_width = 4096 if args.canonical_projector else int(language_model.config.hidden_size)
        projector_config = ProjectorConfig(
            vision_width=vision_width,
            language_width=projector_width,
            merge_factor=merge_factor,
            projector_variant=args.projector_variant,
        )
        projector = (
            seeded_projector(projector_config, seed=args.random_projector_seed)
            if args.random_projector_seed is not None
            else PatchMergerProjector(projector_config)
        )
    else:
        projector = PatchMergerProjector.from_pretrained(args.projector, device=device, dtype=dtype)
        if int(projector.config.vision_width) != vision_width or int(
            projector.config.merge_factor
        ) != merge_factor:
            raise ValueError("projector and feature-cache vision interfaces differ")
    projector.to(device=device, dtype=dtype)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    receiver_width = int(getattr(language_model.config, "hidden_size", 0))
    receiver_adapter = None
    if args.canonical_projector and receiver_width != int(projector.config.language_width):
        receiver_adapter = FixedGroupedReceiverAdapter(
            int(projector.config.language_width),
            receiver_width,
            seed=args.receiver_adapter_seed,
        ).to(device=device)
    model = VisionCausalLM(
        language_model=language_model,
        projector=projector,
        receiver_adapter=receiver_adapter,
        placeholder_token_id=placeholder_token_id,
        backbone_kind="auto",
        freeze_language_model=True,
        pad_token_id=pad_token_id,
    )
    model.eval()
    return model, moonvit, tokenizer, placeholder_token_id, device


def make_scored_row(record: dict, prediction: str, index: int) -> dict:
    """Self-contained per-record output: question, references, prediction, scores.

    Published to HF, so each row must be interpretable without the source JSONL.
    """

    row = {"id": record.get("id", index), "question": record.get("question")}
    if record.get("answers") is not None:
        row["answers"] = record["answers"]
    for key in ("gt_box", "gt_point"):
        if record.get(key) is not None:
            row[key] = record[key]
    row["prediction"] = prediction
    row.update(score_record(prediction, record))
    return row


def build_metadata(
    args: argparse.Namespace,
    *,
    git_sha: str | None = None,
    model=None,
    records: list[dict] | None = None,
    wall_seconds: float | None = None,
    peak_gpu_memory_bytes: int = 0,
) -> dict:
    """Run context embedded in every report; aggregate_eval keys on ``data``."""

    return {
        "text_model": args.text_model,
        "vision_tower": args.vision_tower,
        "vision_weights": args.moonvit_v2_weights if args.vision_tower == "v2" else args.moonvit_model,
        # 兼容旧版调用方：评测元数据仍需记录 revision，但旧的
        # SimpleNamespace/脚本可能还没有该 CLI 字段。
        "vision_revision": getattr(args, "moonvit_revision", None),
        "projector": args.projector or "random",
        "projector_variant": getattr(args, "projector_variant", None),
        "canonical_projector": bool(getattr(args, "canonical_projector", False)),
        "receiver_adapter_seed": getattr(args, "receiver_adapter_seed", None),
        "random_projector_seed": getattr(args, "random_projector_seed", None),
        "data": args.data.name,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "evaluation_max_image_side": getattr(args, "max_image_side", None),
        "feature_cache": (
            str(args.feature_cache) if getattr(args, "feature_cache", None) else None
        ),
        "data_records_manifest_sha256": (
            records_manifest_sha256(records) if records is not None else None
        ),
        "projector_parameters": (
            sum(parameter.numel() for parameter in model.projector.parameters())
            if model is not None
            else None
        ),
        "checkpoint_source": args.projector or "random",
        "vision_tower_instantiated": (
            getattr(args, "feature_cache", None) is None
        ),
        "wall_seconds": wall_seconds,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "dtype": args.dtype,
        "seed": args.seed,
        "git": git_sha,
        "torch": torch.__version__,
        "host": platform.node(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def expanded_length(input_ids: torch.Tensor, placeholder_token_id: int, feature_groups) -> int:
    placeholders = int(input_ids.eq(placeholder_token_id).sum().item())
    image_tokens = sum(group.shape[0] for group in feature_groups)
    return input_ids.shape[1] - placeholders + image_tokens


def feature_groups_for(args, model, moonvit, device, record, feature_cache):
    if feature_cache is None:
        return encode_image(
            moonvit, record, args.max_image_side, base_dir=args.data.parent
        )
    return feature_cache.get(
        str(record["id"]),
        device=device,
        dtype=next(model.projector.parameters()).dtype,
    )


def run_generation(
    args, model, moonvit, tokenizer, placeholder_token_id, device, records,
    feature_cache=None, image_records=None,
):
    scored = []
    for index, record in enumerate(records):
        image_record = record if image_records is None else image_records[index]
        feature_groups = feature_groups_for(
            args, model, moonvit, device, image_record, feature_cache
        )
        input_ids = build_prompt_ids(
            tokenizer, args.prompt_template, record["question"], placeholder_token_id, device
        )
        output = model.generate(
            input_ids=input_ids,
            image_feature_groups=feature_groups,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=model.pad_token_id,
        )
        prompt_length = expanded_length(input_ids, placeholder_token_id, feature_groups)
        prediction = tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)
        scored.append(make_scored_row(record, prediction, index))
        print(f"[{index + 1}/{len(records)}] {record.get('id', index)} -> {prediction!r}")
    return scored


def slice_records(records: list, which: str | None) -> list:
    """Deterministic half-split for eval discipline (review, 2026-08-03).

    ``even`` is the checkpoint-selection half, ``odd`` the final half that is
    run exactly once for the winning checkpoint. Same parity rule across all
    benchmarks and checkpoints, so halves never drift.
    """

    if which == "even":
        return records[::2]
    if which == "odd":
        return records[1::2]
    return records


def summarize_shuffle(rows: list) -> dict:
    """Shuffle-loss summary with spread and relative lift (single delta is not proof)."""

    mean_true = sum(r["true_loss"] for r in rows) / len(rows)
    mean_shuffled = sum(r["shuffled_loss"] for r in rows) / len(rows)
    mean_delta = mean_shuffled - mean_true
    deltas = [r["delta"] for r in rows]
    variance = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
    return {
        "count": len(rows),
        "mean_true_loss": mean_true,
        "mean_shuffled_loss": mean_shuffled,
        "mean_delta": mean_delta,
        "delta_std": variance ** 0.5,
        "relative_improvement": mean_delta / mean_shuffled if mean_shuffled else 0.0,
    }


def run_shuffle_loss(
    args, model, moonvit, tokenizer, placeholder_token_id, device, records, feature_cache=None
):
    rng = random.Random(args.seed)
    records = [record for record in records if record.get("answers")]
    if not records:
        raise ValueError("shuffle-loss mode needs records with an 'answers' field")

    def loss_for(feature_groups, answer_ids, prompt_ids):
        input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
        labels = input_ids.clone()
        labels[:, : prompt_ids.shape[1]] = -100
        outputs = model(
            input_ids=input_ids,
            image_feature_groups=feature_groups,
            labels=labels,
        )
        return float(outputs.loss)

    rows = []
    for index, record in enumerate(records):
        true_groups = feature_groups_for(
            args, model, moonvit, device, record, feature_cache
        )
        prompt_ids = build_prompt_ids(
            tokenizer, args.prompt_template, record["question"], placeholder_token_id, device
        )
        answer_ids = tokenizer.encode(
            " " + record["answers"][0], return_tensors="pt", add_special_tokens=False
        ).to(device)

        true_loss = loss_for(true_groups, answer_ids, prompt_ids)
        shuffled_losses = []
        for _ in range(args.shuffle_repeats):
            other = rng.choice([r for r in records if r is not record])
            other_groups = feature_groups_for(
                args, model, moonvit, device, other, feature_cache
            )
            shuffled_losses.append(loss_for(other_groups, answer_ids, prompt_ids))
        row = {
            "id": record.get("id", index),
            "true_loss": true_loss,
            "shuffled_loss": sum(shuffled_losses) / len(shuffled_losses),
        }
        row["delta"] = row["shuffled_loss"] - row["true_loss"]
        rows.append(row)
        print(f"[{index + 1}/{len(records)}] {row['id']} delta={row['delta']:+.4f}")

    return rows, summarize_shuffle(rows)


def main() -> None:
    args = parse_args()
    started = time.time()
    device_for_stats = torch.device(args.device)
    if device_for_stats.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_for_stats)
    records = load_records(args.data)
    records = slice_records(records, args.record_slice)
    if args.limit:
        records = records[: args.limit]

    feature_cache = FeatureCache(args.feature_cache) if args.feature_cache else None
    model, moonvit, tokenizer, placeholder_token_id, device = build_model(
        args, feature_cache
    )

    if args.shuffle_loss:
        rows, summary = run_shuffle_loss(
            args,
            model,
            moonvit,
            tokenizer,
            placeholder_token_id,
            device,
            records,
            feature_cache,
        )
        report = {"mode": "shuffle_loss", "summary": summary, "records": rows}
    else:
        shuffled_records = None
        if args.shuffled:
            shuffled_records = make_derangements(
                records,
                repeats=1,
                seed=args.seed + 1_000_003,
            )[0]
        scored = run_generation(
            args,
            model,
            moonvit,
            tokenizer,
            placeholder_token_id,
            device,
            records,
            feature_cache,
            image_records=shuffled_records,
        )
        report = {
            "mode": "generation",
            "condition": "shuffled" if args.shuffled else "vision",
            "summary": summarize(scored),
            "records": scored,
        }
        if args.blind:
            blind_records = [
                {**record, "image": None, "question": record["question"]} for record in records
            ]
            # Blind pass: same prompt without the image token, scored identically.
            blind_scored = []
            for index, record in enumerate(blind_records):
                prompt = args.prompt_template.format(image="", question=record["question"])
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                output = model.language_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=model.pad_token_id,
                )
                prediction = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
                blind_scored.append(make_scored_row(record, prediction, index))
            report["blind_summary"] = summarize(blind_scored)
            report["blind_records"] = blind_scored

    report["metadata"] = build_metadata(
        args,
        git_sha=_git_sha(),
        model=model,
        records=records,
        wall_seconds=time.time() - started,
        peak_gpu_memory_bytes=(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    tag = args.run_tag or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if args.out is None and args.upload_repo:
        args.out = Path("eval_results") / f"{args.data.stem}.{tag}.json"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    if args.upload_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.upload_repo, repo_type="model", exist_ok=True)
        path_in_repo = f"eval/{tag}/{args.out.name}"
        api.upload_file(path_or_fileobj=str(args.out), repo_id=args.upload_repo,
                        path_in_repo=path_in_repo)
        print(f"[upload] {path_in_repo} -> {args.upload_repo}", flush=True)
    print(json.dumps(report.get("summary", {}), ensure_ascii=False, indent=2))
    if "blind_summary" in report:
        print("blind:")
        print(json.dumps(report["blind_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
