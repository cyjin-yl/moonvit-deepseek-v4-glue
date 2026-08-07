"""Overfit the projector on a small caption set — the Gate B training signal.

This validates the full training contract on the V100 before any big GPUs
are rented: the loss must fall, and the true-vs-shuffled image loss gap
(shuffle delta) must turn positive. Data uses the same JSONL schema as
``tools/fetch_eval_data.py`` (the ``flickr8k`` entry produces exactly
``{image, question, answers}`` records).

Only the projector is trained; MoonViT and the text LM stay frozen.
Everything runs in fp32 by default because the MoonViT remote code mixes
dtypes internally on the V100 stack.

Example::

    python tools/train_overfit.py --text-model Qwen/Qwen2.5-0.5B-Instruct \
        --data data/eval/flickr8k.jsonl --limit 512 --steps 300 \
        --gradient-accumulation-steps 4 \
        --out checkpoints/overfit-qwen05
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from moonvit_glue import (
    FeatureCache,
    MoonViTEncoder,
    PatchMergerProjector,
    ProjectorConfig,
    VisionCausalLM,
    load_moonvit_v2_encoder,
    resolve_placeholder_token_id,
)
from moonvit_glue.checkpointing import (
    CheckpointUploader,
    load_training_checkpoint,
    save_training_checkpoint,
)
from moonvit_glue.proxy_receiver import FixedGroupedReceiverAdapter
from tools_common import (
    build_prompt_ids,
    encode_image,
    load_records,
    next_batch,
    validate_text_only_backbone_config,
)
from training_protocol import (
    TrainingProgress,
    make_derangements,
    prepare_validation_split,
    records_manifest_sha256,
    resolve_batch_semantics,
    restore_progress_counts,
    select_supervision,
    summarize_validation_losses,
)


class _Tee:
    """Mirror stdout/stderr into a log file so the training log rides with uploads."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--text-model",
        required=True,
        help="Text-only causal LM; native VLM configs with vision_config are rejected",
    )
    parser.add_argument("--moonvit-model", default="moonshotai/MoonViT-SO-400M")
    parser.add_argument("--vision-tower", choices=["v1", "v2"], default="v1",
                        help="v1 = MoonViT-SO-400M from HF; v2 = Kimi K3 MoonViT-V2 from extracted weights")
    parser.add_argument("--moonvit-v2-weights", default=None,
                        help="Path to extracted moonvit_v2.safetensors (required with --vision-tower v2)")
    parser.add_argument("--moonvit-v2-attn", default="eager",
                        choices=["eager", "sdpa", "flash_attention_2"],
                        help="Attention backend for the V2 tower (eager/sdpa on hardware without flash-attn)")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--feature-cache", type=Path, default=None,
                        help="Frozen MoonViT feature cache; avoids repeat tower forwards")
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Deprecated alias for --gradient-accumulation-steps; this trainer's "
             "historical 'batch' was serial microbatch=1 accumulation",
    )
    parser.add_argument("--micro-batch-size", type=int, default=1,
                        help="Examples in one batched forward (currently must be 1)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None,
                        help="Serial microbatches accumulated before each optimizer step; default 4")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Constant LR (no schedule). 5e-4 and short QA pairs match the "
                             "community-validated Baseten GLM-5.2V projector recipe")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-image-side", type=int, default=448)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument(
        "--canonical-projector",
        action="store_true",
        help="Keep the trainable projector at canonical 4096 dims and use a fixed parameter-free receiver adapter.",
    )
    parser.add_argument("--receiver-adapter-seed", type=int, default=20260806)
    parser.add_argument(
        "--projector-variant",
        choices=["legacy_pre_norm", "kimi_k3_v2"],
        default="legacy_pre_norm",
        help="Projector structure; V1 uses legacy_pre_norm and V2 uses kimi_k3_v2.",
    )
    parser.add_argument(
        "--init-projector",
        default=None,
        help="Load the complete frozen step0 projector from a matching directory.",
    )
    parser.add_argument("--image-token", default=None,
                        help="Placeholder token; default auto-detects DeepSeek/Qwen candidates")
    parser.add_argument("--placeholder-token-id", type=int, default=None)
    parser.add_argument("--prompt-template", default="User: {image}\n{question}\nAssistant:")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-samples", type=int, default=32,
                        help="Total fixed validation records, stratified evenly across sources")
    parser.add_argument("--validation-manifest", type=Path, default=None,
                        help="Pinned validation IDs; create deterministically when absent")
    parser.add_argument("--shuffle-repeats", type=int, default=10,
                        help="Seeded random derangements for validation mean/std")
    parser.add_argument("--answer-selection", choices=["canonical", "random"],
                        default="canonical",
                        help="Teacher answer: normalized majority or seeded acceptable-answer sample")
    parser.add_argument("--checkpoint-every", type=int, default=500,
                        help="Save a resumable checkpoint every N steps (0 disables)")
    parser.add_argument("--upload-repo", default=None,
                        help="HF repo id; each checkpoint is uploaded in the background")
    parser.add_argument("--resume", default=None,
                        help="Local checkpoint dir or HF repo id to resume from")
    parser.add_argument("--init-projector-trunk", default=None,
                        help="Donor projector dir: warm-start pre_norm + linear_1 (the "
                             "language-agnostic trunk) from a projector aligned against a "
                             "different text backbone; linear_2 keeps its fresh init. "
                             "--resume overrides this (resume restores the full state)")
    return parser.parse_args()


def build_sample(
    args,
    model,
    moonvit,
    tokenizer,
    placeholder_token_id,
    device,
    record,
    image_record=None,
    *,
    answer_rule="canonical",
    answer_rng=None,
    feature_cache=None,
):
    """Teacher-forced (input_ids, labels, feature_groups) for one record.

    ``image_record`` may supply a different record's image, which is how the
    shuffle-delta check decouples pictures from their (question, answer) pairs.
    """

    source = record if image_record is None else image_record
    if feature_cache is None:
        groups = encode_image(
            moonvit, source, args.max_image_side, base_dir=args.data.parent
        )
    else:
        projector_dtype = next(model.projector.parameters()).dtype
        groups = feature_cache.get(
            str(source["id"]), device=device, dtype=projector_dtype
        )
    prompt_ids = build_prompt_ids(
        tokenizer, args.prompt_template, record["question"], placeholder_token_id, device
    )
    supervision = select_supervision(record["answers"], rule=answer_rule, rng=answer_rng)
    answer_ids = tokenizer.encode(
        " " + supervision.selected_answer, add_special_tokens=False, return_tensors="pt"
    ).to(device)
    eos = torch.tensor([[tokenizer.eos_token_id]], device=device)
    input_ids = torch.cat([prompt_ids, answer_ids, eos], dim=1)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    supervision_metadata = asdict(supervision)
    supervision_metadata["answer_tokens"] = int(answer_ids.numel())
    return input_ids, labels, groups, supervision_metadata


def losses_for(
    args,
    model,
    moonvit,
    tokenizer,
    placeholder_token_id,
    device,
    records,
    image_records=None,
    feature_cache=None,
):
    losses = []
    with torch.no_grad():
        for index, record in enumerate(records):
            image_record = None if image_records is None else image_records[index]
            input_ids, labels, feature_groups, _ = build_sample(
                args,
                model,
                moonvit,
                tokenizer,
                placeholder_token_id,
                device,
                record,
                image_record,
                feature_cache=feature_cache,
            )
            outputs = model(
                input_ids=input_ids,
                image_feature_groups=feature_groups,
                labels=labels,
            )
            losses.append(float(outputs.loss))
    return losses


def main() -> None:
    args = parse_args()
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    log_file = open(args.out / "train.log", "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    batch_semantics = resolve_batch_semantics(
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        legacy_batch_size=args.batch_size,
    )
    accumulation_steps = int(batch_semantics["gradient_accumulation_steps"])
    effective_batch_size = int(batch_semantics["effective_batch_size"])

    text_config = AutoConfig.from_pretrained(args.text_model)
    validate_text_only_backbone_config(text_config)

    records = load_records(args.data)
    records = [record for record in records if record.get("answers")]
    rng.shuffle(records)
    records = records[: args.limit]
    data_records_sha256 = records_manifest_sha256(records)
    if len(records) < effective_batch_size + args.eval_samples:
        raise ValueError(
            f"Need at least effective_batch_size + validation records, got {len(records)}"
        )
    validation_manifest_path = args.validation_manifest or args.out / "validation_manifest.json"
    train_records, eval_records, validation_manifest = prepare_validation_split(
        records,
        manifest_path=validation_manifest_path,
        total_samples=args.eval_samples,
        seed=args.seed,
    )
    if len(eval_records) < 2:
        raise ValueError("validation needs at least two records for derangement")
    supervision_manifest_path = args.out / "supervision_manifest.jsonl"
    with supervision_manifest_path.open("w", encoding="utf-8") as stream:
        for record in sorted(records, key=lambda item: str(item["id"])):
            choice = select_supervision(record["answers"], rule="canonical")
            stream.write(json.dumps({
                "id": str(record["id"]),
                "source": record.get("source"),
                "raw_answers": choice.raw_answers,
                "canonical_answer": choice.canonical_answer,
                "normalization_rule": choice.normalization_rule,
                "training_selection_rule": args.answer_selection,
            }, ensure_ascii=False) + "\n")

    feature_cache = FeatureCache(args.feature_cache) if args.feature_cache else None

    if feature_cache is not None:
        if feature_cache.manifest.get("max_image_side") != args.max_image_side:
            raise ValueError("feature cache and training max-image-side differ")
        moonvit = None
        vision_width = int(feature_cache.manifest["vision_width"])
        merge_factor = int(feature_cache.manifest["merge_factor"])
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
                args.moonvit_model, torch_dtype=dtype
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

    canonical_width = 4096 if args.canonical_projector else int(language_model.config.hidden_size)
    projector = PatchMergerProjector(
        ProjectorConfig(
            vision_width=vision_width,
            language_width=canonical_width,
            merge_factor=merge_factor,
            projector_variant=args.projector_variant,
        )
    )
    projector.to(device=device, dtype=dtype)
    if args.init_projector:
        donor = PatchMergerProjector.from_pretrained(
            args.init_projector, device=device, dtype=dtype
        )
        # None and an explicit flattened width are semantically identical.
        # Compare effective structure so canonical step0 checkpoints remain reusable.
        donor_signature = (
            donor.config.vision_width,
            donor.config.language_width,
            donor.config.merge_factor,
            donor.config.effective_projector_width,
            donor.config.layer_norm_eps,
            donor.config.output_norm,
            donor.config.residual_mode,
            donor.config.projector_variant,
        )
        projector_signature = (
            projector.config.vision_width,
            projector.config.language_width,
            projector.config.merge_factor,
            projector.config.effective_projector_width,
            projector.config.layer_norm_eps,
            projector.config.output_norm,
            projector.config.residual_mode,
            projector.config.projector_variant,
        )
        if donor_signature != projector_signature:
            raise ValueError(
                f"init projector config differs: {donor.config} != {projector.config}"
            )
        projector.load_state_dict(donor.state_dict(), strict=True)
        print(f"loaded complete step0 projector from {args.init_projector}", flush=True)
    if args.init_projector_trunk:
        projector.load_trunk(args.init_projector_trunk)
        print(f"warm-started trunk (pre_norm + linear_1) from {args.init_projector_trunk}", flush=True)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    receiver_adapter = None
    if args.canonical_projector and int(language_model.config.hidden_size) != canonical_width:
        receiver_adapter = FixedGroupedReceiverAdapter(
            canonical_width,
            int(language_model.config.hidden_size),
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
    model.train()

    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    uploader = CheckpointUploader(args.upload_repo) if args.upload_repo else None
    start_step = 0
    if args.resume:
        start_step, history, rng, resume_dir = load_training_checkpoint(
            source=args.resume, projector=projector, optimizer=optimizer, device=device
        )
        print(f"resumed from {resume_dir} at step {start_step}", flush=True)
    else:
        history = []

    last_history = history[-1] if history else {}
    restored_counts = restore_progress_counts(
        start_step=start_step,
        last_history=last_history,
        effective_batch_size=effective_batch_size,
        batch_semantics_explicit=(
            args.gradient_accumulation_steps is not None or args.batch_size is not None
        ),
    )
    progress = TrainingProgress(
        total_training_examples=len(train_records),
        micro_batch_size=int(batch_semantics["micro_batch_size"]),
        gradient_accumulation_steps=accumulation_steps,
        optimizer_steps=start_step,
        examples_seen=int(restored_counts["examples_seen"]),
        answer_tokens_seen=int(restored_counts["answer_tokens_seen"]),
    )
    answer_token_accounting_complete = bool(
        restored_counts["answer_token_accounting_complete"]
    )
    cursor = progress.examples_seen
    training_started = time.time()
    health_path = args.out / "train_health.jsonl"
    if not args.resume:
        health_path.write_text("", encoding="utf-8")
    health_baseline = None
    for step in range(start_step + 1, args.steps + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_started = time.time()
        batch, cursor = next_batch(train_records, cursor, effective_batch_size)
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        projector_rows = []
        receiver_rows = []
        for record in batch:
            input_ids, labels, feature_groups, supervision = build_sample(
                args,
                model,
                moonvit,
                tokenizer,
                placeholder_token_id,
                device,
                record,
                answer_rule=args.answer_selection,
                answer_rng=rng if args.answer_selection == "random" else None,
                feature_cache=feature_cache,
            )
            projected = model.projector(feature_groups)
            projected_tensor = projected[0]
            receiver_tensor = (
                model.receiver_adapter(projected_tensor)
                if model.receiver_adapter is not None
                else projected_tensor
            )
            projector_rows.append(projected_tensor.detach())
            receiver_rows.append(receiver_tensor.detach())
            outputs = model(
                input_ids=input_ids,
                image_embeddings=projected,
                labels=labels,
            )
            (outputs.loss / accumulation_steps).backward()
            step_loss += float(outputs.loss) / accumulation_steps
            progress.record_microbatch(examples=1, answer_tokens=supervision["answer_tokens"])
        grad_before_clip = float(
            math.sqrt(
                sum(
                    float(parameter.grad.detach().float().pow(2).sum())
                    for parameter in projector.parameters()
                    if parameter.grad is not None
                )
            )
        )
        torch.nn.utils.clip_grad_norm_(projector.parameters(), args.grad_clip)
        grad_after_clip = float(
            math.sqrt(
                sum(
                    float(parameter.grad.detach().float().pow(2).sum())
                    for parameter in projector.parameters()
                    if parameter.grad is not None
                )
            )
        )
        optimizer.step()
        progress.record_optimizer_step()
        with torch.no_grad():
            projector_cat = torch.cat([row.reshape(-1, row.shape[-1]).float() for row in projector_rows])
            receiver_cat = torch.cat([row.reshape(-1, row.shape[-1]).float() for row in receiver_rows])
            projector_means = torch.stack([row.float().mean(dim=0) for row in projector_rows])
            receiver_means = torch.stack([row.float().mean(dim=0) for row in receiver_rows])
            projector_output_rms = float(projector_cat.pow(2).mean().sqrt())
            receiver_output_rms = float(receiver_cat.pow(2).mean().sqrt())
            between_image_rms = float(projector_means.std(dim=0).pow(2).mean().sqrt())
            receiver_between_image_rms = float(receiver_means.std(dim=0).pow(2).mean().sqrt())
            within_image_token_rms = float(torch.stack([
                (row.float() - row.float().mean(dim=0, keepdim=True)).pow(2).mean().sqrt()
                for row in projector_rows
            ]).mean())
            relative_spread = between_image_rms / max(projector_output_rms, 1e-12)
            receiver_relative_spread = receiver_between_image_rms / max(receiver_output_rms, 1e-12)
            if health_baseline is None:
                health_baseline = {
                    "projector_output_rms": projector_output_rms,
                    "receiver_output_rms": receiver_output_rms,
                    "relative_spread": relative_spread,
                    "receiver_relative_spread": receiver_relative_spread,
                }
            health_row = {
                "optimizer_step": step,
                "examples_seen": progress.examples_seen,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "projector_output_rms": projector_output_rms,
                "receiver_output_rms": receiver_output_rms,
                "between_image_rms": between_image_rms,
                "receiver_between_image_rms": receiver_between_image_rms,
                "within_image_token_rms": within_image_token_rms,
                "relative_spread": relative_spread,
                "receiver_relative_spread": receiver_relative_spread,
                "mean_direction_fraction": float(projector_means.mean(dim=0).norm() / projector_means.norm(dim=1).mean().clamp_min(1e-12)),
                "projector_gradient_norm_before_clip": grad_before_clip,
                "projector_gradient_norm_after_clip": grad_after_clip,
                "ce_loss": step_loss,
                "geometry_loss": 0.0,
                "total_loss": step_loss,
                "has_nan_or_inf": not all(torch.isfinite(row).all().item() for row in projector_rows + receiver_rows),
            }
            health_row["projector_output_rms_ratio"] = projector_output_rms / max(health_baseline["projector_output_rms"], 1e-12)
            health_row["receiver_output_rms_ratio"] = receiver_output_rms / max(health_baseline["receiver_output_rms"], 1e-12)
            health_row["projector_relative_spread_ratio"] = relative_spread / max(health_baseline["relative_spread"], 1e-12)
            health_row["receiver_relative_spread_ratio"] = receiver_relative_spread / max(health_baseline["receiver_relative_spread"], 1e-12)
            health_row["critical_guard"] = bool(
                health_row["has_nan_or_inf"]
                or health_row["projector_output_rms_ratio"] > 50.0
                or health_row["receiver_output_rms_ratio"] > 50.0
                or (step > 1 and health_row["projector_relative_spread_ratio"] < 0.25)
                or (step > 1 and health_row["receiver_relative_spread_ratio"] < 0.25)
                or not math.isfinite(grad_before_clip)
            )
        with health_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(health_row, ensure_ascii=False) + "\n")
        if health_row["critical_guard"]:
            failure_dir = save_training_checkpoint(
                directory=args.out / "failure-checkpoints" / f"step-{step:06d}",
                projector=projector,
                optimizer=optimizer,
                step=step,
                history=history,
                rng=rng,
            )
            raise RuntimeError(f"collapse/NaN health guard triggered at step {step}: {failure_dir}")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_wall_seconds = time.time() - step_started
        history.append({
            "step": step,
            "loss": step_loss,
            "step_wall_seconds": step_wall_seconds,
            "examples_per_second": effective_batch_size / step_wall_seconds,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            **progress.snapshot(),
        })
        if step % args.log_every == 0 or step == 1:
            window = [row["loss"] for row in history[-args.log_every :]]
            print(
                f"optimizer_step {step}/{args.steps} loss={sum(window) / len(window):.4f} "
                f"examples_seen={progress.examples_seen} "
                f"answer_tokens_seen={progress.answer_tokens_seen} "
                f"effective_epochs={progress.effective_epochs:.4f}",
                flush=True,
            )
        if args.checkpoint_every and step % args.checkpoint_every == 0:
            checkpoint_dir = save_training_checkpoint(
                directory=args.out / "checkpoints" / f"step-{step:06d}",
                projector=projector,
                optimizer=optimizer,
                step=step,
                history=history,
                rng=rng,
            )
            print(f"checkpoint saved: {checkpoint_dir}", flush=True)
            if uploader:
                uploader.upload_async(checkpoint_dir, f"checkpoints/step-{step:06d}")

    # Acceptance check: true-vs-seeded-deranged image loss, overall and by source.
    true_losses = losses_for(
        args,
        model,
        moonvit,
        tokenizer,
        placeholder_token_id,
        device,
        eval_records,
        feature_cache=feature_cache,
    )
    shuffled_record_runs = make_derangements(
        eval_records,
        repeats=args.shuffle_repeats,
        seed=args.seed + 1_000_003,
    )
    shuffled_loss_runs = [
        losses_for(
            args,
            model,
            moonvit,
            tokenizer,
            placeholder_token_id,
            device,
            eval_records,
            image_records=shuffled_records,
            feature_cache=feature_cache,
        )
        for shuffled_records in shuffled_record_runs
    ]
    validation_summary = summarize_validation_losses(
        eval_records,
        true_losses=true_losses,
        shuffled_loss_runs=shuffled_loss_runs,
        shuffled_id_runs=[
            [str(record["id"]) for record in shuffled_records]
            for shuffled_records in shuffled_record_runs
        ],
    )
    true_loss = validation_summary["overall"]["true_loss"]
    shuffled_loss = validation_summary["overall"]["shuffled_loss_mean"]

    args.out.mkdir(parents=True, exist_ok=True)
    projector.save_pretrained(args.out)

    first_window = history[: args.log_every]
    last_window = history[-args.log_every :]
    first = (
        sum(row["loss"] for row in first_window) / len(first_window)
        if first_window
        else None
    )
    last = (
        sum(row["loss"] for row in last_window) / len(last_window)
        if last_window
        else None
    )
    progress_snapshot = progress.snapshot()
    timed_steps = [
        float(row["step_wall_seconds"])
        for row in history
        if "step_wall_seconds" in row
    ]
    report = {
        "text_model": args.text_model,
        "text_model_architectures": getattr(text_config, "architectures", None),
        "text_backbone_native_multimodal": False,
        "records_considered": len(records),
        "records_train": len(train_records),
        "records_eval": len(eval_records),
        "data_records_manifest_sha256": data_records_sha256,
        "steps": args.steps,
        **progress_snapshot,
        "answer_token_accounting_complete": answer_token_accounting_complete,
        "legacy_batch_size_cli": args.batch_size,
        "legacy_batch_size_used": batch_semantics["legacy_batch_size_used"],
        "lr": args.lr,
        "answer_selection": args.answer_selection,
        "supervision_manifest": str(supervision_manifest_path),
        "validation_manifest": str(validation_manifest_path),
        "validation_counts_by_source": validation_manifest["counts_by_source"],
        "shuffle_repeats": args.shuffle_repeats,
        "actual_batched_forward": False,
        "forward_backward_calls_per_optimizer_step": accumulation_steps,
        "feature_cache": str(args.feature_cache) if args.feature_cache else None,
        "canonical_projector": bool(args.canonical_projector),
        "canonical_projector_width": canonical_width,
        "projector_variant": args.projector_variant,
        "init_projector": args.init_projector,
        "receiver_adapter": {
            "kind": "fixed_grouped_signed_projection" if receiver_adapter is not None else "identity",
            "seed": args.receiver_adapter_seed if receiver_adapter is not None else None,
            "receiver_width": int(language_model.config.hidden_size),
            "trainable_parameter_count": 0,
        },
        "vision_tower_instantiated": moonvit is not None,
        "training_max_image_side": args.max_image_side,
        "evaluation_max_image_side": args.max_image_side,
        "projector_parameters": sum(
            parameter.numel() for parameter in projector.parameters()
        ),
        "checkpoint_source": (
            args.resume or args.init_projector_trunk or "scratch"
        ),
        "loss_first_window": first,
        "loss_last_window": last,
        "eval_true_loss": true_loss,
        "eval_shuffled_loss": shuffled_loss,
        "shuffle_delta": validation_summary["overall"]["shuffle_delta_mean"],
        "shuffle_delta_std": validation_summary["overall"]["shuffle_delta_std"],
        "training_and_validation_wall_seconds": round(
            time.time() - training_started, 1
        ),
        "wall_seconds": round(time.time() - started, 1),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "step_time_seconds": (
            {
                "count": len(timed_steps),
                "mean": sum(timed_steps) / len(timed_steps),
                "min": min(timed_steps),
                "max": max(timed_steps),
            }
            if timed_steps
            else None
        ),
        "projector_dir": str(args.out),
    }
    (args.out / "overfit_report.json").write_text(
        json.dumps(
            {"report": report, "validation": validation_summary, "history": history},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    final_dir = save_training_checkpoint(
        directory=args.out / "checkpoints" / f"step-{args.steps:06d}",
        projector=projector,
        optimizer=optimizer,
        step=args.steps,
        history=history,
        rng=rng,
    )
    print(f"final checkpoint: {final_dir}", flush=True)
    if uploader:
        uploader.upload_async(args.out, "")
        uploader.wait()
        if uploader.errors:
            print(f"[uploader] completed with {len(uploader.errors)} error(s):", flush=True)
            for error in uploader.errors:
                print(f"  {error}", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
