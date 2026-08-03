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
        --data data/eval/flickr8k.jsonl --limit 512 --steps 300 --batch-size 4 \
        --out checkpoints/overfit-qwen05
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

from moonvit_glue import (
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
from tools_common import build_prompt_ids, encode_image, load_records, next_batch


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
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--moonvit-model", default="moonshotai/MoonViT-SO-400M")
    parser.add_argument("--vision-tower", choices=["v1", "v2"], default="v1",
                        help="v1 = MoonViT-SO-400M from HF; v2 = Kimi K3 MoonViT-V2 from extracted weights")
    parser.add_argument("--moonvit-v2-weights", default=None,
                        help="Path to extracted moonvit_v2.safetensors (required with --vision-tower v2)")
    parser.add_argument("--moonvit-v2-attn", default="eager",
                        choices=["eager", "sdpa", "flash_attention_2"],
                        help="Attention backend for the V2 tower (eager/sdpa on hardware without flash-attn)")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Constant LR (no schedule). 5e-4 and short QA pairs match the "
                             "community-validated Baseten GLM-5.2V projector recipe")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-image-side", type=int, default=448)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--image-token", default="<|image_pad|>")
    parser.add_argument("--placeholder-token-id", type=int, default=None)
    parser.add_argument("--prompt-template", default="User: {image}\n{question}\nAssistant:")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--shuffle-repeats", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=500,
                        help="Save a resumable checkpoint every N steps (0 disables)")
    parser.add_argument("--upload-repo", default=None,
                        help="HF repo id; each checkpoint is uploaded in the background")
    parser.add_argument("--resume", default=None,
                        help="Local checkpoint dir or HF repo id to resume from")
    return parser.parse_args()


def build_sample(args, model, moonvit, tokenizer, placeholder_token_id, device, record, image_record=None):
    """Teacher-forced (input_ids, labels, feature_groups) for one record.

    ``image_record`` may supply a different record's image, which is how the
    shuffle-delta check decouples pictures from their (question, answer) pairs.
    """

    source = record if image_record is None else image_record
    groups = encode_image(moonvit, source, args.max_image_side, base_dir=args.data.parent)
    prompt_ids = build_prompt_ids(
        tokenizer, args.prompt_template, record["question"], placeholder_token_id, device
    )
    answer_ids = tokenizer.encode(
        " " + record["answers"][0], add_special_tokens=False, return_tensors="pt"
    ).to(device)
    eos = torch.tensor([[tokenizer.eos_token_id]], device=device)
    input_ids = torch.cat([prompt_ids, answer_ids, eos], dim=1)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    return input_ids, labels, groups


def mean_loss_for(args, model, moonvit, tokenizer, placeholder_token_id, device, records, image_records=None):
    losses = []
    with torch.no_grad():
        for index, record in enumerate(records):
            image_record = None if image_records is None else image_records[index]
            input_ids, labels, feature_groups = build_sample(
                args, model, moonvit, tokenizer, placeholder_token_id, device, record, image_record
            )
            outputs = model(
                input_ids=input_ids,
                image_feature_groups=feature_groups,
                labels=labels,
            )
            losses.append(float(outputs.loss))
    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log_file = open(args.out / "train.log", "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    records = load_records(args.data)
    records = [record for record in records if record.get("answers")]
    rng.shuffle(records)
    records = records[: args.limit]
    if len(records) < args.batch_size + 1:
        raise ValueError(f"Need more records than batch size + eval, got {len(records)}")
    eval_records = records[-args.eval_samples :]
    train_records = records[: -args.eval_samples]

    if args.vision_tower == "v2":
        if not args.moonvit_v2_weights:
            raise ValueError("--vision-tower v2 requires --moonvit-v2-weights")
        moonvit = load_moonvit_v2_encoder(
            args.moonvit_v2_weights,
            attn_implementation=args.moonvit_v2_attn,
            torch_dtype=dtype,
        )
    else:
        moonvit = MoonViTEncoder.from_pretrained(args.moonvit_model, torch_dtype=dtype)
    moonvit.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    language_model = AutoModelForCausalLM.from_pretrained(args.text_model, dtype=dtype)
    language_model.to(device)

    if args.placeholder_token_id is not None:
        placeholder_token_id = args.placeholder_token_id
    else:
        placeholder_token_id = resolve_placeholder_token_id(tokenizer, args.image_token)

    projector = PatchMergerProjector(
        ProjectorConfig(
            vision_width=moonvit.vision_width,
            language_width=int(language_model.config.hidden_size),
            merge_factor=moonvit.merge_factor,
        )
    )
    projector.to(device=device, dtype=dtype)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    model = VisionCausalLM(
        language_model=language_model,
        projector=projector,
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

    cursor = start_step * args.batch_size
    started = time.time()
    for step in range(start_step + 1, args.steps + 1):
        batch, cursor = next_batch(train_records, cursor, args.batch_size)
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for record in batch:
            input_ids, labels, feature_groups = build_sample(
                args, model, moonvit, tokenizer, placeholder_token_id, device, record
            )
            outputs = model(
                input_ids=input_ids,
                image_feature_groups=feature_groups,
                labels=labels,
            )
            (outputs.loss / args.batch_size).backward()
            step_loss += float(outputs.loss) / args.batch_size
        torch.nn.utils.clip_grad_norm_(projector.parameters(), args.grad_clip)
        optimizer.step()
        history.append({"step": step, "loss": step_loss})
        if step % args.log_every == 0 or step == 1:
            window = [row["loss"] for row in history[-args.log_every :]]
            print(f"step {step}/{args.steps} loss={sum(window) / len(window):.4f}", flush=True)
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

    # Acceptance check: true-vs-shuffled image loss gap on held-out records.
    true_loss = mean_loss_for(
        args, model, moonvit, tokenizer, placeholder_token_id, device, eval_records
    )
    shuffled_losses = []
    for repeat in range(args.shuffle_repeats):
        shift = repeat + 1
        shuffled_images = eval_records[shift:] + eval_records[:shift]
        shuffled_losses.append(
            mean_loss_for(
                args, model, moonvit, tokenizer, placeholder_token_id, device,
                eval_records, image_records=shuffled_images,
            )
        )
    shuffled_loss = sum(shuffled_losses) / len(shuffled_losses)

    args.out.mkdir(parents=True, exist_ok=True)
    projector.save_pretrained(args.out)

    first = sum(row["loss"] for row in history[: args.log_every]) / min(args.log_every, len(history))
    last = sum(row["loss"] for row in history[-args.log_every :]) / args.log_every
    report = {
        "text_model": args.text_model,
        "records_train": len(train_records),
        "records_eval": len(eval_records),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "loss_first_window": first,
        "loss_last_window": last,
        "eval_true_loss": true_loss,
        "eval_shuffled_loss": shuffled_loss,
        "shuffle_delta": shuffled_loss - true_loss,
        "wall_seconds": round(time.time() - started, 1),
        "projector_dir": str(args.out),
    }
    (args.out / "overfit_report.json").write_text(
        json.dumps({"report": report, "history": history}, ensure_ascii=False, indent=2) + "\n",
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
