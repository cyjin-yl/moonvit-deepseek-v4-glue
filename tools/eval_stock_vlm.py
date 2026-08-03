"""Score a stock (unmodified) HF vision-language model on our benchmark files.

This is the control arm for the glue-built VLM: same records, same metrics,
same report shape as ``tools/eval_vlm.py``, so ``tools/aggregate_eval.py``
can place both on one matrix. The model is loaded with its official weights,
processor and chat template — no glue code, no placeholder surgery.

Precision discipline (project rule): evaluate at the released weight
precision or a lossless upcast (fp8 -> bf16); never at a lower-precision
quantization, which would measure the quantizer instead of the model.

Examples::

    python tools/eval_stock_vlm.py --model Qwen/Qwen3.5-4B \
        --data data/eval/textvqa.jsonl --blind --limit 50

    python tools/eval_stock_vlm.py --model Qwen/Qwen3.5-9B --dtype bfloat16 \
        --data data/eval_v1/screenspot --upload-repo cyjin-yl/moonvit-dsv4-eval
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

from moonvit_glue.metrics import summarize
from tools_common import load_records
from eval_vlm import _git_sha, make_scored_row, slice_records

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks before scoring.

    Thinking variants would otherwise never exact-match a short answer. An
    unclosed trailing block means generation was cut inside reasoning, so
    there is no answer to score.
    """

    cleaned = _THINK_BLOCK.sub("", text)
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.strip()


def build_messages(question: str, with_image: bool) -> list[dict]:
    """One user turn, image first (the Qwen-VL convention)."""

    content = [{"type": "image"}] if with_image else []
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def load_record_image(record: dict, base_dir: Path) -> Image.Image:
    """Packed parquet rows carry ``image_bytes``; JSONL rows a relative path."""

    if record.get("image_bytes"):
        return Image.open(io.BytesIO(record["image_bytes"])).convert("RGB")
    return Image.open(Path(base_dir) / record["image"]).convert("RGB")


def apply_template(processor, messages: list[dict], thinking: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if not thinking:
        try:
            return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            pass  # template has no thinking switch — strip tags post-hoc instead
    return processor.apply_chat_template(messages, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HF id or local path of the stock VLM")
    parser.add_argument("--data", required=True, type=Path, help="Benchmark JSONL or packed parquet dir/file")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--record-slice", choices=["even", "odd"], default=None,
                        help="Same parity rule as eval_vlm: even=selection half, odd=final half")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device-map", default=None,
                        help="Optional accelerate device map (e.g. 'auto') for models too large for one GPU")
    parser.add_argument("--dtype", default="bfloat16",
                        help="Released precision or lossless upcast only (fp8 releases -> bfloat16)")
    parser.add_argument("--thinking", action="store_true",
                        help="Keep thinking enabled (default requests enable_thinking=False and strips think blocks)")
    parser.add_argument("--blind", action="store_true", help="Also score every record without its image")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--upload-repo", default=None, help="HF repo id; report JSON goes to eval/<tag>/")
    parser.add_argument("--run-tag", default=None, help="HF path segment; default UTC timestamp")
    return parser.parse_args()


def build_metadata(args: argparse.Namespace, git_sha: str | None = None) -> dict:
    import transformers

    return {
        "stock": True,
        "model": args.model,
        "data": args.data.name,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "thinking": args.thinking,
        "git": git_sha,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "host": platform.node(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_pass(args, model, processor, records, device, blind: bool) -> list[dict]:
    scored = []
    for index, record in enumerate(records):
        with_image = not blind and bool(record.get("image") or record.get("image_bytes"))
        messages = build_messages(record["question"], with_image)
        text = apply_template(processor, messages, args.thinking)
        if with_image:
            image = load_record_image(record, args.data.parent)
            inputs = processor(text=[text], images=[image], return_tensors="pt")
        else:
            inputs = processor(text=[text], return_tensors="pt")
        if args.device_map is None:
            inputs = inputs.to(device)
        else:
            inputs = inputs.to(model.device)
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        prompt_length = inputs["input_ids"].shape[1]
        raw = processor.batch_decode(output[:, prompt_length:], skip_special_tokens=True)[0]
        prediction = raw if args.thinking else strip_thinking(raw)
        row = make_scored_row(record, prediction, index)
        if raw != prediction:
            row["raw_prediction"] = raw
        scored.append(row)
        tag = "blind" if blind else "vision"
        print(f"[{tag} {index + 1}/{len(records)}] {record.get('id', index)} -> {prediction!r}", flush=True)
    return scored


def main() -> None:
    args = parse_args()
    records = load_records(args.data)
    records = slice_records(records, args.record_slice)
    if args.limit:
        records = records[: args.limit]

    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model)
    if args.device_map:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=dtype, device_map=args.device_map
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype)
        model.to(device)
    model.eval()

    scored = run_pass(args, model, processor, records, device, blind=False)
    report = {"mode": "generation", "summary": summarize(scored), "records": scored}
    if args.blind:
        blind_scored = run_pass(args, model, processor, records, device, blind=True)
        report["blind_summary"] = summarize(blind_scored)
        report["blind_records"] = blind_scored

    report["metadata"] = build_metadata(args, _git_sha())
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
