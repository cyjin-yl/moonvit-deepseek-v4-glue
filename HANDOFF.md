# Handoff

## Current state

- Public repo: https://github.com/cyjin-yl/moonvit-deepseek-v4-glue (`main`, published 2026-08-02).
- Core glue implemented under `src/moonvit_glue/`; **26/26 tests pass on Linux** (torch 2.10.0+cu128, transformers 5.12.1).
- Real `DeepseekV4ForCausalLM` tiny Hash-MoE config passed loss/backward and generation.
- Projector shape fixed at MoonViT `[N,4,1152]` → DeepSeek 4096, 40,119,040 params.
- DeepSeek image placeholder fixed to existing `<｜image｜>` ID 129279; never resize vocab.
- `VisionCausalLM.generate()` exists for both backbone kinds; generic path returns the full expanded sequence.
- Benchmark scaffold landed: `moonvit_glue.metrics` (pure Python), `tools/eval_vlm.py` (generation + `--blind` + `--shuffle-loss`), `tools/fetch_eval_data.py` (TextVQA/DocVQA/OCRBench/ScreenSpot → JSONL + MANIFEST.json). Eval philosophy: always report the blind (no-image) baseline next to capability numbers; grounding reports parse rate, Accuracy@50 on the 0–999 scale, and mean click error, matching the 0xSero GLM-5.2 vision convention.
- Typst report is `report/main.typ`; `report/*.pdf` is gitignored (build artifact) — compile with `typst compile report/main.typ`.
- Read-only Vast snapshot recorded in the report; no instance was created.

## doesworkstation (V100 32GB) operational notes

- SSH alias `doesworkstation` works via Tailscale (100.94.73.9); the Clash TUN fake-IP hijacks the name when TUN is on — use the Tailscale IP if resolution fails.
- **NVML mismatch**: kernel module 580.159.04 vs userland 580.173. `nvidia-smi` fails, but CUDA works fine. Do NOT reload the driver or reboot — the `fastllm` agent's jobs depend on the current state.
- Working env: `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/.venv-1cat/bin/python` — torch 2.10.0+cu128 (**includes sm_70, V100 works**), transformers 5.12.1, safetensors, pillow. No pytest: it was installed into `/run/media/ezra/13D010B6FDBC1A06/moonvit-deps` via `pip --target`; use `PYTHONPATH=$HDD/moonvit-deps:src`.
- HDD (13T, ~3.7T free) is `/run/media/ezra/13D010B6FDBC1A06/` — never write big files to `/home` (89% full). Repo clone lives at `$HDD/moonvit-deepseek-v4-glue`; `HF_HOME=$HDD/huggingface`.
- Workstation proxy: `127.0.0.1:7890` — needed for HF downloads but only ~0.4 MB/s; hf-mirror.com is no faster. MoonViT (~834 MB) downloads in the `moonvit` tmux session.
- The `fastllm` tmux pane runs another agent (GPT-5.6) optimizing Qwen inference; it launches GPU `apiserver` variants. A coordination message was left in its pane. ~15.6 GiB VRAM was free; keep our GPU usage small and short.
- System python3 has torch 2.13.0+cpu only (no CUDA); `~/.conda/envs/tsenv` has no torch.

## Immediate next actions

1. When MoonViT finishes downloading (watch `moonvit` tmux pane for `MOONVIT_DOWNLOAD_DONE`): run `examples/smoke_real_moonvit.py data/bus.jpg` on the V100 with `HF_HOME` and proxy env set.
2. Record V100 SM70-specific failures and pin the oldest compatible PyTorch/FlashAttention-free path.
3. Fetch eval datasets (`tools/fetch_eval_data.py`, needs `datasets` package — install via `pip --target $HDD/moonvit-deps datasets`) and dry-run `tools/eval_vlm.py --shuffle-loss` with a small text model.
4. Re-run the read-only Vast offer search immediately before budgeting; do not create an instance without fresh user approval.

## Main unresolved risk

Official 0731 FP4/FP8 kernels may support inference but not gradient with respect to input embeddings. The glue contract is tested; the large-quantized-backbone Dgrad path is not.

## Safety / repository hygiene

- Never commit HF, GitHub, Vast, or SSH credentials.
- Never commit model shards, datasets, or projector checkpoints (`data/`, `checkpoints/` are gitignored).
- Keep MoonViT, DeepSeek, and projector revisions/hashes separate.
- The supplied Vast credential was not written into this repository.
- The Vast API was used only with `POST /api/v0/bundles/` (offer search), never the instance-creation endpoint.
