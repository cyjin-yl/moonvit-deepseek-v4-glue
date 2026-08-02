# Handoff

## Current state

- Public repo: https://github.com/cyjin-yl/moonvit-deepseek-v4-glue (`main`, published 2026-08-02).
- Core glue implemented under `src/moonvit_glue/`; **26/26 tests pass on Linux** (torch 2.10.0+cu128, transformers 5.12.1).
- Real `DeepseekV4ForCausalLM` tiny Hash-MoE config passed loss/backward and generation.
- **Gate B complete**: real MoonViT-SO-400M forward/backward on the V100 (`[192,4,1152]` at 448px, `[1064,4,1152]` native 640×480), eval harness dry-run end-to-end (generation + blind + shuffle-loss; untrained projector correctly gives `mean_delta = 0.0`).
- **Gate B training signal confirmed (2026-08-02)**: overfit on 109 ComfyUI captions (93 train / 16 eval), frozen MoonViT + frozen SmolLM2-135M-Instruct, projector-only training on the V100. 200 steps (lr 1e-3) was inconclusive (delta +0.007); **1000 steps (lr 2e-3) gives train loss 4.303→3.338, eval true 3.300 vs shuffled 3.642, shuffle_delta = +0.343**. Generation check (8 records): with-image outputs vary per image with content words (token-F1 0.112); blind outputs are byte-identical generic text (0.082). Placeholder = existing `<|endoftext|>` id 0 (SmolLM2 has no reserved image token). Checkpoint on the workstation at `checkpoints/overfit-smollm135-1k` (gitignored). Data path convention: `image` fields are relative to the JSONL file (the comfy JSONL was fixed accordingly).
- Projector shape fixed at MoonViT `[N,4,1152]` → DeepSeek 4096, 40,119,040 params.
- DeepSeek image placeholder fixed to existing `<｜image｜>` ID 129279; never resize vocab.
- `VisionCausalLM.generate()` exists for both backbone kinds; generic path returns the full expanded sequence.
- Benchmark scaffold landed: `moonvit_glue.metrics` (pure Python), `tools/eval_vlm.py` (generation + `--blind` + `--shuffle-loss`, `--max-image-side`), `tools/fetch_eval_data.py` (TextVQA/DocVQA/OCRBench/ScreenSpot → JSONL + MANIFEST.json). Eval philosophy: always report the blind (no-image) baseline next to capability numbers; grounding reports parse rate, Accuracy@50 on the 0–999 scale, and mean click error, matching the 0xSero GLM-5.2 vision convention.
- Typst report is `report/main.typ`; `report/*.pdf` is gitignored (build artifact) — compile with `typst compile report/main.typ`.
- Read-only Vast snapshot recorded in the report; no instance was created.

## doesworkstation (V100 32GB) operational notes

- SSH alias `doesworkstation` works via Tailscale (100.94.73.9); the Clash TUN fake-IP hijacks the name when TUN is on — use the Tailscale IP if resolution fails. With TUN off, the local machine needs `git -c http.proxy=socks5://127.0.0.1:10808 push` for GitHub.
- **NVML mismatch**: kernel module 580.159.04 vs userland 580.173. `nvidia-smi` fails, but CUDA works fine. Do NOT reload the driver or reboot — the `fastllm` agent's jobs depend on the current state.
- Working env: `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/.venv-1cat/bin/python` — torch 2.10.0+cu128 (**includes sm_70, V100 works**), transformers 5.12.1, safetensors, pillow, einops. No pytest: it was installed into `/run/media/ezra/13D010B6FDBC1A06/moonvit-deps` via `pip --target`; use `PYTHONPATH=$HDD/moonvit-deps:src`.
- HDD (13T, ~3.7T free) is `/run/media/ezra/13D010B6FDBC1A06/` — never write big files to `/home` (89% full). Repo clone lives at `$HDD/moonvit-deepseek-v4-glue`; `HF_HOME=$HDD/huggingface`.
- Workstation proxy: `127.0.0.1:7890` — needed for HF downloads but only ~0.4 MB/s; hf-mirror.com is no faster. **Always set `HF_HUB_DISABLE_XET=1`** — Xet transfers hang through the proxy.
- MoonViT quirks on this stack (Transformers 5.x): remote code lacks `all_tied_weights_keys` (shimmed in `moonvit.py`); bf16 has a mixed-dtype layer_norm bug in the remote code → **run MoonViT in fp32 on the V100** (~1.6 GB).
- Small-context text models overflow on native-resolution images (1064 merged tokens from 640×480 + prompt > 1024 positions of tiny-gpt2 → scatter-gather device assert, which surfaces at unrelated async locations; use `CUDA_LAUNCH_BLOCKING=1` to localize). Use `--max-image-side 448` for small models.
- Under load (other agent compiling/running inference servers), torch import from the mechanical disk takes >90 s — budget timeouts generously.
- The `fastllm` tmux pane runs another agent (GPT-5.6) optimizing Qwen inference; it launches GPU `apiserver` variants. Coordination messages were left in its pane, including the user's request to test `bottlecapai/ThinkingCap-Qwen3.6-27B-GGUF` 4-bit/6-bit quants as a replacement for the Fable Fusion model. ~15.6 GiB VRAM was free; keep our GPU usage small and short.

## Immediate next actions

1. Fetch real eval datasets on the workstation (`tools/fetch_eval_data.py`, needs `datasets` — install via `pip --target $HDD/moonvit-deps datasets`; set proxy + `HF_HUB_DISABLE_XET=1`), then dry-run `tools/eval_vlm.py --shuffle-loss` on real data.
2. ~~Overfit check~~ **done (SmolLM2 track, delta +0.343)** — repeat on the Qwen2.5-0.5B + flickr8k track (download was in progress in `tmux moonvit:0.0`: `Qwen2.5-0.5B-Instruct` ~1 GB, then `fetch_eval_data.py --dataset flickr8k --limit 1100`; placeholder auto-resolves to `<|image_pad|>`; suggested `--steps 300+ --limit 1100`).
3. Re-run the read-only Vast offer search immediately before budgeting; do not create an instance without fresh user approval.
4. On the rented multi-GPU box, run Gate D: native 0731 load → single-image forward → single-batch backward (Dgrad verification) before any training loop exists.
5. HF-cache surgery note: direct `curl -C -` resume loop beats `hf download` on this proxy (it stalls); the blob filename in `blobs/` is the content sha256 — the xet-bridge redirect etag is NOT the file hash.

## Main unresolved risk

Official 0731 FP4/FP8 kernels may support inference but not gradient with respect to input embeddings. The glue contract is tested; the large-quantized-backbone Dgrad path is not.

## Safety / repository hygiene

- Never commit HF, GitHub, Vast, or SSH credentials.
- Never commit model shards, datasets, or projector checkpoints (`data/`, `checkpoints/` are gitignored).
- Keep MoonViT, DeepSeek, and projector revisions/hashes separate.
- The supplied Vast credential was not written into this repository.
- The Vast API was used only with `POST /api/v0/bundles/` (offer search), never the instance-creation endpoint.
