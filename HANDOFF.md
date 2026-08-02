# Handoff

## Current state

- Core glue implemented under `src/moonvit_glue/`.
- 11 tests pass under CPU PyTorch 2.13 and Transformers 5.14.1.
- Real `DeepseekV4ForCausalLM` tiny Hash-MoE config passed loss/backward.
- Projector shape fixed at MoonViT `[N,4,1152]` → DeepSeek 4096, 40,119,040 params.
- DeepSeek image placeholder fixed to existing `<｜image｜>` ID 129279; never resize vocab.
- Typst report is `report/main.typ`.

## Immediate next actions

1. Compile the Typst report in an environment with Typst and CJK fonts.
2. Run the full tests on Linux/CUDA.
3. On `doesworkstation`, coordinate with the `fastllm` tmux pane before any GPU use.
4. Set `HF_HOME=/run/media/ezra/1xxxxxxxx/huggingface` before downloads; `/home` lacks space.
5. Run `examples/smoke_real_moonvit.py` with an image.
6. Record V100 SM70-specific failures and pin the oldest compatible PyTorch/FlashAttention-free path.
7. Do a read-only Vast offer search; do not create an instance.

## Main unresolved risk

Official 0731 FP4/FP8 kernels may support inference but not gradient with respect to input embeddings. The glue contract is tested; the large-quantized-backbone Dgrad path is not.

## Safety / repository hygiene

- Never commit HF, GitHub, Vast, or SSH credentials.
- Never commit model shards, datasets, or projector checkpoints.
- Keep MoonViT, DeepSeek, and projector revisions/hashes separate.
- The supplied Vast credential was not written into this repository.
