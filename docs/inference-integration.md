# Inference-side integration: MoonViT-V2 (K3) + projector on DeepSeek-V4-Flash-0731

This document is the contract for using the trained projector at **inference**
time, written for patching inference engines (vLLM / SGLang / llama.cpp /
fastllm) and for publishing the checkpoint on Hugging Face. The executable
reference implementation is `src/moonvit_glue/` (see `model.py`, `merge.py`,
`projector.py`, `moonvit_v2.py`); the unit tests in `tests/` define the
behavior that any engine port must reproduce.

> Vision-tower note (2026-08-03): the project standard is **MoonViT-V2**, the
> Kimi-K3 vision tower (vision width **1024**, `sd2_tpool` merge), extracted
> from `moonshotai/Kimi-K3` and published as `vision_tower_k3/` in our HF
> repo. The earlier MoonViT-SO-400M (width 1152) path remains available for
> comparison; do not mix configs between the two.

## 1. Published artifacts (end-of-rental upload)

| File | Dtype | Size | Purpose |
|---|---|---|---|
| `projector.safetensors` | fp32 | ~134 MB | training master (reproducibility, further fine-tuning) |
| `projector_bf16.safetensors` | bf16 | ~67 MB | serving dtype — 0731 activations are bf16 |
| `projector_config.json` | — | small | structural contract (below) |
| `vision_tower_k3/moonvit_v2.safetensors` | bf16 | 802 MB | extracted K3 vision tower + `MANIFEST.json` (sha256) |
| `eval/*.json` | — | small | benchmark results incl. blind baselines |
| `README.md` | — | — | model card with usage code |

Only the projector and the (separately licensed) vision tower are published,
never the 160 GB DeepSeek backbone — same convention as the community
GLM-5.2V projector checkpoint.

`projector_config.json` (MoonViT-V2 variant, from
`configs/deepseek-v4-flash-0731-projector-moonvit-v2.json`):

```json
{
  "language_width": 4096,
  "layer_norm_eps": 1e-05,
  "merge_factor": 4,
  "projector_width": null,
  "vision_width": 1024
}
```

`projector_width: null` means `effective = vision_width × merge_factor = 4096`.
Parameter count: LN(1024) 2,048 + Linear(4096→4096) 16,781,312 +
Linear(4096→4096) 16,781,312 = **33,564,672** (~33.6 M).

The upload must also record: K3 repo+resolved revision, vision-tower shard
sha256 (`9d10c74f…`), artifact sha256 (`01436a95…`), DeepSeek repo+resolved
revision, placeholder token id, training data manifest, and projector
safetensors sha256.

## 2. The pipeline any engine must implement

```
image (native resolution, RGB)
  → K3 vision preprocessing (NaViT: resize within patch limits, pad,
    normalize 0.5/0.5, patchify 14×14) → pixel_values [N,3,14,14], grid_thws
  → MoonViT3d encoder (27 layers, 1024-wide, rmsnorm, RoPE-2D,
    divided-fixed pos-emb) + sd2_tpool merge → feature groups [G, 4, 1024]
  → PatchMergerProjector → vision embeddings [G, 4096]
       LayerNorm(1024) → flatten(4×1024=4096) → Linear(4096,4096) → GELU → Linear(4096,4096)
  → embedding merge: each `<｜image｜>` placeholder expands to G positions
  → DeepSeek-V4-Flash-0731 (frozen), DSpark/MTP excluded
```

Hard constraints:

- **Placeholder token is the existing `<｜image｜>` = id 129279.** Never resize
  the vocab: resizing would invalidate the frozen input embedding, the LM
  head, and the Hash-MoE `tid2eid` routing tables. (K3's own
  `<|kimi_image_placeholder|>` = 163605 belongs to the K3 tokenizer only —
  irrelevant here.)
- **G (tokens per image) comes from the K3 processor**, not a closed-form
  formula: `navit_resize_image` with `patch_limit_on_one_side=512`,
  `in_patch_limit=65536`, then 2×2 spatial merge — e.g. 1024×1024 →
  74×74 grid → G=1369. Reference: `vendor/kimi_k3/kimi_k3_vision_processing.py`
  driven through `moonvit_glue.moonvit_v2.load_moonvit_v2_processor`.
- **DeepSeek-specific merge rule**: the first layers contain Hash-MoE whose
  expert routing indexes `input_ids`. Keep the expanded placeholder ids in
  `input_ids` for the router, and replace only the *embedding vectors* at
  those positions with the projected vision embeddings. In `model.py` this is
  a one-shot hook on the embedding module; engines with a native multimodal
  merge (vLLM/SGLang) already separate `input_ids` from merged embeddings —
  verify the Hash-MoE path reads `input_ids` (not merged embeddings) and add
  a regression test for it.
- **No gradient requirement at inference**: FP4/FP8 inference kernels are
  sufficient; the Dgrad question only matters for training.
- MoonViT-V2 is natively bf16 (K3 shipped bf16 weights); the fp32 runs in
  this repo are a V100 (sm_70) workaround, not an architecture constraint.

## 3. vLLM patch plan (intended PR)

Both hard pieces already exist upstream (verified 2026-08-03):

- **Kimi-K3 day-0 support** including the MoonViT3d tower and its multimodal
  wiring ([vLLM blog, 2026-07-27](https://vllm.ai/blog/2026-07-27-k3);
  recipes at `recipes.vllm.ai/moonshotai/Kimi-K3`). The vision classes we
  need are the same architecture our extracted weights load into — key/shape
  parity was verified tensor-by-tensor (165/165) against the K3 code our
  `vendor/kimi_k3/` snapshot is cut from.
- **DeepSeek-V4 text stack**: registered as `vllm.models.deepseek_v4` (check
  `architectures: DeepseekV4ForCausalLM`, `model_type: deepseek_v4` in the
  checkpoint config; Flash uses `expert_dtype fp4`).

Patch outline (new file, e.g. `vllm/model_executor/models/deepseek_v4_moonvit.py`):

1. Register `DeepseekV4MoonvitForCausalLM` implementing `SupportsMultiModal`.
2. Vision tower: reuse the K3 MoonViT3d classes from vLLM's Kimi-K3 model
   file; load `vision_tower_k3/moonvit_v2.safetensors` (verify
   `MANIFEST.json` sha256 first). Run the encoder with
   `--mm-encoder-tp-mode data` (small encoder, avoid TP overhead).
3. Projector: new `MoonViTPatchMerger` module matching section 2 shapes
   (pre-LN → flatten → 4096→4096 → GELU → 4096→4096); load
   `projector_bf16.safetensors`.
4. Input processor: port the K3 NaViT preprocessing to compute G per image
   and expand the single `<｜image｜>` placeholder into G positions of id
   129279. (Do **not** reuse K3's placeholder expansion — different tokenizer,
   different token id.)
5. Embedding merge: standard vLLM multimodal merge; then the DeepSeek
   check — Hash-MoE routing in the first layers must consume `input_ids`
   (still 129279 at vision positions), never the merged embeddings. Add a
   unit test asserting placeholder positions route identically with/without
   image replacement.
6. Serve with the FP4/FP8 0731 weights exactly as the text-only DeepSeek-V4
   recipe; vision adds only encoder + merge. DSpark/MTP stays out of the
   vision path.

## 4. SGLang patch plan

Same shape as vLLM. SGLang natively implements K3
([LMSYS blog, 2026-07-27](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support/))
and serves DeepSeek-V4 (see the NVFP4 integration notes, e.g. SGLang PR
25820 for NVFP4 autodetect). New model file under
`python/sglang/srt/models/`: reuse its MoonViT3d implementation + multimodal
registry, reuse the engine's existing 0731 path, port section 2's projector
and preprocessing, and mirror the section 3 Hash-MoE regression test.

## 5. llama.cpp (GGUF) — out of scope for v1, patch points documented

llama.cpp's multimodal path is `libmtmd` (`tools/mtmd/`, vision encoders in
`clip.cpp`, projector in the `mmproj` GGUF). There is community work on
DeepSeek-V4-Flash support (e.g. the `antirez/llama.cpp-deepseek-v4-flash`
fork), but a working port would require, in order:

1. **Text side**: a DeepSeek-V4 GGUF conversion (`convert_hf_to_gguf.py`
   support for `deepseek_v4` incl. Hash-MoE `tid2eid` tables) plus a
   compatible FP4/FP8 quantization path — as of 2026-08-03 there is no
   confirmed upstream support; the fork above is the closest reference.
2. **Vision side**: a new MoonViT3d architecture in `clip.cpp` (2D RoPE,
   divided-fixed positional embedding, `sd2_tpool` merge) — none of the
   existing clip architectures match.
3. **Projector**: fold our PatchMerger into the mmproj GGUF (its tensor
   layout is a trivial 2-layer MLP + LN; the converter script needs a new
   mapping).
4. **Merge logic**: `mtmd` must keep placeholder ids in the token stream for
   Hash-MoE routing while substituting embeddings — this separation does not
   exist in mtmd today and is the deepest change.

Estimated effort: weeks. The published fp32 master keeps this path open.

## 6. fastllm — not recommended for v1

fastllm targets low-VRAM/CPU-offload serving of mid-size models; neither
Kimi-K3 (2.8T) nor DeepSeek-V4-Flash (304B) is in its usual envelope, and we
found no confirmed support for either as of 2026-08-03. A port would mean
(a) registering `deepseek_v4` in its model factory with NVFP4 dequant
kernels, and (b) reimplementing MoonViT3d + projector in its C++ operator
layer. For local smoke-testing of the *vision side only*, our PyTorch
reference (`examples/`, `tools/eval_vlm.py` on any 8–24 GB GPU with a small
text backbone) is the supported path — that is what the V100 workstation
runs.

## 7. Acceptance checks for any engine port

- Placeholder/vision-token count mismatch raises loudly (no silent
  truncation).
- Same image + prompt produces logit-equivalent output (within dtype
  tolerance) to the reference implementation on a small config.
- Blind (no-image) output equals the text-only model on identical text
  prompts.
- `shuffle-loss` sanity on a few samples: true-image loss < shuffled-image
  loss.
- V2 tower in bf16 vs the fp32 reference: per-feature max abs diff within
  bf16 tolerance (reference eager-vs-sdpa fp32 diff is 3.1e-05; bf16 should
  land ~1e-2 relative — record the measured value in the PR).
- Hash-MoE regression: placeholder positions route to the same experts with
  and without image embedding replacement.
