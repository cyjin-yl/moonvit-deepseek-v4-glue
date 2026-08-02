# Inference-side integration: MoonViT projector on DeepSeek-V4-Flash-0731

This document is the contract for using the trained projector at **inference** time,
written for patching inference engines (vLLM / SGLang) and for publishing the
checkpoint on Hugging Face. The executable reference implementation is
`src/moonvit_glue/` (see `model.py`, `merge.py`, `projector.py`); the unit tests in
`tests/` define the behavior that any engine port must reproduce.

## 1. Published artifacts (end-of-rental upload)

| File | Dtype | Size | Purpose |
|---|---|---|---|
| `projector.safetensors` | fp32 | ~160 MB | training master (reproducibility, further fine-tuning) |
| `projector_bf16.safetensors` | bf16 | ~80 MB | serving dtype — 0731 activations are bf16 |
| `projector_config.json` | — | small | structural contract (below) |
| `eval/*.json` | — | small | benchmark results incl. blind baselines |
| `README.md` | — | — | model card with usage code |

Only the projector is published, never the 160 GB backbone — same convention as the
community GLM-5.2V projector checkpoint.

`projector_config.json` (current, from `configs/deepseek-v4-flash-0731-projector.json`):

```json
{
  "language_width": 4096,
  "layer_norm_eps": 1e-05,
  "merge_factor": 4,
  "projector_width": 4608,
  "vision_width": 1152
}
```

The upload must also record: MoonViT repo+resolved revision, DeepSeek repo+resolved
revision, placeholder token id, training data manifest, and safetensors sha256.

## 2. The pipeline any engine must implement

```
image (native resolution, RGB)
  → MoonViT-SO-400M preprocessing (pinned remote code, revision-locked)
  → MoonViT encoder        → feature groups [G, 4, 1152]
  → PatchMergerProjector   → vision embeddings [G, 4096]
       LayerNorm(1152) → flatten(4×1152=4608) → Linear(4608,4608) → GELU → Linear(4608,4096)
  → embedding merge: each `<｜image｜>` placeholder expands to G positions
  → DeepSeek-V4-Flash-0731 (frozen), DSpark/MTP excluded
```

Hard constraints:

- **Placeholder token is the existing `<｜image｜>` = id 129279.** Never resize the
  vocab: resizing would invalidate the frozen input embedding, the LM head, and the
  Hash-MoE `tid2eid` routing tables.
- **G (token count per image) is whatever the pinned MoonViT preprocessing produces**,
  not a closed-form formula — e.g. 448×448 → 192 groups, 640×480 → 1064 groups. The
  input processor must run the same preprocessing to compute G per image, exactly like
  Kimi-VL's processor does. Our `tools_common.encode_image` is the reference.
- **DeepSeek-specific merge rule**: the first layers contain Hash-MoE whose expert
  routing indexes `input_ids`. Keep the expanded placeholder ids in `input_ids` for the
  router, and replace only the *embedding vectors* at those positions with the projected
  vision embeddings. In `model.py` this is done with a one-shot hook on the embedding
  module; engines with a native multimodal merge (vLLM/SGLang) already separate
  `input_ids` from merged embeddings for other VLMs — verify the Hash-MoE path reads
  `input_ids` (not merged embeddings) and add a regression test for it.
- **No gradient requirement at inference**: FP4/FP8 inference kernels are sufficient;
  the Dgrad question only matters for training.
- MoonViT runs fine in bf16 at inference; the fp32 requirement in this repo is a V100
  (sm_70) training-stack workaround, not an architecture constraint.

## 3. vLLM patch plan (intended PR)

vLLM already ships the two hardest pieces:

- MoonViT vision encoder — `vllm/model_executor/models/kimi_vl.py` (and the K2.5
  variant `kimi_k25_vit`), i.e. the same tower our weights target.
- A multimodal wiring pattern for DeepSeek-family text stacks (`SupportsMultiModal`,
  input processors, `merge_multimodal_embeddings`); the community note for porting
  Kimi-VL even recommends "replace DeepseekV2Model from vllm" as the closest base.

Patch outline (new file `vllm/model_executor/models/deepseek_v4_moonvit.py`):

1. Register `DeepseekV4MoonvitForCausalLM` implementing `SupportsMultiModal`.
2. Vision tower: reuse the Kimi-VL MoonViT classes, load `moonshotai/MoonViT-SO-400M`
   (pinned revision) — run it with `--mm-encoder-tp-mode data` (small encoder, avoid TP
   overhead, same flag Kimi-VL uses).
3. Projector: new `MoonViTPatchMerger` module matching section 2 shapes; load
   `projector_bf16.safetensors` from the published repo.
4. Input processor: for each image, run the pinned preprocessing to get G, expand the
   single `<｜image｜>` placeholder into G positions of id 129279.
5. Embedding merge: standard vLLM multimodal merge; then the DeepSeek-specific check —
   Hash-MoE routing in the first layers must consume `input_ids` (which still hold
   129279), never the merged embeddings. Add a unit test asserting placeholder positions
   route identically with/without image replacement.
6. Serve with the FP4/FP8 0731 weights exactly as the text-only DeepSeek-V4 recipe;
   vision adds only encoder + merge. DSpark/MTP stays out of the vision path.

## 4. SGLang patch plan

SGLang's supported-models documentation lists Kimi-VL (MoonViT encoder), so a
MoonViT reference exists there as well. Same outline: new model file under
`python/sglang/srt/models/`, reuse its MoonViT + multimodal registry, reuse the engine's
existing 0731 FP4 path, and mirror the section 3 Hash-MoE regression test.

## 5. llama.cpp / GGUF (explicitly out of scope for v1)

Would require a MoonViT→GGUF mmproj converter plus a compatible FP4 GGUF of 0731 and a
custom tensor mapping — a separate project. The published fp32 master keeps this path
open.

## 6. Acceptance checks for any engine port

- Placeholder/vision-token count mismatch raises loudly (no silent truncation).
- Same image + prompt produces logit-equivalent output (within dtype tolerance) to the
  reference implementation on a small config.
- Blind (no-image) output equals the text-only model on identical text prompts.
- `shuffle-loss` sanity on a few samples: true-image loss < shuffled-image loss.
