# Package 15Q — projector output-normalization structure screen

Package 15P tested the frozen step0-geometry objective at four pre-registered
gradient doses. Every arm stopped at optimizer step 2 with the same
representation collapse. Package 15Q changes one structural variable at a
time: a parameter-free normalization applied after `linear_2` at the canonical
4096 boundary.

The frozen arms are:

* `baseline_none`: existing projector, CE-only control;
* `post_layernorm`: affine-free LayerNorm at the 4096 output;
* `post_rmsnorm`: affine-free RMSNorm at the 4096 output.

All arms reuse the exact step0 MLP weights, MoonViT-V2 cache, Qwen2.5-3B
contract, receiver, data order, 100-step/800-example budget and projector-health
schedule. The normalization adds no trainable parameters and keeps the output
width 4096, so the candidates are directly transferable to DeepSeek after
runtime validation.

Each run must pass the independent structure verifier before training. Health
preservation is a screening result only; no arm can claim visual ability without
the fixed ScreenSpot, TextVQA, DocVQA and OCRBench contract. If none survives
the early health/CE rule, the next registered candidate is a residual or gated
residual projector, with no 500-step expansion.
