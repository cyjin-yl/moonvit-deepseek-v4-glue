# Qwen2.5-3B grounding-enriched fixed-budget training

Package 15K records the one-seed training screen preregistered in Package 15I
and fed by the independently verified Package 15J cache. The treatment starts
from the exact canonical step0 projector and runs 500 AdamW optimizer steps,
micro batch 1, accumulation/global batch 8, 4,000 examples and 0.06756985 full
pack effective epochs. Every global batch contains four ShowUI grounding rows
and four short-answer rows.

The frozen FP16 Qwen has 3,085,938,688 parameters. The fixed receiver and Qwen
both have zero trainable parameters; only the 33,564,672-parameter FP32
projector is optimized. The run sees 36,589 assistant answer tokens. All six
projector tensors have finite nonzero gradients at steps 1 and 500, while the
language model has exactly zero gradient tensors. Loss moves from 4.14400 to
1.91563 with mean 2.44623; this is optimization evidence only.

Training takes 489.606 seconds, total process wall time is 529.299 seconds and
peak V100 allocation is 8,973,374,976 bytes. Five checkpoints at steps
100/200/300/400/500 bind FP32/BF16 projector state, AdamW state, history and
RNG. `INDEPENDENT_VERIFICATION.json` reconstructs all batches and answer-token
counts, rehashes every checkpoint file and confirms six optimizer parameter
states. The 25 checkpoint payloads total 2,351,007,317 bytes. Final FP32
projector SHA-256 is `62f69393…3df4`, distinct from step0 `efd942e0…b06b0`.

The canonical root remains at
`/run/media/ezra/13D010B6FDBC1A06/data/qwen3b_contract/train_grounding_enriched_4k_v1`.
The full 40-file inventory is 2,353,629,390 bytes and remains on the local
V100. This package establishes train/save/resume integrity. It makes no visual
capability claim, leaves previous-best at exact step0, does not evaluate a
final half and uses no paid resource. Its implementation transfer label is
`transferable_with_runtime_validation`; the learned checkpoint enters no
DeepSeek candidate list until causal evaluation passes.
