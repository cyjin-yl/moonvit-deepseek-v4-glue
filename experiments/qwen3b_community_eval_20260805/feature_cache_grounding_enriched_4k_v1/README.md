# Qwen2.5-3B grounding-enriched 4k MoonViT feature cache

Package 15J records the canonical frozen-MoonViT cache for the exact Package
15I alternating training order. The clean runner at
`aa933ca1cf5a9386f60cd67658083d4e79b2b376` is bound to training-order
manifest `d632ecc2…0bf1`, MoonViT-V2 weight SHA `01436a95…ced24`, eager
attention, float32 CPU storage, maximum image side 448 and at most 256 visual
groups per row.

All 4,000 rows completed with zero failures in 299.1416 seconds and
1,947,973,120 peak allocated GPU bytes. Content addressing required 2,013 real
tower forwards and reused 1,987 later rows whose encoded images matched an
earlier canonical occurrence. The 63 safetensors shards contain 5,943,468,912
bytes and remain on the local V100 workstation.

`INDEPENDENT_VERIFICATION.json` rehashed all 63 shards, loaded every logical
row, rejected non-finite or shape-mismatched features, checked all aliases
against their first occurrence, and matched all 4,000 rows to the Package 15I
record/image order. It verified 2,742,976,512 logical float values and
1,485,864,960 unique float values. The largest sequence is exactly the frozen
256-token limit. The full remote inventory contains 70 files / 5,946,091,225
bytes and binds each omitted shard by SHA-256.

The canonical full cache remains at
`/run/media/ezra/13D010B6FDBC1A06/data/qwen3b_contract/feature_cache_grounding_enriched_4k_v1`.
This is engineering and input-identity evidence. It contains no trained
candidate or capability score, leaves previous-best at exact step0, does not
evaluate a final half and uses no paid resource. Its DeepSeek migration label
is `directly_transferable`: these frozen MoonViT features and ordered bindings
do not depend on Qwen internals.
