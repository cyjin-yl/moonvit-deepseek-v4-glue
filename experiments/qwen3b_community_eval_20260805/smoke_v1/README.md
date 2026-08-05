# Qwen2.5-3B real-image glue smoke

This package records the first post-contract run of the pinned pure-text
`Qwen/Qwen2.5-3B-Instruct` proxy with the real MoonViT-V2 tower, canonical
4096-wide projector, fixed parameter-free 4096→2048 receiver, and a frozen
public ScreenSpot image.

`failures/attempt01_checkpoint_verifier/` is intentionally invalid. The model
load, generation, backward, optimizer step, and checkpoint write completed,
but the final verifier compared an AdamW scalar on CPU with the restored scalar
on CUDA. The raw failure remains immutable.

`valid_retry1/` changes only that verifier and is the first valid closure.
Submission review then found that input hashes were externally guaranteed by
Package 15A but not rechecked inside the smoke runner. `valid_retry2/` is the
canonical result: before loading, it verifies all 9 Qwen files and the extracted
MoonViT-V2 weight file against the frozen contract. Its checkpoint hashes and
generation rows exactly match retry1. It proves a real-image forward, finite
nonzero gradients in every projector tensor, zero language-model parameter
gradients, one optimizer step, and exact projector/optimizer/RNG/history
restoration. The untrained step0 projector emits the same center click for
vision and blind, so this package makes no visual-capability or benchmark claim.

The complete 470 MB roots remain on the V100 workstation:

- invalid attempt: `/run/media/ezra/13D010B6FDBC1A06/data/qwen3b_contract/smoke_v1`
- first valid retry: `/run/media/ezra/13D010B6FDBC1A06/data/qwen3b_contract/smoke_v1_retry1`
- canonical identity-checked retry: `/run/media/ezra/13D010B6FDBC1A06/data/qwen3b_contract/smoke_v1_retry2`

The checked-in source manifests retain every large-file SHA-256. The package
verification independently rehashed all 12/12 invalid-attempt files and all
13/13 files in each valid retry before the curated small evidence was copied.
