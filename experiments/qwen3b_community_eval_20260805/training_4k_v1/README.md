# Qwen2.5-3B fixed-budget 4k projector training

This package records the first formal projector-only training run under the
preregistered Qwen2.5-3B community evaluation contract. The clean runner at
`97e9c03403b2673e68f2ed7fc421630add8a9d3a` consumed the exact Package-15C
4,000-record prefix and the independently verified Package-15D MoonViT cache.
It ran 500 AdamW optimizer steps with micro batch 1, gradient accumulation 8,
real global batch 8, constant learning rate `5e-4`, and no weight decay.

The run saw 4,000 examples and 21,532 answer tokens, equal to one pass over the
frozen prefix and 0.06757 effective epochs of the 59,198-row source mix. The
3,085,938,688 Qwen parameters and fixed 4096-to-2048 receiver remained frozen;
only the 33,564,672 FP32 projector parameters were trainable. Every projector
tensor had finite nonzero gradients at steps 1 and 500, and the language model
had zero gradient tensors. Training took 532.81 seconds, total process time was
905.39 seconds, and peak V100 allocation was 8,979,616,768 bytes.

Five exact-resume checkpoints were written at steps 100 through 500. Their 25
payload files total 2,351,006,545 bytes and remain on the V100 workstation. The
checked-in `INDEPENDENT_VERIFICATION.json` binds every file by size and SHA-256,
reconstructs the 500 batches and 21,532 supervised tokens, verifies optimizer
and RNG state, and confirms that the final FP32 projector differs from step0.
The final projector SHA-256 is
`566830f3b6f85f5aa66b13566054022bcffce3660d5b2210fc5ee192834ca89f`.

The first verifier attempt exposed a field-routing bug before producing an
accepted verification. Its failure record is retained under `failures/`; the
fixed independent verifier is commit
`075f3e5889aa445a9ca748bb8ecfc21ba96abacc`.

Falling loss is an optimization result only. Visual ability is decided by the
paired ScreenSpot controls in the adjacent `screenspot_glm50_4k_v1` package.
The training machinery is `transferable_with_runtime_validation`; this learned
checkpoint is not promoted to a DeepSeek candidate after grounding evaluation.
No paid resource or final-half evaluation was used.
