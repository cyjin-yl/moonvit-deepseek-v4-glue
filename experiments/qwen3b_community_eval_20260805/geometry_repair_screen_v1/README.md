# Package 15P — projector geometry-repair short screen

Package 15O found that the grounding-enriched projector is already grossly
collapsed at the first saved checkpoint (step 100 / 800 examples). This package
therefore tests a training-only geometry objective that acts from optimizer
step one at the canonical 4096 boundary.

The objective compares each real global batch against the exact frozen step0
projector on the same cached MoonViT features. It combines log-scale,
relative-spread, and normalized centered-Gram losses. The Gram term preserves
cross-image geometry while permitting a shared orthogonal rotation, so the
projector can still align to a language space. No Qwen tokenizer, chat-template,
native-vision, or cross-attention feature is used.

Before training, the unweighted auxiliary gradient is calibrated on the frozen
step100/batch100 state. The contract deterministically derives three fixed
lambdas whose auxiliary-gradient norms are 5%, 20%, and 80% of the recorded CE
gradient norm. A lambda-zero control and all three arms then see the exact same
first 800 records in the same order for 100 optimizer steps.

The smallest arm that clears both projector and receiver representation guards
while keeping final-20-step CE loss within 1.25× control advances to the exact
500-step budget. A short-screen pass is representation evidence only; capability
still requires paired preference and generation gates. No paid resource or
frozen final half is allowed.

Calibration is complete and independently verified under `calibration/`. The
derived auxiliary-gradient ratios are 0.05, 0.20, and 0.80 with fixed λ values
0.01018730507868909, 0.04074922031475636, and 0.16299688125902545. The first
shell logging attempt is retained under `calibration/failures/`; it was an
output-directory ordering error and did not invalidate the GPU result.
