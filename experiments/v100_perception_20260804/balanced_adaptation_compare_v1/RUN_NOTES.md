# Package 8: matched extra-projector versus top-12 LoRA

## Registered comparison

Both arms start from the verified balanced-projector step-100 weights. They see the same 2,400 synthetic records in the exact same order (`a0929326…2f5`), true batch 24, four records per task per step, and 100 steps. The projector arm restores the step-100 AdamW state so that it is a real continuation. The top-12 rank-8 LoRA arm starts from an exact zero delta and leaves the projector frozen.

The projector arm trains 20,454,272 parameters and takes 159.1 s / 11.80 GB peak. The LoRA arm trains 442,368 parameters and takes 153.4 s / 9.93 GB peak. Step 1 loss is exactly 1.221496 for both arms. At step 100 the projector loss is 0.950940 with pre-clip gradient norm 0.590; LoRA loss is 1.306617 with pre-clip gradient norm 8.639. Both runs complete without OOM or non-finite values, and all checkpoint tensors reload under the independent verifier.

## Canonical bf16 endpoint result

The canonical run contains 21,600 preference rows and 3,600 generation rows across frozen base, LoRA step 100, and projector step 100. All cells contain complete counterfactual pairs and no failure rows. Confidence intervals use 2,000 pair bootstrap resamples.

Overall strict paired preference is 0.224 at the base, 0.247 after LoRA, and 0.511 after the extra projector epoch. The projector gain is +0.287 [0.258, 0.318]; the LoRA gain is +0.023 [-0.003, 0.049]. Overall paired generation is 0.063/0.080/0.257; projector gain +0.193 [0.147, 0.240], LoRA gain +0.017 [-0.020, 0.050].

The projector arm unlocks paired generation for color (0.160, gain +0.160 [0.060, 0.260]), coordinate (0.240, +0.240 [0.120, 0.360]), and spatial (1.000, +0.780 [0.660, 0.880]). OCR gains teacher-forced strict preference (+0.085 [0.040, 0.130]) but remains at zero generation. Count remains unresolved. Shape regresses in strict preference by -0.125 [-0.180, -0.065] and does not improve generation.

LoRA produces a task-specific shape gain: strict preference +0.320 [0.250, 0.395] and paired generation +0.320 [0.200, 0.460]. It does not close color/coordinate/count/OCR generation, and it significantly erases spatial generation (-0.220 [-0.340, -0.120]) while reducing count preference. This refutes a broad top-layer-use fix and exposes task competition in both adaptation mechanisms.

## Precision sensitivity

The first full endpoint evaluation was internally matched but used the fp32 projector training dtype. Package 7 used bf16 for canonical evaluation. The fp32 baseline moved several thresholded metrics, including spatial strict preference/generation from 0.250/0.220 to 0/0. That valid diagnostic is retained as v1. The bf16 v2 baseline exactly reproduces all package-7 per-task values and is authoritative for package-8 conclusions.

## Retained failures and next decision

The first two one-step smoke directories failed before model loading because the remote config copy had been interrupted; they contain only the resulting `FileNotFoundError` and contribute no metrics. Replacement v2 smokes have identical first-batch loss and order SHA, then the full runs pass.

Additional projector training is the stronger broad direction, but its shape regression and the LoRA endpoint gradient spike make a single endpoint insufficient for scheduling the next long run. The next local experiment is a balanced step-25/50/100 trajectory screen at canonical bf16 precision. It will test early LoRA peaks and projector task interference before choosing learning-rate, replay weighting, auxiliary objective, or Qwen2.5-1.5B capacity controls. Final odd halves remain unscored and no paid resource was used.
