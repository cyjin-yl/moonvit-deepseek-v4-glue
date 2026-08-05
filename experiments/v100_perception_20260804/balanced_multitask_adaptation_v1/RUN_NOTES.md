# Package 7: balanced six-task projector continuation

## Decision carried from package 6

Package 6 showed that 40 shape training pairs produce a narrow task mapping with no validated transfer to the other five synthetic tasks. Package 7 therefore keeps the MoonViT tower and Qwen2.5-0.5B language model frozen, starts from the historical step-1500 projector, and gives every task equal supervision before testing a larger language backbone or a larger projector.

## Cache and true-batch training

The complete 2,400-record train split was encoded once at 256 px into 75 float32 safetensor shards. Independent verification rehashed 3,932,170,800 shard bytes and read all 2,400 tensors / 983,040,000 values; every shape and value was finite. Caching took 251.5 s and peaked at 1,946,604,032 GPU bytes.

Training uses true batch 24 with exactly four records from each of color, coordinate, count, OCR, shape, and spatial at every optimizer step. Step 100 sees every train record exactly once: 400 records / 200 pairs per task. Loss is 2.7778/1.4495/1.4594/1.3490 at steps 1/25/50/100. The run took 144.5 s and peaked at 11,797,675,520 GPU bytes. Independent verification rehashed four complete checkpoints, checked 24 projector tensors, and confirmed 400 examples per task.

## Teacher-forced capability trajectory

The canonical preference run contains 38,400 rows, 16 checkpoint/condition cells, 1,200 complete pairs per cell, and zero failures. All confidence intervals use 2,000 complete-pair bootstrap resamples.

Validated tasks accumulate with training: coordinate at step 25; color and shape at step 50; all six tasks at step 100. At step 100, strict paired preference and shuffled-image controls are:

| task | vision | shuffled | checkpoint gain (95% CI) | vision minus shuffle (95% CI) |
|---|---:|---:|---:|---:|
| color | 0.230 | 0.095 | +0.230 [0.175, 0.290] | +0.135 [0.065, 0.205] |
| coordinate | 0.055 | 0.000 | +0.055 [0.025, 0.090] | +0.055 [0.025, 0.090] |
| count | 0.115 | 0.060 | +0.115 [0.075, 0.160] | +0.055 [0.005, 0.105] |
| OCR | 0.135 | 0.050 | +0.135 [0.085, 0.185] | +0.085 [0.040, 0.140] |
| shape | 0.560 | 0.155 | +0.430 [0.330, 0.525] | +0.405 [0.325, 0.480] |
| spatial | 0.250 | 0.000 | +0.250 [0.190, 0.310] | +0.250 [0.190, 0.310] |

The trajectory is not monotonic at short horizons. Shape drops by `-0.130 [-0.180, -0.085]` at step 25, then recovers to 0.435 at step 50 and 0.560 at step 100. This is transient multi-task interference, not a terminal failure.

## Teacher forcing versus free generation

The canonical generation run contains 12,000 synthetic rows plus 128 held-out shuffle-loss rows, zero failures, and four checkpoints. At step 100, only shape and spatial have validated paired-generation improvement: shape `+0.160 [0.080, 0.280]`, spatial `+0.220 [0.120, 0.340]`. Their vision-minus-shuffled paired effects are respectively `+0.140 [0.060, 0.240]` and `+0.220 [0.100, 0.340]`. Color, coordinate, count, and OCR remain at zero paired generation despite validated teacher-forced visual selection.

This supports insufficient task coverage as the main cause of the teacher-forced floor and refutes a general projector information-loss explanation. It also exposes a remaining use/decoding bottleneck in the frozen language stack. The next discriminating experiment is a matched small top-layer LoRA screen from balanced step 100 versus additional projector-only epochs, evaluated with the same paired matrix.

## Failures retained

1. `balanced_multitask_projector_smoke_v1` stopped before optimizer step 1 because Qwen padding and the image placeholder share token ID 151643. Mixed-length padding was counted as extra images. The merge path now counts placeholders only under an active attention mask; the replacement batch-24 smoke passed at 11.43 GB peak allocation.
2. `balanced_multitask_preference_v1` wrote only the frozen baseline's 9,600 rows before an adapted-checkpoint provenance field was found missing. It is invalid; v2 reran all 38,400 rows under one config hash.
3. `balanced_multitask_generation_v1` stopped before any evaluation row because the trajectory runner required a random checkpoint. The runner now allows saved-checkpoint-only trajectories; the replacement smoke and full v2 run passed.

No failed run contributes to any metric. The final odd halves were never scored and no paid resource was used.
