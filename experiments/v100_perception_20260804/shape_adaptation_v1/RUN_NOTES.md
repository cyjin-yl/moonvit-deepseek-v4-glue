# Package 5: shape adaptation diagnostic

This package tests the package-4 localization result with two matched local V100 interventions. Both arms start from the step-1500 projector, see the same 400 shape-training records in the exact same true-batch order, use batch size 8, and save checkpoints at 0/50/100/200 optimizer steps. The selection split is disjoint by record and pair ID.

## Arms

- `lora_top12`: rank-8 LoRA on `q_proj`, `v_proj`, and `o_proj` in Qwen layers 12–23; 442,368 trainable parameters.
- `projector_continuation`: continue all projector weights; 20,454,272 trainable parameters.
- Frozen step 1500 is evaluated in the same process as the shared baseline.

The two training-order files have identical SHA-256 `993f1b2eb7015c042fcb7d76bbb475fe6aae865cd43be8d194e56aa118476988`.

## Decisive results

Teacher-forced strict paired preference / free-generation paired accuracy:

| state | examples seen | preference | generation |
|---|---:|---:|---:|
| frozen step 1500 | 0 | 0.130 | 0.000 |
| LoRA step 50 | 400 | 0.180 | 0.000 |
| LoRA step 100 | 800 | 0.605 | 0.080 |
| LoRA step 200 | 1,600 | 0.430 | 0.080 |
| projector step 50 | 400 | 1.000 | 1.000 |
| projector step 100 | 800 | 0.945 | 0.880 |
| projector step 200 | 1,600 | 0.945 | 0.880 |

Projector step 50 improves strict paired preference over frozen by `+0.870 [0.825, 0.915]` and paired generation by `+1.000 [1.000, 1.000]`. Its vision-minus-shuffled strict paired gap is `+0.820 [0.765, 0.870]`; shuffled-image strict paired accuracy is 0.180. At equal 400 examples, LoRA minus projector is `-0.820 [-0.870, -0.765]` for strict paired preference and `-1.000 [-1.000, -1.000]` for paired generation.

Layerwise replay explains the output change. LoRA step 100 retains the original layer-12 peak (balanced 0.816) and raises final assistant/native readout only to balanced 0.500. Projector step 50 creates a sustained late-layer path: layer 17 reaches 1.000, the final assistant probe is 0.945, and the native LM-head is 1.000. Every primary vision probe has pair-permutation `p=1/2001`.

## Hypothesis update

- Supported: the frozen tower and step-1500 projector already expose shape information, while the trained interface has not yet made that information robustly usable by the frozen language head.
- Supported: upper-layer language adaptation can partially recover instruction-visible behavior, so the frozen language stack contributes a real use/decoding bottleneck.
- Strongly supported: for this shape regime, insufficient projector-interface training is the dominant correctable bottleneck. A short 400-example continuation outperforms the larger-example LoRA trajectory and restores a late-layer native path.
- Refuted for this diagnostic: a larger projector architecture is required before any local improvement is possible.
- Unresolved: transfer to OCR, counting, coordinates, spatial reasoning, and natural-image domain shift. This package trains shape only and cannot establish broad visual alignment.

## Audit and storage

The independent verifier re-read four checkpoints per arm, 8400 preference rows, 1400 generation rows, 63 bootstrap contrasts, two 2000-row representation suites, 448 probe-metric rows, and 179,200 probe-prediction rows. Full projector checkpoints and the two 286 MB representation runs remain on the V100 HDD; `RAW_DATA_LOCATION.json`, run summaries, and checkpoint manifests bind their exact paths and hashes. The first LoRA smoke run is retained with an explicit invalidation caused by an examples-seen accounting defect; its replacement smoke and the projector smoke are retained under `screening/`.

## Next local decision

The next high-information experiment is a multi-task projector continuation screen with the same cached true-batch path. It will test whether the 400-example shape recovery transfers across the six synthetic tasks or reflects a narrow shape-only fit. If broad recovery holds, auxiliary-objective and initialization ablations become worthwhile; if it stays shape-specific, use balanced multi-task supervision before moving to the 1.5B capacity control.
