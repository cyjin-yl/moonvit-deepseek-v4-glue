# Package 6: shape-only projector cross-task transfer

## Question and preregistered decision

This package applies the package-5 `shape-projector-step50` checkpoint to the frozen six-task synthetic selection suite without any additional training. Broad transfer requires pair-bootstrap lower bounds above zero for both checkpoint-minus-`step-001500` and vision-minus-shuffled-image strict paired preference on at least three of the five non-shape tasks. A validated shape result with fewer than two validated non-shape tasks is classified as shape-specific.

## Canonical runs

- Teacher-forced preference: 28,800 rows, 12 checkpoint/condition cells, 1,200 complete counterfactual pairs per cell, zero failures. The run took 1,117.6 s and peaked at 3,965,700,608 GPU bytes.
- Free generation: 9,000 rows, 96 held-out shuffle-loss rows, zero failures. The run took 235.2 s and peaked at 1,530,525,184 GPU bytes.
- Joint analysis: 567 metric rows and 525 pair-bootstrap contrasts, 2,000 resamples per contrast. The final odd halves were never scored.

Every checkpoint is evaluated under the same synthetic IDs. Preference uses 200 pairs per task; the disjoint free-generation selection uses 50 pairs per task. The causal claim requires correct-image performance to exceed shuffled-image performance, so task priors alone cannot pass the rule.

## Result

Only shape transfers. Strict paired preference rises from 0.130 to 1.000, checkpoint improvement `+0.870 [0.820, 0.915]`, while vision minus shuffled image is `+0.820 [0.765, 0.875]`. Strict paired free generation rises from 0 to 1.000, improvement `[1.000, 1.000]`; vision minus shuffled image is `+0.980 [0.940, 1.000]`.

Color, coordinate, count, OCR, and spatial have no validated causal transfer. Their strict paired preference improvements are respectively `+0.010`, `0`, `+0.005`, `+0.015`, and `0`, with every lower confidence bound equal to zero. No task has significant negative preference transfer.

This supports a narrow shape-task mapping after 40 training pairs. It refutes broad interface correction from shape-only continuation. The next experiment therefore uses balanced six-task projector continuation before capacity expansion or projector-structure ablation.

## Failure retained

The legacy preference analyzer assumed a `blind` teacher-forced condition that the preregistered preference matrix did not contain. It stopped before producing a valid summary. The empty output directory is retained under `invalid/legacy_preference_analysis/INVALIDATION.json`; the raw preference run remains valid and was re-analyzed by `tools/analyze_multitask_transfer.py`.

## Verification

`PACKAGE_VERIFICATION.json` independently re-reads all raw row counts, binds both source summaries to the joint analysis by SHA-256, verifies every analysis artifact hash, and checks the canonical decision. `ARTIFACT_MANIFEST.json` binds the complete package after report-ready charts and notes are finalized.
