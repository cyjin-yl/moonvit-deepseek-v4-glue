# Package 11: matched projector retention objective

## Question

Can a single task-conditioned representation objective preserve the step-50 count/shape abilities while the exact original step-51→100 continuation still acquires coordinate/spatial ability?

## Matched design

- Start from `balanced_compare_projector_v1/checkpoints/step-000050`.
- Restore the exact step-50 AdamW state (`57e9ddb…ac2f32`).
- Read the original step-51→100 batches from training-order SHA `a0929326…f2f5`.
- Keep batch size 24, learning rate, examples seen, record pool, seed, frozen language model, and all six task counts unchanged.
- Compare the unregularized continuation with count/shape-only MSE anchoring against the frozen step-50 projector outputs at weights `1e-4`, `1e-3`, and `1e-2`.
- Select a candidate only if count and shape remain within 0.05 of step 50, coordinate and spatial both strictly improve over step 50, and macro strict preference remains within 0.02 of the better endpoint.

The control reproduces all six target tensors exactly. Its final tensor SHA is `7b731cff…a76`, and the serialized projector file SHA also exactly matches the original step-100 file (`05f19079…092d`).

## Result

No coefficient passes the preregistered retention rule.

| state | macro strict | worst strict | macro paired generation | count | shape | coordinate | spatial |
|---|---:|---:|---:|---:|---:|---:|---:|
| step 50 | 0.5267 | 0.180 | 0.2333 | 0.42 | 0.80 | 0.44 | 0.74 |
| control step 100 | 0.5167 | 0.120 | 0.2567 | 0.12 | 0.48 | 0.54 | 1.00 |
| anchor `1e-4` | 0.4967 | 0.160 | 0.2467 | 0.16 | 0.42 | 0.48 | 1.00 |
| anchor `1e-3` | **0.5700** | 0.100 | **0.3833** | 0.10 | 0.54 | 0.66 | 1.00 |
| anchor `1e-2` | 0.5433 | 0.160 | 0.3000 | 0.16 | 0.54 | 0.72 | 0.76 |

The `1e-3` arm is a real Pareto diagnostic even though it fails the target. Relative to the exact control, overall strict preference improves by **+0.0533 [0.0200, 0.0867]** and paired generation by **+0.1267 [0.0900, 0.1633]**. Relative to step 50, count falls **−0.32 [−0.46, −0.18]** and shape falls **−0.26 [−0.38, −0.14]**. Color, coordinate, and spatial generation improve, while count remains 0.02 and OCR remains 0.

The result refutes the claim that a task-conditioned full projector-output MSE is sufficient to preserve the old decision boundaries. It supports a narrower claim: the auxiliary objective changes the reachable trade-off and can improve macro preference and generation, so the optimization path is controllable. Representation distance alone is not aligned with task retention.

## Vision controls and endpoint reproducibility

The screen contains 9,000 preference rows and 6,000 generation rows with zero failures. The `1e-3` overall vision-minus-shuffle strict effect is **+0.4300 [0.3667, 0.4933]**, but count vision-minus-shuffle is −0.04 [−0.18, 0.08]; its nonzero count preference therefore does not establish retained visual counting.

Both endpoint teacher-forced evaluations reproduce package 10 exactly across 1,800 rows each, including logp, NLL, and margins. Repeated free generation is structurally identical but not text-bit-exact: 42/1,200 frozen and 62/1,200 control predictions differ across runs; correct flags differ on 18 frozen rows and zero control rows. Generation is reported from the matched same-run screen and is not part of the candidate-selection rule.

## Preserved failures

The first attempt accidentally selected the earlier `balanced_multitask_projector_v1` step 50 rather than the package-9 `balanced_compare_projector_v1` step 50. Its unexpectedly low endpoint preference exposed the mismatch. Four training runs, their comparison index, and their evaluation remain on disk; all six have `INVALIDATION.json`. The first strict analysis attempt also remains invalidated after requiring text-bit-exact repeated GPU generation. Seven invalidations are independently verified and the old results must not be used.

## Next local experiment

Run the newly requested exact matched comparison of six-task stratified batches against one global random permutation. Only if stratification has a measurable advantage may the rental contract require capability coverage in every physical batch. After that, run preregistered forgetting-triggered replay from the known step-50 trade-off. Per-task gradient-conflict intervention follows only if replay also fails.

No paid resource was used. Final odd halves remain untouched.
