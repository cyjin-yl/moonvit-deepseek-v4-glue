# Package 9: balanced adaptation trajectory and full step-50 confirmation

## Registered screen

The canonical-bf16 screen evaluates frozen base plus LoRA/projector steps 25, 50, and 100. Teacher forcing uses the first 50 complete authoritative-selection pairs per task; free generation uses the pre-registered 50-pair/task manifest. All seven states use identical IDs, prompts, scorers, dtype, and control definitions. Training comes from package 8: both arms saw the same 2,400 records in exact order (`a0929326…2f5`) with true batch 24.

The valid screen contains 12,600 preference rows and 8,400 generation rows, 21/14 state-condition cells, zero failed rows, and 693 contrasts using 2,000 complete-pair bootstrap resamples. Projector step 50 is the only screened checkpoint whose strict paired-preference point estimate exceeds frozen base on all six tasks. LoRA remains shape-specific and erases count/spatial behavior.

## Full canonical step-50 confirmation

The full confirmation contains 21,600 preference rows and 3,600 generation rows over frozen, LoRA step 50, and projector step 50. It ran for 1,137.6 seconds at 12.72 GB peak GPU memory. Independent verification re-read all cells, the 24/288 projector/LoRA checkpoint tensors, and the common training-order hash. Final odd halves were never scored.

Overall strict paired preference is base/LoRA/projector 0.2242/0.2433/0.5117. Projector step 50 gains +0.2875 [0.2583, 0.3167] over base; LoRA gains +0.0192 [-0.0058, 0.0442]. Overall paired generation is 0.0633/0.1000/0.2267; projector gains +0.1633 [0.1200, 0.2100], while LoRA gains +0.0367 [0.0000, 0.0733].

Projector step-50 strict-preference gains over base are color +0.405 [0.335, 0.475], coordinate +0.360 [0.280, 0.435], count +0.265 [0.200, 0.330], OCR +0.020 [-0.025, 0.065], shape +0.175 [0.120, 0.230], and spatial +0.500 [0.430, 0.570]. OCR therefore does not confirm the screen's base-relative lower bound. Its vision-minus-shuffle gain is nevertheless +0.075 [0.020, 0.130], and all other task-level vision-minus-shuffle lower bounds are also positive. This is evidence of image-dependent OCR ranking without stable base-relative improvement or free generation.

Projector step-50 paired-generation gains over base are color +0.140 [0.060, 0.240], coordinate +0.020 [0.000, 0.060], count +0.020 [0.000, 0.060], OCR 0, shape +0.240 [0.100, 0.380], and spatial +0.560 [0.420, 0.700]. Count and OCR retain a clear teacher-forced/free-generation gap.

LoRA step 50 confirms a shape-only effect: shape strict preference +0.340 [0.275, 0.405] and paired generation +0.440 [0.300, 0.580]. Count strict preference falls -0.105 [-0.155, -0.060], and spatial strict/generation fall -0.250 [-0.310, -0.190] / -0.220 [-0.340, -0.120]. No non-shape task gains paired generation.

## Full step-100 minus step-50 paired comparison

The cross-run comparator first requires exact equality of every sample ID, pair ID, pair variant, task, and condition, then bootstraps complete pairs. It produces 210 contrasts. Projector step 100 minus step 50 is effectively zero overall for strict preference (-0.0008 [-0.0283, 0.0275]) and inconclusive for generation (+0.0300 [-0.0133, 0.0767]), while redistributing task ability sharply:

- strict preference: color +0.100 [0.025, 0.175], coordinate +0.160 [0.095, 0.225], count -0.280 [-0.345, -0.220], OCR +0.065 [0.015, 0.120], shape -0.300 [-0.360, -0.240], spatial +0.250 [0.190, 0.315];
- paired generation: coordinate +0.220 [0.120, 0.340], spatial +0.220 [0.120, 0.340], shape -0.280 [-0.420, -0.160], with color/count/OCR intervals not excluding zero.

This refutes a single monotonic global stopping rule. Equal task sampling did not prevent interference, so sampling imbalance alone cannot explain the trade-off. The current evidence supports a multi-objective gradient/representation conflict along one projector trajectory.

## Retained analysis defect and next decision

Trajectory analysis v2 correctly computed metrics and bootstrap contrasts but labeled a flat all-zero curve as a nonmonotonic peak because its tie-break selected the earliest state. It is retained under `balanced_compare_trajectory_analysis_v2_invalid_flat_peak/` with an explicit invalidation record. Test-first v3 changes the label rule to require a strictly higher earlier value; no evaluation matrix or numeric contrast changed. An initial verifier invocation used obsolete CLI flag names and exited before writing output; the corrected invocation produced the committed valid verification.

The next local screen will interpolate projector step-50 and step-100 weights at fixed coefficients. If an interpolant retains count/shape while gaining coordinate/spatial, it supplies a zero-training, theoretically grounded checkpoint-merging improvement for full confirmation. If the trade-off remains monotonic, proceed to an anti-forgetting auxiliary target or gradient-conflict intervention anchored at step 50. Paid Gate D remains paused.
