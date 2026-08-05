# Package 10: step-50/100 projector interpolation

## Theory and registered decision

Package 9 found nearly identical aggregate scores but complementary task abilities at projector steps 50 and 100. Linear checkpoint interpolation tests the lowest-cost same-basin hypothesis: if both endpoints encode compatible solutions, an interior weight point may retain step-50 count/shape while acquiring step-100 coordinate/spatial.

The decision was fixed before reading interior results. A non-endpoint must (1) keep count and shape strict paired preference within 0.05 of alpha 0, (2) strictly improve coordinate and spatial over alpha 0, (3) improve the better endpoint's worst-task preference, and (4) keep macro preference within 0.02 of the better endpoint. Only a passing point would receive full confirmation.

## Construction and exact endpoint controls

The interpolator constructs `(1-alpha) * P50 + alpha * P100` at alpha 0/.25/.50/.75/1. It validates identical non-empty tensor keys, shapes, and dtypes; saves and reloads each safetensors file; and hashes both serialized files and ordered tensor contents. Alpha 0 exactly reproduces tensor SHA `fd7b07e6…d192`; alpha 1 exactly reproduces `7b731cff…a76`.

The canonical-bf16 screen contains frozen base plus all five interpolation states, 10,800 preference rows and 7,200 generation rows, zero failures, 630 metrics, and 651 complete-pair bootstrap contrasts. It ran for 541.3 seconds at 12.72 GB peak. Independent verification proves alpha 0/1 exactly reproduce package-9 step 50/100 across 1,800 preference and 1,200 generation rows per endpoint, including raw logp/NLL/margins and generation strings.

## Result: interpolation follows the same trade-off

No point passes the registered merge rule. Alpha .25 is the best balance diagnostic: macro strict preference 0.5333, worst-task 0.160, macro generation 0.2700, endpoint regret 0.520. It improves spatial preference from 0.74 to 1.00 (+0.26 [0.14, 0.38]) and macro generation over alpha 0 by +0.0433 [0.0100, 0.0767]. However, count drops from 0.42 to 0.26 (-0.16 [-0.28, -0.04]) and shape from 0.80 to 0.70 (-0.10 [-0.20, 0.00]); it therefore violates retention and does not improve the endpoint worst-task value of 0.18. Its small macro gains over alpha 1 are inconclusive: preference +0.0167 [-0.0267, 0.0633], generation +0.0133 [-0.0233, 0.0533].

Alpha .50/.75 further reduce count to 0.14. Spatial reaches 1.00 already at alpha .25, while coordinate and color generally rise and shape declines along the path. The curve is evidence for a connected basin with smooth aggregate behavior, but it refutes the proposed zero-training capability union. No full interpolation confirmation was run, exactly as registered.

## Next local experiment

Start from the verified step-50 checkpoint and continue over the exact remaining step-51–100 record order while adding a task-conditional anti-forgetting target. The cheapest first screen is projector-output anchoring on count/shape, with the unregularized continuation reproduced as a control. If anchoring preserves count/shape while coordinate/spatial still rise, confirm the best coefficient; if it simply blocks all learning, move to per-task gradient-conflict handling. Paid Gate D remains paused.
