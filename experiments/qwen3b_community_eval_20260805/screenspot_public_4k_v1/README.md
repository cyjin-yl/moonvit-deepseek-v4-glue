# Qwen2.5-3B 4k complete public ScreenSpot result

This package is the first complete 1,272-row public ScreenSpot result under the
fixed pure-text `Qwen/Qwen2.5-3B-Instruct` contract. It uses the same immutable
dataset revision, image preprocessing, 1024-pixel maximum side, visual-token
limit, prompt, greedy decoding, strict click parser, sample order, projector
initialization, and 4,000-example budget as the preregistered GLM-format public
50 subset.

All seven registered roles are present. `vision` aliases the trained
`current_candidate`; `previous_best` aliases exact `step0`; blind keeps the same
semantic prompt; shuffled uses the frozen derangement; random-projector uses the
separately seeded initialization control. Formal generation took 2,807.66
seconds and peaked at 7,247,035,392 V100 bytes. The frozen Qwen backbone has
3,085,938,688 parameters and zero trainable parameters. Scoring uses 2,000
paired bootstrap replicates with seed 20260805.

The trained vision condition parses 1,227/1,272 outputs (96.46%). Its all-sample
metrics are Accuracy@50 1.73%, Accuracy@100 4.87%, Accuracy@200 11.79%,
click-in-box 2.67%, mean center distance 565.18, and median center distance
553.42. Blind reaches 3.07% click-in-box and mean distance 395.52. Step0 reaches
3.30% and 391.12. Shuffled reaches 2.75% and 566.26.

Relative to blind, trained vision changes click-in-box by -0.39 points (95% CI
[-1.65, 0.79]), Accuracy@200 by -3.22 points (CI [-5.98, -0.24]), and worsens
mean center distance by 169.66 points (improvement CI [-185.68, -154.17]).
Relative to step0, it changes click-in-box by -0.63 points (CI [-1.81, 0.63]),
loses 3.54 points of parse rate (CI [-4.56, -2.59]), and worsens mean distance
by 174.06 points (CI [-189.67, -157.44]). Vision and shuffled are statistically
indistinguishable on click and all three distance thresholds.

The trained result is stronger on text targets than icon/widget targets:
click-in-box is 4.16% versus 0.87%, while all-sample mean distance is 516.39
versus 624.33. The platform breakdown is retained in full. macOS is the largest
failure region, with 77.33% parse, 1.16% click-in-box, and mean distance 726.97.

The public community reference is metric-aligned only; its private 50 samples
are not this dataset. The current full-public result clears the 92% parse
threshold, but misses Accuracy@50 4.3%, Accuracy@100 8.7%, Accuracy@200 15.2%,
and mean distance 563.7. It also fails the required vision-over-blind and
vision-over-shuffled causal gates. The candidate remains rejected and
`previous_best` remains step0.

The package keeps every generated prediction and every per-row scored record.
The 593,342,933 image bytes and 7,609,930,976 cached feature bytes remain on the
local V100 disk and are bound by checked-in manifests and the remote artifact
index. The initial new-output invocation incorrectly supplied `--resume`; it
failed before model load or prediction and is preserved under `failures/`.
No paid resource was used.
