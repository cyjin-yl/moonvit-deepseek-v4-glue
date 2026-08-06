# Qwen2.5-3B grounding-enriched 4k GLM50 generation

This package completes the frozen GLM-format public-50 generation contract for
the Package-15K checkpoint. The treatment uses 2,000 ShowUI grounding and
2,000 short-answer records at the same exact step0, 500 optimizer steps, 4,000
examples, image preprocessing, fixed receiver, prompt, parser and decoding
settings as the first 4k baseline. It evaluates vision, blind, shuffled,
random-projector, step0, previous-best and current-candidate in the same sample
order. Scoring uses 2,000 paired bootstrap replicates with seed 20260805.

Vision is fully parseable, but Accuracy@50/@100/@200 is 2%/2%/14%,
click-in-box is 6%, and mean/median center distance is 502.06/494.94. Blind is
6%/6%/16%, 12% click and 392.59 mean distance. Exact step0 is 4%/6%/14%, 10%
click and 398.59 mean distance. Vision-minus-blind click is -6 points with 95%
CI [-16, 2], while mean-distance improvement is -109.47 with CI [-171.64,
-44.59]. Current-minus-step0 mean-distance improvement is -103.47 with CI
[-168.28, -38.91].

Correct and shuffled images have identical 6% click accuracy. Their mean
distances are 502.064 and 502.082; the paired improvement is 0.018 with CI
[-3.544, 3.213]. Accuracy@200 is four points higher with the correct image, but
its CI [0, 10] does not have a strictly positive lower bound. The checkpoint
therefore fails both registered causal requirements. It passes the community
parse-rate and mean-distance thresholds, while missing Accuracy@50/@100/@200
and both image-causality guards, so it does not reach the community
metric-aligned baseline.

Generation also exposes a narrow output collapse. Vision emits only six unique
coordinates and returns `click(start_box=[125, 345])` on 31/50 rows; shuffled
uses the same mode on 23/50 and matches the exact vision output on 30/50. This
is not the training-label mode: 2,000 grounding targets contain 1,066 unique
coordinate pairs, have median x/y 513/320, and contain no exact `[125,345]`
target. Image identity produces small output changes, but they are not aligned
with target location.

Package 15L had already rejected the checkpoint by internal correct-coordinate
preference; free generation now independently agrees. Exact step0 remains
previous-best, the checkpoint cannot advance to full ScreenSpot or three seeds,
and no capability claim is allowed. The shortest diagnostic is a small
projector/fixed-receiver information-retention screen on these 50 records. It
will determine whether the next matched-budget change should directly add the
training-only correct-versus-counterfactual margin objective or first repair a
representation-collapse defect. No paid resource or final evaluation half was
used.
