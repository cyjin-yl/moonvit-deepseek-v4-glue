# Qwen2.5-3B 4k ScreenSpot GLM-format result

This package is the first real grounding result under the fixed Qwen2.5-3B
contract. It uses the preregistered `screenspot_glm50_v1` public subset, 50
samples stratified across platform and text/icon-widget type, and must be called
a **GLM-format metric-aligned public subset**. It is not the community private
50-sample set.

All seven registered roles are present. `vision` aliases the trained
`current_candidate`; `previous_best` aliases exact `step0`; blind preserves the
semantic prompt; shuffled uses the frozen derangement; random-projector uses the
separately seeded control. Generation is greedy with the fixed chat template,
32-token cap, exact click parser, identical sample order, and 1024-pixel
MoonViT preprocessing. Generation took 240.30 seconds and peaked at
7,245,852,672 V100 bytes. Scoring uses 2,000 paired bootstrap replicates with
seed 20260805.

The trained vision condition parses 48/50 outputs (96%). Its all-sample metrics
are Accuracy@50 2%, Accuracy@100 4%, Accuracy@200 16%, click-in-box 4%, mean
center distance 554.53, and median center distance 568.37. It clears the public
reference on parse rate, Accuracy@200, and mean distance in isolation, while it
misses Accuracy@50 and Accuracy@100 and fails every required causal-improvement
gate.

Blind reaches 12% click-in-box and mean distance 392.59. Step0 reaches 10% and
398.59. Relative to blind, trained vision changes click-in-box by -8 points
(95% CI [-20, 2]) and worsens mean distance by 161.94 points (improvement CI
[-246.70, -89.24]). Relative to step0, it changes click-in-box by -6 points
(CI [-16, 4]) and worsens mean distance by 155.94 points (CI
[-246.74, -75.50]). Vision-minus-shuffled click-in-box is -2 points (CI
[-6, 0]); no threshold-accuracy CI has a positive lower bound.

The candidate is therefore rejected and `previous_best` remains step0. This
result refutes the hypotheses that the 0.5B backbone was the sole grounding
bottleneck, that a 3B frozen receiver plus 4,000 examples is sufficient, and
that falling teacher-forced loss establishes image use. It supports the narrower
claims that the complete 3B train/eval path is operational and that the current
objective can learn output format while degrading a strong text/center prior.

The result package keeps every prediction and the full score tree. Extracted
images and 50 large feature shards remain on the V100 and are bound by the
remote artifact index and checked-in cache manifest. The next registered action
is the complete 1,272-sample public ScreenSpot run. If it confirms this failure,
teacher-forced paired preference and projector-information diagnostics will
separate missing visual information from a generation/alignment failure before
any 8k extension. No paid resource or final-half evaluation was used.
