# Qwen2.5-3B 4k ScreenSpot teacher-forced preference diagnosis

This package tests whether the failed 4k checkpoint internally distinguishes a
correct ScreenSpot coordinate from a deterministic counterfactual coordinate,
even when greedy generation cannot emit the correct click. It uses the frozen
50-row GLM-format metric-aligned public subset. The correct answer is the
rounded center of the sample bbox. The counterfactual answer is the rounded
center of the sample selected by the preregistered shuffled-image derangement.
The two candidate answers are scored in the same batch under the same prompt.

Preference is strict token-normalized assistant-answer log probability,
including the terminating `im_end` token: `correct_logp_mean >
counterfactual_logp_mean`. The run evaluates blind, trained correct image,
trained shuffled image, step0 correct/shuffled, and random-projector
correct/shuffled. It retains the registered aliases and performs 2,000 paired
bootstrap replicates with seed 20260805. Formal scoring took 99.28 seconds and
peaked at 7,652,064,768 V100 bytes.

The trained vision condition prefers the correct coordinate on 23/50 samples
(46%). Blind is 56%, shuffled is 52%, step0 is 54%, and random-projector is
50%. Vision-minus-blind is -10 points (95% CI [-22, 2]);
vision-minus-shuffled is -6 points (CI [-14, 0]); trained-minus-random is -4
points (CI [-20, 12]); current-minus-step0 is -8 points (CI [-26, 8]). The
vision-minus-shuffled mean correct-margin change is -0.00725 with CI
[-0.01287, -0.00186].

Training strongly raises the absolute probability of a coordinate answer:
correct-answer mean NLL falls from 2.50769 at step0 to 1.22362, an improvement
of 1.28407 with CI [1.13713, 1.43892]. The same effect appears with the wrong
image: trained correct-image and shuffled-image mean correct log probabilities
are -1.22362 and -1.22329, with a paired improvement CI of [-0.03752, 0.03906].
The checkpoint therefore learned a strong coordinate-answer soft prompt while
failing to make the coordinate choice depend on image identity.

Step0 shows a +10-point correct-versus-shuffled preference difference (CI [2,
20]), but its mean-margin and correct-logp CIs cross zero; the separately seeded
random projector shows 50% versus 52%. This isolated binary step0 effect is kept
as an initialization-sensitivity observation and is not treated as visual
ability. The trained checkpoint removes it rather than amplifying it.

The result selects a new training/data objective path. Extending the same 4k
stream or changing decoding is not supported. The next one-variable screen will
keep the exact step0, 500 optimizer steps, 4,000 examples, resolution, model,
receiver, and evaluator, while increasing explicit ShowUI grounding supervision
within the fixed budget. If correct-versus-shuffled preference still does not
improve, the next candidate is an auxiliary counterfactual-margin objective that
is discarded after training and leaves the canonical 4096 projector directly
transferable to DeepSeek. No paid resource was used.
