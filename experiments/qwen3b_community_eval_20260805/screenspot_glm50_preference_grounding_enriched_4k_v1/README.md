# Qwen2.5-3B grounding-enriched 4k paired preference

This package evaluates the Package-15K checkpoint on the frozen 50-row
GLM-format metric-aligned public ScreenSpot subset. The training treatment
changed only the exact 4,000-example mix: 2,000 ShowUI grounding and 2,000
short-answer records, strictly alternating in every global batch. The model,
exact step0 projector, 500 optimizer steps, image preprocessing, fixed
4096-to-2048 receiver, prompts, candidate targets and evaluator match the first
4k baseline.

Teacher forcing scores the correct rounded bbox center against the
preregistered derangement's center in one batch. The score is strict
token-normalized assistant-answer log probability including `im_end`. Seven
physical conditions and their registered aliases cover blind, current correct
and shuffled image, exact step0 correct and shuffled image, and separately
seeded random projector correct and shuffled image. All comparisons use 2,000
paired bootstrap replicates with seed 20260805.

The current checkpoint prefers the correct coordinate on 26/50 samples (52%).
Blind is 56%, shuffled is 54%, step0 is 54%, and random projector is 50%.
Vision-minus-blind is -4 points with 95% CI [-18, 10];
vision-minus-shuffled is -2 points with CI [-6, 0]; trained-minus-random is +2
points with CI [-14, 18]; current-minus-step0 is -2 points with CI [-20, 14].
The correct-image versus shuffled-image mean-margin change is -0.002378 with CI
[-0.006099, 0.001248].

The treatment strongly raises the absolute probability of coordinate answers.
Correct-answer NLL falls from 2.50769 at exact step0 to 1.05915, an improvement
of 1.44854 with CI [1.29793, 1.60698]. The same checkpoint produces shuffled
NLL 1.05752. Its correct-image minus shuffled-image correct-logp change is
-0.001633 with CI [-0.005786, 0.002342]. Doubling grounding to half of the
fixed budget therefore strengthens the coordinate soft prompt without making
the chosen coordinate depend on image identity.

This paired-preference gate rejects the checkpoint and keeps exact step0 as
previous-best. It refutes insufficient grounding proportion as the sole cause
of the first 4k failure under the present budget and frozen-backbone
cross-entropy objective. The matched GLM50 generation run is still required to
complete the checkpoint contract, but this candidate cannot advance to full
ScreenSpot or three seeds. If generation agrees, the next one-variable screen
is a training-only correct-versus-counterfactual margin auxiliary objective;
the auxiliary head/loss is discarded after training so the canonical 4096
projector stays directly transferable to DeepSeek. No paid resource or final
evaluation half was used.
