# Qwen2.5-3B representation-retention preregistration

Package 15N freezes a short diagnostic before extracting any new
representation result. It uses the already frozen `screenspot_glm50_v1` sample
order and feature cache, exact step0, Package-15K step 500 and the fixed
4096-to-2048 receiver. It never loads or changes Qwen weights and performs no
training.

For each image it concatenates the cached MoonViT groups and mean-pools the
visual token sequence at three boundaries: raw flattened MoonViT, canonical
4096 projector output and fixed-receiver 2048 output. It records sample RMS,
between-image RMS, relative between-image spread, participation/entropy
effective rank, top-1 variance fraction, within-image token RMS, all pairwise
RMS distances/cosines, linear CKA and pairwise-distance correlation. Pooled
float64 tensors, every pair and every per-sample norm are retained for
independent recomputation.

The decision is deliberately narrow. At the fixed-receiver boundary, gross
collapse requires both current/step0 relative-spread ratio below 0.25 and
current/step0 participation-rank ratio below 0.5. If both fire, the next
matched-budget experiment must repair projector/receiver representation before
margin training. Otherwise the next treatment is the already selected
training-only correct-versus-counterfactual margin auxiliary objective.

The projector has no text query, so an image-only target-coordinate probe would
confound representation retention with instruction-conditioned selection. This
contract explicitly forbids using such a probe as capability evidence.
Diversity retention also cannot prove grounding; Package 15L/15M preference and
generation remain the ability gates. No paid resource or final evaluation half
is allowed.
