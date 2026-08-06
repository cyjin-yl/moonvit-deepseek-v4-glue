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

## Formal result

The preregistered fixed-receiver decision fired both gross-collapse guards.
Current/step0 relative-spread and participation-rank ratios are 0.1372 and
0.0846; at the projector boundary they are nearly identical at 0.1384 and
0.0859. Effective rank falls from 13.28 to 1.14 and the top component grows
from 17.48% to 93.46% before the receiver. Sample RMS grows from 0.124 to 97.31
and within-image token RMS from 0.139 to 18.45. Absolute variation therefore
did not vanish: it became a large, nearly collinear common-direction soft
prompt with approximately rank-one cross-image differences.

The registered action is to repair projector representation before spending a
matched 4k budget on counterfactual margin. The immediate zero-training follow-
up is the same screen across steps 0/100/200/300/400/500 to locate collapse
onset; that trajectory will set the smallest scale/geometry-preservation
treatment. The receiver is not the primary source because it preserves the
projector ratios.

The first independent verifier failed only because safetensors enumerated
tensor names in a different block order. Its failure log and frozen source hash
are retained. The corrected verifier sorts by stable row identity and exactly
recomputed five pooled tensors, 6,125 pairwise rows, 50 per-sample rows, both
decisions and all artifact hashes.

The V100 host also had a loaded-driver/user-library mismatch. The run used
matching 580.159.04 user-space libraries extracted on the HDD and scoped only
through `LD_LIBRARY_PATH`; no system file, desktop GPU client or kernel module
was changed. `RUNTIME_RECOVERY.json` binds the official repository checksum and
the two library hashes.
