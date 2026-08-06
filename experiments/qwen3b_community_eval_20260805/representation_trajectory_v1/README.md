# Package 15O — checkpoint-aware representation trajectory

This package freezes a zero-training diagnostic over the existing Qwen2.5-3B
grounding-enriched checkpoints at steps 0, 100, 200, 300, 400, and 500. It
reuses the frozen `screenspot_glm50_v1` order, MoonViT feature cache, projector
initialization, and fixed receiver from Package 15N.

Package 15N had already established gross collapse at step 500 before this
trajectory was frozen. The preregistered unknown is the earliest saved
checkpoint at which both collapse guards fire. Training-history loss and
gradient windows are bound to the same checkpoint schedule.

The registered receiver-stage action is:

- step 100 already collapsed: protect geometry from the first optimizer step;
- later onset: use the last saved pre-collapse checkpoint to localize the
  intervention window;
- no saved checkpoint collapsed: proceed to the counterfactual-margin screen.

This diagnostic cannot establish grounding or visual ability. It uses no paid
resources and does not score the frozen final half.

## Result

Both the projector and fixed-receiver guards already fire at step 100, after
800 examples. At the projector boundary, relative spread and participation
rank are 0.12985 and 0.07721 of step0; at the receiver they are 0.12873 and
0.07596. Projector sample RMS has simultaneously increased from 0.124 to
35.74 and top-1 variance fraction from 17.48% to 98.76%.

All later saved checkpoints remain collapsed. The registered action is
therefore to apply scale/geometry protection from the first optimizer step and
run a matched small-lambda screen. Extending the same CE-only training is not a
repair candidate. The formal root includes all 13 pooled tensors, 15,925 pair
rows, 50 per-sample rows, and the bound 500-row training history; the independent
verifier recomputed the result exactly.
