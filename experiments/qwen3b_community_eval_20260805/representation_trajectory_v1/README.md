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
