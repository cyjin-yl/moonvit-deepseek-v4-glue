# Package 15Q — high-frequency projector structure results

This directory contains the Git-sized records from the pre-registered
`projector_structure_screen_v1` experiment. Full checkpoints, optimizer state
and RNG state stay in the content-verified raw archive under
`D:/V100-artifacts/projector_structure_screen_hf_v1/` and on the V100 HDD.

Every arm uses the same step0 MLP weights, Qwen2.5-3B receiver, MoonViT cache,
training order, 100-step budget and projector-health contract. A health pass
only permits the fixed real-vision evaluation contract to run; it does not
establish visual ability by itself.

All three arms auto-stopped at optimizer step 2 with onset `[1, 2]`, and every
run passed the independent health verifier. The passing set is empty, so the
pre-registered 500-step expansion is cancelled. `DECISION.json` records the
comparison and sends the next screen to residual/gated-residual structures.
