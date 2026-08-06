# Package 15Q — high-frequency projector structure results

This directory contains the Git-sized records from the pre-registered
`projector_structure_screen_v1` experiment. Full checkpoints, optimizer state
and RNG state stay in the content-verified raw archive under
`D:/V100-artifacts/projector_structure_screen_hf_v1/` and on the V100 HDD.

Every arm uses the same step0 MLP weights, Qwen2.5-3B receiver, MoonViT cache,
training order, 100-step budget and projector-health contract. A health pass
only permits the fixed real-vision evaluation contract to run; it does not
establish visual ability by itself.

The matched `baseline_none` control reproduced the step-2 auto-stop and is
independently verified. LayerNorm and RMSNorm arms are evaluated sequentially
and added without changing the frozen contract.
