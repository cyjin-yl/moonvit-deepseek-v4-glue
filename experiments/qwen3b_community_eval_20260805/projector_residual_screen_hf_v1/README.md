# Package 15R — residual structure screen results

This directory records completed arms of the frozen residual/gated-residual
screen. The `baseline_none` control reuses the exact step0 projector
and the same Qwen2.5-3B data, order, receiver, budget and projector-health
contract as the earlier structure screens.

The control reached optimizer step 2 and was automatically stopped. CE fell
from 4.1440037 to 2.4380192 while projector and receiver spread/rank ratios
fell below the fixed safety boundary; the independent verifier recomputed all
three probes and all 22 health artifacts successfully. This is a health
failure record, not a visual-capability result, and it does not enter
ScreenSpot/TextVQA/DocVQA/OCRBench promotion.

The complete raw copy, including checkpoint tensors, optimizer state, RNG
state, probe logs and launcher log, is outside Git at
`D:/V100-artifacts/projector_residual_screen_hf_v1/baseline_none` and on the
V100 HDD. The next arm is `zero_init_residual`, under the same contract.

`zero_init_residual` also stopped at optimizer step 2. Its residual branch
passed the registered binding and received a non-zero first-step gradient, but
projector/receiver geometry degraded faster than the control. It remains a
health-screen failure and has no capability claim.
