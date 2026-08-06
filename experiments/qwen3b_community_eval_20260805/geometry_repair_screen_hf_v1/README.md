# Package 15P high-frequency health trajectory

This directory records the first valid run under the frozen projector-health
contract. It is the `control` arm (`lambda=0`) of the matched 100-step / 800
example geometry-repair screen. The run was intentionally stopped at optimizer
step 2 when the pre-registered adverse-trend guard detected rising output RMS
and falling cross-image spread at both the canonical projector boundary and the
Qwen receiver.

The result is representation-health evidence, not a visual-capability result.
The loss fell from 4.1440 to 2.4380, while the collapse onset was narrowed to
steps 1–2. The runner saved failure and healthy checkpoints, optimizer/RNG
state, current batch IDs, JSONL health/probe logs, and rollback metadata. The
independent verifier recomputed all three probe decisions and rehashed the
22-file health artifact tree.

`control/` contains the small, reviewable metadata and logs. The complete raw
copy, including the 1.1 GB checkpoint/optimizer payload, is retained at
`D:/V100-artifacts/geometry_repair_screen_hf_v1/control` and on the V100 HDD;
`control/RAW_ARTIFACT_POINTER.json` binds both roots and the manifest SHA-256.

The matched `ratio005` arm is now also archived here. It stops at the same
`[1,2]` onset, so the smallest pre-registered geometry dose does not rescue the
early trajectory. Its raw pointer and result are under `ratio005/`.

This result keeps Gate D at `NO-GO`, leaves `previous_best=step0`, and requires
the matched `ratio005`, `ratio020`, and `ratio080` arms to run under the same
early-stop contract before any 500-step expansion or capability evaluation.
