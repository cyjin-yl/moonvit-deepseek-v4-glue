# Package 15R — residual and gated-residual structure screen

Package 15Q showed that affine-free output normalization does not preserve
cross-image geometry. Package 15R keeps the original 4096 projector path and
adds one full-width residual branch after `linear_2`:

* `baseline_none`: the unchanged CE-only control;
* `zero_init_residual`: a bias-free residual Linear whose weights are all zero;
* `gated_residual`: a normally initialized residual Linear multiplied by a
  scalar gate initialized to zero.

Both candidates produce exactly the frozen step0 output before training and
share all base MLP tensors. The branch is projector-only, keeps the canonical
4096 boundary and leaves the Qwen receiver and language model frozen. The
fixed 100-step/800-example health screen is the only first-stage budget; no
arm may enter capability evaluation on health metrics alone.

The pre-result initialization and launcher attempts exposed a scalar-hash bug,
a gated-gradient test expectation, an output-directory creation mistake and
the legacy runner's canonical-projector-only SHA gate. Every repair and raw
log is preserved under `failures/`; none created an optimizer step. The
repaired initialization manifest is stored outside Git and bound by SHA in
the contract. The generic projector binding now validates registered variants
before CUDA model load.
