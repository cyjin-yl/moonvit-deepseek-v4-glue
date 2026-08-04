# Package 3 run notes

## Boundary and starting state

- Starting repository HEAD: `1cc6f631cb2477e8ebfbed62f156fec8018e19f3`.
- Workstation: `doesworkstation`, Tesla V100-PCIE-32GB, torch 2.10.0+cu128, CUDA build 12.8, bf16 evaluation.
- All runs used the fixed selection half. `final_half_scored=false` is recorded and independently verified.
- No server was rented and no paid resource was used.
- The pre-run worktree was dirty only with package 3 implementation files; each run embeds the complete porcelain status.

## Valid teacher-forced run

- Canonical run: `$HDD/data/perception_v1/runs/preference_v2`.
- Git copy: `preference_run/`.
- Matrix: 5 independent projector states × 8 conditions × 2,400 authoritative selection samples = 96,000 rows.
- Wall time: 3,337.034 s; peak allocated GPU memory: 3,965,700,608 bytes; failures: 0.
- Raw SHA-256: `f7bf09aa3bc7f383ecca7abf9febf16e37ef0fdec8a04f5b19af8aab2d3b54c8`.
- Independent verification checked exact denominators, finite log-probabilities, config/raw hashes, pair membership, aliases, visual-source provenance, and `final_half_scored=false`.
- `current-final` is a bit-identical alias of `step-002000`; it was not rerun.

## Fixed free-generation selection

- Manifest: `controls/synthetic_generation_selection/MANIFEST.json`.
- Selection algorithm: per-task SHA-256 ranking of `seed:task:pair_id`, seed 20260804, retaining both variants.
- 50 complete pairs / 100 images for each of color, shape, count, spatial, OCR, and coordinate; total 300 pairs / 600 images.
- Authoritative source selection SHA-256: `51ce274183bd4b7adc6e7201d9aeed259071d380551bf55f656119d201a936d6`.
- Subset JSONL SHA-256: `210c9d93c37f6300088c7d2387b3fb1bb752509037d5cdcf421685a94dfed758`.
- Logical dataset SHA-256 remains `122ae820381e11e12c7dd9db03a525b801a5ebee5adbef037919185e65cbaa71`.

## Valid free-generation and benchmark trajectory

- Canonical run: `$HDD/data/perception_v1/trajectory/trajectory_v5`.
- Git copy: `generation_run/`; derived bootstrap decisions are in `generation_analysis/`.
- Matrix: five independent projector states, six synthetic conditions plus two diagnostic extensions, 600 fixed synthetic rows, five benchmark selection sets, and 32 historical heldout rows with ten shuffle repeats.
- Exact raw totals: 37,300 generation rows and 160 heldout shuffle-loss rows; failures: 0.
- Wall time: 2,745.863 s; peak allocated GPU memory: 9,817,912,832 bytes.
- Generation raw bytes/SHA-256: 30,247,129 / `8d67f5548df3cb71ec9aacb92e542a09c14d9e3095664a843ec24d54a822fb0b`.
- Independent verification checked all raw hashes, exact denominators, condition aliases, causal visual-source relationships, finite scores, the fixed generation-subset manifest, and `final_half_scored=false`.
- Synthetic vision sample accuracy reaches 0.1367/0.1550/0.1350/0.1417 at steps 500/1000/1500/2000, while strict paired generation and answer-flip accuracy stay exactly 0 at every checkpoint. At step 2000, vision−blind is +0.1417 [0.1167, 0.1683], but vision−same-image is −0.0033 [−0.0183, 0.0100] and vision−shuffled-image is −0.0033 [−0.0150, 0.0083].
- Historical heldout shuffle delta is −0.0095 at matched random, then +1.0263/+0.6703/+0.9533/+1.1886 at steps 500/1000/1500/2000. This confirms learned global image conditioning while the correct-image paired result remains absent in free decoding.
- Step-1500 shape is the decisive teacher-forced/free-generation split: strict paired preference 0.130 versus paired generation 0. The next package therefore starts with layerwise probes and activation patching around shape step 1500→2000.

## Derived charts and failure audits

- `charts/` contains all preregistered SVG/CSV figures and a hash manifest, including the task-level teacher-forced-paired versus paired-generation comparison.
- Canonical step-1500 and step-2000 failure audits contain 374 and 318 deterministic records. Step 1500 has 30 shape cases with positive teacher-forced margin but failed generation.
- The first audit invocation used non-canonical IDs `step1500`/`step2000`; the old tool silently emitted empty audits. Those outputs are preserved under `invalid/failure_audit_checkpoint_alias_*`, the tool now rejects zero checkpoint matches, and the canonical audits were rerun with `step-001500`/`step-002000`.

## Invalid and screening runs

- `preference_v1`: numeric tensors are bit-identical to v2, but 36,000 control rows recorded the wrong `visual_source_id`. The whole run is invalid; v2 was rerun from the start. Invalidation and comparison are preserved.
- `trajectory_v3`: stopped before completing step-0 benchmark after batch-2 1024px throughput measured about 0.8 sample/s. Partial raw data remains on the HDD and is bound by `invalid/trajectory_v3/INVALIDATION.json` hashes.
- `trajectory_v4`: stopped below 1% after batch-16 synthetic throughput measured about 8.3 sample/s. Partial raw data remains on the HDD and is bound by `invalid/trajectory_v4/INVALIDATION.json` hashes.
- `trajectory_smoke_v5_batch64`: exited before generation because screening `--limit 128` incorrectly overrode the fixed 32-record heldout denominator. The attempt produced no scores, was invalidated, and the cap-at-source-count behavior now has a regression test.
- `failure_audit_checkpoint_alias_step1500/step2000`: empty audit outputs caused by non-canonical checkpoint IDs; preserved as invalid and replaced by canonical reruns.
- Valid batching screens are under `batch_screening/`: 64/16 took 221.270 s at 9,817,912,832 bytes peak; 128/32 took 215.425 s at 18,568,209,408 bytes peak. The 2.6% speed gain did not justify nearly doubling memory, so the final run uses synthetic batch 64 and benchmark batch 16.

## Storage

Large frozen-feature caches and the invalid partial raw streams remain under `$HDD/data/perception_v1/`. Git contains the complete valid primary raw evaluations, all manifests/hashes, summaries, analysis CSV/JSON, charts, audit tables, logs, and compact invalidation evidence.

## Code and report verification

- Targeted package-3 tests: 33 passed.
- Full workstation suite: 150 passed; only known torch/Pillow deprecation warnings and the documented NVML userland mismatch warning.
- `report/main.pdf` builds to 25 pages; package-3 pages 12–15 and all four embedded SVG panels were render-checked.
