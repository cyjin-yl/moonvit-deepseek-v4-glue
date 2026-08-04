# Package 4 run notes

## Scope and frozen denominator

- Starting repository HEAD: `1f161789b06b60ee10e47fbfa0fbeff72303966e`.
- Workstation: `doesworkstation`, Tesla V100-PCIE-32GB, torch 2.10.0+cu128, bf16 language/projector inference.
- The task is the package-3 decisive shape subset. Train and selection each contain 200 complete minimal pairs / 400 records with zero ID and pair overlap. Patching uses a preregistered SHA-ranked 50-pair subset.
- No final odd half was inspected, no server was rented, and no paid resource was used.

## Frozen representations and calibrated probes

- Canonical representation run: `$HDD/data/perception_v1/mechanisms/shape_layerwise_v1`.
- Matrix: matched random, step 1500, and step 2000; train vision plus selection vision, paired-counterfactual, shuffled-image, and patch-permutation; 59 tensors per cell. This includes three tower pools, three projector pools, and assistant/image-span states for all 25 hidden-state indices.
- Extraction completed in 122.266 s with 3,612,558,336 bytes peak allocated GPU memory. The 30 tensor/metadata files occupy about 899 MB and remain on the V100 data disk; `representations/SUMMARY.json` binds every file by bytes and SHA-256.
- Canonical analysis: `$HDD/data/perception_v1/mechanisms/shape_layerwise_v2_analysis`. It contains 672 metric rows, 728 paired-bootstrap intervals, 268,800 per-record predictions, and 1,344 serialized probe tensors. The raw prediction JSONL is committed as a deterministic gzip stream; decompression yields the exact 108,164,550-byte file with SHA-256 `8dbaa77b16e3f253965c724b98afa6217ea506796e78dc8fd702122b6c00ed7c`.
- Pair-label permutation is the primary association null. The earlier v1 analysis used one random-training-label probe as if it were a calibrated null; one cell reached 0.49. That run is invalidated and fully superseded by v2, while the random-label probe remains only as an overfit diagnostic.

## Probe result

- Tower and projector representations reach 1.000 balanced shape accuracy at every checkpoint under at least one preregistered pooling. Because matched-random projector features also reach 1.000, this proves information preservation, not a training benefit by itself.
- At the assistant position, step 1500 peaks at layer 12: raw accuracy 0.790, balanced accuracy 0.816, pair-permutation `p=1/2001`, null 95% upper bound 0.285. Step 2000 peaks later and lower at layer 14: 0.605 / 0.632, `p=1/2001`, null upper bound 0.278.
- The decisive probes follow the actual visual source. At step1500/layer12, paired-counterfactual target accuracy falls to 0.075 while source accuracy stays 0.790; shuffled target is 0.230 while source stays 0.790. Step2000/layer14 gives the same pattern at 0.1125/0.605 and 0.2025/0.605.
- Trained final hidden states collapse to balanced 0.250 and pair-permutation `p=1`; the native LM-head readout is also balanced 0.250 for vision, paired-counterfactual, and shuffled image. The frozen tower/projector therefore retain shape while the trained upper language layers erase or ignore it.

## Activation patching result

- Canonical run: `$HDD/data/perception_v1/mechanisms/shape_patching_v2`; 18,300 raw rows, 183 cells, 50 pairs / 100 directions per checkpoint, all 24 decoder layers.
- The layer donor is captured from the exact decoder-layer hook output before final RMSNorm. The final-layer assistant patch reproduces every clean margin within `1e-6`; paired clean/counter margins are exactly antisymmetric.
- Step 1500 correct-image-span replacement peaks at layer 11 with margin effect +0.3538 [0.2506, 0.4569]. Subtracting a different-pair, wrong-label donor leaves +0.2194 [0.1463, 0.2994].
- Step 2000 peaks at layer 6 with raw effect +0.1531 [0.1125, 0.1931]; the preregistered layer-5 correct-minus-wrong contrast is +0.0856 [0.0519, 0.1181]. Step1500 minus step2000 at layer 11 is +0.2219 [0.1338, 0.3094].
- The final-layer image-span effect is zero because no later attention operation can transmit those patched token positions. The final assistant patch is the positive control and recovers +0.2925 [0.1875, 0.3950] at step 1500 and +0.0950 [0.0712, 0.1200] at step 2000.
- At projector-token input positions, step1500 center replacement is -0.0044 [−0.0119, 0.0025], outer replacement +0.2969 [0.1963, 0.4088], and full replacement +0.2925 [0.1913, 0.4013]. Outer-minus-center is +0.3013 [0.1988, 0.4188]. MoonViT projector tokens are globally contextualized, so this localizes token positions in the receiver and cannot localize source pixels or prove a background-pixel mechanism.

## Defects and invalid runs

- `shape_layerwise_smoke_v1` failed before producing output because the first extractor imported `training_protocol` instead of `tools_common`. The empty log and explicit local failure record are retained; the corrected smoke and full extraction passed.
- `shape_patching_smoke_v1` and `shape_patching_v1` used post-final-RMSNorm `hidden_states[-1]` as the layer-23 donor at a pre-final-RMSNorm hook. Both are invalid. The raw full v1 run remains on the data disk and its hashes are bound by `INVALIDATION.json`; v2 reran all 18,300 rows after the exact-hook correction.
- `shape_layerwise_v1_analysis` is superseded because its random-label significance procedure was underpowered and high-variance. The representation tensors were valid and unchanged; v2 recalculated every metric and prediction with pair-unit permutation.

## Independent verification and next decision

- Independent verification checks 30 representation files / 6,000 representation rows, all 672 metric rows, 268,800 predictions, 18,300 patch rows / 183 cells, file hashes, tensor keys/shapes/finiteness, visual-source provenance, exact denominators, pair units, final-layer reproduction, and `final_half_scored=false`.
- Package 4 refutes projector information loss as the shape bottleneck. It supports mid-layer access followed by upper-layer erasure, with a weaker and earlier causal path at step 2000.
- The next discriminating local experiment is a small top-layer LoRA diagnostic anchored at step 1500, with the projector frozen and projector-only continuation as a matched control. It will score teacher-forced paired preference and free generation at checkpoints, then determine whether upper-layer adaptation restores use of already-preserved shape information before testing the 1.5B capacity arm.

