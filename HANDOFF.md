# Handoff

## Current authority (2026-08-07)

Read [`docs/current-status.md`](docs/current-status.md) first for the live state and
[`docs/architecture-matrix.md`](docs/architecture-matrix.md) for architecture IDs.
The immutable evaluation rules remain in
[`docs/qwen2.5-3b-community-eval-contract.md`](docs/qwen2.5-3b-community-eval-contract.md).
The final target is `MoonViT-V2 → 4096 projector → DeepSeek-V4-Flash-0731`.
Qwen2.5-3B is a low-cost pure-text proxy; exact K3 V2 and V1 family controls are
the current local comparison. Gate D is **NO-GO**. The dedicated workstation
pane is `moonvit:0.0`; use it for the next V100 screen and capture its logs.

Token boundary already audited: Qwen2.5 `<|image_pad|>` is ID 151655 and its
reserved rows are dummy pure-text initialization; DeepSeek `<｜image｜>` is ID
129279 and is retained for Hash-MoE routing. Neither path extends the vocabulary.
Qwen3.5 native token rows belong to a diagnostic receiver-prior control and never
replace the pure-text Qwen contract.

## 2026-08-07 plain-language status and DeepSeek ETA

The project has a working software seam: real MoonViT features enter a frozen
receiver through a trainable projector, gradients reach the projector, and tiny
DeepSeek-shaped FP32/BF16 loops save, resume, and generate exactly. The central
product claim is still open: Qwen2.5-3B and the matched V1/V2 screens do not show
stable correct-image attribution. Qwen2.5-7B runs on the V100 and its full
1,272-row public ScreenSpot diagnostic now has a weak vision-vs-shuffled
click-in-box gain (`+0.629` points, CI `[+0.157,+1.179]`), while vision remains
slightly below blind (`-0.157` points, CI `[-0.943,+0.629]`). The 50-row
GLM-format screen still crossed zero; the larger receiver changes coordinate
priors without producing reliable click grounding. Qwen3.5-9B gives a useful
receiver-prior signal but is too memory heavy for projector-only training on this
V100 and is not a leaderboard result.

This leaves a credible engineering path with an unresolved scientific bottleneck:
the projector/receiver/target interface must make `vision - shuffled` positive on
real data. Before DeepSeek-V4-Flash-0731 training, six gates remain: load the
resolved 0731 weights and image-token routing; verify finite non-zero input
gradients through the real FP4/FP8 path; run a full-model image forward/backward/
generate micro-loop; pass a 20-step stability and memory pilot with the online
collapse guard; prove exact full-checkpoint save/resume; and finally pass the
fixed ScreenSpot/TextVQA/DocVQA/OCRBench causal contract. Local software work is
estimated at 1–2 working days; after explicit paid-hardware authorization, the
minimum 0731 pilot is roughly 2–5 working days, conditional on kernels and
weight access. Gate D is **NO-GO** until these artifacts exist.

The historical mechanism record remains part of the handoff: falling CE without
visual attribution, V1/V2 early geometry collapse, low-LR geometry preservation
without causal gain, token-count sensitivity, and Qwen2.5/Qwen3.5 receiver-prior
differences. These observations guide the next local screen and are not replaced
by a single final benchmark number.

The matched V1/exact-K3 V2 initialization contract is now executable. Both
step0 (seed 20260805) and random-projector (seed 20260806) were serialized and
verified by regeneration plus strict save/load. V1 step0 is
`f24f677f…786cf`; exact K3 V2 step0 is `bec6e8bf…54815`. The V1 snapshot weight
set is bound to aggregate SHA `51a39391…f0ef`. Machine-readable manifests are
under `experiments/qwen3b_community_eval_20260805/architecture_controls/`;
complete 589 MB of projector tensors remain on the workstation HDD under
`data/qwen3b_contract/architecture_controls/`. The next pane command should
materialize the two effective contracts and build the V1 50-row health cache.
The first cache attempt passed the snapshot directory directly to Transformers
5.12.1 and hit its symlink-relative-import bug before writing any tensor. The
tracked fix loads by pinned model ID/revision and keeps the snapshot for hashing
only; the failure record is next to the V1 architecture-control artifacts.

The architecture control is now complete: both V1 and exact-K3 V2 high-frequency
health screens stop by step 2, and neither has entered the capability leaderboard.
Do not follow the older text below that says the V1 benchmark is still pending;
the current decision is that both versions have failed the current 3B health or
causal screen, while the full-public 7B result is recorded at the end of this
handoff.

## Community GLM-5.2V architecture audit (2026-08-06)

The public `baseten/GLM-5.2-Vision-NVFP4` page has now been checked against
the resolved files and the independent projector tensor header. It explicitly
uses the MoonViT-3d vision tower from Kimi-K2.6: 27 layers, width 1152, 2x2
merge, frozen vision and frozen GLM-5.2 text. The newly trained component is a
49.5M-parameter PatchMerger with affine pre-LayerNorm and bias-bearing
`4608 -> 4608 -> 6144` linear layers. The standalone projector file is bound by
SHA-256 `e7c6ce8c27424f292e708e7bbb48ade57ea9f1aaddd28bd6a1020a860d9db80c`.

This separates the two meanings of “来自 K2.6”: the tower is copied from the
K2.6 lineage; the GLM-compatible projector is trained for GLM's 6144-wide text
space. It is not a directly reusable K2.6 projector because K2.6 targets 7168.

The audit also found a naming/structure boundary in our code. The current
`PatchMergerProjector` used with the `[tokens,4,1024]` MoonViT-V2 tower is a
legacy V1-style implementation (`pre_norm + bias-bearing MLP`). The vendored
Kimi-K3/MoonViT-V2 reference uses `PatchMergerMLPV2`: bias-free MLP followed by
trainable post-RMSNorm. Package 15P therefore remains a valid failure record
for the implementation that was trained, but it cannot be presented as a
failure of the exact K3 V2 projector. The machine-readable source and hash
record is under
`experiments/qwen3b_community_eval_20260805/community_architecture_audit_v1/`.

Gate D remains NO-GO. Before promoting any checkpoint, the next matched screen
must compare an exact K3-V2 projector variant with a MoonViT-SO-400M V1
projector on Qwen2.5-3B under the same fixed evaluation contract.

The V1 control is now explicit in `configs/qwen2.5-3b-projector-moonvit-v1-community.json`.
Its projector emits canonical 4096 and reuses the same frozen Qwen 4096-to-2048
receiver as V2; an earlier draft emitted 2048 directly, was rejected as
non-comparable, and is not a valid result.
`tools/cache_moonvit_features.py` accepts `--vision-tower v1` with the pinned
MoonViT-SO-400M revision, while the existing `--vision-tower v2` path remains
the default K3 cache route. V1 and V2 caches/projector checkpoints are kept
separate; the V1 run is an architecture control and cannot become the
DeepSeek candidate without the same causal evaluation evidence.

## Projector health contract (2026-08-06, mandatory for the next V100 runs)

The practical objective is easy to state: MoonViT-V2 reads the screenshot, the
4096-wide projector translates those visual tokens, and a frozen text model must
change its answer because of the image. A falling cross-entropy number is only an
optimization signal. The earlier Qwen2.5-3B run showed why: projector RMS grew
from about 0.124 to 35.74 while effective rank collapsed to about 1, and vision
and shuffled-image conditions were indistinguishable. That run learned a common
coordinate prompt, not image-dependent grounding.

The next runs are therefore bound to
`configs/qwen3b-projector-health-v1.json` and the immutable 50-row
`health_probe_v1/PROBE_MANIFEST.json`. Every optimizer step writes
`train_health.jsonl`; steps 0, 1, 2, 5, 10, 20, 30, 50, 75, 100 and then every
50 steps write representation and eight-row teacher-forced causal probes. Both
the canonical 4096 boundary and the Qwen receiver are measured. The fixed hard
thresholds are spread ratio >= 0.25 and effective-rank ratio >= 0.50; top-1
variance > 0.80 / > 0.90 and RMS ratio > 10 / > 50 are warning / critical.

A critical guard now saves the failure checkpoint, optimizer/RNG state, current
batch IDs, health/probe logs, the last healthy checkpoint and an onset interval,
then rolls the in-memory model back to the last healthy state. The independent
`tools/verify_qwen3b_training_health.py` recomputes guard decisions and rehashes
the artifact tree. This is a training-safety mechanism, not a visual-ability
claim: only full ScreenSpot click-in-box and TextVQA/DocVQA/OCRBench causal gains
can promote a checkpoint. If all four Package-15P geometry arms fail the frozen
health/CE screen, the 500-step expansion is cancelled and projector structure
redesign is the next local experiment. DeepSeek-V4-Flash-0731 remains a later,
unpaid migration target; Gate D is still NO-GO.

### First valid high-frequency result: control arm (2026-08-06 UTC)

The corrected control run loaded the real 3B model and MoonViT cache, then
stopped automatically at optimizer step 2. The onset interval is now `[1, 2]`:
projector RMS rose `0.1235 → 0.6598`, projector spread ratio fell to `0.2690`,
and receiver effective-rank ratio fell to `0.3622`. CE still fell `4.1440 →
2.4380`; the two trends therefore give a clean counterexample to using loss as
the visual-success criterion. The run saved failure/healthy checkpoints,
optimizer and RNG state, current batch IDs, rollback metadata and all JSONL
logs. The independent verifier reports `verified` (3 probes, 3 checkpoints,
22 hashed health artifacts, 1,141,300,055 bytes). Complete raw tensors are
kept outside Git at `D:/V100-artifacts/geometry_repair_screen_hf_v1/control`;
the committed pointer and manifest binding are under the matching experiment
directory.

This result supports early common-direction collapse and the usefulness of
automatic stop/rollback. It does not establish visual ability, does not promote
`previous_best`, and does not justify a DeepSeek run. The matched `ratio005`,
`ratio020`, and `ratio080` arms remain required; if they all stop or exceed the
frozen CE/geometry rule, the next experiment is projector structure repair.

The first matched repair arm, `ratio005` (`lambda=0.01018730507868909`), also
stopped at step 2 with onset `[1,2]`. At that point total loss was 2.45268
(CE 2.43802 plus geometry 1.43947), while projector/receiver spread ratios
were `0.2691/0.2254` and receiver rank ratio was `0.3622`. The tiny loss change
relative to control did not preserve image geometry. This refutes the idea that
the minimum calibrated dose alone is enough; `ratio020` is the next matched
run.

The 20% arm (`lambda=0.04074922031475636`) has now been independently verified
and shows the same onset `[1,2]`. Its step-2 total loss is 2.49668, while
projector/receiver spread ratios are `0.2692/0.2255` and receiver rank ratio is
`0.3623`. Thus both nonzero low-to-moderate doses fail the frozen health screen
before step 5. The final `ratio080` arm also stops at `[1,2]` (total loss
2.67265; receiver spread/rank `0.2258/0.3628`). All four arms therefore fail;
`DECISION.json` records an empty passing set and cancels the 500-step expansion.
The contract now points to a matched projector output-normalization/residual
structure screen.

### Package 15Q is frozen before results

The next local screen is registered under
`experiments/qwen3b_community_eval_20260805/projector_structure_screen_v1/`.
It tests affine-free post-output LayerNorm and RMSNorm against the unchanged
CE-only projector. The operation sits after `linear_2` at width 4096, adds no
trainable parameters, and reuses the exact step0 weight bytes. The same 50-image
health probe and auto-stop/rollback contract applies from step 0 onward. A
structure candidate can proceed only if it avoids critical guards through the
fixed 100-step screen and stays within the preregistered CE ratio; no health
pass alone can promote a checkpoint to visual evaluation.

Before GPU results, three mechanical issues were preserved in the package
failure archive: the RMSNorm test tolerance, the verifier's legacy omitted
`output_norm` default, and a launcher variable-scope mistake. The first two
were repaired in code/tests; the launcher repair only changes process setup.
No optimizer step or capability result was created by any of them.

### Package 15Q baseline result

The freshly matched `baseline_none` control also auto-stopped at optimizer
step 2, with collapse onset `[1, 2]`. CE fell from `4.14400` to `2.43802`,
while projector spread/rank ratios reached `0.2690/0.5022` and receiver
spread/rank ratios reached `0.2254/0.3622`; both RMS-rising/spread-falling
critical guards fired. The independent verifier recomputed all three probe
points and all three checkpoints. This is a health failure result, not a
grounding result; no ScreenSpot or capability checkpoint was promoted. The
complete raw copy is outside Git at
`D:/V100-artifacts/projector_structure_screen_hf_v1/baseline_none`.

The matched `post_layernorm` candidate also auto-stopped at step 2 with the
same onset and the same two critical trend guards. Its CE was `4.92825` at
step 1 and `3.60105` at step 2; projector spread/rank ratios were
`0.1998/0.6452`, and receiver ratios were `0.1559/0.5178`. The independent
verifier passed all three probes/checkpoints. Output normalization held scale
at the 4096 boundary, yet did not preserve cross-image spread, so this arm is
not promoted to capability evaluation. Its raw copy is outside Git at
`D:/V100-artifacts/projector_structure_screen_hf_v1/post_layernorm`.

The final `post_rmsnorm` arm also stopped at step 2. It reached projector and
receiver spread/rank ratios of `0.2110/0.7540` and `0.1656/0.6285`, and it
additionally triggered the causal critical guard because
vision-minus-shuffle correct-answer log-prob was `-0.21164`. Its independent
verifier passed all three probes/checkpoints; the raw copy is outside Git at
`D:/V100-artifacts/projector_structure_screen_hf_v1/post_rmsnorm`.

### Package 15R is frozen before results

Package 15Q showed that output-only normalization does not preserve image
geometry. Package 15R therefore keeps the original 4096 path and adds either
a zero-initialized full-width residual branch or a normally initialized branch
with a scalar gate initialized to zero. Both candidates reproduce the exact
step0 output before training and share all base MLP tensors. The fixed
100-step/800-example health screen, high-frequency probes and auto-stop/
rollback rules are unchanged.

The initialization files and SHA-bound manifest are outside Git at
`D:/V100-artifacts/projector_residual_initializations_v1_retry1`; the pointer
and two pre-result repair records are in the preregistration package. No arm
may enter capability evaluation on health metrics alone.

Package 15Q therefore has no passing arm. The pre-registered 500-step
expansion is cancelled. The evidence supports a step-one update-direction or
receiver-interface problem that output-only normalization cannot repair. The
next local package will test a residual/gated-residual projector with a
matched CE-only control, then return to the fixed real-vision evaluation
contract only if the health screen survives.

### Package 15R baseline control result (2026-08-06 UTC)

The first arm of the frozen residual screen re-ran the unchanged step0
projector under the new runner commit `5682265c`. It reproduced the known
failure exactly: the health probe found collapse onset `[1, 2]`, CE fell from
`4.14400` to `2.43802`, projector probe RMS rose `0.1235 → 0.6598`, and
projector spread/rank ratios fell to `0.2690/0.5022`. Receiver ratios fell to
`0.2254/0.3622`. The two fixed RMS-rising/spread-falling critical guards
stopped the run at step 2 and rolled it back to the healthy step-1 checkpoint.

The independent verifier recomputed 3 probes, 3 checkpoints and 22 hashed
health artifacts (`1,141,294,624` bytes) with status `verified`. The complete
raw copy, including optimizer/RNG/checkpoint state, is bound by the committed
pointer at `D:/V100-artifacts/projector_residual_screen_hf_v1/baseline_none`;
the Git-sized result files are under
`experiments/qwen3b_community_eval_20260805/projector_residual_screen_hf_v1/results/baseline_none/`.
No capability score was run, no checkpoint was promoted, and Gate D remains
NO-GO.

This control supports that the early collapse is reproducible after the 15R
code sync and that the automatic stop/rollback path is trustworthy. It
refutes any claim that the residual candidates can be judged by a lower CE
alone. The next arm is `zero_init_residual`; it must keep the same first-step
geometry healthy before we spend time on full grounding evaluation.

### Package 15R zero-initialized residual result (2026-08-06 UTC)

After the variant-binding repair, `zero_init_residual` passed the pre-GPU
contract check and ran on the same V100 budget. Its step-1 projector gradient
norm before clipping was `189.33`, confirming that the new branch was trainable.
The health trajectory still stopped at `[1, 2]`: CE fell `4.14400 → 2.88565`,
while step-2 projector output RMS reached `1.4244`, spread/rank ratios fell to
`0.1838/0.1799`, and receiver RMS reached `1.9779` with ratios
`0.1672/0.1358`. The same two fixed critical guards fired and rollback restored
the healthy step-1 checkpoint.

The independent verifier passed 3 probes, 3 checkpoints and 22 artifacts
(`1,711,724,384` bytes); the local raw copy and SHA recheck are bound under
`experiments/qwen3b_community_eval_20260805/projector_residual_screen_hf_v1/results/zero_init_residual/`.
This arm is not a capability result. It supports the narrower conclusion that
the residual branch can receive gradient while the receiver-facing update still
collapses image geometry, and it refutes “zero initialization alone preserves
the visual subspace.” `gated_residual` remains the final pre-registered arm;
if it also stops at step 2, the package moves to a frozen-base residual-only or
learning-rate-cap intervention rather than a 500-step expansion.

### Package 15R variant-binding repair before the first candidate

The first `zero_init_residual` launch stopped before CUDA model load because
the legacy trainer accepted only the canonical projector SHA. The failure is
archived as `attempt04_variant_sha_gate`; it created no optimizer step or
checkpoint. The repaired runner now accepts a variant only when a registered
contract binds its config/weights SHA, base projector SHA, parameter count,
common base tensors and exact step0 output. The binding module and runner
source hashes are recorded in the 15R config and preregistration. This changes
the runner interface only; data, budget, health thresholds and model remain
fixed. The candidate will be relaunched after this repair is committed.

## ⚠️ ACTIVE V100 REAL-VISION BRIDGE (2026-08-06)

**Current task / hard boundary**: the engineering mainline is a fixed real-data bridge on pure-text `Qwen/Qwen2.5-3B-Instruct`, followed by transfer of a validated MoonViT/projector/data/eval contract to DeepSeek-V4-Flash-0731. The 0.5B/synthetic line remains mechanism evidence. The public GLM-5.2V audit now requires two architecture controls before any checkpoint promotion: exact Kimi-K3/MoonViT-V2 `PatchMergerMLPV2`, and MoonViT-SO-400M/K2.6-lineage V1. Do not rent any server, create a paid resource, run the full DeepSeek weights, or inspect final evaluation halves without explicit authorization. Exact V100 environment evidence is under `experiments/v100_perception_20260804/infra/environment/`.

**Experiment packages 1–14 COMPLETE; Packages 15A–15R establish, diagnose and redirect the 3B bridge**: packages 1–12 established auditable caching and mechanism controls. Package 13 fixes preventive replay; Package 14 fixes the smallest reliable sentinel and V100 cost. Package 15A freezes the Qwen2.5-3B model/data/generation/scoring/budget contract before any 3B output. Package 15B closes the real-image load/gradient/optimizer/checkpoint smoke. Package 15C freezes the exact first-4,000 training prefix and teacher targets. Package 15D independently verifies the content-addressed MoonViT cache. Package 15E completes exact 4k projector-only training. Package 15F rejects the first GLM-format ScreenSpot50 candidate. Package 15G confirms the failure on all 1,272 public ScreenSpot rows. Package 15H shows that training raises coordinate-answer probability without learning correct-image coordinate preference. Package 15I preregisters the matched 2,000-grounding/2,000-short-answer treatment before its cache or training result exists. Package 15J independently verifies its content-addressed MoonViT cache. Package 15K completes the exact matched-budget training and checkpoint audit. Packages 15L/15M reject it by preference and free generation: vision/blind/shuffled preference is 52%/56%/54%, while click is 6%/12%/6%. Package 15N localizes a scale/rank collapse at the projector output; Package 15O shows that both collapse guards already fire at the first saved checkpoint, step 100/800 examples. Package 15P is a calibration and health-only screen: all four geometry arms stop at steps 1–2, so no repair arm is promoted and the 500-step expansion is cancelled. Package 15Q tests output LayerNorm/RMSNorm and stops all three arms early. Package 15R records the residual screen; its historical baseline and zero-init arms also stop early. The architecture audit now marks the old V2 label as legacy-pre-norm and blocks promotion until exact K3-V2 and V1 controls are run.

**Pre-rental go/no-go (authoritative top-level table)**:

| 条件 | 状态 | 证据 |
|---|---|---|
| Synthetic projector 可学习多任务视觉信号 | 通过（代理） | packages 7–9；六任务 teacher-forced vision−shuffle 均转正，不能代替真实 benchmark |
| 固定预算抗遗忘保护 | **本地通过，正式域待桥接** | package 13 fixed replay：目标任务 +0.255 [0.210, 0.300]，donor 合并 +0.0038 [−0.0125, 0.0188] |
| Sentinel 功效与成本 | 通过（V100 小主干） | package 14 Tiny=25 pairs/task；recall 0.975，false trigger 0.040，teacher median 22.501 s |
| Qwen2.5-3B 纯文本真实视觉 baseline | **4k 训练通过 / grounding 候选拒绝** | 500 steps、4,000 examples、21,532 answer tokens；GLM50 vision click 4%，blind 12%，step0 10%；vision−blind 距离显著恶化 |
| 完整公共真实评测合同 | **ScreenSpot50/full 完成 / 候选拒绝** | 1,272 条七条件与 2,000 bootstrap 已落盘；vision click 2.67%，blind 3.07%，step0 3.30%，vision−blind 距离显著恶化 |
| Paired preference 机制判定 | **完成 / 无 content-specific readout** | trained vision/blind/shuffled 为 46%/56%/52%；训练显著降低坐标 NLL，但正确图与错图的 correct-logp 无差异 |
| Grounding-enriched fixed-budget treatment | **训练通过 / preference+generation 拒绝** | preference 52%/56%/54%；generation click 6%/12%/6%，vision−blind mean distance 显著恶化 109.47 |
| Projector representation retention | **step1–2 已触发在线止损** | Package 15P control 在 step2 停止；projector spread/rank ratio 0.2690/0.5022，receiver 0.2254/0.3622，RMS 0.1235→0.6598；CE 仍 4.144→2.438 |
| Package 15P geometry calibration | **四臂高频轨迹全部止损，500-step 取消** | λ=0/0.0101873/0.0407492/0.162997 的 onset 全为 [1,2]；四个独立 health verifier `verified`；转 projector 结构筛选 |
| 通用 train/save/resume/generate 链路 | 通过（含 3B 代理） | Qwen2.5-3B + 真 MoonViT 图像完成生成、梯度、一步 AdamW 与 checkpoint/optimizer/RNG 精确恢复 |
| 完整 DeepSeek-V4-Flash-0731 闭环 | **未通过** | 完整权重从未完成图像 forward/backward/train/save/resume/generate |
| Gate D 量化 DGRAD | **未通过（本地 harness 完成）** | 三模式接口/reference 通过；真实 FP8/FP4 module 仍 `hardware_pending` |
| 可以请求用户授权租机 | **否** | V100 仍可完成 3B baseline、固定真实评测、projector/分辨率/数据筛选与语言保持 |

**Package 12 decisive result**: the only treatment is batch ordering over the same 2,400 records. Stratified true batches are 100/100 exactly 4 records per task; global random is 0/100 balanced, with a maximum of 11 records from one task. Both arms use the same base projector (`98a566…37ef3`), step-0 tensor (`d2b413…f70e69`), AdamW source (`47ceb2…97196`), seed, hyperparameters, record set and examples seen. At step 50, stratified macro preference/generation is **0.512/0.233** versus global **0.389/0.167**; at step 100 global reverses to **0.531/0.320** versus stratified **0.511/0.257**. Endpoint stratified−global overall is **−0.020 [−0.0442, 0.0025]**, so the preregistered verdict is `mixed_or_underpowered`.

**Package 12 task/gradient result**: endpoint coordinate favors stratified **+0.165 [0.115, 0.220]**, while color/shape favor global **−0.090 [−0.165, −0.025] / −0.245 [−0.315, −0.175]**. Stratified forgets count/shape by 0.28/0.30 after step 50. On six fixed 8-record task batches, stratified-step100 has 6/15 negative projector-gradient cosines, led by count–shape **−0.1704**; global-step100 has 0/15 and mean cosine 0.1185. Per-batch stratification is therefore not a DeepSeek hard requirement. The rental contract now requires fixed-window domain coverage plus sentinels/replay, while retaining stratification only as a short-calibration candidate.

**Package 12 verification/storage**: independent verification re-read 50,400 preference rows, 8,400 generation rows, 735 metrics, 525 paired contrasts, 42 gradient norms and 105 task-pair cosines; all 14 matched-order invariants and every declared file hash pass. The full repository suite is **220/220** green. The report builds to **43 pages**; package-12 pages 31–33 and the revised rental/Gate-D pages 34–37 passed text/render inspection. Eight checkpoint manifests are committed without 937 MB of duplicate weights/optimizer payload. Final halves remain untouched. Remote run metadata records the workstation baseline `d5944d9…`; the synchronized scripts and full artifact manifest are anchored by the Package 12 commit.

**Package 13 fixed-budget design**: ordinary, fixed, and triggered policies all inherit the exact package-12 stratified step-50 projector and AdamW state. Every full policy trajectory is exactly 50 steps × batch 24 = **1,200 training examples**. Ordinary uses 200/task. Fixed replay replaces 80 donor slots with 40 historical count and 40 shape records, yielding `180/180/240/180/240/180`. Triggered shares ordinary steps 51–75, then reallocates only the remaining 600 slots. No policy adds an optimizer step or training example. The ordinary control exactly reproduces the historical step-100 projector across all six tensors and both recorded hashes.

**Package 13 sentinel/result**: ordinary step 50→75 raises overall paired preference by +0.040 [0.0108, 0.0675] while count collapses **0.380→0.075**, gap **−0.305 [−0.365, −0.245]**. The frozen trigger (`drop ≥0.10`, CI upper <0, at most two tasks) selects count only. At step 100, ordinary/fixed/triggered macro preference is **0.5108/0.5983/0.5358** and macro generation is **0.2567/0.3567/0.2600**. Fixed count/shape preference versus ordinary is **+0.255 [0.210, 0.300]**; donor tasks combine to +0.00375 [−0.0125, 0.01875]. Target generation improves **+0.120 [0.050, 0.190]**. Triggered count gains +0.175 [0.125, 0.230] but ends at 0.275, outside the step-50 recovery band 0.380±0.05. Fixed beats triggered overall by **+0.0625 [0.0425, 0.0833]**. Recommendation: `fixed_preventive_replay`; late triggering remains a useful fallback.

**Package 13 verification/storage**: the package verifier re-read 21,600/3,600 sentinel rows, 50,400/8,400 final rows, 735 metrics, 223 contrasts, 18 trajectories, eight checkpoint manifests and every declared file hash. It confirms zero extra training examples, 80/20 reallocated examples for fixed/triggered, exact ordinary reproduction, fixed target preference/generation CIs, bounded donor cost and the trigger-decision SHA. The first analyzer run failed only because it treated dictionary order as semantic state order; the failure log is retained, the corrected run checks the exact state set. The full repository suite is **231/231** green. The report builds to **45 pages**; package-13 pages 33–35 passed render inspection. Final halves remain untouched and no paid resources were used.

**Package 14 sentinel power result**: 200 deterministic trials at each 8/16/25/50/100 pairs/task show that 25 pairs/task is the smallest preregistered profile passing all guards: count recall **0.975** (Wilson `[0.943, 0.989]`), exact count-only decision **0.935** (`[0.892, 0.962]`), familywise false trigger **0.040** (`[0.020, 0.077]`). The 8/16-pair profiles fail on recall; Medium is fixed at 50 pairs/task.

**Package 14 timing/action**: three V100 repeats per profile give synchronized teacher medians **22.501 s Tiny / 43.881 s Medium**, end-to-end medians 31.215/52.537 s and 6.886 GB peak memory. Tiny raw rows exactly reproduce the same Package-13 `(state,id)` rows. At the Package-13 median train-step time, resident Tiny needs at least 476/226 steps for 5%/10% overhead, rounded to 512/256. Fixed preventive replay is now the default protection, Tiny is a sparse checkpoint audit, and Medium confirms Tiny alerts. Replay-dose, Fisher and EWC expansion is deferred unless the fixed real-data contract would change the formal recipe. The manifest contains 51 files / 3,708,513 bytes with 51/51 independent hash matches; the full suite is **240/240** green, and the 49-page report's package-14/Gate-D pages 35–38 passed render inspection.

**Package 15A fixed contract**: pure-text `Qwen/Qwen2.5-3B-Instruct` is pinned to `aa8e7253…04d1`; all 9 required files pass SHA-256, including weight shards `67347b…06c2` and `a40d94…bfc1`. Tokenizer bundle/chat template hashes are `69a5cf…93c6` / `cd8e94…527f`; `<|image_pad|>` already exists at ID 151655, so the tokenizer is not extended. Public ScreenSpot revision `0be08781…d5d` has 1,272 rows and three verified shards. `screenspot_glm50_v1` has exactly 5 samples in each platform×type stratum; manifests self-hash to `9583a75e…632b5` / `e556ac52…3d7cc`. Strict parser, official click metrics, parsed/all denominators, 2,000 paired bootstrap, seven conditions and 4k/8k/16k/32k/64k budgets are implemented. Language retention adds 140 balanced MMLU-Pro plus 100 GSM8K rows, manifest `c518403a…b6a8`.

**Package 15A exact initialization controls**: canonical step0/random-projector are two separately seeded FP32 states with the same 33,564,672-parameter structure and initialization distribution. Their 134,259,248-byte safetensors SHA-256 values are `efd942e0…b06b0` / `7bd4aacf…fc44`; exact tensor-state hashes, same-seed regeneration and save/restore all match. HF commit `65639da5…a010` publishes five files under `contract/qwen2.5-3b-community-eval-v1/projector_initializations_v1`; both LFS SHA values and all small-file downloads match. Every horizontal method comparison loads the exact step0 file. First-baseline `previous_best` aliases step0, then advances only to an accepted full-contract checkpoint.

**Package 15A V100 precision decision**: official Qwen shards remain byte-exact BF16 source artifacts. A fixed 4096×4096 GEMM probe on the Tesla V100-PCIE-32GB found BF16 callable and finite, but its median time was 9.16× FP16 (FP32 was 6.82× FP16). The preregistered runtime therefore loads Qwen in FP16, keeps projector master weights FP32, casts only at the embedding splice, and fails on any non-finite loss/gradient. Raw timings are under `contract/hardware/`.

**4096 boundary decision**: the exact K3 V2 projector is bias-free `4096 → 4096 → 4096` with trainable post-RMSNorm and 33,558,528 parameters; the historical legacy V2 projector has 33,564,672. Qwen receives either canonical output through a parameter-free signed-pair orthogonal 4096→2048 readout; all 4096 inputs receive gradient, buffers are frozen in a 37,072-byte safetensors (`1cecc883…a6d47`) and every architecture control shares them. The readout is discarded for DeepSeek, so Qwen checkpoint transfer is `transferable_with_runtime_validation`. No direct-2048 projector is a formal result.

**Package 15A failures retained**: aria2 multi-range produced a correct-size corrupt ScreenSpot blob (`0c793b…e534` versus expected `ff06d3…aa8fb`) after cross-shard range responses/403; retry used `HF_HUB_DISABLE_XET=1`, one worker, and all shards verified. The first Qwen command fetched only config and was replaced by the same single-worker method. Language manifest construction hit two Python/JSON boolean spelling errors; the first V100 precision probe had a Python syntax error. No partial model result was written; corrected runs succeeded. Logs and summaries are under `contract/failures/`; the corrupt 134.5 MB blob remains only on HDD.

**Package 15A verification**: full repository suite is **262/262** green. The artifact manifest declares 18 files / 1,967,831 bytes with 18/18 independent SHA matches. The report builds to **50 pages**; package-15A/Gate-D pages 38–39 passed text extraction and rendered-page inspection. `contract/VERIFICATION.json` confirms all manifests, source/model hashes, exact initialization controls, receiver save/restore, V100 precision choice and result-before-contract boundary. No paid resource was used and no Qwen3B output exists before this freeze.

**Package 15B real Qwen3B smoke**: the pinned `Qwen2ForCausalLM` loaded in FP16 with all 3,085,938,688 parameters frozen after 9/9 Qwen files and the extracted MoonViT weight passed in-run SHA-256 checks. A frozen public ScreenSpot image (`screenspot-0be08781-0472`, SHA `f079f4…4041`) passed through the real MoonViT-V2 tower to `[128,4,1024]`, then the exact 33,564,672-parameter FP32 step0 projector and fixed 4096→2048 receiver. Assistant-only loss was 2.19208. All six projector parameter tensors had present, finite, nonzero gradients; the canonical 128×4096 visual embedding gradient had 524,236/524,288 nonzero elements, while language-model gradient tensors were exactly zero. Peak GPU memory was 8,367,393,280 bytes; canonical wall time including ~7 GB frozen-input hashing was 174.476 s.

**Package 15B save/resume and claim boundary**: one AdamW step produced a 469,922,601-byte checkpoint. FP32 projector tensors, optimizer state, Python RNG and history restored exactly; serving BF16 weights were also emitted. Step0 vision and blind both generated `click(start_box=[500, 250])`, so `visual_ability_established=false`: this supports the real engineering/gradient path only and does not support grounding improvement. The transfer label remains `transferable_with_runtime_validation` because DeepSeek discards the fixed Qwen receiver.

**Package 15B failures/storage**: attempt 1 completed the expensive path but its final verifier called `torch.equal` across CPU/CUDA AdamW scalar devices. The invalid 470,219,506-byte run is retained; a focused cross-device regression now covers equal and changed states. A preceding helper-test import error is also recorded. Retry1 changed only the verifier and passed. Submission review then moved all model/tower SHA checks into the runner and tightened every projector tensor to nonzero; canonical retry2 passed with the same checkpoint hashes and byte-identical generation rows as retry1. Independent workstation rehash matched invalid 12/12, retry1 13/13 and retry2 13/13 files, totaling 470,219,506 / 470,232,302 / 470,235,478 bytes. Complete roots stay under `$HDD/data/qwen3b_contract/smoke_v1`, `smoke_v1_retry1` and `smoke_v1_retry2`; curated evidence is under `experiments/qwen3b_community_eval_20260805/smoke_v1/`. Final halves remain untouched and no paid resources were used.

**Package 15B verification**: the checked-in smoke manifest contains 26 files / 73,000 bytes and is rehashed by a repository test. Focused chat/checkpoint/artifact tests are 6/6 green; the full repository suite is **269/269** green. The report builds to **51 pages**; package-15B page 39 and the revised Gate-D page 40 passed rendered inspection. `smoke_v1/VERIFICATION.json` preserves the independent full-HDD rehash counts and prohibits a capability claim.

**Package 15C exact 4k order**: the first matched budget is now an immutable prefix of the Package-15A 59,198-row training pack: first 4,000 rows, no shuffle, no holdout removal, micro batch 1, accumulation 8, real global batch 8, and exactly 500 optimizer steps. This is one subset pass and 0.0675698503 full-pack effective epochs. Sources are TextVQA 1,985, DocVQA 1,160, OCRBench-derived `train` 516 and ShowUI desktop 339. All 4,000 IDs and paths are unique; 3,534 image SHA values reflect intentional same-image multi-question rows. The manifest self-hash is `ddca738e…c2fd`; its ordered-record hash is `61fa7360…315e`.

**Package 15C target/data audit**: all 339 ShowUI rows arrived as legacy `click(start_box=[x,y])` supervision and are deterministically converted to canonical `click(start_box=[x, y])`, while raw-answer hashes remain bound. Two TextVQA answers, `(` and `a`, normalize to empty under the VQA metric and use an explicit raw-majority fallback. Independent verification re-read the 59,198-row JSONL and matched 4,000/4,000 record hashes, teacher targets, image SHA values and dimensions, covering 1,523,324,154 image bytes. Two fail-closed attempts preserve the discoveries above; neither wrote a manifest or training result. The Git package contains 9 files / 3,446,266 bytes plus its artifact manifest. This evidence is `directly_transferable` to DeepSeek because it binds shared data, order, targets and accounting. It makes no capability claim, leaves previous-best at step0, does not touch final halves and uses no paid resource.

**Package 15C verification**: four target/order unit tests and two checked-in artifact tests pass. The complete V100 suite is **268 passed, 3 skipped**; skips are optional/environment-gated tests, with no failure. `report/main.pdf` rebuilds to **53 pages**; package-15C page 40 and the revised Gate-D page 41 passed text extraction and rendered-page inspection.

**Package 15D canonical feature cache**: the clean runner commit is `1e4c400…4142`; it is bound to Package-15C manifest `ddca738e…c2fd`, MoonViT-V2 weight SHA `01436a95…ced24`, max side 448, eager attention and float32 CPU storage. It cached 4,000/4,000 records with zero failures in 503.5901 s and 1,949,755,904 peak GPU bytes. Content addressing required 3,534 real tower forwards and reused 466 same-byte images. The final cache has 3,534 unique tensor spans, 111 safetensors shards and 10,372,103,792 shard bytes; the largest visual sequence is exactly 256 groups.

**Package 15D independent verification/failure boundary**: verifier commit `a9bd07b…a646` rehashed all 111 shards, read every record, checked 2,921,816,064 logical float values for finite/shape identity, verified 2,593,021,952 unique values, canonical first-occurrence aliases, all three recorded runtime source hashes, clean runner provenance and exact per-row Package-15C binding. The full V100 root manifest covers 118 files / 10,374,552,697 bytes. Attempt 1 reached 1,128 rows before the provenance audit found uncommitted runner files under HEAD `018b798…`; it was terminated, preserved as 33 shards, and is forbidden for training. The curated Git package contains 14 files / 2,728,827 bytes plus its manifest. This supports cache feasibility, exact order identity and a directly transferable frozen-MoonViT input pipeline; it does not support Qwen3B visual ability, grounding improvement or any Gate-D claim. Final halves remain untouched and no paid resource was used.

**Package 15D verification**: four focused cache-verifier tests and three checked-in artifact tests pass. The complete V100 suite is **288 passed** with zero failure. `report/main.pdf` rebuilds to **54 pages**; Package-15D page 41 and the revised Gate-D page 42 passed text extraction and rendered-page inspection.

**Package 15E exact 4k training**: clean runner commit `97e9c03…a9d3a` trained only the canonical 33,564,672-parameter FP32 projector over the exact Package-15C prefix and verified Package-15D cache. The frozen FP16 Qwen has 3,085,938,688 parameters and zero gradient tensors; the fixed 4096→2048 receiver is also non-trainable. The run is exactly 500 AdamW steps, micro batch 1, accumulation/global batch 8, 4,000 examples, 21,532 answer tokens and 0.06756985 full-pack effective epochs. Training wall is 532.810 s, total wall 905.390 s and peak GPU allocation 8,979,616,768 bytes. All six projector tensors have finite nonzero gradients at steps 1 and 500. Loss falls 4.60169→2.47889, which is recorded as optimization evidence only.

**Package 15E resume/verification/failure boundary**: checkpoints 100/200/300/400/500 contain FP32/BF16 projector, optimizer, RNG and history; 25 payload files total 2,351,006,545 bytes. Independent verifier commit `075f3e5…acc` rebuilds all 500 batches and answer-token counts, rehashes every checkpoint, confirms six optimizer states and exact final step 500. Final projector SHA is `566830f3…a89f`, distinct from step0 `efd942e0…b06b0`. The first verifier attempt raised `KeyError: training_order_manifest_sha256` because it read identity from the budget-only `RUN_CONFIG.binding`; the failure is preserved, the verifier now reconstructs identity from checkpoint manifests, and training artifacts were never modified. This establishes a reproducible train/save/resume path. It does not establish image use.

**Package 15F GLM-format public-50 grounding result**: all seven registered roles were generated with the fixed parser, greedy decoding and 1024-pixel cache. Trained vision parses 48/50 and reports all-denominator Accuracy@50/100/200 **2%/4%/16%**, click-in-box **4%**, mean/median center distance **554.53/568.37**. Blind is **12%** click and mean distance **392.59**; step0/previous-best is **10%** and **398.59**; shuffled is **6%** click. Vision−blind click is **−0.08 [−0.20, 0.02]** and mean-distance improvement is **−161.94 [−246.70, −89.24]**. Current−step0 click is **−0.06 [−0.16, 0.04]** and mean-distance improvement is **−155.94 [−246.74, −75.50]**. Vision−shuffled click is **−0.02 [−0.06, 0]**; no threshold-accuracy CI has a positive lower bound.

**Package 15F decision**: `current_candidate` is rejected and `previous_best` remains exact step0. The result refutes “0.5B capacity was the sole grounding bottleneck,” “3B plus 4k projector-only supervision is sufficient,” and “lower training loss establishes causal vision.” It supports an operational 3B train/eval path and shows that the present objective can learn format while degrading the text/center prior. The method implementation remains `transferable_with_runtime_validation`; this learned checkpoint does not enter the DeepSeek candidate list. Full public ScreenSpot is the immediate confirmation run. Its 1,272 records were materialized from all three pinned shards and cached with 606 real MoonViT forwards, 666 content-addressed aliases, zero failures, 387.349 s wall and 2,081,363,968 peak bytes; seven-condition generation is active on the local V100.

**Package 15G complete public ScreenSpot**: all seven roles now cover 1,272/1,272 rows. Trained vision parse/Accuracy@50/@100/@200/click/mean distance is **96.46% / 1.73% / 4.87% / 11.79% / 2.67% / 565.18**. Blind click/mean is **3.07% / 395.52**; step0 is **3.30% / 391.12**; shuffled is **2.75% / 566.26**. Vision−blind click is **−0.0039 [−0.0165, 0.0079]**, Accuracy@200 is **−0.0322 [−0.0598, −0.0024]**, and mean-distance improvement is **−169.66 [−185.68, −154.17]**. Current−step0 loses 3.54 parse points and worsens mean distance by **174.06 [157.44, 189.67]**. Vision−shuffled click and distance CIs cross zero. The full set therefore confirms the 50-row failure; current remains rejected.

**Package 15G systems/failure boundary**: complete generation took 2,807.658 s and 7,247,035,392 peak V100 bytes. Materialization covers 606 unique images / 593,342,933 bytes. The cache contains 606 real forwards, 666 aliases and 7,609,930,976 feature bytes. Every prediction and per-row score is checked in; image/tensor payloads stay on the V100 under SHA manifests. The first new-output command mistakenly supplied `--resume` and failed at cache/config verification before model load or prediction; it is retained separately.

**Package 15H teacher-forced paired preference**: under correct versus frozen-derangement counterfactual coordinates, trained vision is **23/50 = 46%**, blind **56%**, shuffled **52%**, step0 **54%**, random projector **50%**. Vision−blind is **−0.10 [−0.22, 0.02]**; vision−shuffled is **−0.06 [−0.14, 0]** and its mean-margin change is **−0.00725 [−0.01287, −0.00186]**. Current−step0 preference is **−0.08 [−0.26, 0.08]**. Yet correct-answer NLL improves from 2.50769 at step0 to 1.22362, gain **1.28407 [1.13713, 1.43892]**; correct and shuffled images have indistinguishable correct-answer logp. Training learned an image-agnostic coordinate soft prompt, not hidden grounding that greedy decoding failed to express.

**Package 15H action/failure boundary**: the same-stream 8k extension and decoding-only work are unsupported. The next one-variable screen keeps exact step0, 500 steps, 4,000 examples, resolution, receiver and evaluators while increasing explicit ShowUI grounding inside the fixed budget. If paired preference remains non-causal, add a discard-after-training counterfactual-margin auxiliary objective. The first one-row preference attempt correctly failed after raw scoring because aliases were aggregated under source condition names; commit `6c23722…09ba` adds role labeling and a regression test, the failure is retained, and the clean retry/formal run passed. No paid resource or final-half evaluation was used.

**Package 15I pre-result grounding-enriched order**: runner `c43c161…f3a7` freezes exactly 2,000 first-in-source-order ShowUI rows and 2,000 first-in-source-order short-answer rows, merged grounding-first in strict alternation. Every global batch of eight has four rows from each route. The treatment changes only training mix/order: exact step0, 4,000 examples, 500 optimizer steps, resolution, Qwen, receiver and evaluators remain fixed. Sources are ShowUI 2,000, TextVQA 1,080, DocVQA 649 and OCRBench-derived `train` 271. Manifest/order hashes are `d632ecc2…0bf1` / `f3c3dec1…15ab`.

**Package 15I audit/claim boundary**: the independent verifier reconstructed the registered selection from all 59,198 source rows, then matched 4,000/4,000 records, canonical targets, image hashes and dimensions, covering 1,255,969,179 encoded-image bytes. The complete V100 suite is 317/317 green. This is `directly_transferable` pre-result evidence; it establishes no visual capability and cannot advance previous-best. No paid resource or final-half evaluation was used. The next operation is a content-addressed MoonViT cache bound to this manifest, followed by the exact 500-step screen.

**Package 15J grounding-enriched cache**: clean runner `aa933ca…b376` cached 4,000/4,000 rows with zero failures in 299.142 s and 1,947,973,120 peak V100 bytes. Content addressing performed 2,013 real MoonViT forwards and reused 1,987 later rows. Independent verification rehashed all 63 shards, loaded and checked 2,742,976,512 logical float values / 1,485,864,960 unique values, and matched every row to Package 15I; maximum visual tokens are 256. Shards total 5,943,468,912 bytes, while the full remote inventory is 70 files / 5,946,091,225 bytes. This is `directly_transferable` engineering evidence with no capability claim, paid resource or final-half evaluation. The exact 500-step projector-only run is next.

**Package 15K grounding-enriched training**: clean runner/verifier `f0afdae…e307` starts from exact step0 and completes 500 AdamW steps, 4,000 examples and 36,589 answer tokens over the strict 4/4 route batches. Qwen has 3,085,938,688 frozen FP16 parameters, the receiver is frozen, and only the 33,564,672 FP32 projector parameters train. Projector gradients are finite/nonzero at steps 1/500; language gradient tensors remain zero. Loss is 4.14400→1.91563, training/total wall is 489.606/529.299 s and peak allocation is 8,973,374,976 bytes. These are optimization/system results only.

**Package 15K checkpoint/claim boundary**: five checkpoints at steps 100–500 contain 25 payloads / 2,351,007,317 bytes. Independent verification reconstructs the exact 500 batches and 36,589 tokens, rehashes every payload, confirms six optimizer states and final projector `62f69393…3df4` distinct from step0. Full remote inventory is 40 files / 2,353,629,390 bytes. Package 15L now supplies the pending preference verdict; previous-best remains step0 and `visual_ability_established=false`. No paid resource or final-half evaluation was used.

**Package 15L grounding-enriched preference result**: the frozen 50-row teacher-forced gate scores correct versus preregistered counterfactual coordinates under blind, current correct/shuffled, step0 correct/shuffled and random correct/shuffled. Vision/blind/shuffled are **52%/56%/54%**; vision−blind is **−0.04 [−0.18, 0.10]**, vision−shuffled **−0.02 [−0.06, 0]**, trained−random **+0.02 [−0.14, 0.18]**, and current−step0 **−0.02 [−0.20, 0.14]**. Correct-image minus shuffled-image mean margin is **−0.002378 [−0.006099, 0.001248]**. The checkpoint is rejected at the paired-preference gate and cannot advance to full ScreenSpot or three seeds.

**Package 15L mechanism/action boundary**: correct-answer NLL falls from step0 2.50769 to **1.05915**, improvement **1.44854 [1.29793, 1.60698]**, while shuffled NLL is 1.05752 and correct-image minus shuffled-image correct-logp is **−0.001633 [−0.005786, 0.002342]**. Half-budget explicit grounding therefore strengthens the image-agnostic coordinate soft prompt and refutes grounding proportion as the sole first-run bottleneck. Complete the matched GLM50 generation contract, then screen the training-only counterfactual-margin auxiliary objective. The 14-file formal root is 685,140 bytes; the checked-in manifest rehashes all 17 package files / 694,076 bytes. Two focused artifact tests pass, the 61-page report builds, and Package-15L/Gate-D pages 48–49 passed rendered inspection. No paid resource or final-half evaluation was used.

**Package 15M GLM50 generation result**: vision parse/A@50/A@100/A@200/click/mean distance is **100% / 2% / 2% / 14% / 6% / 502.06**. Blind is 100% / 6% / 6% / 16% / 12% / 392.59; shuffled is 100% / 2% / 2% / 10% / 6% / 502.08; step0 click/mean is 10%/398.59. Vision−blind click is **−0.06 [−0.16, 0.02]** and mean-distance improvement **−109.47 [−171.64, −44.59]**. Vision−shuffled click is exactly **0 [0,0]** and mean-distance improvement **0.018 [−3.544, 3.213]**. Parse and community mean-distance thresholds pass, but all three accuracy thresholds and both causality guards fail. The checkpoint does not reach the community metric-aligned baseline.

**Package 15M collapse/action boundary**: vision emits only six unique coordinates and returns `[125,345]` on 31/50 rows; shuffled uses the same mode on 23/50 and exactly matches vision on 30/50. The 2,000 grounding labels contain 1,066 unique pairs and zero exact `[125,345]` targets, so this is a learned output collapse rather than a copied majority label. The next task is a minimal step0/current projector→fixed-receiver information-retention screen over the frozen 50 rows. It decides between representation-preserving projector repair and the matched training-only counterfactual-margin target. The two remote roots contain 19 files / 565,923 bytes; the package manifest rehashes 23 files / 576,071 bytes. Three focused artifact tests pass. No paid resource or final-half evaluation was used.

**Package 15N pre-result representation contract**: the frozen 50-row order, exact step0, `62f69393…3df4` and fixed 4096→2048 receiver are bound before activation extraction. The screen records pooled float64 representations, sample/between-image RMS, relative spread, participation/entropy rank, top-1 variance fraction, within-image token RMS, all pairwise distances/cosines, linear CKA and distance correlation at MoonViT/projector/receiver boundaries. Gross receiver collapse requires both current/step0 relative spread below 0.25 and participation rank below 0.5. The projector has no text query, so image-only coordinate probes are forbidden as capability evidence. This package contains no new representation result, no training, no paid resource and no final-half evaluation.

**Package 15N decisive result**: both registered guards fire before and after the receiver. At projector output, current/step0 relative spread is **0.1384** and participation-rank ratio **0.0859**; effective rank is **13.28→1.14**, top-1 variance fraction **17.48%→93.46%**, sample RMS **0.124→97.31**, and within-image token RMS **0.139→18.45**. Absolute pairwise distances increase, so the failure is a huge nearly collinear common-direction soft prompt with approximately rank-one cross-image differences, not zero-valued features. Receiver ratios are **0.1372/0.0846**, ruling it out as the primary collapse source. CKA/pairwise-distance correlation are 0.436/0.425 at projector and 0.428/0.416 after receiver.

**Package 15N verification/runtime boundary**: the corrected independent verifier exactly recomputes five pooled tensors, 6,125 pair rows, 50 per-sample rows, both actions and all hashes. Its first failure compared safetensors tensor blocks in enumeration order; the log, frozen verifier hash and post-result sorting repair are retained. The first full suite then passed 348 tests and failed only because a Windows-generated nested manifest counted CRLF while Git committed LF; the generic writer now forces LF, the failure is retained, and the canonical V100 suite is **347/347** green. The package manifest binds **17 files / 8,120,202 bytes**. The host's loaded 580.159.04 kernel module had mismatched 580.173.02 system user libraries. Matching 580.159.04 libraries were extracted on HDD, verified against official RPM Fusion primary-metadata SHA, and scoped only to the run with `LD_LIBRARY_PATH`; no reboot, system-file change or GPU-client stop occurred. The formal screen took 7.494 s and 402,776,064 peak GPU bytes. It supports projector scale/geometry repair, defers margin, establishes no visual ability, uses no paid resource and does not inspect final halves.

**Package 15O pre-result boundary**: the trajectory contract binds the frozen ScreenSpot50 order, exact step0, grounding-enriched checkpoints 100/200/300/400/500, every projector/checkpoint SHA and the 500-row training-history SHA. Package 15N's step500 collapse is explicitly disclosed as known at freeze; steps 100–400 and the earliest saved onset are the registered unknowns. A pre-result unit test found an action-key interface mismatch, and its full log is retained. The repair adds only local placeholder actions before selecting the preregistered trajectory action; schedule, thresholds, onset rule and history binding are unchanged, and no GPU result existed before repair.

**Package 15O decisive result**: the projector and fixed receiver both satisfy both collapse guards at step100/800 examples. Projector relative-spread/effective-rank ratios are **0.12985/0.07721**, sample RMS is **0.124→35.74**, and top-1 variance is **17.48%→98.76%**. Receiver ratios are **0.12873/0.07596**. Every later saved checkpoint stays collapsed; step500 exactly reproduces Package 15N. The first 100-step loss mean is still 3.916 and the last loss is 2.276, so gross geometry collapse is an early optimization effect and unchanged longer training does not repair it. Exact onset is only bounded to steps 1–100 because step100 is the first saved trained state.

**Package 15O action/verification boundary**: the registered action is `apply_geometry_protection_from_initial_step_and_run_matched_lambda_screen`. The next training screen starts from exact step0 and applies a small, directly transferable 4096-boundary scale/geometry regularizer from optimizer step one against a fully matched CE-only control. Counterfactual margin remains deferred. The independent verifier recomputed 13 pooled tensors, 15,925 pair rows, 50 per-sample rows, 500 history rows, all geometry and both onsets. The V100 repository suite is **361/361** green; the package manifest binds **17 files / 20,683,664 bytes**. Formal analysis took 56.603 s and 939,810,816 peak V100 bytes; no visual-capability claim, paid resource, final-half score or previous-best promotion is allowed.

**Package 15P calibration result**: before any arm was trained, the frozen step100/batch100 calibration recomputed the three geometry terms against the exact step0 projector. The unweighted auxiliary gradient norm is **3.8781849597** versus recorded CE norm **0.7901650667**. The preregistered target ratios produce fixed λ values **0.0101873051 / 0.0407492203 / 0.1629968813** for `ratio005/ratio020/ratio080`; control is exactly zero. The raw pooled tensor, calibration config, logs and independent verification are checked in under `geometry_repair_screen_v1/calibration/` and the verifier status is **verified**. A first shell logging attempt failed before directory creation and is retained under `calibration/failures/`; it did not alter or replace the valid GPU result. This is calibration evidence only: no visual-capability claim, candidate promotion, final-half score or paid resource.

**Package 15P pre-result repair**: the first `control` launch was rejected before optimizer step 1 because the calibration SUMMARY omitted `screen_contract_file_sha256`. The full supervision/failure output is retained under `geometry_repair_screen_v1/failures/attempt01_calibration_binding/`; no checkpoint, capability metric or training result was created. The repair exposes all calibration input hashes and record IDs in SUMMARY and makes the independent verifier compare them with RUN_CONFIG and the supplied files. The schedule, thresholds, objective and λ derivation are unchanged; corrected calibration must be regenerated before the four-arm screen.

A second pre-result focused-test attempt found a test-only `tools/` import-path omission; its stderr and manifest are under `failures/attempt02_test_import/`. The test path is repaired before GPU rerun, with no change to the runtime source or experiment contract. Corrected calibration has now been regenerated under `geometry_repair_screen_v1/calibration_v2/` and independently verified; its λ and geometry values match the first mathematical calibration while every trainer binding is present.

**Packages 15G/15H verification/storage**: the full V100 repository suite reaches 100% with zero failure after the alias fix. Ten focused local artifact/metric tests pass. Package 15G contains 30 files / 11,012,700 bytes, including every prediction and per-row score; Package 15H contains 45 files / 813,872 bytes, including all preference rows, the failed development attempt and corrected smoke. Both artifact manifests independently rehash every checked-in byte. `report/main.pdf` builds to 58 pages; Package-15G/15H and revised Gate-D pages 44–46 passed rendered inspection. Large image/cache tensors remain on the local V100 under checked-in SHA bindings.

**Package 3 decisive result**: shape at step 1500 is the first and only causally validated content signal. Full-matrix teacher forcing gives strict paired preference **0.130**, trained−random **+0.075 [0.025, 0.135]**, vision−shuffled **+0.115 [0.070, 0.160]**, and vision−paired-counterfactual margin **+0.266 [0.219, 0.313]**. The signal collapses at step 2000: step1500−step2000 **+0.130 [0.085, 0.180]**. The other five tasks have no validated onset by 16,000 examples.

**Teacher-forced/free-generation split**: the valid generation run has 37,300 rows plus 160 heldout shuffle rows, 0 failures, and passed independent exact-denominator/hash/source verification. Synthetic vision sample accuracy is 0.1367/0.1550/0.1350/0.1417 at steps 500/1000/1500/2000, yet strict paired generation and answer-flip accuracy are **0 at every checkpoint and task**. Step 2000 vision−blind is +0.1417 [0.1167, 0.1683], while vision−same-image and vision−shuffled-image are both −0.0033 with intervals crossing zero. The model learns a global visual-conditioning interface; current decoding does not select the answer tied to the correct image content. Historical heldout shuffle delta rises from random −0.0095 to **+1.1886** at step 2000.

**Package 3 verification**: 150/150 full repository tests pass on the V100 workstation. The report builds to 25 pages and the new package-3 pages/figures were render-checked. The first failure-audit invocation exposed a zero-row acceptance bug from non-canonical checkpoint IDs; invalid empty outputs are retained, the tool now fails closed, and canonical reruns contain 374/318 records for steps 1500/2000.

**Package 4 decisive result**: MoonViT/projector information loss is ruled out for shape. Tower and projector reach balanced accuracy **1.000** at random, step 1500, and step 2000 under at least one frozen pooling. At the assistant position, step1500 peaks at layer 12 with raw/balanced **0.790/0.816** and step2000 peaks at layer 14 with **0.605/0.632**; both have pair-permutation `p=1/2001`, while trained final states and the native LM-head fall to balanced **0.250, p=1**. The paired-counterfactual and shuffled controls follow the actual visual source rather than the target label.

**Package 4 causal result**: on 50 frozen pairs, step1500 correct-image span patching peaks at layer 11 with margin effect **+0.3538 [0.2506, 0.4569]**; correct donor minus different-pair wrong-label donor remains **+0.2194 [0.1463, 0.2994]**. Step2000's raw peak is layer 6 **+0.1531 [0.1125, 0.1931]** and preregistered layer-5 content-specific effect is **+0.0856 [0.0519, 0.1181]**. The final assistant positive control exactly reproduces the clean margin. This supports a real mid-layer causal path that weakens and moves earlier by step 2000, followed by upper-layer erasure/non-use.

**Package 4 verification/storage**: independent verification passed 30 representation files / 6,000 representation rows, 672 metric rows, 268,800 predictions, and 18,300 patching rows / 183 cells. The full repository suite is **167/167** green. The report builds to **27 pages** and pages 13–22, including all package-4 figures and the transition back to Gate C, were render-checked. The 108 MB prediction JSONL is committed losslessly as a 3.2 MB deterministic gzip. Approximately 899 MB of activation tensors remain under `$HDD/data/perception_v1/mechanisms/shape_layerwise_v1`; `representations/SUMMARY.json` binds every file. The first extractor import failure, the high-variance v1 probe significance run, and the pre/post-final-RMSNorm patching v1 defect are all retained with explicit invalidation and replacement runs.

**Package 5 decisive result**: LoRA and projector continuation start from step 1500 and share the exact 400-record shape train order (`training_order` SHA `993f1b2e…6988`). Top-12 rank-8 LoRA (442,368 trainable parameters) peaks at 800 seen examples: strict paired preference **0.605** versus frozen **0.130**, mean gap **+0.475 [0.405, 0.545]**; free-generation paired rises from 0 to **0.080 [0.020, 0.160]**. This proves the frozen upper stack has a correctable use/decoding bottleneck.

**Package 5 matched-control result**: projector continuation (20,454,272 trainable parameters) reaches strict paired preference **1.000** and paired generation **1.000** after only 400 seen examples. Relative to frozen, strict paired improves **+0.870 [0.825, 0.915]**; vision−shuffled strict paired is **+0.820 [0.765, 0.870]** with shuffled at 0.180. At equal 400 examples, LoRA−projector is **−0.820 [−0.870, −0.765]** strict paired and **−1.000 [−1.000, −1.000]** paired generation. The leading local explanation is insufficient projector-interface training for shape, with an additional smaller upper-stack bottleneck.

**Package 5 layerwise/verification result**: LoRA step 100 retains the original layer-12 peak (balanced **0.816**) but final assistant/native head only reach **0.500/0.500**. Projector step 50 reaches balanced **1.000** at layer 17, **0.945** at the final assistant probe, and **1.000** at the native LM-head. The independent verifier re-read 8 training checkpoints, 8,400 preference rows, 1,400 generation rows, 63 bootstrap contrasts, 4,000 representation metadata rows, and 179,200 probe predictions. The full repository suite is **174/174** green. The report builds to **29 pages** and pages 17–19 were render-checked. Full projector raw data is at `$HDD/data/perception_v1/adaptation/`; `shape_adaptation_PACKAGE_VERIFICATION.json` is valid.

**Package 6 decisive result**: zero-additional-training transfer is shape-specific. On 200 preference pairs per task, shape strict paired preference rises from **0.130 to 1.000**, checkpoint gain **+0.870 [0.820, 0.915]**, and vision−shuffled is **+0.820 [0.765, 0.875]**. On the disjoint 50-pair generation selection, shape paired accuracy rises from **0 to 1.000**, and vision−shuffled is **+0.980 [0.940, 1.000]**. None of color/coordinate/count/OCR/spatial passes the joint checkpoint-gain and visual-causality rule; none shows significant negative transfer. This refutes broad interface correction from 40 shape training pairs and supports a narrow task mapping.

**Package 6 verification/storage**: teacher forcing contains 28,800 rows and generation 9,000 rows, both with zero failures; the joint table has 567 metrics and 525 contrasts with 2,000 complete-pair bootstrap resamples. `PACKAGE_VERIFICATION.json` independently re-read the raw counts and SHA bindings; the full suite is **178/178** green. The report builds to **30 pages** and the new pages 19–20 were render-checked. The legacy analyzer's absent-`blind` assumption is retained as an invalid run; the replacement analysis is canonical. The final odd halves remain untouched and no paid resources were used.

**Package 7 decisive result**: one balanced synthetic epoch establishes visually causal teacher-forced signal on all six tasks. Step-100 strict paired preference vision/shuffle is color **0.230/0.095**, coordinate **0.055/0**, count **0.115/0.060**, OCR **0.135/0.050**, shape **0.560/0.155**, spatial **0.250/0**. Checkpoint gains have positive 95% lower bounds for every task; vision−shuffle lower bounds are also positive. Validated tasks accumulate from coordinate at step 25, to color/shape at step 50, to all six at step 100. Shape transiently falls by **−0.130 [−0.180, −0.085]** at step 25 before recovering, so short-horizon multi-task interference is real.

**Package 7 generation split**: only shape and spatial cross the free-generation paired threshold at step 100: **+0.160 [0.080, 0.280]** and **+0.220 [0.120, 0.340]** over step 1500; their vision−shuffle effects are **+0.140 [0.060, 0.240]** and **+0.220 [0.100, 0.340]**. Color/coordinate/count/OCR remain at zero paired generation despite validated teacher-forced visual selection. This supports training coverage as the main teacher-forced bottleneck and isolates a remaining frozen-language use/decoding bottleneck.

**Package 7 systems/verification result**: caching 2,400 records took 251.5 s / 1.95 GB peak; 100 true-batch steps took 144.5 s / 11.80 GB peak, loss 2.778→1.349. Independent checks rehashed 75 cache shards / 3,932,170,800 bytes, read 983,040,000 values, verified 100 balanced steps and four checkpoints, then re-read 38,400 preference rows, 12,000 generation rows, 756 metrics, and 714 pair-bootstrap contrasts. The full suite is **184/184** green; the report builds to **33 pages** and pages 21–23 were render-checked. Three implementation failures are preserved and invalidated: masked padding counted as images, missing checkpoint provenance, and an unnecessary random-checkpoint requirement.

**Package 8 decisive result**: both arms start from balanced step 100 and share the exact 2,400-record order (`a0929326…2f5`). The projector arm restores AdamW state; top-12 rank-8 LoRA starts at exact zero delta. Canonical bf16 overall strict preference is base/LoRA/projector **0.224/0.247/0.511**; projector gain **+0.287 [0.258, 0.318]**, LoRA **+0.023 [−0.003, 0.049]**. Overall paired generation is **0.063/0.080/0.257**; projector gain **+0.193 [0.147, 0.240]**, LoRA **+0.017 [−0.020, 0.050]**.

**Package 8 task result**: the extra projector epoch unlocks paired generation for color **0.160 [+0.060, +0.260]**, coordinate **0.240 [+0.120, +0.360]**, and spatial **1.000, gain +0.780 [0.660, 0.880]**. OCR strict preference improves +0.085 [0.040, 0.130] but OCR/count generation remains zero-level. Shape strict preference regresses −0.125 [−0.180, −0.065]. LoRA improves shape strict/generation by **+0.320 [0.250, 0.395] / +0.320 [0.200, 0.460]**, leaves the four original generation gaps closed, reduces count preference, and erases spatial generation −0.220 [−0.340, −0.120]. Broad upper-stack repair is refuted; added projector training is supported with real task interference.

**Package 8 systems/precision result**: projector/LoRA training took 159.1/153.4 s at 11.80/9.93 GB peak; 100 steps and 24/288 checkpoint tensors passed independent reload checks. Canonical eval has 21,600 preference and 3,600 generation rows; analysis has 315 metrics and 189 contrasts with 2,000 pair bootstraps. The first full endpoint evaluation used fp32 projector and moved spatial base strict/generation from the package-7 **0.250/0.220** to **0/0**. It is retained as a valid sensitivity diagnostic. bf16 v2 exactly reproduces every package-7 base value and is authoritative. The full suite is **195/195** green; the report builds to **35 pages** and package-8 pages 23–25 were render-checked.

**Package 9 decisive result**: the canonical-bf16 screen covers seven states with 12,600 preference / 8,400 generation rows. Full step-50 confirmation has 21,600 / 3,600 rows. Projector step 50 reaches overall strict preference **0.5117**, gain **+0.2875 [0.2583, 0.3167]**, and paired generation **0.2267**, gain **+0.1633 [0.1200, 0.2100]**. OCR checkpoint−base strict is only +0.020 [−0.025, 0.065], but OCR vision−shuffle is **+0.075 [0.020, 0.130]** with generation 0; retain this as weak image-dependent ranking evidence, not a solved OCR capability. Count strict gains **+0.265 [0.200, 0.330]** while generation remains 0.020.

**Package 9 Pareto result**: full exact-ID projector step 100−50 is **−0.0008 [−0.0283, 0.0275]** overall strict and **+0.0300 [−0.0133, 0.0767]** generation. The unchanged aggregate hides large task movement: count/shape strict fall **−0.280/−0.300**, coordinate/spatial rise **+0.160/+0.250**; shape generation falls **−0.280**, coordinate/spatial rise **+0.220/+0.220**. Equal six-task sampling therefore does not eliminate interference. LoRA step 50 remains shape-only and significantly harms count/spatial. A monotonic global stop and a broad top-LoRA remedy are refuted.

**Package 9 verification/defect result**: trajectory/full verifiers re-read 7/3 states, 34,200 preference rows, 12,000 generation rows, 882 within-run contrasts, both training manifests, and 24/288 projector/LoRA checkpoint tensors. The cross-run analyzer adds 210 exact-sample-identity contrasts. Analysis v2's numeric metrics were valid, but flat zero trajectories were mislabeled as nonmonotonic peaks; it is retained and invalidated, while test-first v3 is authoritative. The full step-50 run took 1,137.6 s at 12.72 GB peak. The full suite is **199/199** green; the report builds to **38 pages** and package-9 pages 25–27 were render-checked. Final odd halves remain untouched.

**Package 10 decisive result**: linear `(1−alpha)P50 + alpha P100` at alpha 0/.25/.50/.75/1 does not merge the endpoints. The preregistered rule requires count/shape retention within 0.05, coordinate/spatial gains, a better endpoint worst-task score, and a macro floor. No point passes. Alpha .25 is the best balance diagnostic (macro strict **0.5333**, worst **0.160**, macro generation **0.2700**) but count falls **−0.16 [−0.28, −0.04]** and shape −0.10 [−0.20, 0], while spatial rises **+0.26 [0.14, 0.38]**. Alpha .50/.75 reduce count to 0.14. Linear weight averaging is refuted as a capability-union fix.

**Package 10 verification/result boundary**: alpha 0/1 reproduce source tensor SHAs exactly and independently reproduce package-9 endpoint evaluation across 1,800 preference plus 1,200 generation rows each, including raw logp/NLL/margins and strings. The screen has 10,800 preference / 7,200 generation rows, 630 metrics, 651 contrasts, 541.3 s wall time, and 12.72 GB peak. Alpha .25 improves macro generation over alpha 0 by +0.0433 [0.0100, 0.0767], but its small macro gains over alpha 1 are inconclusive and it fails the retention/worst-task rule; no full confirmation was run, exactly as preregistered. The full suite is **203/203** green; the report builds to **39 pages** and package-10 pages 27–29 were render-checked. Final odd halves remain untouched.

**Package 11 decisive result**: the unregularized continuation restores AdamW state `57e9ddb…ac2f32`, exact order `a0929326…f2f5`, and reproduces all six original step-100 tensors plus the serialized projector file exactly (`7b731cff…a76`, file `05f19079…092d`). Count/shape-only frozen-step50 projector-output MSE anchoring at `1e-4/1e-3/1e-2` does not pass the preregistered rule. Even the strongest arm ends at count/shape **0.16/0.54** versus step-50 **0.42/0.80**. Full-output representation distance therefore does not preserve the old answer decision boundaries.

**Package 11 Pareto/verification result**: `1e-3` is a useful diagnostic: relative to exact control, overall strict preference is **+0.0533 [0.0200, 0.0867]** and paired generation **+0.1267 [0.0900, 0.1633]**; relative to step 50, count/shape are **−0.32 [−0.46, −0.18] / −0.26 [−0.38, −0.14]**. OCR generation remains 0 and count 0.02. The screen has 9,000 preference / 6,000 generation rows, zero failures, 525 metrics, and 399 contrasts. Endpoint teacher-forced values reproduce package 10 exactly; repeated GPU free generation has 42/62 text differences at frozen/control, with 18/0 correct-flag differences, so candidate selection uses same-run strict preference. A wrong-source v1 and an over-strict generation verifier are preserved through seven verified invalidations. The full suite is **208/208** green; the report builds to **41 pages**, and package-11 pages 29–31 were render-checked.

**Synthetic package 2 valid result**: authoritative images are at `$HDD/data/perception_v1/synthetic/synthetic-v1-seed20260804`; Git metadata/raw records are at `experiments/v100_perception_20260804/synthetic_data_v1/`. Six tasks × 200 base pairs × 2 splits gives 2,400 base questions / 4,800 rendered images. Every base has a byte-identical question and answer-changing a/b image; train/selection image hashes, OCR strings, pair IDs, and template IDs have intersection 0. Independent verification read and SHA-checked all 4,800 PNGs, all 2,400 pairs, and all 4,800 blind/blank/same-image/shuffled/patch-permutation assignments. Logical dataset SHA is `122ae820…cbaa71`; 0 failures. OCR glyphs are the stimulus, but no extra answer label/hint is rendered. The full deterministic PNG set stays on the data disk; Git contains complete record/control JSONL, hashes, code/config, and representative pairs.

**Invalid results preserved, never use them**: `failed_cache_attempt_01` stopped after three computed rows because foreground SSH closed stdout and progress printing raised `BrokenPipeError`; no cache manifest, invalid. `step_time_failed_network_attempt_01` completed 0 steps due remote HF HEAD timeout; invalid. `step_time_failed_offline_cache_attempt_02` completed 0 steps because offline mode used the wrong default cache root; all three arm logs plus the zero-row CSV driver failure are preserved. Immutable local Qwen snapshot fixed retry2.

**Exact next task after Package 15P calibration**: run the four-arm 100-step geometry screen (`control`, `ratio005`, `ratio020`, `ratio080`) from exact step0 with the same order, cache, receiver, optimizer and 800 examples. Select the smallest nonzero arm only if both projector/receiver guards clear and final-20-step CE is at most 1.25× control; otherwise redesign the objective without spending a full 4k run. A short-screen pass remains representation evidence only. In parallel, close fixed-receiver TextVQA/DocVQA/OCRBench and language retention before any candidate can replace previous-best.

**New pre-rental queue is binding**: the Qwen2.5-3B fixed real-data contract now has highest priority. Projector, data, replay, resolution, initialization and training changes must report ScreenSpot50/full plus TextVQA, DocVQA, OCRBench, synthetic and language-retention controls at matched budgets. Only `directly_transferable` or `transferable_with_runtime_validation` methods enter the DeepSeek candidate list. Fixed-revision DeepSeek source audit, GPU matrix and rental contract remain in `docs/dsv4-runtime-source-audit.md`, `docs/gpu-runtime-matrix.md` and `docs/deepseek-rental-training-contract.md`; actual FP8/FP4 DGRAD remains `hardware_pending`.

**Go/no-go**: NO-GO for renting, NO-GO for claiming Gate D, and NO-GO for promoting either 4k checkpoint. The local glue path has formal Qwen2.5-3B train/save/resume plus paired, free-generation and representation evidence. Both CE-only data mixes lack correct-image dependence; Package 15O shows projector collapse is already present by step100, and Package 15P calibration now fixes the preregistered geometry-arm scales. The existing V100 can still complete the matched short screen, any selected 500-step candidate, TextVQA/DocVQA/OCRBench and language retention. The full 0731 weights have never completed image forward/backward/train/save/resume/generate.

## ⚠️ TAKEOVER NOTE (2026-08-04 ~18:52, Codex recovery — read this first)

**Corrected stock positive-control suite is COMPLETE**: repaired Qwen3.5-4B eval finished with `STOCK_EVAL_REPAIRED_ALL_DONE` in `/tmp/stock_eval_repaired.log`; all five repaired JSONs are under `$HDD/data/stock_eval_qwen35_4b_repaired/`. Selection-half vision/blind results at 1024px: TextVQA soft-VQA **0.820 / 0.031**, DocVQA ANLS **0.926 / 0.071**, OCRBench exact **0.900 / 0**, ScreenSpot parse **0.86 / 0.99** and accuracy **0.760 / 0.010**, MMMU-Pro exact **0.300 / 0.280**. MMMU's +0.020 vision gap shows most of its raw 30% is text/options knowledge, not image contribution.

**Interpretation red line (user correction, 2026-08-04)**: stock Qwen3.5-4B is a native VLM (`Qwen3_5ForConditionalGeneration`, `vision_config` present). It uses its own vision tower/alignment and therefore validates only the eval data/processor/scorer; it is not evidence that MoonViT-V2 can be mapped into text-only DeepSeek. Gate B did use a genuinely text-only receiver (`Qwen2ForCausalLM`, no `vision_config`), so its learned-vs-random-vs-blind and shuffle-delta results remain valid interface-learnability evidence, but only full 0731 Gate D/training can establish target feasibility/capability. `train_overfit.py` now rejects native multimodal `--text-model` configs by default; native VLMs belong only in `eval_stock_vlm.py`.

The 16:27 run in `/tmp/stock_eval.log` and `$HDD/data/stock_eval/` is **invalid and must never be reported/uploaded**. Root cause: local `model.safetensors-00002-of-00002.safetensors` had the correct byte size but wrong SHA-256 (`547d2f…8627`, official `cb544b…e188`); the model index places the entire Qwen3.5 visual tower in that shard, so it coherently answered from question text while describing every real image as blank. The old runner also had no image-size cap (large samples OOM; the workstation's NVML library mismatch masked the true OOM) and asked no short-answer/coordinate-format instruction (a repaired one-image smoke correctly saw “Dakota Digital” but verbose output scored 0). Uploads from that run failed under `HF_HUB_OFFLINE=1`, so bad files did not reach HF.

Commit `dbfacd5` fixes all three: optional local weight-manifest verification (`configs/qwen3.5-4b-hf-sha256.json`), `--max-image-side`, and metric-aware output contracts (short answer / option letter / normalized `(x, y)`). The damaged 3.99 GB shard was re-downloaded via single-connection ModelScope at ~13 MB/s and atomically installed only after matching `cb544b…e188`; repaired smoke: prediction `dakota digital`, soft-VQA **1.0**. The live suite verified both 8.8 GB shards once, then runs selection half × vision/blind at 1024px (same resolution as the Gate-B table), bf16, 32-token cap, without upload.

**Closeout completed**: the repaired directory was uploaded and re-downloaded from HF for verification at `eval/stock-qwen35-4b/` (8 files; final HF commit `ba6e98ef0f108ece266447bd1b76fa30336ff7b0`; SUMMARY is `native_vlm`, five benchmarks, no spurious skipped entries). `report/main.pdf` was rebuilt and render-checked (18 pages); 96/96 full tests pass; actual Qwen2/DeepSeek/Qwen3.5 config guard smoke passes. Code guard commit is `41a9bae` (on top of stock-eval repair `dbfacd5`).

**Next task boundary**: finish full public ScreenSpot, then use paired teacher forcing and projector-information diagnostics to select the next matched-budget change; do not extend to 8k solely because training loss fell. Continue local Qwen2.5-3B real-data work without pausing for rental. **Do NOT create a Vast instance or any paid resource without explicit authority.** Gate D runbook remains `docs/gate-d-runbook.md`; re-query offers only when all useful V100 work is exhausted and a fresh paid proposal is needed. Native 9B/27B VLM controls do not close the MoonViT-to-text-backbone gap and are not scheduled by default.

**0.5B capacity caveat and 3B switch (updated 2026-08-06)**: Gate B's Qwen2.5-0.5B is genuinely text-only, so its learned visual dependency remains valid, while low capacity confounds every absolute benchmark and dense optimization does not predict DeepSeek Hash-MoE convergence. Gate B is an engineering/signal gate only. Pure-text `Qwen/Qwen2.5-3B-Instruct`, with no `vision_config`, is now the mandatory bridge under the fixed ScreenSpot/TextVQA/DocVQA/OCRBench/synthetic/language contract. Its first 4k benchmark exists and is negative: vision is causally weaker than blind/step0 on GLM50. This shows that capacity alone does not repair the current objective; no 0.5B/0.6B result may substitute for the 3B contract. Native Qwen2.5-VL/Qwen3.5 remains evaluator evidence only.

**Training-accounting correction + ablation protocol (user correction, 2026-08-04)**: full-mix Gate B's historical `2000 steps × batch 8` was 8 serial single-example forward/backward passes per optimizer update, not one batched forward. It therefore saw 16,000 examples, only ~0.27 epoch of the 59,198-row mix; exact historical answer-token count is unavailable. Reclassify it as early alignment/interface learnability, never a fully trained capability ceiling. `tools/train_overfit.py` now uses explicit micro-batch/gradient-accumulation/effective-batch terminology, records optimizer steps/examples/answer tokens/effective epochs, writes raw/canonical answer provenance, creates a reusable source-stratified validation manifest, and reports overall/per-source loss over 10 seeded derangements with pair IDs and mean/std. True `micro_batch_size > 1` is deliberately rejected until padded multi-example forward is implemented and measured. The prioritized capacity/projector/LoRA/resolution/causal-control/synthetic plan is `docs/ablation-protocol.md`; Gate D time and resolution estimates are no longer locked before those preflight results.

Everything below this note is current through the pre-repair project history; newer facts above override stale stock-control/download statements below. The HF model repo remains reorganized: root only `README.md` + `.gitattributes`; controls under `gate_b_qwen05_v100/` and `gate_b_smoke_smollm135_v100/`; no DeepSeek weights exist yet.

---


## Current state

- **HF namespace renamed (2026-08-03)**: the user's HF account is now `cyjin-yl` (was `255doesnotexist`). Both repos verified live under the new name: `cyjin-yl/DeepSeek-V4-Flash-0731-Vision` (model: vision tower, projector checkpoints, eval results) and `cyjin-yl/moonvit-dsv4-data` (dataset: train/eval data).

- Public repo: https://github.com/cyjin-yl/moonvit-deepseek-v4-glue (`main`; do not hard-code a stale HEAD here—verify with `git ls-remote`).
- Core glue implemented under `src/moonvit_glue/`; **96/96 tests pass on Linux** (torch 2.10.0+cu128, transformers 5.12.1). Actual-config guard smoke also passes: Qwen2 text-only + DeepSeek-V4 accepted; Qwen3.5 native VLM rejected.
- **MoonViT-V2 (Kimi K3 tower) ported (2026-08-03)**: vision-only code vendored at `src/moonvit_glue/vendor/kimi_k3/` (text-model dep `modeling_kimi_linear` removed — it needs `OutputRecorder` from a newer transformers; `KimiK3ForConditionalGeneration` dropped). `moonvit_glue.moonvit_v2` wraps it in the existing `MoonViTEncoder` contract. Contract deltas vs SO-400M: **vision width 1024** (not 1152), input `[total_patches,3,14,14]` + `grid_thws` (not flattened), `sd2_tpool` merge (identity at t=1), output `[G,4,1024]` — PatchMerger glue unchanged. K3 registers only flash-attn-2/eager; we add `sdpa` varlen attention (numerically equal to eager, tested). Random-weight forward verified on the workstation; 6 new tests in `tests/test_moonvit_v2.py`.
- **K3 weight extraction — DONE (2026-08-03)**: shard `model-00096-of-000096.safetensors` (802,448,352 B, sha256 `9d10c74fc10161be…`) downloaded to `$HDD/staging/k3/`; `tools/extract_moonvit_v2.py` extracted 165 tensors / 401.2M params with strict-load pass → artifact at `$HDD/staging/k3/extracted/` (`moonvit_v2.safetensors` bf16, sha256 `01436a9593996518…`, + `MANIFEST.json` with both hashes + vision/preprocessor configs + code snapshot). **Real-weight V100 smoke passed**: 1024×1024 image → `[1369,4,1024]`, all finite, mean 0.0003 / std 0.0511 / absmax 0.60, bitwise deterministic, eager-vs-sdpa max diff 3.1e-05; 34/34 tests green. Upload to `cyjin-yl/DeepSeek-V4-Flash-0731-Vision` under `vision_tower_k3/` (HF token in local `.env`, write access verified via whoami-v2 + upload/delete probe).
- Local machine cannot reach GitHub (TUN off, socks5:10808 down, direct reset): **push via the workstation** — `git bundle` the commits, scp, `git pull /tmp/main.bundle main` in the clone, `HTTPS_PROXY=127.0.0.1:7890 git push`.
- Real `DeepseekV4ForCausalLM` tiny Hash-MoE config passed loss/backward and generation.
- **Gate B complete**: real MoonViT-SO-400M forward/backward on the V100 (`[192,4,1152]` at 448px, `[1064,4,1152]` native 640×480), eval harness dry-run end-to-end (generation + blind + shuffle-loss; untrained projector correctly gives `mean_delta = 0.0`).
- **Gate B training signal confirmed (2026-08-02)**: overfit on 109 ComfyUI captions (93 train / 16 eval), frozen MoonViT + frozen SmolLM2-135M-Instruct, projector-only training on the V100. 200 steps (lr 1e-3) was inconclusive (delta +0.007); **1000 steps (lr 2e-3) gives train loss 4.303→3.338, eval true 3.300 vs shuffled 3.642, shuffle_delta = +0.343**. Generation check (8 records): with-image outputs vary per image with content words (token-F1 0.112); blind outputs are byte-identical generic text (0.082). Placeholder = existing `<|endoftext|>` id 0 (SmolLM2 has no reserved image token). Checkpoint on the workstation at `checkpoints/overfit-smollm135-1k` (gitignored). Data path convention: `image` fields are relative to the JSONL file (the comfy JSONL was fixed accordingly).
- Projector shape fixed at MoonViT-V2 `[N,4,1024]` → DeepSeek 4096, **33,564,672 params** (config `configs/deepseek-v4-flash-0731-projector-moonvit-v2.json`; fp32 ~134 MB, bf16 ~67 MB). V1 alternative (MoonViT-SO-400M `[N,4,1152]`) is 40,119,040 params — never mix configs across towers.
- DeepSeek image placeholder fixed to existing `<｜image｜>` ID 129279; never resize vocab.
- `VisionCausalLM.generate()` exists for both backbone kinds; generic path returns the full expanded sequence.
- Benchmark scaffold landed: `moonvit_glue.metrics` (pure Python), `tools/eval_vlm.py` (generation + `--blind` + `--shuffle-loss`, `--max-image-side`), `tools/fetch_eval_data.py` (TextVQA/DocVQA/OCRBench/ScreenSpot/**MMMU-Pro single-image subset** → JSONL + MANIFEST.json). Eval philosophy: always report the blind (no-image) baseline next to capability numbers. The community GLM-5.2V recipe ([0xSero/fable-glm-vision](https://github.com/0xSero/fable-glm-vision), reproducing Harry Partridge) uses the **same MoonViT tower** we do and its headline metric is **MMMU-Pro 55%**; its two training findings shape our data plan: *grokking* needs **short answers** (batch 64, lr 5e-4, sharp loss drop ~step 900; long captions prevent it) and *warm-starting* from an aligned projector skips most of the plateau.
- **Baseten recipe is a prior, not a locked schedule (corrected 2026-08-04)**: the source writeup used constant lr 5e-4, global batch 64, ~66k short-QA pairs and 2 epochs ≈ 2070 optimizer steps. Our historical trainer did draw distinct records, but executed them as serial microbatch-1 gradient accumulation; it did not reproduce a true batch-64 forward or its timing. Dataset production is complete. Preserve constant-lr/short-answer as hypotheses, but derive steps from examples/tokens and re-estimate time only after true batching and the pre-rental ablations.
- **Train data pipeline landed (2026-08-03, GUI revision)**: train-split specs (`textvqa_train` full 34.6k, `docvqa_train` cap 25k, `showui_desktop` 8k GUI grounding, `flickr8k_train`) in `tools/fetch_eval_data.py` with mechanical `max_answer_words ≤ 20` (short-answer red line), JPEG for photo datasets, and `save_max_side` downscale (ShowUI screenshots are 3k+ px). GUI answers use the 0xSero/ShowUI action format `click(start_box=[x,y])` (0..999 scale) — `metrics.extract_point` parses it natively (tested). GUI inclusion was the user's call for computer-use capability; consequence: **ScreenSpot becomes an in-domain benchmark and must be labeled as such in all reports**. 0xSero's own screenshots/multistep rows are NOT consumed (renamed image files can't be re-joined to sources; multistep is trajectory format) — we fetch from the same public source (showlab/ShowUI-desktop) instead. `tools/build_train_mix.py` assembles the final mix (per-source caps, 0xSero conversations-schema normalization, **average-hash decontamination vs all eval images**, hamming ≤ 6 → drop, report published). flickr8k stays out of the real mix (captions too long), Gate B smoke only. Dataset repo: **cyjin-yl/moonvit-dsv4-data**. Downloads running in tmux `moondata` on the workstation (`/tmp/fetch_data.log`); assembly + upload + README pending. 46/46 tests green.
- **Eval publication pipeline (2026-08-03)**: `tools/eval_vlm.py` reports are now self-contained (per-record question + reference answers/gt_box + raw prediction + per-record scores) with a metadata block (model/weights/projector/git/torch/host/timestamp); `--upload-repo <model-repo> --run-tag <tag>` uploads the report to `eval/<tag>/` on HF immediately after the run. `tools/aggregate_eval.py` builds `SUMMARY.json` (benchmark × vision/blind/gap matrix, ScreenSpot flagged `in_domain`) and uploads the whole results dir. Requirement source: user — all benchmark raw outputs AND final aggregates must be publicly on HF before the rental ends. 53/53 tests green.
- **V100 dress rehearsal COMPLETE (2026-08-03) — the rental gate**: full pipeline (offline fetch → split → 400-step train with streaming HF checkpoints → trained/blind/random evals → aggregate → upload) ran in tmux `smokepipe` on the V100 with SmolLM2-135M + real K3 MoonViT-V2 (eager), 1000 flickr8k train / 200 eval (jxie/flickr8k is one row per image; caption_0 only → row split = image split, no leakage). Results: **trained vision token-F1 0.1565 vs blind 0.0391 (gap +0.117); random-projector control 0.0584 (gap +0.019)** — pipeline is discriminative, training signal real. Loss 4.30 → 3.06 over first 200 steps. Verified on HF: `checkpoints/step-000200|000400/` (projector fp32+bf16+training_state+history), `eval/v100-smoke-smollm135/` (trained+random+SUMMARY), `train.log`, `overfit_report.json`. 400 steps is below the grokking regime — these numbers are the report's control group, not a capability claim.
- **Download channel resolved (2026-08-03)**: hf_transfer stalls at 0 B through the proxy, hf-mirror rate-limits, and the workstation's datasets+dill stack cannot pickle pyarrow's MonthDayNano (breaks `load_dataset`/`cast_column` on any local file). Working path: **aria2 `-x8 -c --max-tries=0` prefetch of parquet shards** (the xet-bridge CDN randomly TLS-resets and 403s range-signed URLs — unlimited retries grind through; API listing retried 5×) + **offline fetch via raw pyarrow** (`fetch_eval_data.py --data-files`, zero hub traffic, image structs decoded eagerly). New tools: `tools/prefetch_parquet.py`, `tools/fetch_art_data.py` (offline 0xSero art port, schema-compatible with `build_train_mix.normalize_0xsero_row`, tested). flickr8k's two train parquets were already fully cached (sha256-verified vs API lfs oids). **MMMU-Pro config fixed**: the repo has no bare `standard` config — spec now pins `standard (10 options)` (canonical). Prefetch running in tmux `moondata`/`moonart`. 62/62 tests green.
- **Workstation-direct owns all downloads (2026-08-03 late, user call)**: the Windows relay was retired at the user's request ("小水管太慢" — the 8×scp streams saturated the home link for zero throughput gain, since scp ≈ proxy ≈ 250 KB/s). Current ownership: tmux `moondata` runs `/tmp/prefetch.sh` (all 8 HF datasets via Clash proxy, skip-complete + sha256 verify + infinite retry; note ScreenSpot now comes from the `bevaya/ScreenSpot` mirror), tmux `moonart` continues WikiArt → fashion, `moonspot` killed (screenspot consolidated into moondata; stray manual aria2s killed by PID; scp partials in mmmu_pro/docvqa_val deleted, aria2-completed textvqa_val shards 0–1 kept). Proxy is total-capped ~250 KB/s (2-file parallel test: 116+94 KiB/s, no scaling) → remaining ~15 GB ≈ 16 h. **The big untapped lever: switching the Clash node in the mihomo-party GUI (no external-controller on the core, so only the user can click it) — the current node is congested to ~50 KB/s/conn; a healthy node would 10× everything.** Workstation-direct ModelScope remains dead (per-IP CDN ban), IPv6 has no route to ModelScope, MS token irrelevant to the ban. If the relay is ever revived: `scratch/ms_relay.py` (3-level resume), and never `pkill -f` a pattern that appears in the invoking command line (self-kill, exit 255).
- **Upload channel measured (2026-08-03)**: 315 MB test file → 2.0 min = **~2.6 MB/s through the proxy** using the stock `huggingface_hub` upload (the deprecated `HF_HUB_ENABLE_HF_TRANSFER` is ignored — new-hub chunked upload does the work; note this is a different code path from downloads, which still need aria2). Full ~15 GB dataset upload ≈ 1.5–2 h, free and in background — the on-box rebuild fallback is no longer needed. Scratch test file deleted from the data repo.
- **Bandwidth crisis resolved via ModelScope (2026-08-03 afternoon)**: the Clash/mihomo proxy node collapsed to ~50 KB/s aggregate (single-conn HF range requests timed out entirely; hf-mirror 91 KB/s; no `external-controller` on the mihomo core so no API node-switch). Two moves fixed it: (1) the fastllm agent's Q6_K GGUF download (22 GB, 16 aria2 conns, auto-respawning via a hub background job) was paused at 97.5% after tmux coordination — resume later with `aria2c -c` in `/1CatVLLM/models/ThinkingCap-Qwen3.6-27B-GGUF/`; **notify the fastllm pane when our downloads finish**. (2) **ModelScope direct (no proxy) is 50–100× faster** (8.8 MB/s measured, warm CDN edges burst higher). Mirrors with byte-identical layout to HF: `lmms-lab/textvqa` (train 20 + val 3 shards), `lmms-lab/DocVQA` (`DocVQA/` subtree: train 12 + val 6), `AI-ModelScope/MMMU_Pro` (`standard (10 options)/test-*` ×2), `showlab/ShowUI-desktop` (34 shards). NOT on ModelScope (stay on proxy path, tmux `moonart`): WikiArt_Full, fashion-product-images, ScreenSpot — `AI-ModelScope/wikiart-zero-shot` exists but is 1000 rows, unusable for training. New script `/tmp/prefetch_ms.sh` (tree-API listing + aria2 `-x8 -c` direct, same staging layout, 8× retry per file) runs in tmux `moondata` (recreated session — the old one died when its pane shell exited, killing its screenspot aria2; screenspot re-chained behind moonart in tmux `moonspot` via `/tmp/prefetch_spot.sh`). Integrity: ModelScope shard sizes verified equal to HF tree listing; full sha256-vs-HF-lfs-oid check deferred until proxy recovers (pyarrow row-count read in the fetch phase catches truncation). Fetch pipeline validated on real data: textvqa val-00000 → 500 samples + images + MANIFEST via `fetch_eval_data.py --data-files`.
- **Decontamination hardened (2026-08-03, review)**: `build_train_mix.py` now runs three independent mechanisms with per-mechanism drop counts in `decontamination_report.json`: perceptual aHash (hamming ≤ 6, catches resized/recompressed dups) + exact RGB-pixel sha256 (catches same content across containers) + normalized-question text near-dup (`--eval-jsonl`, catches cross-split text leakage). Tests in `tests/test_train_mix.py`.
- **All data downloads COMPLETE + verified (2026-08-04)**: 93 formal parquet shards (12 sources) in `$HDD/staging/parquet/`, every shard sha256-checked against its HF LFS oid with a `.sha256ok` idempotence marker, **0 mismatch**. Root-caused the silent-corruption incident: xet-bridge signs URLs per-range, so aria2 `-x8` segmented retries with a stale URL wrote wrong-range bytes at wrong offsets (valid TLS, correct size, 25 bad shards) — fix is single-connection `-x1 -s1` + per-shard sha256 verify (commits f6337fc…248ec64; also aria2 ignores uppercase `HTTPS_PROXY`, 081ef8a). Fetch products: eval 1,400 rows at exact targets (textvqa 500 / docvqa 200 / ocrbench 200 / screenspot 200 / mmmu_pro 300) + train_raw (textvqa_train 34,602 / showui_desktop 7,496) + sft_art 71,780 train / 2,220 val. **docvqa_train (25k target) was the last fetch** — offline from staging, tmux `moonart`. `pack_to_parquet.py` (union-keys schema, embedded PNG bytes) + `load_records` transparent JSONL/parquet loading landed (6/6 tests). Post-fetch pipeline scripted: `/tmp/post_docvqa.sh` (mix → pack train+eval) and `/tmp/gate_b.sh` (2000-step full-mix train → 5-benchmark trained/random × vision/blind eval on packed parquet → aggregate → upload), fired from tmux.
- **Stock-model positive control arm COMPLETE (2026-08-04; evaluator validation only)**: `tools/eval_stock_vlm.py` runs an unmodified native HF VLM (official weights + processor + chat template) on the same eval records with the same metrics/report shape as `eval_vlm.py` (reuses `make_scored_row`/`slice_records`/`score_record`), thinking disabled by default (template `enable_thinking=False` + think-block strip, raw prediction kept), blind pass text-only. Qwen3.5-4B is explicitly multimodal (`Qwen3_5ForConditionalGeneration`, `vision_config` present), so these scores are not projector/DeepSeek evidence. Repaired bf16 selection-half results are recorded in the takeover note and report; suite marker `STOCK_EVAL_REPAIRED_ALL_DONE`. Do not schedule 9B/27B merely as more native-VLM controls unless a later question requires them.
- **Trunk warm-start control arm (2026-08-04)**: `PatchMergerProjector.load_trunk(donor_dir)` transfers `pre_norm` + `linear_1` (language-agnostic, vision-side dims only) from a projector aligned against a small backbone; `linear_2` keeps fresh init (its output width is backbone-specific). `train_overfit.py --init-projector-trunk DIR` (`--resume` still wins). This is the legal cross-backbone warm-start arm for the rental (small-model projector cannot plug into 0731 directly: hidden 896/576 vs 4096).
- **Report sec. 11 added (2026-08-04, commit c7f20b5)**: dataset selection process + build pipeline — selection principles, eval/train tables with HF sources, exclusions (long captions, 0xSero screenshots/multistep, below-release-precision controls), download-channel postmortem, triple decontam, parquet packing, serial-upload rule. PDF rebuilt and committed.
- **Gate B full-mix early-alignment run COMPLETE (2026-08-04) — engineering/signal gate only**: 2000 optimizer steps (`micro_batch_size=1`, serial gradient accumulation 8, 16,000 examples seen ≈ 0.27 epoch, constant lr 5e-4, `--max-image-side 448`) on the packed train_v1 mix (59,198 rows), frozen Qwen2.5-0.5B-Instruct + frozen K3 MoonViT-V2 (eager), projector-only. Loss 6.413→3.006 (windowed; single-step 10.19→3.01, min 2.718 @ 1750); historical held-out (32 records, old circular-shift protocol) true 3.175 vs shuffled 3.902 → **shuffle_delta +0.727**. This is unambiguous interface-learnability evidence but not a converged capability measurement; TextVQA 8.1% / DocVQA 3.9% / OCRBench 0% are not architecture ceilings. Two incidents: (1) **all 4 streaming checkpoint uploads failed** (proxy SSL resets on S3 multipart PUT during training) — local checkpoints intact at `$HDD/data/gate_b/ckpt/checkpoints/step-000500..002000`; (2) initial evals crashed on a placeholder default mismatch — fixed as candidate auto-detect. **Final five-benchmark table (selection half, 1024px; raw outputs on HF `eval/v100-fullmix-qwen05/`)**: textvqa **0.081** / random 0 / blind 0; docvqa **0.039** / 0 / 0; ocrbench 0/0/0; screenspot parse **0.51** + accuracy 0.01 / 0 / 0; MMMU-Pro **0.073** / 0 / 0. The scoring audit found two MMMU input-format bugs and the repaired strict score is 7.3% (old 2.0% voided). Treat every absolute score as a low-capacity, undertrained control result.
- Typst report is `report/main.typ`; the built `report/main.pdf` **is committed** (only that file is exempted from the `report/*.pdf` ignore rule) so readers can grab the PDF without a Typst install. Rebuild after report edits with `typst compile report/main.typ` or, on the Windows box without a Typst CLI, `pip install typst` into `.venv` and `python -c "import typst; typst.compile('report/main.typ', output='report/main.pdf')"` (verify a page render afterwards — bare `<` in prose breaks the build with "unclosed label").
- Read-only Vast snapshot recorded in the report; no instance was created.

## doesworkstation (V100 32GB) operational notes

- SSH alias `doesworkstation` works via Tailscale (100.94.73.9); the Clash TUN fake-IP hijacks the name when TUN is on — use the Tailscale IP if resolution fails. With TUN off, the local machine needs `git -c http.proxy=socks5://127.0.0.1:10808 push` for GitHub.
- **NVML mismatch**: kernel module 580.159.04 vs userland 580.173. `nvidia-smi` fails, but CUDA works fine. Do NOT reload the driver or reboot — the `fastllm` agent's jobs depend on the current state.
- Working env: `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/.venv-1cat/bin/python` — torch 2.10.0+cu128 (**includes sm_70, V100 works**), transformers 5.12.1, safetensors, pillow, einops. No pytest: it was installed into `/run/media/ezra/13D010B6FDBC1A06/moonvit-deps` via `pip --target`; use `PYTHONPATH=$HDD/moonvit-deps:src`.
- HDD (13T, ~3.7T free) is `/run/media/ezra/13D010B6FDBC1A06/` — never write big files to `/home` (89% full). Repo clone lives at `$HDD/moonvit-deepseek-v4-glue`; `HF_HOME=$HDD/huggingface`.
- Workstation proxy: `127.0.0.1:7890`. **Current integrity rule**: never use aria2 multi-range against HF Xet redirects; it has twice produced correct-size cross-shard corruption. Use `HF_HUB_DISABLE_XET=1` with `hf download --max-workers 1` or a single-connection resume, then require SHA-256 equality with the HF LFS oid before load. Offline dataset builds use raw pyarrow because the datasets+dill stack cannot pickle pyarrow's MonthDayNano.
- MoonViT quirks on this stack (Transformers 5.x): remote code lacks `all_tied_weights_keys` (shimmed in `moonvit.py`); bf16 has a mixed-dtype layer_norm bug in the remote code → **run MoonViT in fp32 on the V100** (~1.6 GB).
- Small-context text models overflow on native-resolution images (1064 merged tokens from 640×480 + prompt > 1024 positions of tiny-gpt2 → scatter-gather device assert, which surfaces at unrelated async locations; use `CUDA_LAUNCH_BLOCKING=1` to localize). Use `--max-image-side 448` for small models.
- Under load (other agent compiling/running inference servers), torch import from the mechanical disk takes >90 s — budget timeouts generously.
- The `fastllm` tmux pane runs another agent (GPT-5.6) optimizing Qwen inference; it launches GPU `apiserver` variants. Coordination messages were left in its pane, including the user's request to test `bottlecapai/ThinkingCap-Qwen3.6-27B-GGUF` 4-bit/6-bit quants as a replacement for the Fable Fusion model. ~15.6 GiB VRAM was free; keep our GPU usage small and short.
- **ComfyUI service check during Package 15A returned `inactive`**; Package 15A did not stop or restart it. The runtime-only override at `/run/user/1000/systemd/user/comfyui.service.d/zz-stop.conf` still disappears on reboot. Re-check before GPU work and coordinate if another session has started a workload.

## Immediate next actions

1. ~~Fetch real eval datasets / mix / pack / upload~~ **DONE (2026-08-04)**: all 12 sources staged and sha256-verified; mix 59,198 rows; `train_v1`/`eval_v1`/`sft_art_v1` all on `cyjin-yl/moonvit-dsv4-data`. Historical Gate B early-alignment run DONE (shuffle_delta +0.727 at only ~0.27 epoch). Repaired stock 4B positive control and artifact publishing are also complete.
1b. **Qwen2.5-3B contract, real-image smoke and exact 4k order/targets are complete in Packages 15A/15B/15C**. The exact `efd942e0…b06b0` step0, pinned BF16-source→FP16-runtime Qwen, 4096 receiver gradient, checkpoint round-trip and `ddca738e…c2fd` training-order manifest all pass. The next architecture screen is now two matched arms: exact K3/MoonViT-V2 `PatchMergerMLPV2` and MoonViT-SO-400M V1/K2.6-lineage PatchMerger, both under the same 3B health and community evaluation contract. Legacy-pre-norm V2 checkpoints remain diagnostic only. Expand to 8k/16k only from contract evidence; every candidate also runs TextVQA, DocVQA, OCRBench, synthetic and language retention.
2. ~~Overfit check~~ **done on three tracks** — SmolLM2-135M/comfy delta +0.343, Qwen2.5-0.5B/comfy delta +0.282, Qwen2.5-0.5B/flickr8k-1100 delta +0.148 (checkpoint `checkpoints/overfit-qwen05-flickr8k` on the workstation). flickr8k data note: `jxie/flickr8k` (nlphuji is gated); hub fetch kept dying on the proxy, so the 1100 records were decoded offline from the two fully-cached train parquets (script `$HDD/staging/rescue_flickr8k.py`), MANIFEST.json records the resolved revision `56f58c9`. Validation/test parquets were never downloaded — don't need them.
3. Re-run the read-only Vast offer search immediately before budgeting; do not create an instance without fresh user approval. **Ampere correction**: 4×A100 cannot load the native NVFP4 weights (no FP8/FP4 path; bf16 dequant = 568GB > 320GB) — Gate D primary is 4×H100 PCIe ($6.93/h); 8×A100 SXM4 ($10.30/h) is the scenario-B dequant fallback only; 4×B200 ($21.25/h) is scenario A′ when H100 cannot even run the FP4 forward. Final bill in the report ("最终账单"): scenario A **$55 optimistic / $75 baseline / $110 pessimistic** (download-time sensitivity included; budget $120), scenario A′/B ~$210–310, ceiling $350. Bandwidth and storage are minor (~$5 total) but **destroy the instance at the end — stopped instances keep billing storage ($0.107/GB/mo ≈ $53/week for 2TB)**. The report also carries the full Gate D risk register (R1–R11) with per-risk mitigation and abort points; there is no "guaranteed success" — the guarantee is that failure costs ≤ ~$20 and every failure mode has a planned next step.
4. On the rented multi-GPU box, run Gate D: native 0731 load → single-image forward → single-batch backward (Dgrad verification) before any training loop exists.
5. Rental closed-loop shape remains setup → Gate D → measured alignment budget → full benchmarks ON the box → projector + eval JSON upload, but the hours/steps/global batch estimate is **suspended** until true batching and pre-rental ablations finish. Stop criterion remains benchmark-minus-blind gap plateau, not loss. Deliverable is the projector only — never the 160GB backbone — in **two dtypes: fp32 master (~134MB) + bf16 serving (~67MB)**. Checkpoints remain fully resumable and stream to HF when bandwidth permits. Inference-side usage is in `docs/inference-integration.md`.
6. **Vision tower on the rented box = partial download only** (verified 2026-08-03 in a clean-room import test with no K3 staging on PYTHONPATH): the server needs (a) `git clone` of our repo — vendored V2 code depends only on stdlib/torch/transformers/numpy/PIL, no `trust_remote_code`; (b) `vision_tower_k3/` from `cyjin-yl/DeepSeek-V4-Flash-0731-Vision` (~800MB: `moonvit_v2.safetensors` + `MANIFEST.json` with sha256 + processor/vision configs + code snapshot). Verify `sha256sum moonvit_v2.safetensors` against MANIFEST before training. The full 1.56TB K3 repo is never touched on the server. Train/eval with `--vision-tower v2 --moonvit-v2-weights <path>/moonvit_v2.safetensors` (both `train_overfit.py` and `eval_vlm.py` support it; `--moonvit-v2-attn sdpa` on Hopper, `eager` is the reference). The CLI still defaults to V1 for backward-compatible comparison smoke only; every production command must select V2 explicitly.
5. HF-cache surgery note: direct `curl -C -` resume loop beats `hf download` on this proxy (it stalls); the blob filename in `blobs/` is the content sha256 — the xet-bridge redirect etag is NOT the file hash.

## Main unresolved risk

Official 0731 FP4/FP8 kernels may support inference but not gradient with respect to input embeddings. The glue contract is tested; the large-quantized-backbone Dgrad path is not.

## Training memory arithmetic (verified 2026-08-03, table in the report)

Optimizer states exist **only for the projector** (`AdamW(projector.parameters())`; LLM + tower are `requires_grad_(False)`, tower forward under `no_grad`): full optimizer+weights+grads ≈ 0.55 GB. Per-card totals with TP-sharded LLM weights: **scenario A (4×H100, FP4, TP=4): ~47–55 / 80 GB — healthy** (25–32 GB headroom); **scenario B (8×A100, bf16 dequant, TP=8): ~77–79 / 79.1 GB — ceiling-level**, needs micro-batch 1 + aggressive checkpointing, escalate to 4×B200 (142 GB/card) on OOM. bf16 on any 4-card box is arithmetically impossible (568/4 = 142 GB > 80/141 GB). New risk R12: 4×H100 PCIe has no NVLink, TP all-reduce over PCIe may blow past the 3–5 s/step estimate — Gate D must measure actual step time; >15 s/step → NVLink host or pipeline parallelism.

## Safety / repository hygiene

- Never commit HF, GitHub, Vast, or SSH credentials.
- Never commit model shards, datasets, or projector checkpoints (`data/`, `checkpoints/` are gitignored).
- Keep MoonViT, DeepSeek, and projector revisions/hashes separate.
- The supplied Vast credential was not written into this repository.
- The Vast API was used only with `POST /api/v0/bundles/` (offer search), never the instance-creation endpoint.

## Matched V1/V2 health screen result (2026-08-06)

The public MoonViT-SO-400M V1 lineage and the exact Kimi-K3/MoonViT-V2
PatchMergerMLPV2 were finally compared on the same frozen Qwen2.5-3B contract.
Both use the same 4,000-row order, receiver, learning rate, probe schedule and
automatic rollback.

V1 has a complete 4,000-row cache: 3,534 real tower forwards, 466 image-hash
reuses, zero failures. It stops at step 2 with projector/receiver rank ratios
`1.000 -> 0.264` and `1.000 -> 0.212`. V1 therefore does not rescue the
early common-direction collapse.

Exact K3 V2 uses the correct bias-free MLP plus post-RMSNorm. Its existing cache
was rebound to the current frozen order with 111 hard links after all 4,000
IDs, image hashes, shapes and spans matched. It also stops at step 2. Geometry
is healthier (`1.000 -> 0.910` projector rank ratio and `1.000 -> 0.830`
receiver ratio), yet correct-image log-prob is below shuffled at every
post-step probe (`-0.240 -> -0.204 -> -0.098`). The causal guard stops it.

The result weakens “V2 compression alone is the root cause”. V1 and V2 share
an early optimization/receiver-interface problem on the frozen 3B proxy.
V2 remains the migration candidate because its exact structure preserves
geometry better; it has no visual-capability claim. Raw archives and hashes
are under the two architecture-control health-run directories, with compact
summaries and independent verifier records committed in Git.

Next local screen: exact V2 with a substantially smaller projector learning
rate, same data/order/guards and a matched CE-only control. If geometry survives,
run the smallest image-vs-shuffle counterfactual objective. Do not increase the
dataset or run full ScreenSpot generation until a trajectory passes both health
and causal gates.

The next exact-V2 exploration used a projector learning-rate override of
5e-5 while retaining the main-contract value 5e-4 in the logged RUN_CONFIG.
Rank/spread stayed essentially at step 0 through step 2 (projector rank ratio
1.000, 1.000, 0.999; receiver 1.000, 1.000, 0.999), so update scale
contributed to the high-LR geometry collapse. Correct-image preference stayed
tied with shuffled (0.625, 0.625, 0.500), and vision-minus-shuffled log-prob
stayed negative (-0.240, -0.211, -0.285). The causal guard stopped at step 2
and the independent verifier was verified. The next registered direction is
an image-vs-shuffle supervision screen, with the same geometry-safe LR and a
matched CE-only control.

The first paired image-shuffle margin screen used the geometry-safe learning
rate 5e-5 with margin 0.1 and lambda 0.1. Projector/receiver rank ratios stayed
near 1.0 through step 2, and vision-minus-shuffle correct-logp moved from
-0.240 to -0.061. It did not cross zero: shuffled preference reached 0.750
while vision stayed 0.625, so the causal guard stopped and rolled back the run.
The independent verifier is `verified`; the raw archive is bound by the
committed pointer and hashes. This supports the supervision direction having a
small effect, but rejects lambda 0.1 as sufficient evidence of image use.

The next discriminating branch is now a capacity/receiver-prior screen. Audit
and try a pure-text Qwen2.5-7B projector-only arm first; gate any 9B/14B arm on
V100 memory and finite input gradients before downloading/training. In
parallel, a stripped-native Qwen3.5 diagnostic may retain its visual-pretrained
language weights while bypassing the native vision tower, merger and
cross-attention, accepting only MoonViT-V2/projector embeddings. Such a result
is receiver-prior evidence and cannot be promoted as a DeepSeek capability
result. All arms keep the fixed cache, health guards and community evaluation
contract.

== Package 15S-capacity：Qwen3.5 receiver-prior 结果

Qwen3.5 的 native vision、merger 和 visual forward 被绕过，外部输入来自同一份 exact-K3 V2 cache/projector；三个成功/失败运行都记录了 `native_vision_forward_calls=0`。4B BF16/16-token 能有限回传梯度，但 `vision−shuffle=-0.0597`；4B FP16 full-token 在首次更新后出现 NaN/Inf；9B BF16/16-token 使用 4096 identity receiver，得到 `vision−shuffle=+0.6265`，CE 从 1.5632 降至 0.9113。

9B 的官方 revision、config 和四个权重 SHA 已冻结。这个正 margin 支持“视觉预训练过的接收器比纯文本 3B 更容易接收新的视觉塔”这一假设，也说明当前 3B 失败并非 V2 版本一个因素。它仍只有一个样本、16 个 token、一步更新，属于 receiver-prior numerical diagnostic；`capability_claim_allowed=false`，不能替代 ScreenSpot、TextVQA、DocVQA、OCRBench，也不能写成 DeepSeek 已获得视力。

下一位执行者先跑 9B BF16 32/64/128/240 token 短筛选，再跑 Qwen2.5-7B 纯文本 matched control。若长 token 仍稳定且 margin 为正，才值得把相同 projector/目标带回固定 3B 社区评测；否则先处理 token 压缩和数值尺度。Qwen3.5 只作为迁移判断材料，不改变正式 DeepSeek 配方。

9B token sweep 已完成：32/64/128/240 token 均 finite，`vision−shuffle` 分别为 `+0.1781/-0.4574/+0.2881/-0.8842`。这证明长序列输入梯度在 V100 上可控，但图像因果方向不随长度稳定。Qwen2.5-7B 纯文本 matched control（官方 revision `a09a35458c702b33eeacc393d103063234e8bc28`）在 FP16 下 16/240 token 均 finite，margin 为 `-1.0731/-0.8335`。因此“模型规模从 3B 增到 7B 就能读懂外部视觉 token”被反驳。

下一步不要直接扩大 9B 训练。先用固定 9B receiver 做至少 8--32 个 probe 样本的 16/32/64/128/240 token 小筛选，并加入 random projector；若正向差异仍不稳定，优先检查视觉 token 顺序、压缩和尺度。Qwen3.5 的所有结果仍为 `transferable_with_runtime_validation` diagnostic，不能替代 DeepSeek Gate D。

8-sample probe 已完成：16 token 的 `vision−shuffle` 为 `+0.0447 ± 0.3729`，240 token 为 `-0.0748 ± 0.4520`；`vision−blind` 分别为 `+0.1993 ± 0.2142` 和 `+0.6753 ± 0.3335`。因此 9B receiver-prior 能感知“有视觉 token”，但不能稳定归因到正确图片。下一条应优先做 token ordering/压缩或输入尺度的小设计筛选，而不是把 9B 直接扩成长训或把它写成 VLM 成功。

V1 交叉检查也完成：同一 9B、8 个样本、240 token 下，V1 `vision−shuffle=+0.0620 ± 0.4185`，V2 `-0.0748 ± 0.4520`；V1 `vision−blind=+0.3780 ± 0.1962`，V2 `+0.6753 ± 0.3335`。V1 只略高于零，差异没有超过方差，不能宣称 V1 胜出。主要嫌疑移向 token 顺序/压缩、输入尺度和监督接口。

Qwen3.5 native 3D mRoPE 诊断也完成：V2、8 samples、240 tokens 下 `vision−shuffle=-0.0375 ± 0.4537`、`vision−blind=+0.6680 ± 0.2896`，和普通连续位置的 `-0.0748/+0.6753` 几乎一致。mRoPE 不是当前差异缺失的主因；该分支为 `qwen_specific_not_transferable`，不能改变 DeepSeek 方案。

9B projector-only training 的两次修复均撞到 V100 32GB 边界：240-token 首次 forward 是 NVML allocator assert；修复 health graph retention、缩到 16 token 后仍在约 25.9GiB allocated 时 OOM。9B 保留 inference/input-gradient gate，不再在本机强行长训；3B/7B 承担 projector 训练与正式合同筛选。

Qwen2.5-7B 的 3-step CE-only screen 已完成：8 samples、16 tokens、FP16 全 finite；CE `0.2381→0.0094`，RMS/spread 基本不动，`vision−shuffle` `+0.0333→-0.1027`。这确认 7B 有足够显存完成 projector backward，却重复“loss 下降、图像归因变差”的路线。CE-only 结果不进入 candidate；下一条训练若继续，必须用 matched image-vs-shuffle objective，并用相同 7B CE-only control 对照。

matched margin arm（λ=0.1）在 7B 上的 16-token 轨迹为 `+0.0333/-0.0568/+0.0365/+0.0984`，240-token 轨迹为 `+0.0263/+0.0033/+0.0086/+0.0090`。这是一条有限但重要的监督方向证据；240-token 仍接近零，不能进入 ScreenSpot。下一条只在 7B 上做 token compression/scale-safe 的单变量筛选，保持 CE-only control 与 health guard。

## 2026-08-07 审计修正与当前大方向

审计发现，以上 stripped-receiver 的旧 3B/7B/9B 运行只使用 feature-cache manifest；它没有问题和答案字段，旧 `build_inputs()` 静默回退到统一 prompt 与 `click(start_box=[500,500])`。这些 raw 结果仍保留，能证明加载、反向、保存和健康日志链路，但所有 `vision−shuffle`、capacity 和 token-length attribution 只能标记为伪监督 receiver 扰动诊断，不能作为真实视觉能力或 projector 优劣证据。

已新增冻结真实答案 manifest：`experiments/qwen3b_community_eval_20260805/capacity_controls/qwen25_7b_real_probe_manifest.json`，SHA-256 `9fb216e...de130`。训练/探针现在按 ID 连接该 manifest，逐条校验 image SHA，并在缺少 question/instruction 或 target answer 时硬失败；不再允许默认坐标回退。

修复后的 Qwen2.5-7B 8-sample matched runs（FP16、exact step0、3 steps、prefix 16）仍健康，但真实答案归因为负：CE-only `6.9373→4.4393`、`vision−shuffle -0.2741→-0.8746`；paired margin λ=`0.1` 为 `6.9373→4.5825`、`-0.2741→-0.1613`。因此“7B 能训练”成立，“7B 已经学会看图”不成立；当前不能进入 ScreenSpot 或 DeepSeek 候选。

机制经验必须与 benchmark 同步保留：CE 下降不等于视觉归因提升；3B 高 LR 的 common-direction collapse 可在首步出现；小 LR/几何保护可保留表示却没有自动产生 image attribution；V1/V2/mRoPE 对照尚未消除 paired gap；监督接口和真实答案 manifest 本身是 Gate D 前置条件。

真实答案合同下的后续 token screen：240-token CE-only `vision−shuffle +0.3338→+0.2444`，paired margin λ=`0.1` `+0.3338→+0.3375`；16-token prefix `-0.2741→-0.1613`，uniform `-0.2421→+0.0630`，mean-pool `-0.2036→-0.1363`。这些结果只说明 token 覆盖会影响 receiver 的局部答案归因；8 samples、3 steps、无 bootstrap，不能替代 ScreenSpot。mean-pool 梯度峰值约 4,292，需保留为数值风险记录。

最短后续是对四个 token 条件做逐样本 probe、random-projector 和 bootstrap，再决定是否扩大样本。只有真实 benchmark 通过，才能替代 previous-best 或进入 DeepSeek 候选。旧 cache-only receiver-prior 数字均已降级，见 `docs/current-status.md` 的审计修正节。

32-sample frozen probe 已完成：固定 seed `20260805`，TextVQA、DocVQA、ShowUI、普通 VQA 各 8 条，manifest SHA `c726ebfd...a5a629f`。2,000 bootstrap 后 full240/prefix16/uniform16/mean_pool16 的 `vision−shuffle` 均值与 CI 分别为 `-0.22[-0.64,0.13]`、`-0.07[-0.35,0.21]`、`-0.05[-0.31,0.22]`、`+0.14[-0.12,0.39]`；`vision−blind` 全为正。receiver 感知视觉 token，正确图像归因仍不稳定。

32-sample mean-pool matched training：CE-only `+0.1351→+0.2051`，margin λ=`0.1` `+0.1351→+0.1722`，margin 没有优于 CE-only，梯度峰值约 3,000。scale sweep（projector RMS 约 0.994、文本 embedding RMS 0.01364）在 `0.01/0.03/0.1/0.25/1.0` 上均未让 paired CI 脱离 0。下一步只做 scale=`0.1` matched training；如果仍失败，停止扩展 Qwen 训练量，转 projector 结构/辅助目标。

## 2026-08-07 scale=0.1 训练结论与交接

scale=`0.1` 的 32-sample matched screen 已完成，使用冻结 manifest SHA `c726ebfd...a5a629f`、mean-pool 16、同一 derangement、同一 exact step0 和冻结 Qwen2.5-7B receiver。CE-only：CE `6.9045→5.8405`，`vision−shuffle -0.0167→+0.1297`，全程 finite，gradient peak 约 `781`。paired margin (`lambda=0.1, margin=0.1`)：CE `6.9045→5.9001`，`vision−shuffle -0.0167→+0.2487`，同样 finite。

训练后 32 条 probe 的 2,000 次 bootstrap：

- trained CE `vision−shuffle=+0.1297`，CI `[-0.3042,0.5542]`；`vision−blind=+1.8596`，CI `[1.2702,2.5173]`。
- trained margin `vision−shuffle=+0.2487`，CI `[-0.1099,0.6001]`；`vision−blind=+1.7932`，CI `[1.2040,2.4167]`。
- trained margin 的 `vision−random_projector=-0.5038`，CI `[-1.0500,-0.0247]`；margin-minus-CE 配对差 `+0.1190`，CI `[-0.0429,0.2881]`。

结论：scale 和 paired supervision 能改善点估计与随机 projector 对照，但正确图/打乱图的 paired CI 仍跨零，不能称真实 grounding 改进，不能替换 `previous_best`，也不能启动完整 ScreenSpot/通用 VQA 晋升。该结果支持“数值尺度和监督方向是有效变量”，拒绝“单纯乘常数就能修复归因”。当前下一项只做一个结构/辅助目标变量加 matched CE-only control；若 CI 继续跨零，冻结 Qwen 训练量，整理迁移合同并转 DeepSeek runtime Gate 设计。

研究记录必须继续同时保存训练健康与真实能力：RMS、spread、rank、Gram、gradient、NaN/Inf 与 vision/blind/shuffled/random-projector paired 指标分开报告；旧 cache-only receiver-prior 结果已经降级为伪监督扰动诊断，真实答案 manifest 运行才可作监督接口证据。

Gate D 仍为 **NO-GO**。Qwen 代理已证明 V100 上 projector-only 训练和安全止损链路可重复；完整 DeepSeek-V4-Flash-0731 的权重加载、FP4/FP8 input DGRAD、完整 Hash-MoE 图像 forward/backward、20-step 稳定 save/resume 和真实 benchmark 仍未通过。没有付费硬件授权前，不租卡、不下载完整 0731。

## 2026-08-07 λ=0.5 训练结果

在预注册 `configs/qwen25-7b-real-answer-scale01-margin-screen-v1.json` 下，固定 scale=`0.1`、mean-pool 16、32-sample manifest 和 exact step0，运行 CE-only control 与 paired margin λ=`0.5`。CE-only 的 CE `6.9045→5.8405`、`vision−shuffle -0.0167→+0.1297`；λ=0.5 的 CE `6.9045→5.9831`、`vision−shuffle -0.0167→+0.4874`。两臂 finite，gradient peak 约 781/459，between-image RMS 稳定。

训练后 probe 的 2,000 bootstrap：λ=0.5 的 `vision−shuffle=+0.4874`，CI `[+0.1423,+0.8786]`；`vision−blind=+1.9574`，CI `[+1.3909,+2.5699]`；相对同批 CE-only 的 paired 提升 `+0.3577`，CI `[-0.1287,+0.8397]`；`vision−random_projector=-0.3397`，CI `[-0.7099,+0.0082]`。

这是第一条通过 32-sample paired image attribution CI 的训练轨迹，证据仍限于 teacher-forced、16 visual tokens、Qwen2.5-7B receiver-prior diagnostic。`capability_claim_allowed=false`，不能替代 ScreenSpot、TextVQA、DocVQA、OCRBench，也不能直接改写 DeepSeek 配方。它支持“paired 监督强度能修正局部图像归因”，下一项优先做 7B formal evaluator 的统一 parser/blind/shuffled/random-projector 与自由生成检查；只有自由生成方向一致，才考虑把 λ=0.5 带回 3B 社区合同。

新训练的 385 MB projector/optimizer 原始目录留在远端 V100 数据根，SHA 在 `qwen25_7b_scale01_margin05_20260807_RAW_POINTER.json`；Git 保留摘要、health、probe、bootstrap 和可审查指针。Gate D 仍为 **NO-GO**。

7B λ=0.5 checkpoint 的自由生成 companion 已完成。固定 8 条 ShowUI 样本、社区 grounding prompt、greedy decoding、`max_new_tokens=32` 下，四种条件都能解析 `click(start_box=[x, y])`（8/8），但 vision/blind/shuffled/random 的到目标点平均距离为 `491.73/514.31/493.97/499.97`；vision 相对 shuffled 的逐样本距离改善只有 `+2.24`，8 条样本不足以做 bootstrap。输出大多落在 `[450,300]` 一类窄坐标附近，说明 teacher-forced 的正向归因尚未转成可靠自由 grounding。

第一次 generic prompt 运行全部无法解析 click，随后发现这是 prompt route 不匹配；第一次 generator 还错误假设 manifest 含 shuffled_sample_id。两次失败/修复均有 `FAILURE.json` 和 retry raw 目录。结论：λ=0.5 仍只能列为 receiver-prior mechanism candidate，不能替换 ScreenSpot previous-best；下一条实验应先修复/统一 7B formal evaluator，避免把 prompt 退化或 coordinate prior 当视觉能力。

随后完成 50 条 `screenspot_glm50_v1` 的 7B stripped ScreenSpot 诊断（16 mean-pool tokens、scale 0.1、四条件、固定 parser 和 2,000 bootstrap）。四条件 parse rate 均为 50/50；vision/blind/shuffled/random 的 click-in-box 均 10%，Accuracy@50/@100/@200 均为 2%/6%/18%；中心距离均值为 380.73/415.11/384.45/390.22。vision-blind 中心距离差为 -34.38，CI [-70.63,-3.30]；vision-shuffled 为 -3.72，CI [-9.91,+1.54]。因此视觉 token 让输出更接近某些目标，但正确图和 shuffled 图的 grounding 指标完全相同，Qwen7B 候选拒绝晋升。当前停止增加 λ、数据和训练步数，转文档归纳与 DeepSeek runtime Gate 设计。

## Qwen3.5-9B stripped receiver result (2026-08-07)

The first 50-row run is retained as a decoding-contract failure: Qwen3.5's
default reasoning template spent the 32-token budget before emitting a click,
so every condition parsed 0/50. After adding `enable_thinking=false`, an 8-row
repair parsed 8/8 in every condition, yet click-in-box was 0%; vision-minus-
blind center distance was `-42.74` with CI `[-127.74,+13.77]`, while
vision-minus-shuffled was `+88.69` with CI `[+3.00,+199.15]`. Positive means
the correct-image prediction was farther from the target. This receiver-prior
diagnostic rejects an automatic capacity or visual-pretraining rescue, while
remaining outside the formal Qwen leaderboard and DeepSeek capability claims.
Raw summaries, JSONL rows and the initial format failure remain on the V100
artifact path; the next design variables are placeholder semantics, projector
scale, position encoding and receiver-distribution alignment.

The local DeepSeek DGRAD preflight also completed. The reference mode passed an
ordinary frozen Linear input-only autograd check; the candidate mode matched the
same mathematical reference and was deliberately recorded as `hardware_pending`.
No complete DeepSeek-V4-Flash-0731 weights, real FP4/FP8 kernel, or Hash-MoE
routing was executed. Gate D remains NO-GO. The next local work is the
placeholder/position/routing/save-resume verifier; real quantized targets wait
for explicitly authorized hardware.

The tiny DeepSeek-V4 end-to-end retry then passed on the V100. It used the real
Transformers `DeepseekV4ForCausalLM` implementation with batch 2 and 20
projector-only optimizer steps. Projector gradients were finite and non-zero,
language gradients stayed `None`, greedy generation returned shape `[2, 8]`,
and step-10 save/resume matched an uninterrupted run with projector and loss
maximum absolute deltas both `0.0`. The initial grouped-feature shape failure
was preserved before retry. This closes the software tiny seam, while complete
0731 weights and real FP4/FP8 input-DGRAD remain pending.

The same tiny loop also passed in `bfloat16` on the V100 with batch 2, 20 steps,
exact save/resume deltas of `0.0`, and successful generation. This covers the
local BF16 seam only; it is not evidence for the target 0731 FP4/FP8 kernels.

## Exact DeepSeek placeholder-ID seam (2026-08-07)

The tiny software loop was rerun with the target `<｜image｜>` placeholder ID
`129279` rather than the earlier low test ID `63`; the tiny vocabulary was grown
only for this fixture. On the V100 in BF16, 20 batch-2 projector-only steps had
finite non-zero projector gradients and no language gradients, exact step-10
save/resume (`0.0` projector and loss deltas), and generation retained the two
expanded `129279` routing IDs. The raw pointer is
`experiments/qwen3b_community_eval_20260805/capacity_controls/deepseek_gate_d_tiny_e2e_placeholder129279_20260807_RAW_POINTER.json`.
This closes a numeric placeholder/routing software check; it does not pass the
full 0731, FP4/FP8 or Gate D requirements.

## Full public ScreenSpot result for the 7B lambda=0.5 checkpoint

The fixed Qwen2.5-7B checkpoint was evaluated on all 1,272 public ScreenSpot
samples with 16 mean-pooled visual tokens, projector scale 0.1, greedy decoding,
and the frozen vision/blind/shuffled/random-projector conditions. All 5,088
outputs parsed. Vision/blind/shuffled/random click-in-box was
`3.30%/3.46%/2.67%/2.91%`; Accuracy@50 was
`1.18%/1.02%/1.02%/1.26%`, Accuracy@100
`5.19%/5.03%/4.87%/4.87%`, Accuracy@200
`15.33%/15.09%/15.02%/14.94%`, and mean center distance
`404.38/409.71/406.10/405.74`.

Vision minus shuffled click-in-box was `+0.629` percentage points. The
independent category verifier's 2,000-bootstrap 95% CI was
`[+0.157,+1.179]` percentage points. Vision minus blind was `-0.157` points,
CI `[-0.943,+0.629]`. This is the first full-public result with a weak positive
correct-image click signal against shuffle, while the required blind control
still fails. Do not promote the checkpoint or claim the community GLM-5.2V
metric-aligned baseline. It meets or approaches parse rate, Accuracy@200 and
mean-distance references, but misses Accuracy@50/100 and causal vision-over-
blind evidence.

Category analysis shows the gain is not uniform. iOS vision-minus-shuffled
click was `+1.96` points with CI `[+0.39,+3.92]`; Android vision-minus-blind
click was `-1.62` points with CI `[-3.24,-0.40]`. Preserve these strata as
mechanism/failure evidence.

Raw artifacts live at:

`/run/media/ezra/13D010B6FDBC1A06/data/qwen3b_contract/capacity_controls/qwen25_7b_stripped_screenspot_public_margin05_scale01_20260807`

and the category verifier output at the sibling
`qwen25_7b_stripped_screenspot_public_margin05_scale01_categories_20260807`.
The row SHA is `bd783eb...f6d3`; the full summary SHA is `b41b275...b164`;
the category summary SHA is `f5da80b...75be`.

Mechanism evidence is now consolidated in
`docs/experiment-mechanism-findings.md`. Keep that document current whenever
receiver, token count/order, projector health, attribution, V1/V2 or
teacher-forced/free-generation behavior changes.

Next discriminating local task: run the same checkpoint on the frozen
ScreenSpot50 with 240 full-sequence tokens and compare it to the existing
16-token mean-pool result. Expand to all 1,272 samples only if correct-image
causal metrics improve without a blind regression. If token count does not
help, test one DeepSeek-transferable projector/auxiliary variable with a matched
CE-only control.

Gate D remains NO-GO. The tiny target-ID BF16 seam is complete. Full 0731 still
needs resolved-weight loading, real FP4/FP8 input DGRAD, 43-layer Hash-MoE
forward/backward and routing checks, target memory/throughput and activation
checkpointing, a stable 20-step exact resume, and the fixed real benchmark.
Local candidate/verifier work is estimated at 1--2 working days. After explicit
paid-hardware authorization, the minimal Gate D and first small real training
judgment are estimated at 3--5 working days if weights and kernels work as
expected.

## 240-token matched ScreenSpot50 result (2026-08-07)

The same 7B lambda=0.5 checkpoint was rerun on the frozen 50-row GLM-format
subset with 240 full-sequence tokens. Parse was `50/50` in all conditions;
vision/blind/shuffled/random click-in-box was `10%/10%/10%/8%`, Accuracy@50
`0%/2%/2%/0%`, Accuracy@100 `6%/6%/6%/6%`, and Accuracy@200
`18%/18%/20%/18%`.

The independent category verifier recomputed center-distance means
`399.51/415.11/396.78/397.02`. Vision-blind distance improvement was `+15.59`
with CI `[-13.51,+47.15]`; vision-shuffled was `-2.74` with CI
`[-13.58,+9.96]`. Both click paired differences were exactly `0` with CI
`[-6,+6]` percentage points. Full sequence therefore does not rescue the
grounding gap. Stop expanding token count and move to one projector or
auxiliary-objective variable that can transfer to DeepSeek, with a matched
CE-only control.

The raw evaluator has complete `center_distance` statistics; an older summary
consumer used an incompatible key/schema. The category verifier cross-checks
the distances and categories, with the schema boundary recorded in
`...full240_20260807_RAW_POINTER.json`. The run took 1,310.98 s,
including cold Transformers import and 339-shard CPU weight loading. That
startup cost must be separated from generation throughput in future batching
optimization.

## Package 15R gated residual repair (2026-08-07)

15R is now closed as a geometry rejection. The parent preregistration remains
unchanged. Three provenance failures are preserved separately: source drift
before training, the exact-frozen gated run rejecting the mathematically
expected zero `residual.weight` gradient at gate=0, and a runner-hash mismatch
when the repair was first copied into the frozen worktree. The repair contract
and runner now allow only that zero branch gradient; the scalar gate and every
other projector parameter remain hard finite/non-zero checks, with a real
backward unit test.

The clean-main matched runs are
`baseline_none_repair_v3` and `gated_residual_repair_v2`. Both passed the
independent health verifier and stopped at optimizer step 2, onset `[1,2]`,
with `projector_rms_rising_spread_falling` and
`receiver_rms_rising_spread_falling`. Gated step 2 had projector/receiver
spread ratios `0.2690/0.2254`, rank ratios `0.5008/0.3611`, and
`vision_minus_shuffle_correct_logp=+0.1071`; the positive local log-prob did
not rescue the representation guard. No 500-step expansion or capability
evaluation is allowed. Raw artifacts remain on the V100 HDD and are bound by
`experiments/qwen3b_community_eval_20260805/projector_residual_screen_v1/REPAIR_RESULT_POINTER_20260807.json`.

Decision: gated residual does not address the shared early receiver-facing
collapse. Keep `previous_best` unchanged and move to one DeepSeek-transferable
projector scale/auxiliary-objective variable with a matched CE-only control.
Gate D remains **NO-GO**.

## Package 15T exact K3 causal margin 0.5 (2026-08-07)

Before this screen, the formal Qwen3B supervision path was audited. The frozen
order joins real `train_mix.jsonl` rows by ID/source SHA; it contains 2,000
grounding and 2,000 short-answer records, 1,066 unique grounding coordinates,
zero `[500,500]` targets, and no same-image cyclic negatives in an 8-row batch.
The target is point-derived click text from ShowUI; the current source pack does
not preserve an independent bbox field, so future prose must say
“point-derived click supervision,” not “independent bbox join.” Historical
cache-only stripped-receiver runs that used the old `[500,500]` fallback remain
diagnostic only. See `SUPERVISION_PROVENANCE_AUDIT_20260807.json`.

The preregistered 15T screen held exact K3/MoonViT-V2, step0, order, cache,
receiver, resolution and health schedule fixed. It used projector LR `5e-5`
and increased the within-batch correct-vs-shuffled hinge lambda from `0.1` to
`0.5`. Raw artifacts are at
`/run/media/ezra/13D010B6FDBC1A06/data/qwen3b_contract/architecture_controls/local_v2_exact_k3/health_run_100_v2_exact_causal_l05_20260807`,
with pointer
`architecture_controls/local_v2_exact_k3/CAUSAL_MARGIN05_RAW_POINTER_20260807.json`.
Independent health verification passed, but the arm stopped at step 2 with
onset `[1,2]`. Geometry stayed healthy (projector/receiver spread ratios
`0.990/0.986`, rank ratios `1.000/1.000`), while
`vision_minus_shuffle_correct_logp` improved only from `-0.2404` to `-0.0515`
and final vision/shuffled preference was `0.625/0.625`. The loss rose
`4.8526→5.6925`; no capability claim or ScreenSpot run is allowed.

Interpretation: stronger paired supervision moves the causal direction toward
zero but does not create correct-image grounding in the frozen 3B receiver.
Combined with 15R, geometry preservation and grounding are independent gates.
Do not sweep lambda or expand to 500 steps. Keep `previous_best=step0`, retain
the exact K3 projector as the structural candidate, and next test one
DeepSeek-transferable placeholder/position or receiver-distribution alignment
variable before deciding whether the Qwen proxy has exhausted its value.
