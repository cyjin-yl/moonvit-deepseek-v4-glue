# ARCHIVED — Community-scale ablation matrix

The registered matrix attempt set is archived. The completion rule was satisfied for the 11 MATRIX_SUMMARY rows by a valid result or immutable failure artifact, but this does not mean every receiver completed the full community two-epoch budget. Qwen2.5-7B V2 is the only stable external arm that reached 57,600 examples; all external arms failed the visual capability gate.

This is the complete comparison table for the current V100 phase. A row is a planned model condition; it is not a
result until its checkpoint has been trained and evaluated under the same contract.

## 项目级完成标准：完整对比矩阵（2026-08-08 修订）

“至少一组”不再是本项目的成功条件。当前目标是完成并发布整个已注册对比矩阵：配置中的每个 active arm 都必须在同一社区规模合同下实际尝试，并留下正式结果或不可变的失败记录；不能因为某一臂表现较好就提前结束，也不能静默跳过显存不足、实现失败或因果门失败的臂。

每个外部 MoonViT arm 都按 receiver 单独重新训练 projector，并统一运行相同的数据顺序、examples-seen 节点、图像预处理、prompt、parser、greedy decoding 和 vision/blind/shuffled/random_projector 条件；同时保存 step0、previous_best、current_candidate、健康日志、checkpoint、逐样本 prediction、正式 ScreenSpot50 paired bootstrap 和 artifact manifest。原生 Qwen VLM 只作独立阳性对照，历史 0.5B/3-step/replay/geometry 结果只作 archived 机制证据。

矩阵完成的含义是：所有 active rows 都出现在 MATRIX_SUMMARY.json，每行都标注 result、causal_pass、failure_reason 或 resource_limit。只有矩阵整体完成后，才能选择 transferable candidate；DeepSeek-V4-Flash-0731 Gate D 仍是独立的最终门，不会被 Qwen 代理结果替代。

## Model and receiver matrix

| ID | Receiver condition | Visual path | Trainable part | Role | Current status |
|---|---|---|---|---|---|
| `qwen25_05b_historical` | Qwen2.5-0.5B pure text | MoonViT V1/V2 | projector only | old toy capacity reference | archived; not community-comparable |
| `qwen25_3b_v1` | Qwen2.5-3B-Instruct pure text | MoonViT V1 | projector only, canonical 4096 + fixed receiver adapter | official 3B proxy baseline | queued |
| `qwen25_3b_v2` | Qwen2.5-3B-Instruct pure text | MoonViT V2 | projector only, canonical 4096 + fixed receiver adapter | official 3B proxy baseline | queued |
| `qwen25_7b_v1` | Qwen2.5-7B-Instruct pure text | MoonViT V1 | projector only, canonical 4096 + fixed receiver adapter | first active capacity run | running after cache |
| `qwen25_7b_v2` | Qwen2.5-7B-Instruct pure text | MoonViT V2 | projector only, canonical 4096 + fixed receiver adapter | matched tower control | queued |
| `qwen35_4b_v1_stripped` | Qwen3.5-4B language receiver, native vision disabled | MoonViT V1 | projector only | tests whether visual pretraining prior helps read an external tower | queued |
| `qwen35_4b_v2_stripped` | Qwen3.5-4B language receiver, native vision disabled | MoonViT V2 | projector only | matched V2 control | queued |
| `qwen35_9b_v1_stripped` | Qwen3.5-9B language receiver, native vision disabled | MoonViT V1 | projector only | larger receiver capacity probe | queued if V100 memory permits |
| `qwen35_9b_v2_stripped` | Qwen3.5-9B language receiver, native vision disabled | MoonViT V2 | projector only | matched V2 control | queued if V100 memory permits |
| `qwen35_4b_native` | Qwen3.5-4B native VLM intact | native Qwen vision tower + native merger | frozen positive control | confirms the benchmark can see a working VLM | separate leaderboard |
| `qwen35_9b_native` | Qwen3.5-9B native VLM intact | native Qwen vision tower + native merger | frozen positive control | larger native-VLM reference | separate leaderboard if runnable |
| `qwen14b_capacity` | approximately 14B receiver | matched external MoonViT V1/V2 | projector only | capacity-only probe | only if a local checkpoint and V100 memory permit |
| `deepseek_v4_flash` | DeepSeek-V4-Flash-0731 real receiver | matched selected MoonViT tower | projector only | final target | blocked by real-weight Gate D; no paid run |

The native Qwen VLM rows are positive controls, never external-MoonViT projector results. The stripped-native rows use
the Qwen language receiver and its tokenizer only; they bypass Qwen's visual tower, merger and multimodal forward.

## Visual and training ablations

| Factor | Fixed levels | Interpretation |
|---|---|---|
| MoonViT version | V1 (1152-d output), V2 (1024-d output) | separates version/compression from receiver and supervision effects |
| visual input | correct image, blind text-only, deterministic shuffled image, random projector | separates image use from text prior and random signal |
| projector structure | V1 community-shaped `LayerNorm -> flatten -> Linear -> GELU -> Linear`; V2/K3 exact shape; later normalization/residual/gated residual one variable at a time | tests whether the failure is structural rather than capacity-only |
| objective | CE-only community baseline; paired-margin; fixed-budget replay; geometry auxiliary objective | margin/replay/geometry never replaces the CE-only matched control |
| token selection | prefix, uniform, mean-pool, 16, 64, 128, 240/256, full cache | tests whether grounding fails from token placement or token budget |
| resolution/domain | fixed 448 training side; fixed evaluation preprocessing; registered higher/lower resolution and background-shift arms | no post-hoc image preprocessing changes |
| trainable language weights | projector-only mainline; top-layer LoRA or top-layer unfreeze only as matched diagnostics | every LoRA/unfreeze arm retains a projector-only control and cannot replace it |
| initialization | exact step0, random projector, previous best, current candidate | checkpoint identity is recorded; cross-receiver projector reuse is not a training result |

## Community-matched training contract

Every comparable arm uses the same frozen data order and preprocessing:

- about 66k short-answer image-text examples, with the local 59,198-row pack reported as an explicit shortfall;
- global batch 64, constant learning rate `5e-4`, AdamW, zero weight decay, projector-only;
- same prompt template, image placeholder, token cap, answer masking, greedy decoding and checkpoint format;
- exposure nodes `4k/8k/16k/32k/57.6k/66k/132k` examples seen, reporting the actual maximum when the local data pack is short;
- the community-reported grokking reference near optimizer step 900 / 57.6k examples seen is a checkpoint, not a success claim.

## Community-matched benchmark contract

At every candidate checkpoint, run the same four conditions: `vision`, `blind`, `shuffled`, and `random_projector`, plus
`step0`, `previous_best`, and `current_candidate` roles.

- `screenspot_glm50_v1`: frozen 50-row GLM-format metric-aligned public subset;
- complete public ScreenSpot: 1,272 rows, split by overall/text/icon-widget and platform;
- exact generated grammar: `click(start_box=[x, y])`, integer coordinates in `[0, 999]`, greedy decoding;
- report parse rate, mean/median/p90/minimum center distance, Accuracy@50/@100/@200, and click-in-box accuracy;
- report paired `vision-blind`, `vision-shuffled`, `trained-random_projector`, and candidate delta with at least 2,000 bootstrap replicates;
- after the first causal screen, run TextVQA soft accuracy, DocVQA ANLS, OCRBench accuracy, synthetic paired preference/generation,
  and language-retention tests;
- a row cannot be called a visual improvement unless vision beats both blind and shuffled with the registered paired CI and
  the matched benchmark budget is respected.

The current active run is only the first row that can be executed on the V100 immediately. The table is the goal; the first
run is not the conclusion.

## Evaluation cadence and growth curves (2026-08-08)

Training health and capability evaluation are separate streams. Health is written every optimizer step and may stop a broken
run. At the frozen early/periodic nodes (`1/2/5/10/20/30/50/75/100`, then every 50 steps), the runner also writes a cheap
teacher-forced attribution probe to `train_eval.jsonl`: fixed held-out rows, correct image versus deterministic shuffled image,
answer loss, and the loss gap. This is an early warning signal, not a free-generation capability score.

At the community examples-seen nodes (`4,096/8,192/16,384/32,768/57,600/59,136/132,480`), each surviving checkpoint gets a
multi-task generation evaluation rather than a ScreenSpot-only check. The curve includes ScreenSpot GLM-format and click-in-box
metrics, TextVQA soft accuracy, DocVQA ANLS, OCRBench accuracy, synthetic paired metrics, and language retention. Every task is
run under the same `vision/blind/shuffled/random_projector` conditions where its contract supports them; raw per-sample JSONL,
summary JSON, CSV rows and SVG growth charts are retained. A falling training loss or a healthy projector cannot substitute for
an improving multi-task curve.
