# Projector / receiver 机制实验记录

更新：2026-08-07

本文是持续更新的机制账本，和最终 benchmark 表并行保存。它记录哪些变量改变了训练健康、哪些变量改变了正确图像归因，以及哪些看似积极的数值没有转化为自由生成能力。所有能力判断仍以冻结的 ScreenSpot、TextVQA、DocVQA、OCRBench 和 synthetic 合同为准。

## 当前总体判断

MoonViT 特征可以经过 canonical 4096 projector 注入纯文本 Qwen2.5 和 stripped-native Qwen3.5 receiver，projector-only backward、自动止损、checkpoint、恢复和生成链路均已跑通。当前最强的 Qwen2.5-7B 轨迹在 32 条真实答案 teacher-forced probe 上得到正的 correct-image 对 shuffled margin；同一 checkpoint 的自由生成只呈现很弱的 ScreenSpot 因果增益，仍受文本与坐标先验主导。

项目已经从“接口能否运行”推进到“正确图片是否稳定改变答案”的阶段。还没有任何 checkpoint 满足正式晋升合同，`previous_best` 不变。

## Receiver 差异

| Receiver | 本机能力边界 | 观察到的现象 | 当前解释 |
|---|---|---|---|
| Qwen2.5-3B 纯文本 | projector-only 训练可运行 | 高学习率在 step 1--2 出现 common-direction collapse；降低学习率可保住 rank/spread，正确图归因仍没有稳定出现 | 3B 同时暴露更新尺度和 receiver readout 瓶颈 |
| Qwen2.5-7B 纯文本 | FP16、16/240 visual tokens 可训练；完整 1,272 条 ScreenSpot 四条件评测可运行 | CE 快速下降；paired margin λ=0.5 能改善 teacher-forced attribution；自由生成只有弱 vision-shuffle 增益，vision 不胜 blind | 增大纯文本容量有帮助，但不足以自动解读外部视觉 token |
| Qwen3.5-4B stripped-native | 16-token BF16 input-gradient 可运行；full-token FP16 曾 NaN | 原生视觉模块完全绕过后，外部 token 没有稳定 correct-image 优势 | 视觉预训练 prior 在 4B 规模没有自动救活接口 |
| Qwen3.5-9B stripped-native | 16--240 token BF16 input-gradient/probe 可运行；projector 训练在 V100 OOM | vision-blind 常为正，vision-shuffle 在样本和 token 长度间换符号；修复 decoding 后 8 条 ScreenSpot click 为 0% | receiver 会响应视觉 token，内容归因仍未对齐；仅作 prior 诊断 |

Qwen3.5 路径绕过原生 vision tower、merger 和 visual forward。它只能回答视觉预训练过的语言权重是否更容易接收新的视觉 token，不进入纯文本 Qwen 排行榜，也不能替代 DeepSeek 证据。

## V1 / V2 与社区 projector 形状

MoonViT V1 使用 1152 维输入和 community-shaped 4608 hidden MLP；V2 使用 1024 维输入和 K3/DeepSeek 目标的 exact projector 变体。两者都输出 canonical 4096，以保持 DeepSeek-V4-Flash-0731 接口。

在同一 3B 健康合同下，V1 在 step 2 的 projector/receiver effective-rank ratio 降到约 `0.264/0.212`；exact V2 在合同学习率下约为 `0.910/0.830`。同一 Qwen3.5-9B、8-sample、240-token probe 中，V1 的 vision-shuffle 为 `+0.0620 ± 0.4185`，V2 为 `-0.0748 ± 0.4520`。差异被样本方差覆盖。

现有证据削弱了“V2 embedding 压缩是失败根因”的判断。社区 GLM-5.2V 的 K2.6/MoonViT3d 视觉侧和 6144 receiver 输出仍与本项目 4096 DeepSeek 目标不同，因此当前实现是社区形状参考加 DeepSeek 接口约束，尚无“完全同构社区 projector”的结论。

## Token 数量、压缩和位置

测试覆盖 16、32、64、128、240 个 visual tokens，以及 prefix、uniform、mean-pool 和 Qwen3.5 3D mRoPE。主要现象如下：

- 单样本与 8-sample 结果会随 token 数换符号，短上下文正 margin 不能外推到完整序列。
- 32-sample step0 probe 中，full240、prefix16、uniform16、mean-pool16 的 vision-shuffle 95% CI 全部跨 0；mean-pool 点估计最高。
- mean-pool 的短训练曾出现约 4,292 的梯度峰值，覆盖更多图像区域同时带来数值风险。
- Qwen3.5 3D mRoPE 与普通连续位置的 8-sample attribution 接近，没有单独修复 paired gap。

这组结果支持 token 覆盖、顺序和尺度共同影响 receiver alignment。下一项固定同一 checkpoint，在冻结 ScreenSpot 子集上直接比较 16-token mean-pool 与 240-token full sequence，避免用不同训练记录或不同模型混淆结论。

## CE、表征健康与真实能力的分离

项目反复观察到 CE loss 下降而视觉归因不升：

- 3B baseline 在两步内 CE 下降约 41%，projector/receiver RMS 同时上升，rank 和 spread 同时下降，自动 guard 在 `[1,2]` 止损。
- 3B 小学习率能保住 rank/spread，vision-shuffle correct-logp 仍为负。
- 7B CE-only 3-step 可把 CE 从 `0.2381` 降到 `0.0094`，旧伪监督条件下 attribution 反而变差；真实答案合同也出现 CE 改善而 paired CI 跨 0。
- λ=0.5 paired objective 在 32-sample teacher-forced probe 上得到 vision-shuffle `+0.4874`，95% CI `[+0.1423,+0.8786]`，自由生成仍集中在窄坐标先验。

因此健康指标和能力指标分开判定。RMS、spread、rank、Gram、gradient 与 NaN/Inf 只回答表示是否可训练；vision/blind/shuffled/random-projector、click-in-box、VQA 和 OCR 指标回答模型是否真正使用正确图片。

## Qwen2.5-7B 完整公共 ScreenSpot

λ=0.5 checkpoint 使用 16 个 mean-pool tokens、scale `0.1`、固定 grounding prompt、贪心解码和四条件完成完整公共 ScreenSpot 1,272 条评测。所有条件 parse rate 都为 100%。

| 条件 | click-in-box | Accuracy@50 | Accuracy@100 | Accuracy@200 | mean center distance |
|---|---:|---:|---:|---:|---:|
| vision | 3.30% (42/1272) | 1.18% | 5.19% | 15.33% | 404.38 |
| blind | 3.46% (44/1272) | 1.02% | 5.03% | 15.09% | 409.71 |
| shuffled | 2.67% (34/1272) | 1.02% | 4.87% | 15.02% | 406.10 |
| random projector | 2.91% (37/1272) | 1.26% | 4.87% | 14.94% | 405.74 |

vision-shuffled click-in-box 的 paired improvement 为 `+0.629` 个百分点，独立分层 verifier 的 2,000-bootstrap 95% CI 为 `[+0.157,+1.179]` 个百分点。vision-blind 为 `-0.157` 个百分点，CI `[-0.943,+0.629]`。距离和 Accuracy@50/100/200 的关键 paired CI 均未同时通过。

分层上，iOS 的 vision-shuffled click 改善为 `+1.96` 个百分点，CI `[+0.39,+3.92]`；Android 的 vision-blind click 为 `-1.62` 个百分点，CI `[-3.24,-0.40]`。Web 的 vision-blind Accuracy@50 也显著下降。整体弱正信号并不均匀，不能当作通用 GUI grounding。

社区 GLM-format 参考中，当前 parse rate、Accuracy@200 和 mean distance 表面达到或接近公开数值；Accuracy@50、Accuracy@100 未达到，vision 也没有显著胜过 blind。项目不得声称达到社区 GLM-5.2V metric-aligned baseline。

完整原始生成保存在 V100 数据盘；Git 保存分类 summary、SHA 指针和 verifier。该 checkpoint 维持 `reject_current_candidate`，`capability_claim_allowed=false`，迁移标签为 `transferable_with_runtime_validation`。

## 设计依据与下一步

当前证据把候选瓶颈排序为：

1. 视觉 token 与 receiver 训练分布的对齐，包括尺度、顺序、placeholder 和位置语义。
2. projector 目标只优化答案 CE，容易形成 coordinate soft prompt；paired image attribution objective 已显示局部价值。
3. 16-token mean-pool 可能丢失 grounding 所需的局部布局；需要与 240-token full sequence 做严格匹配的生成对照。
4. 纯文本 receiver 的容量影响存在，7B 仍受文本先验支配；视觉预训练 receiver prior 也没有自动解决问题。
5. V1/V2 版本差异目前弱于监督接口和 receiver alignment 差异。

最近的最小实验顺序：

1. 在冻结 ScreenSpot50 上运行同 checkpoint 的 16-token mean-pool 对 240-token full sequence。
2. 若 240 token 的 vision-shuffled 改善且没有 vision-blind 退化，再扩到完整 1,272 条。
3. 若 token 数无效，测试一个与 DeepSeek 兼容的 projector/辅助目标变量，并保留 matched CE-only control。
4. 候选只有同时通过 vision-blind、vision-shuffled、完整 ScreenSpot 和通用 VQA/OCR 合同，才进入 DeepSeek 正式配方。

任何付费 DeepSeek 操作继续等待明确授权。

## 2026-08-07 token-count matched screen

同一个 Qwen2.5-7B λ=`0.5` checkpoint 在冻结的 GLM-format 50 条 subset 上改用 240-token full sequence，其他配置完全不变。四条件 parse rate 都为 100%；vision/blind/shuffled/random 的 click-in-box 为 `10%/10%/10%/8%`，Accuracy@50 为 `0%/2%/2%/0%`，Accuracy@100 为 `6%/6%/6%/6%`，Accuracy@200 为 `18%/18%/20%/18%`。

独立 verifier 重算的中心距离均值为 `399.51/415.11/396.78/397.02`。vision-blind 的距离改善均值 `+15.59`，CI `[-13.51,+47.15]`；vision-shuffled 为 `-2.74`，CI `[-13.58,+9.96]`。click-in-box 的 vision-blind 和 vision-shuffled 都是 `0`，CI `[-6,+6]` 个百分点。

这条结果没有支持“16-token mean-pool 的压缩是单一主要 grounding 瓶颈”这一假设。它支持停止扩大 token 数量筛选，优先测试 projector/辅助目标与 receiver 分布对齐。原始 evaluator 的 `center_distance` 统计实际存在；先前消费者读取了不兼容的摘要 key，独立 category verifier 用于分类与交叉核对，raw pointer 记录了这一 schema 边界。

## Package 15R gated residual：修复实现缺陷后仍被几何守卫否决

15R 的原始预注册 contract 保持不变。失败记录明确分开：源码漂移在训练前被拒绝；严格冻结源码的 gated arm 在 gate=0 时把 `residual.weight` 的链式法则零梯度误报成错误；将修复 runner 混入冻结 worktree 后又被 runner SHA 绑定拒绝。修复实现只放宽 `residual.weight` 在 `residual_gate==0` 的首步零梯度，gate 本身和其他 projector 参数仍必须 finite、non-zero，并新增了真实 backward 回归测试。

在当前 main 源码和独立 repair contract 下，matched canonical CE-only control (`baseline_none_repair_v3`) 与 gated residual (`gated_residual_repair_v2`) 都通过独立 health verifier，然后在 optimizer step 2 自动止损。两者都出现 projector/receiver RMS 上升、relative spread 下降的 adverse trend；gated 的 step-2 projector/receiver spread ratios 为 `0.2690/0.2254`，effective-rank ratios 为 `0.5008/0.3611`，虽然 `vision_minus_shuffle_correct_logp=+0.1071`，仍未通过几何硬门槛。首步允许的零梯度只在 `residual.weight`，step 2 已无允许零梯度参数。

这条证据支持“修复训练守卫后，gated residual 仍然没有改变共同的 receiver-facing collapse”，反驳“zero-init residual 旁路本身足以保留图像差异”。健康指标仍然只是训练安全证据，不能当作 grounding 能力；该 arm 不进入 500-step、ScreenSpot 或 DeepSeek 候选。后续应把研究预算转向一个可迁移的输出尺度/辅助目标单变量，并保留 matched CE-only control。
