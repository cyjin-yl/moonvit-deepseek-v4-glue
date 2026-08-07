# Projector / receiver 机制实验记录

更新：2026-08-08

本文是持续更新的机制账本，和最终 benchmark 表并行保存。它记录哪些变量改变了训练健康、哪些变量改变了正确图像归因，以及哪些看似积极的数值没有转化为自由生成能力。所有能力判断仍以冻结的 ScreenSpot、TextVQA、DocVQA、OCRBench 和 synthetic 合同为准。

## 当前总体判断

MoonViT 特征可以经过 canonical 4096 projector 注入纯文本 Qwen2.5 和 stripped-native Qwen3.5 receiver。需要分开理解各路径：3B formal runner 跑通自动止损和绑定 checkpoint/resume；7B/Qwen3.5 多数是 3-step diagnostic；DeepSeek 只在 tiny 软件模型上跑通 20-step save/resume/generate。完整 0731 尚未运行。当前最强的 Qwen2.5-7B 轨迹在 32 条真实答案 teacher-forced probe 上得到正的 correct-image 对 shuffled margin；同一 checkpoint 的自由生成只有弱 ScreenSpot vision−shuffle 信号且不胜 blind，仍受文本与坐标先验主导。

项目已经从“接口能否运行”推进到“正确图片是否稳定改变答案”的阶段。还没有任何 checkpoint 满足正式晋升合同，`previous_best` 不变。

## 新增数值机制记录：Qwen2.5-7B V1 step-2 NaN

社区规模 V1 重跑在第一步完成了真实 7B forward/backward：loss 12.7393，projector
gradient norm 259.8，裁剪后为 1.0，输出 RMS 0.199，relative spread
0.0671，没有 NaN。第二个 optimizer step 的 projector/receiver 输出、loss 和梯度
全部变成 NaN，hard health guard 在 128 examples seen 处停止并保存了
failure-checkpoints/step-000002、optimizer/RNG state 和两行 health log。检查保存的
optimizer state 后，exp_avg/exp_avg_sq 已经是 FP16 NaN；因此这不是“图片没有
信息”的能力结论，而是 V100 上直接用 FP16 projector + AdamW state 跑 5e-4 的
数值路径失败。冻结的 Qwen receiver 仍保持 FP16，projector 和 optimizer state 已改为
FP32（视觉输出在 merge 边界再转成 receiver dtype），V1 retry4 将在 V2 后重跑。
这个修复不改变视觉塔、数据、token、prompt、global batch 或学习率；它只消除
低精度 optimizer state 的实现性混淆。该失败记录与 raw artifact 已提交到
failure_artifacts/qwen25_7b_v1_attempt3/，不能写进能力排行榜。

### V1 retry4 step-2 数值修复验证（2026-08-08）

retry4 保持同一视觉塔、初始化、数据顺序、学习率和 global batch，���把 projector 与 AdamW
state 从 FP16 改为 FP32。它在 step1/2（64/128 examples seen）均保持 finite；step2 loss
为 `12.4078`，projector/receiver RMS ratio 为 `3.0678/3.2231`，relative-spread ratio
为 `0.4167/0.4029`，未触发 critical guard。换句话说，之前的 step-2 NaN 是低精度优化器
状态的工程故障，修复后能继续训练；RMS 上升和 spread 变化仍只说明训练健康，不能说明模型已经
学会使用正确图片。原始两行 health、训练日志、validation manifest 与 SHA 清单保存在
`experiments/community_scale_model_ablation_20260808/interim_artifacts/qwen25_7b_v1_retry4_step2/`。

截至 step17（1,088 examples seen），CE loss 继续下降到 `3.3788`，但 projector/receiver RMS ratio 已到 `30.58/32.41`；relative-spread ratio 仍为 `0.4605/0.4633`，且所有值 finite。这个轨迹把“loss 降低”和“表示尺度膨胀”同时展示出来：即使尚未触发 50× critical guard，也不能把它解释成视觉对齐。step17 的原始快照保存在 `experiments/community_scale_model_ablation_20260808/interim_artifacts/qwen25_7b_v1_retry4_step17/`，后续要看它是否在前 100 steps 内触发自动止损或保持健康。

### V1 retry4 自动止损结果（step33）

step33（2,112 examples seen）触发了预注册的 critical guard：receiver output RMS ratio 为 `50.7792`，超过 `50×`；projector ratio 为 `47.8930`，relative-spread ratio 为 `0.4563/0.4593`，CE 为 `3.4664` 且没有 NaN/Inf。也就是说，FP32 只修复了“第二步就数值爆炸”，没有修复 projector 的尺度漂移：它可以把 CE 降下来，却在 33 步内把 receiver 输入放大到不可接受范围。该臂现在是正式的 `failed_health_guard`，不能继续从 failure checkpoint 训练，也没有进入视觉能力排行榜。

此次运行暴露一个 checkpoint 合同缺口：`checkpoint-every=64` 导致 onset（step33）之前没有周期性 healthy checkpoint；failure checkpoint 和 step0 projector 均已保留，但后续矩阵 runner 必须在 step0/每个早期 health 节点保存可回滚的 optimizer/RNG 状态，不能只依赖 64-step 周期。
该缺口已在后续 runner 中修复：健康 checkpoint 采用冻结的 step0/1/2/5/10/20/30/50/75/100 及每50步 schedule，guard failure 会额外写出 `STOP_REASON.json` 指向最近健康回滚点；三组相关测试在正确环境下 `12 passed`。

## 主线重置：用社区训练量做条件消融，而不是继续堆 verifier（2026-08-08）

历史脚本、权重和数据 verifier 已经足够支撑可信实验；继续重复它们不会回答当前最重要的问题：视觉塔版本、
receiver 容量和视觉预训练先验中，到底哪一个决定外接 projector 能否学会图像。后续把模型消融放在主线，
把本节的健康诊断放在护栏位置。

固定 arm 包括：V1（MoonViT-SO-400M/K2.6-lineage）、V2（K3/MoonViT-V2）、无视觉、random projector；
纯文本 Qwen2.5-3B/7B；去掉原生视觉层的 Qwen3.5-4B/9B；以及独立的原生 Qwen VLM 阳性对照。每个
receiver×tower 都重新初始化 projector，不能把旧 receiver 的 projector 直接搬过来。所有 arm 共享真实图文
数据顺序、预处理、token 上限、prompt、parser 和 greedy decoding，并在 vision/blind/shuffled/random 四条件
下报告结果。

训练量按社区 GLM-5.2V 的数量级复现：约 66k 条短 QA、global batch 64、constant LR `5e-4`、约 2 epochs，
约 2,070 optimizer steps；社区报告的能力突变约在 step 900（约 57.6k examples seen）。因此 step 20/100
只用于发现 NaN 或塌缩，不能作为“没有视力”的结论。主节点为 4k/8k/16k/32k/57.6k/66k/132k examples seen，
每个节点保留完整 raw rows、bootstrap CI 和语言保持结果。

表征 rank/spread/RMS、梯度和 NaN/Inf 只用于在线止损与回滚；CE 下降或表示健康都不等于视觉能力。只有
ScreenSpot click-in-box、Accuracy@50/100/200、TextVQA/DocVQA/OCRBench 以及 vision−blind/shuffled 的配对 CI
同时支持，才允许称为 grounding 改进。旧 3-step/32-row/geometry/replay 结果在后文统一视为 **archived**，
用于解释失败机制，不再阻塞社区规模训练。

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

这组结果支持 token 覆盖、顺序和尺度共同影响 receiver alignment。固定同一 checkpoint 的 16-token mean-pool 与 240-token full-sequence 对照已经完成：240 token 没有改善 vision−shuffle click，也没有解决 blind 竞争，因此 token 压缩不是单一主要根因。

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
3. 16-token mean-pool 可能丢失局部布局，但 matched 240-token 生成对照没有改善因果 grounding；继续扩大 token 数已降为低优先级。
4. 纯文本 receiver 的容量影响存在，7B 仍受文本先验支配；视觉预训练 receiver prior 也没有自动解决问题。
5. V1/V2 版本差异目前弱于监督接口和 receiver alignment 差异。

冻结后的最小实验顺序：

1. 把 3B formal runner 的 health/stop/rollback/bound-checkpoint 能力抽到共享 7B 训练入口。
2. 运行 7B 100-step formal causal screen；任一 health critical 或两个因果 CI 下界不为正就停止。
3. 只有 100-step 通过，才按同预算进入 500/2000；否则不再用加训练量包装失败。
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

## Package 15T：更强 paired objective 只修正方向，不建立因果

在 exact K3/MoonViT-V2 projector 的 geometry-safe 小学习率 `5e-5` 上，固定 step0、order、cache、receiver 和 100-step health 合同，只把 within-batch correct-vs-shuffled hinge 的 lambda 从 `0.1` 提到 `0.5`。独立 verifier 确认 step 0/1/2 的 3 个 probe 与 3 个 checkpoint 完整存在，训练在 `[1,2]` 自动止损。

step 2 的 projector/receiver relative-spread ratio 为 `0.9903/0.9859`，effective-rank ratio 为 `1.0002/1.0001`，所以表示几何没有崩；但 vision/shuffled preference 都为 `0.625`，`vision_minus_shuffle_correct_logp=-0.05155`，仍未通过 causal guard。相较 parent lambda=`0.1` 的 `-0.06098`，λ=`0.5` 只带来有限的方向改善；CE 从 `4.8526` 升到 `5.6925`，没有进入 ScreenSpot 或通用能力评测。

这组结果把“几何健康”和“真实 grounding”进一步分开：更强的 paired loss 能把错误方向推近零，却不能让冻结纯文本 3B 选择正确图像。继续增大 λ 或延长该 arm 没有判别力；下一项应测试 placeholder/位置语义、输出尺度与 receiver 分布的可迁移单变量，或者冻结 Qwen 代理训练配方推进 DeepSeek runtime Gate。

## 正式 Qwen3B 监督 provenance 审计

当前 `train_qwen3b_proxy.py` 的正式路径按 order manifest 的 ID、source row 和 SHA 重建 `train_mix.jsonl`，4,000 行中 grounding/short-answer 各 2,000；grounding 有 1,066 个唯一 click 坐标、0 个 `[500,500]`，每个训练 batch 的循环负例都没有同图 SHA。这个证据支持“15R/15T 的 CE supervision 不是统一坐标 fallback”。数据上游把 ShowUI `point` 转成 click 字符串，却没有把独立 bbox/raw point 字段保存在训练 manifest；因此当前结论只能写成 point-derived click supervision，不能声称独立 bbox join。旧 stripped-receiver cache-only 运行的 `[500,500]` fallback 已在文档中单独标为历史诊断，不得混入正式排行榜。
## DeepSeek 接口旁路的可归因验证

tiny DeepSeek screen 的首版失败已保留：随机 tiny 配置的 `tid2eid` 全零，routing ID 改动没有任何输出差异。修复版在预注册 synthetic 非退化 hash 表上重跑，结果确认两条旁路都实际进入 receiver：embedding 固定时 routing-ID 消融的最大 logit 差为 `0.0015277863`，embedding/routing 固定时 position-ID 消融为 `0.0193590522`。同时，视觉 placeholder 展开、连续位置、视觉 label mask、projector 输入梯度和冻结语言梯度均通过。

机制含义有明确边界。Qwen 训练里的“image token 能改变 receiver”可以由这条软件接缝复现，但 full DSV4 仍需加载其真实 `tid2eid` 表，确认 `129279` 及视觉 span 在 43 层 Hash-MoE 中的实际路由，并在目标 FP4/FP8 runtime 上复测。tiny synthetic route table 不能作为视觉能力证据；它只排除了 wrapper 丢失 routing/position 信息这一类工程错误。

这条经验与既有现象合并后的工作假设是：projector collapse、CE 降低而 vision/shuffle attribution 不升、token 数量扩展无效和 receiver prior 差异都指向“接缝连通后，冻结语言主干能否把视觉 token 解码成正确任务监督”这一瓶颈。后续实验仍必须把健康指标和真实 grounding 指标分开记录。

## Qwen2.5-7B：V1 family proxy 与 exact V2 的直接对照（2026-08-08）

为了回答“社区 GLM-5.2V 使用的 K2.6/MoonViT-3d 家族是否更容易被 receiver 读取”，固定 Qwen2.5-7B、32 条真实答案、同一 derangement、mean-pool 16、scale `0.1`、BF16、LR `5e-5`、3 steps 和 receiver adapter，仅替换 V1/V2 视觉塔与 projector。V1 使用 `MoonViT-SO-400M` family proxy 的 1152 维输入、community-shaped pre-norm MLP；V2 使用 exact K3/MoonViT-V2。

V1 CE-only 的 `vision−shuffle` 为 `+0.00615`，CI `[-0.01760,+0.03182]`；V1 λ=`0.5` 为 `+0.01145`，CI `[-0.02580,+0.04766]`。两臂 `vision−blind` 的 CI 为正，random-projector 差值显著为负，说明 projector 输出会改变 receiver，且训练后的 projector 不是随机映射；正确图与 shuffled 图仍没有稳定区别。

V1 λ=`0.5` 相对 V1 CE-only 的差为 `+0.00530`，CI `[-0.04882,+0.05758]`；相对 V2 λ=`0.5` 的差为 `-0.47600`，CI `[-0.87349,-0.13102]`。这是一条重要的版本结论：V1 family proxy 没有改善 grounding，V2 的 embedding 压缩不能单独解释失败，V2 在同一 receiver 和同一监督合同下反而更强。后续优先级转向 receiver readout/alignment、监督目标和 projector 输出尺度，停止继续做 V1/V2 token 数量扫参。
## Qwen3.5-4B：完整 external-MoonViT 对照表（2026-08-08）

本轮把 Qwen3.5 的“有视觉预训练的 receiver”从单样本/8 样本诊断提升到固定
`screenspot_glm50_v1` 的完整 50 条四条件对照。原生 visual tower、merger 和
pixel path 全部绕过，外部 Kimi-K3/MoonViT-V2 特征经 exact K3 projector 输出
4096 维，再通过固定 4096→2560 grouped-signed adapter 送入 Qwen3.5 language
receiver。三种 projector 角色共享 step0、数据顺序、32 条真实答案训练 probe、
16-token mean-pool、scale、prompt、parser 和 greedy decode。

| 角色 | V click | blind click | shuffled click | random click | V A50/A100/A200 | V mean distance | V−B click CI | V−S click CI |
|---|---:|---:|---:|---:|---|---:|---|---|
| step0 | 2% | 2% | 2% | 4% | 0/6/24% | 469.54 | 0 `[-6,+6]` pp | 0 `[0,0]` pp |
| CE-only | 0% | 2% | 4% | 4% | 0/8/22% | 470.49 | −2 `[-6,0]` pp | −4 `[-10,0]` pp |
| λ=0.5 paired margin | 4% | 2% | 4% | 4% | 2/10/20% | 484.96 | +2 `[-4,+10]` pp | 0 `[0,0]` pp |

训练 health 与自由生成再次分离。CE-only 的 CE `8.3974→7.5631`，
`vision−shuffle` `-0.1361→-0.0165`；paired-margin 的 CE `8.3974→8.1195`，
`vision−shuffle` `-0.1361→+0.0162`。这两个训练信号都没有转化成正的
ScreenSpot paired causal CI。paired-margin 在 teacher-forced probe 上把方向推近零，
生成时仍无法区分正确图与确定性错误图。三臂均 `native_vision_forward_calls=0`，
所以这组结果确实测到 external projector 路径，不包含 Qwen3.5 原生视觉塔的结果。

机制结论：视觉预训练 receiver prior 能让外部 token 改变分布，但“receiver 有视觉
预训练”本身不足以完成 projector alignment；projector 仍需要针对 receiver 重新训练，
而训练目标必须直接约束 correct-image attribution。由于固定 50 条 causal gate 未通过，
本轮不扩展完整 1,272 条公共集，也不把 Qwen3.5 checkpoint 送入 DeepSeek 候选。

## DeepSeek-V4-Flash-0731：多模态残留接口假设

公开 tokenizer 的 added-token 区保留图像与区域标记，包括 `<｜image｜>` ID `129279`、
`<｜image2｜>`、`<｜rl_image_start｜>`、`<｜rl_image_pad｜>`，以及 415 个
`place_holder_mm_span`、box/point/ref/polygon 标记。公开 config 没有 `vision_config`，
HF 文件树没有视觉塔或 projector。这种“词表/路由接口保留、公开视觉模块缺失”的组合
使内部曾经接过多模态训练成为合理假设，但它仍缺少权重层面的证据。

可检验预测是：真实 DSV4 step0 在正确图/盲图/打乱图之间应出现比纯文本 Qwen 更稳定的
teacher-forced 差异；若出现，projector-only 小训练的样本效率应更高。Gate D 因此新增
step0 receiver-prior 表和 trained-vs-step0 配对表，使用同一 ScreenSpot parser 与 CI；
tiny synthetic route 结果只证明 wrapper 传递了 placeholder/routing/position，不能替代真实权重。
## DeepSeek 0731 预留 token 的权重侧证据（2026-08-08）

公开 `DeepSeek-V4-Flash-0731` 的 tokenizer 保留了 415 个 multimodal span placeholder 及 image/region 标记。我们进一步对 `embed.weight` 做了 range 抽样：预留行均值范数约为普通 token 的 6.8%，且与普通 token 均值几乎正交。这个模式更像零初始化或专门保留、等待运行时视觉向量写入的槽位；它能解释为什么 DeepSeek 可能比纯文本 Qwen 更容易接入 MoonViT projector，也说明直接把 image token 当普通词表 token 训练并不合理。

这条证据的边界很重要：embedding 行低范数不能说明 hidden layers 具备视觉能力，也不能替代真实权重上的图像 forward、视觉梯度和 vision/blind/shuffle 对照。当前 DeepSeek Gate D 仍为 NO-GO；下一项最短路径是用真实 0731 权重完成 step0 receiver-prior、placeholder 替换、input-gradient 与 checkpoint round-trip gate，合格后才进入最小 projector-only 训练。

同时审计了公开 `inference/model.py`：forward 只接收 `input_ids`，没有图像到 embedding 的公开注入实现；`config.json` 没有 `vision_config`。这使“曾经有私有视觉训练环节”的猜想仍然可能，但公开发布物本身不能直接复现该环节。工程上应把它当作一个有利的接口先验，不能当作已经存在的视觉能力。

## Qwen3.5-4B：MoonViT V1/V2 回归对照（2026-08-08）

执行审计发现首次 V1 runner 只使用索引 `0–7`，而 V2 使用完整 `0–31`；旧 V1 结果因此降级为 8-row pilot。按预注册修复合同重跑后，V1 与 V2 现在真正共享 32 条真实答案训练 probe、16-token mean-pool、projector scale `0.1`、BF16、3 steps 和 50 条 ScreenSpot 合同。full32 V1 CE-only 和 paired-margin 都 finite，末步 teacher-forced vision−shuffle 分别为 `+0.0008` 与 `-0.0153`。

ScreenSpot50 仍没有形成能力增益：V1 full32 CE-only 的 vision/blind/shuffled click 为 `2%/2%/2%`；paired-margin 为 `2%/2%/0%`。paired-margin 的 vision−blind click CI 为 `[-6,+6] pp`，vision−shuffled 为 `[0,+6] pp`。exact V2 paired-margin reference 为 `4%/2%/4%`，同样没有正的 paired causal CI 下界。

这条回归反驳“换回社区 V1 就能直接修复当前失败”。当前优先级转向 receiver-facing 分布对齐、视觉 token 监督/位置和输入尺度；V1 仍保留为 DeepSeek runtime validation 的可迁移架构候选，不进入 previous best。
## Fixed regression baseline matrix

`regression_baseline_matrix_v1.json` is the machine-readable index for the current Qwen contract. It binds every row to its receiver, tower, evaluation scope and evidence pointer. The old Qwen2.5-3B full-public result is explicitly a legacy V2 proxy, not exact K3 V2; the native Qwen3.5 VLM remains a separate positive control. Qwen2.5-7B gives a weak positive vision-minus-shuffled click interval but fails vision-minus-blind, while Qwen3.5-4B external V1 and V2 both fail the 50-row causal gate.

This is why a receiver response, lower CE, or positive teacher-forced margin cannot be promoted to visual ability. The next experiment changes one DeepSeek-transferable interface/scale/target-alignment variable and keeps a matched CE-only control; no replay or token-count sweep is opened before that causal screen improves.
## Projector scale 0.03: numerical health is not generation validity

The preregistered V1 scale screen changed only the projector runtime scale from `0.1` to
`0.03`. Three projector-only steps remained finite and the final teacher-forced
vision-minus-shuffle was `+0.0306`, but the fixed ScreenSpot50 parser returned zero parsed
vision and shuffled samples. Blind stayed parseable at 100% and clicked 2%; vision,
blind and shuffled click-in-box were `0%/2%/0%`, with paired CIs `[-6,0]` and `[0,0]`
percentage points. The scale arm is therefore a format-collapse rejection, not a visual
capability result. It reinforces the separation between training health, teacher-forced
attribution and free-generation grounding.
