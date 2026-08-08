# ARCHIVED — 租卡前能力归因与消融协议

该协议已冻结为历史设计；归档后不再自动启动付费消融。

> 2026-08-05 方向更新：第 3–7 节原先的 0.5B/1.5B/3B 宽筛选队列已被纯文本 `Qwen/Qwen2.5-3B-Instruct` 固定社区可比合同取代。0.5B 只保留历史容量下界，不再继续分配新训练预算；1.5B 不进入当前主线。任何 projector、数据、replay、sentinel、分辨率或训练改进都必须先在固定 ScreenSpot50/full、TextVQA、DocVQA、OCRBench、synthetic、语言保持与 vision/blind/shuffled/random-projector 条件下报告。新合同完成后将替代本文第 2–7 节；包 8–14 的机制证据仍有效。

本协议把当前低分拆成可分别回答的问题，不把继续放大原生 VLM 对照当作主线。正式目标仍是：冻结 MoonViT-V2 与纯文本语言主干，仅训练视觉接口；原生 Qwen3.5-4B 只用于证明数据、processor 和评分器健康。

## 1. 当前 Gate B 的准确定位

历史 full-mix Gate B 使用 59,198 条数据、2,000 个 optimizer step，以及旧参数 `batch_size=8`。旧训练器每个 step 实际串行执行 8 次单样本 forward/backward 后再更新一次，因此其真实计量是：

- `micro_batch_size = 1`
- `gradient_accumulation_steps = 8`
- `effective_batch_size = 8`
- `optimizer_steps = 2,000`
- `examples_seen = 16,000`
- `effective_epochs ≈ 0.27`
- 历史 `answer_tokens_seen` 未记录，不能事后精确补齐

所以它是 **early alignment / 视觉接口可学习性验证**，不是充分训练后的能力评测。`shuffle_delta = +0.727` 说明答案 loss 已依赖正确图片；TextVQA 8.1%、DocVQA 3.9%、OCRBench 0% 不能解释成架构上限。

从本协议开始，任何训练结果必须报告 `optimizer_steps`、`examples_seen`、`answer_tokens_seen`、`effective_epochs`、`micro_batch_size`、`gradient_accumulation_steps` 和 `effective_batch_size`。跨配置只按相同 `examples_seen` 或 `answer_tokens_seen` 比较，不按同名 `steps` 比较。

当前训练器已经如实暴露上述语义，并暂时拒绝 `micro_batch_size > 1`。这意味着 `effective_batch_size=64` 仍会产生 64 次串行 forward/backward，绝不能直接套用社区 recipe 的 step-time。实现并实测真正的 padded multi-example forward 之前，不锁定正式租期的训练时长。

## 2. 固定验证口径

训练器自动创建并复用 ID 固定的 `validation_manifest.json`，按 TextVQA、DocVQA、ShowUI 和 art 分层抽样。所有配置必须共享同一 manifest，并保存：

- overall 和每个 source 的 true loss；
- 10 组 seeded random derangement 的 shuffled loss；
- `shuffle_delta` 的均值和总体标准差；
- 每组 `record_id → shuffled image_id` 配对；
- supervision manifest 中的 raw answers、canonical answer 和选择规则。

旧的最后 32 条随机留出和循环平移 1/2/3 位只作为历史数据保留，不与新协议的标准差直接比较。

## 3. 第一优先级：固定 Qwen2.5-3B 代理

主干固定为无 `vision_config` 的纯文本 `Qwen/Qwen2.5-3B-Instruct`。MoonViT-V2、数据、分辨率、projector 初始化、答案规则、验证 manifest、记录顺序和 `examples_seen` 全部冻结。历史尺寸只作上下文：

| 实验 | 主干 | 问题 |
|---|---|---|
| B0 | Qwen2.5-0.5B-Instruct | 历史容量下界 |
| B1 | Qwen2.5-1.5B-Instruct | 暂缓，不分配当前预算 |
| B2 | Qwen2.5-3B-Instruct | 当前唯一主代理；验证真实视觉能力与 DeepSeek 迁移价值 |

在 4k/8k/16k/32k/64k `examples_seen` 保存并按固定合同评测；数据不足时报告实际最大值。探索先用一个固定 seed，任何替代 previous best 或改变 DeepSeek 配方的结论至少用三个独立 seed 复核。

解释规则：

- 3B 仍低且视觉因果区间跨零：优先检查 projector、特征层、分辨率、监督和数据覆盖；
- 3B 明显超过历史 0.5B：确认低容量是历史绝对分数的重要混杂，但不直接外推 DeepSeek Hash-MoE；
- loss / shuffle delta 改善但 benchmark 不改善：配对依赖已经形成，但未转化为可解码能力；
- OCR 随尺寸增长仍为零：优先检查分辨率与答案监督。

Qwen3.5-4B 是原生 VLM，不进入 B0/B1/B2；它不能替代纯文本主干容量消融。

## 4. 第二优先级：Projector 与冻结上界

在容量实验选出的最小可用主干上只跑三种 projector：

| 实验 | Projector | 目的 |
|---|---|---|
| P0 | 单层 Linear | 判断两层 MLP 是否必要 |
| P1 | 当前两层 MLP，scratch | 主基线 |
| P2 | 当前两层 MLP，warm-start trunk | 判断是否提高样本效率 |

P2 只迁移 `pre_norm + linear_1`，语言宽度相关的最后一层重新初始化。比较达到同一 validation loss 所需的 `examples_seen`，以及 10%/25%/50%/100% epoch 的 benchmark、shuffle delta 和 3-seed 稳定性。只有跨至少两个纯文本主干稳定加速时，才把 warm-start 带入昂贵的 DeepSeek 实验。

然后在 Qwen2.5-3B 上比较：

- A：projector only；
- B：projector + 文本主干顶部若干层 LoRA。

LoRA 若显著提升 TextVQA/DocVQA/OCRBench，说明严格冻结限制了 LM 对新视觉 token 分布的适应；若仍无改善，则把排查重点移到 projector 信息损失、MoonViT 特征层、分辨率、视觉 token 压缩和监督格式。LoRA 是定位工具，不自动改变正式 DeepSeek 路线。

## 5. 分辨率与监督

在固定 TextVQA + DocVQA 子集上、按相同 `examples_seen` 比较：

| 训练上限 | 评测 448 | 评测 640 | 评测 1024 |
|---|---:|---:|---:|
| 448 | ✓ | ✓ | ✓ |
| 640 | ✓ | ✓ | ✓ |

同时报告视觉 token 数分布、吞吐、峰值显存、答案上下文长度和各基准变化。正式计划中的“训练 640 / 评测 1024”只有在该矩阵证明小字 OCR 收益且分布失配可接受后才成立。

多答案监督不再固定取 `answers[0]`：默认用 VQA 归一化后的多数 canonical answer；可选模式在可接受 raw answers 中做 seeded random sample。每条样本保留 raw answers、canonical answer、normalization rule 和训练选择规则。

## 6. 因果控制与最小诊断集

在 trained / random projector / blind / shuffled image 之外补充：

- blank image：排除“只需要一段视觉 token”的可能；
- fixed image：所有问题使用同一张图，控制序列长度和 token 分布；
- patch permutation：保持值和 token 数不变，只打乱空间顺序，重点观察 ScreenSpot；
- synthetic minimal pairs：程序化生成颜色、计数、上下左右、短字符串 OCR 和坐标任务。

synthetic 集报告 single accuracy、paired accuracy、answer-flip accuracy、blind accuracy 和 shuffled accuracy。answer-flip 使用完全相同的问题、只改变图片属性，是当前最直接的因果视觉诊断。

原生 Qwen3.5 的 vision/blind 输出还可把正式评测切成：image-required（vision 对、blind 错）、language-prior（都对）、hard/ambiguous（都错）等子集。MoonViT glue 的主要能力读数应优先看 image-required 子集；该切分是由阳性对照定义的诊断视图，不替代原始 benchmark 总分。

## 7. 执行顺序与租卡门槛

1. 冻结 Qwen2.5-3B 模型哈希、ScreenSpot50/full manifests、严格 parser、生成与 bootstrap 配置。
2. 解决 canonical 4096 projector 与 Qwen receiver width 的显式接口，再完成真实 multi-example forward 的 micro batch 1/2/4 测量。
3. 在最小真实预算上运行 vision/blind/shuffled/random-projector/step0，确认图像因果增益。
4. 按 4k/8k/16k/32k/64k 节点建立 ScreenSpot、TextVQA、DocVQA、OCRBench、synthetic 和语言保持轨迹。
5. 只在同一合同与 matched budget 下完成 P0/P1/P2、projector-only/LoRA、数据、replay 和分辨率实验。
6. 冻结可迁移候选后再估算 DeepSeek Gate D；Gate D 仍需单独验证 Hash-MoE 路由与量化 kernel input gradient。

任何阶段都不得用原生 VLM 高分替代 MoonViT-V2 → 纯文本主干的接口证据，也不得把 0.27 epoch 的历史 Gate B 低分写成能力上限。

## 8. Checkpoint 合并与抗遗忘边界（包 10–11）

包 10 对同一 projector 训练 basin 的 step 50/100 做了 `alpha=0/.25/.50/.75/1` 线性权重插值。两端 tensor 与原始评测精确复现；所有中间点都未同时保留 count/shape 并获得 coordinate/spatial。最佳折中 `alpha=.25` 仍使 count 相对 step 50 下降 0.16、shape 下降 0.10。当前结果否定的是同一训练 basin 中两个 checkpoint 的简单线性平均，不等于否定所有模型合并方法。正式 DeepSeek 训练不得把 checkpoint averaging 设为默认抗遗忘方案。

包 11 从 step 50 精确恢复权重、optimizer 与数据游标，测试 count/shape-only frozen-step50 projector-output MSE。MSE 随系数增强显著下降，旧任务 paired preference 仍未保住；完整表示距离不能代替答案决策边界或任务级哨兵。`1e-3` 改善 macro preference 与 generation，说明辅助目标可改变 Pareto 路径。后续按顺序执行：严格匹配的分层 batch 对全局随机 batch、固定 replay 对遗忘触发 replay；两者都失败后才引入 per-task gradient-conflict 方法，避免同时堆叠多个机制。

## 9. Batch-order 证据边界（包 12）

包 12 只改变 2,400 条记录的 batch-order constraint：分层臂每个 true batch 六任务各 4 条，全局臂对相同记录做 seeded random permutation；初始化、optimizer state、seed、超参数与 examples seen 均精确匹配。分层在 step 50 的 macro preference/generation 为 0.512/0.233，高于 global 的 0.389/0.167；step 100 的 global macro/generation 反超为 0.531/0.320，对分层 0.511/0.257。终点 overall gap −0.020 的 paired CI `[−0.0442, 0.0025]`，coordinate 分层更好，color/shape 显著偏向 global，预注册判定为 `mixed_or_underpowered`。

逐 batch 分层不能写成 DeepSeek 的硬规则。正式主线采用固定窗口领域覆盖，并用 sentinel/replay 管理任务交换；分层只保留为短校准候选。gradient diagnostic 显示分层终点 6/15 个负 task-pair cosines，global 为 0，当前结果也未支持“分层必然降低干扰”。下一项按既定顺序进入 matched replay，不追加块状 curriculum。

## 10. 固定预算 replay 与 sentinel 边界（包 13–14）

包 13 从分层 step 50 精确恢复 projector、AdamW 和数据游标，把每条完整策略锁为 50 steps、batch 24、1,200 examples。ordinary 使用六任务各 200；fixed replay 在两个 25-step 窗口各给 count/shape 重放 10 个历史 complete pairs，同时等量换出 donor pairs，最终分配为 `180/180/240/180/240/180`。训练预算没有增加。ordinary 的 step-100 六个 tensor 与历史 checkpoint 逐张量一致，排除了训练器或恢复误差。

触发规则在结果前冻结：step 75 相对 step 50 的 paired preference 下降至少 0.10，且 current-minus-reference paired-bootstrap `ci95_high < 0`，最多取下降最大的两个任务。整体 step 50→75 上升 +0.040 [0.0108, 0.0675] 时，count 仍从 0.380 坍塌到 0.075，gap −0.305 [−0.365, −0.245]；只有 count 触发。该结果要求所有正式 sentinel 保留 per-domain 指标，macro 不能覆盖局部退化。

终点 ordinary/fixed/triggered macro preference 为 0.5108/0.5983/0.5358。fixed 的 count+shape 相对 ordinary 提升 +0.255 [0.210, 0.300]，donor 四任务合并 +0.00375 [−0.0125, 0.01875]；目标自由生成提升 +0.120 [0.050, 0.190]。triggered 的 count 提升 +0.175 [0.125, 0.230]，endpoint 0.275 仍未回到 step-50 参考 0.380±0.05。fixed 相对 triggered overall 为 +0.0625 [0.0425, 0.0833]。

正式训练准备据此采用以下顺序：

1. optimizer steps 与 examples seen 双重锁定；replay 只替换下一固定窗口内的槽位。
2. 已知高风险域使用小比例预防性 replay，触发式 replay 保留为 fallback。
3. 每个替换记录 source ID、donor ID、完整 pair 状态、窗口计数与 wall time。
4. 恢复带固定为参考值下方 0.05；未恢复时只能使用预注册的下一剂量，禁止看结果后手调。
5. 小模型每目标每 25-step 窗口 10 complete pairs 是机制证据；正式域配额仍需按域规模、pair 可用性与 sentinel 功效换算。

包 14 已完成上述成本校准：Tiny=25 pairs/task，count recall 0.975、exact count-only 0.935、familywise false trigger 0.040；Medium=50 pairs/task。V100 teacher-only 中位时间为 22.501/43.881 s，峰值显存 6.886 GB。fixed preventive replay 作为默认保护，Tiny 只作稀疏 checkpoint audit，Medium 只确认告警。replay 剂量、trigger、Fisher 与 EWC 扩展暂缓，除非固定真实视觉合同显示它们会直接改变 DeepSeek 正式配方。
