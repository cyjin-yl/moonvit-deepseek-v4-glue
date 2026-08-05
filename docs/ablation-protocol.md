# 租卡前能力归因与消融协议

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

## 3. 第一优先级：语言主干容量

保持 MoonViT-V2、数据、分辨率、scratch projector、答案规则、验证 manifest 和 `examples_seen` 完全相同，只更换**无 `vision_config` 的纯文本**主干：

| 实验 | 主干 | 问题 |
|---|---|---|
| B0 | Qwen2.5-0.5B-Instruct | 历史容量下界 |
| B1 | Qwen2.5-1.5B-Instruct | 容量增加是否迅速改善 |
| B2 | Qwen2.5-3B-Instruct | 更强语言/推理能力能否利用视觉 token |

先在固定 10k 分层训练子集上做等 `examples_seen` 筛选，再让最好的两组跑完整 1 epoch。每组至少在 10%、25%、50%、100% epoch 保存并评测；关键结论尽量用 3 个种子复核。

解释规则：

- 三个尺寸都低：优先怀疑 projector、特征层、分辨率或监督，而不是继续放大 LLM；
- 分数随尺寸清晰上升：当前低分主要是 0.5B 容量混杂，不能归咎于 MoonViT-V2；
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

然后在 Qwen2.5-1.5B 上比较：

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

1. 完成真实 multi-example forward，并测量 micro batch 1/2/4 的 LLM step time、视觉塔时间和峰值显存。
2. 运行 B0/B1/B2 的 10k 等量筛选，决定是否存在明显语言容量瓶颈。
3. 在胜出主干上完成 P0/P1/P2 与 projector-only/LoRA 定位实验。
4. 完成 448/640 训练 × 448/640/1024 评测矩阵。
5. 加入 causal controls 与 synthetic minimal pairs，确认模型使用图片内容和空间信息。
6. 只有上述结果确定训练分辨率、主干敏感性和真实 step time 后，才重新估算 DeepSeek Gate D 租期；Gate D 仍需单独验证 Hash-MoE 路由与量化 kernel 的 input gradient。

任何阶段都不得用原生 VLM 高分替代 MoonViT-V2 → 纯文本主干的接口证据，也不得把 0.27 epoch 的历史 Gate B 低分写成能力上限。

## 8. Checkpoint 合并与抗遗忘边界（包 10–11）

包 10 对同一 projector 训练 basin 的 step 50/100 做了 `alpha=0/.25/.50/.75/1` 线性权重插值。两端 tensor 与原始评测精确复现；所有中间点都未同时保留 count/shape 并获得 coordinate/spatial。最佳折中 `alpha=.25` 仍使 count 相对 step 50 下降 0.16、shape 下降 0.10。当前结果否定的是同一训练 basin 中两个 checkpoint 的简单线性平均，不等于否定所有模型合并方法。正式 DeepSeek 训练不得把 checkpoint averaging 设为默认抗遗忘方案。

包 11 从 step 50 精确恢复权重、optimizer 与数据游标，测试 count/shape-only frozen-step50 projector-output MSE。MSE 随系数增强显著下降，旧任务 paired preference 仍未保住；完整表示距离不能代替答案决策边界或任务级哨兵。`1e-3` 改善 macro preference 与 generation，说明辅助目标可改变 Pareto 路径。后续按顺序执行：严格匹配的分层 batch 对全局随机 batch、固定 replay 对遗忘触发 replay；两者都失败后才引入 per-task gradient-conflict 方法，避免同时堆叠多个机制。

## 9. Batch-order 证据边界（包 12）

包 12 只改变 2,400 条记录的 batch-order constraint：分层臂每个 true batch 六任务各 4 条，全局臂对相同记录做 seeded random permutation；初始化、optimizer state、seed、超参数与 examples seen 均精确匹配。分层在 step 50 的 macro preference/generation 为 0.512/0.233，高于 global 的 0.389/0.167；step 100 的 global macro/generation 反超为 0.531/0.320，对分层 0.511/0.257。终点 overall gap −0.020 的 paired CI `[−0.0442, 0.0025]`，coordinate 分层更好，color/shape 显著偏向 global，预注册判定为 `mixed_or_underpowered`。

逐 batch 分层不能写成 DeepSeek 的硬规则。正式主线采用固定窗口领域覆盖，并用 sentinel/replay 管理任务交换；分层只保留为短校准候选。gradient diagnostic 显示分层终点 6/15 个负 task-pair cosines，global 为 0，当前结果也未支持“分层必然降低干扰”。下一项按既定顺序进入 matched replay，不追加块状 curriculum。
