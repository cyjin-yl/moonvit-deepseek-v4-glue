# 当前工程状态与下一步

更新日期：2026-08-06

## 一句话结论

Qwen2.5-3B 代理的真实图像 glue、projector 梯度、checkpoint 保存恢复和生成
链路已经跑通；真实视觉能力尚未建立。此前 Package 15P–15R 主要测试的是
`local_v2_legacy`，因此它们的失败不能外推到精确的 Kimi-K3 V2 projector。
当前最有判别力的本地任务，是在同一 Qwen3B 社区评测合同下完成
`local_v2_exact_k3` 与 `local_v1_family_proxy` 的 matched architecture screen。

## 当前状态表

| 问题 | 当前证据 | 状态 | 允许的结论 |
|---|---|---|---|
| 图像能否进入小模型 | 真实 MoonViT 图像 → projector → frozen Qwen2.5-3B；projector 有 finite/non-zero gradient；save/load/resume/generate 已验证 | 通过（工程） | 通用 glue pipeline 可运行 |
| 3B 是否已经获得视觉能力 | legacy V2 的 ScreenSpot50/full 与 paired preference 中，vision 没有稳定优于 blind/shuffled；candidate 被拒绝 | 未通过 | 不能声称 Qwen 或 DeepSeek 已“看懂图像” |
| 失败发生在哪里 | legacy V2 训练在很早期 common-direction collapse：projector effective rank 13.28→1.14，top-1 variance 17.48%→93.46%，RMS 约 0.124→35.74/97.31（不同 trajectory） | 已定位到 projector 输出动力学 | CE/loss 下降不能作为视觉成功证据 |
| geometry repair | Package 15P 的 control、ratio005、ratio020、ratio080 都在 step 1–2 止损；500-step expansion 取消 | 失败并已止损 | 同一 geometry λ 剂量不值得继续堆训练量 |
| output normalization | Package 15Q 的 CE-only、post-LayerNorm、post-RMSNorm 都在 step 2 止损 | 失败 | 输出归一化单变量不足以保留跨图像几何 |
| residual repair | Package 15R baseline 与 zero-init residual 都在 step 2 止损；gated arm 尚待结果 | 进行中 | 已证明 zero-init branch 可收梯度，尚未证明能保留图像差异 |
| exact K3 V2 projector | `kimi_k3_v2` 已实现，bias-free MLP、post-RMSNorm 与 vendored K3 forward parity 单测通过；step0/random 权重已冻结 | 高频 screen 在 step 2 止损 | 几何相对健康，但 vision−shuffled log-prob 连续为负；仍未建立视觉能力 |
| V1 family proxy | SO-400M revision、1152 维特征、权重集合 SHA、canonical-4096 step0/random 权重和 4,000-row cache 已冻结 | 高频 screen 在 step 2 止损 | V1 也发生相同早期 RMS/spread/rank 恶化；V1 版本单独不能解释失败 |
| Qwen3.5-4B | 原生 VLM positive control 已在 TextVQA、DocVQA、OCRBench、ScreenSpot 等 selection 上得到阳性 | 诊断完成 | 证明 scorer/processor 健康；不进入 projector 排名 |
| TextVQA/DocVQA/OCRBench/language retention | 3B architecture candidate 尚未完成全套候选评测 | 待补 | ScreenSpot 单项不能替代完整能力合同 |
| DeepSeek-V4-Flash-0731 | tiny DeepSeek 类和 routing harness 通过；完整 0731 图像 forward/backward/train/save/resume/generate 未运行 | 未通过 | Gate D 仍需真实量化 input DGRAD 和完整主干证据 |
| 付费资源 | 当前没有租机或产生账单 | 未授权 | 继续 V100；不自动执行 Gate D 租机 |

## Placeholder 与词表边界

当前两条目标路径都使用主干词表中已有的 placeholder，不扩充词表：

| 主干 | placeholder | token ID | 备注 |
|---|---|---:|---|
| Qwen2.5-3B-Instruct | `<|image_pad|>` | `151655` | 固定社区合同；训练和评测都使用同一 pinned tokenizer |
| DeepSeek-V4-Flash-0731 | `<｜image｜>` | `129279` | 目标路径已有 token；Hash-MoE routing 需要保留扩展后的 ID |

旧 0.5B 机制包中出现的 ID `151643` 属于历史 artifact 身份，不能进入当前
Qwen3B 合同或 V1/V2 architecture screen。

`configs/qwen2.5-3b-community-eval-v1.json` 的
`contract_status=preregistered_no_qwen3b_results` 是冻结时的 provenance 标记，
不表示当前没有 Qwen3B 结果；当前结果状态只在本文和各实验 manifest 中维护，
避免修改已冻结合同的身份字段。

## Matched screen 冻结进度

两臂 projector 已在同一 PyTorch 2.10.0 CPU 构造环境中按 seed `20260805`
冻结 step0，并按 seed `20260806` 冻结 random-projector。V1 step0 权重 SHA 为
`f24f677f…786cf`，exact K3 V2 为 `bec6e8bf…54815`；完整 tensor-state、文件大小、
serialized config SHA 和 save/load/regeneration 一致性记录位于
`experiments/qwen3b_community_eval_20260805/architecture_controls/`。V1 视觉塔
snapshot 的权重集合聚合 SHA 为 `51a39391…f0ef`。训练入口现在同时校验 source
config、serialized config 与 step0 权重，避免结构 JSON 和 safetensors 静默错配。
首次正式 V1 probe-cache 尝试暴露了 Transformers 5.12.1 对 HF snapshot 符号链接
的 dynamic-module 相对导入缺陷；失败记录已保存。加载入口已改为 pinned model ID
和 revision，snapshot 只用于离线文件身份与哈希，随后从空目录重跑。

### V1/V2 matched high-frequency health screen（2026-08-06）

两条架构使用同一 Qwen2.5-3B、同一 4,000-row order、同一 receiver、同一
learning rate 和前 100-step guard。V1 真实 cache 为 4,000/4,000、0 failures、
3,534 次 MoonViT forward；V2 旧的 4,000-row cache 通过逐记录 SHA/尺寸/顺序校验，
以 111 个硬链接 shard 重绑定到当前 order，未复制特征数据。两条训练都在
step 2 自动止损，完整 raw archive、failure checkpoint、optimizer/RNG、batch IDs、
rollback 和独立 verifier 均已保存到 V100 HDD，Git 只保存摘要与哈希指针。

| arm | step 0→2 effective-rank ratio（projector / receiver） | step 0→2 vision−shuffle log-prob | stop reason | 判定 |
|---|---|---|---|---|
| V1 SO-400M | 1.000→0.264 / 1.000→0.212 | -0.155→+0.031 | RMS 上升、spread 下降；step 1 已低于 receiver rank floor | V1 也失败 |
| exact K3 V2 | 1.000→0.910 / 1.000→0.830 | -0.240→-0.098 | RMS 上升、spread 下降；causal critical 连续触发 | 几何较稳，视觉因果仍失败 |

这组结果削弱了“V2 的 embedding 压缩是唯一根因”。V1 与 V2 都无法在冻结
纯文本 3B 上形成稳定的正确图像优势；V2 的表示健康比 V1 好，却仍然更偏好
shuffled 或与 shuffled 持平。当前最强共同解释是 projector 更新尺度与冻结文本
receiver 的读出接口，训练目标也没有提供足够强的 image-vs-shuffle 因果约束。
这仍然是 health screen，不能把几何保留称为视觉能力。

### Exact V2 小学习率控制（2026-08-06）

把 projector LR 从合同默认 5e-4 降到 5e-5，其余字段完全不变。step 1/2
的 projector/receiver rank ratio 约为 1.000/1.000 和 0.999/0.999，spread
ratio 也保持在 0.996–0.998；这说明默认 LR 确实放大了前一条 V2 trajectory
的几何退化。可惜 causal signal 仍没有出现：vision 与 shuffled preference
保持 0.625/0.625 后到 0.500/0.500，vision-minus-shuffle correct-logp
为 -0.240/-0.211/-0.285。因果 guard 在 step 2 自动停止，独立 verifier
仍为 verified。结论是“更新尺度是几何塌缩的原因之一”，同时“降低 LR 就能
获得视觉能力”被反驳；下一项应直接测试 image-vs-shuffle 的监督/目标接口。

### Paired image-vs-shuffle margin λ=0.1（2026-08-06）

在 geometry-safe 的 V2 LR `5e-5` 上加入 within-batch derangement hinge，margin
`0.1`、lambda `0.1`，同一 batch 的正确图与轮换错误图共享 prompt 和答案。step 1/2
的 hinge loss 为 `0.753/0.440`，vision-minus-shuffle 从 `-0.240` 改善到 `-0.061`，
projector/receiver rank ratio 仍约 `1.000/1.000`。但 vision preference 没有超过
shuffled，step 2 为 `0.625/0.750/0.625`（vision/shuffled/blind），因果 guard 仍止损。
这支持“监督方向有一点作用”，反驳“当前权重已足以建立视觉因果”；该 derangement
只覆盖 batch 内配对，暂时不能当作能力结果。完整 raw archive 和独立 verifier 已保存。

## 为什么前一套方案没有改进

### 1. 训练目标奖励了“会输出坐标”，没有奖励“从正确图片读取坐标”

legacy V2 的 CE 和 correct-answer NLL 持续下降，vision 与 shuffled image 的
paired preference 却没有正向差异。自由生成还出现窄坐标模式。模型找到的是
image-agnostic coordinate soft prompt，训练指标因此给出假阳性。

### 2. projector 表示在早期变成近共线方向

RMS 上升、跨图 spread 下降、effective rank 下降同时发生，receiver 端保留了
相同趋势。健康合同把 onset 从“step 1–100 某处”缩到 `[1,2]`，自动止损保存了
failure checkpoint、optimizer/RNG、batch IDs 和最近健康 checkpoint。

### 3. 几何损失和输出归一化没有改变首步更新方向

四个预注册 λ 都无法通过 projector/receiver 的 spread/rank 门槛；LayerNorm 和
RMSNorm 也无法保留跨图像差异；zero-init residual 仍在第二步塌缩。继续增加
同一训练量的判别力很低，下一步必须先区分 projector 版本和视觉塔版本。

### 4. 结构审计发现旧结果使用了错误的“V2”标签

旧实现是 affine pre-LayerNorm 加 bias MLP。Kimi-K3/MoonViT-V2 reference 是
bias-free MLP 加 trainable post-RMSNorm。旧结果仍然有价值，它们准确描述了已
训练代码的失败；结果不能当作 exact K3 V2 的失败证明。

## 下一步最短路径

### 阶段 A：架构与 cache 接口

1. `local_v2_exact_k3` 的 state/forward parity、strict save/load、参数量和两组
   初始化冻结已经完成。
2. 已用 pinned `MoonViT-SO-400M` 生成正式 1152 维 V1 probe/training cache；每张图的
   SHA、feature shape、processor revision 和视觉 token 数已写入 manifest。
3. V1/V2 独立 step0、random-projector、同 50 个 probe ID 和匹配 4,000-row order/cache
   已冻结；V1/V2 高频 screen 已完成并独立复核。

### 阶段 B：廉价高频筛选

两臂都在相同 3B budget 下从 step 0/1/2 开始高频运行；两臂都在 step 2 自动止损，
因此没有把已触发 critical guard 的 checkpoint 推到 5/10/20/30/50/75/100。每步写
`train_health.jsonl`；probe 写 `probe_metrics.jsonl`。任一 critical guard 触发
就自动保存并回滚，禁止继续到 500 steps。健康通过只允许进入评测队列，不等于
视觉能力通过。

### 阶段 C：固定真实 benchmark

健康臂运行完整 ScreenSpot（GLM-format 50 和 1,272-row public test，含
click-in-box、Accuracy@50/100/200、距离与 paired bootstrap），再运行 TextVQA、
DocVQA、OCRBench、synthetic 和 language-retention。所有结果同时报告 vision、
blind、shuffled、random projector、step0 和 previous-best 角色。

### 阶段 D：根据结果选择解释

| 结果模式 | 最强解释 | 后续动作 |
|---|---|---|
| V1 健康/因果通过，exact V2 失败 | 视觉塔版本、预处理或 V2 压缩映射问题 | 当前未发生；V1 也失败，保留两条 raw trajectory |
| V1 与 exact V2 都失败且早期塌缩 | 纯文本 3B receiver、优化尺度或监督接口是共同瓶颈 | 先跑 exact V2 的更小 projector LR / scale-safe control，再做冻结 receiver 的 image-vs-shuffle objective screen；不扩大数据预算 |
| 两者 health 通过但 causal benchmark 失败 | 表示仍有差异，语言主干没有形成可解码读出 | 检查 receiver、prompt/loss mask、答案格式和训练数据覆盖 |
| exact V2 causal 通过，V1 失败 | K3 V2 是当前更合适的最终视觉塔 | 固定 exact V2 配方，准备 DeepSeek runtime validation |

## Gate D 边界

当前 Gate D 为 **NO-GO**。V1/V2 screen 证明了真实 cache、训练、自动止损、checkpoint 恢复和独立 verifier 可控；
仍缺：

- 固定 revision 的完整 DeepSeek-V4-Flash-0731 真实权重加载；
- 真实 FP4/FP8 module 的有限、非零 input gradient；
- 图像 token 插入完整 Hash-MoE 主干后的 forward/backward/generate；
- batch、activation checkpointing、routing、20-step 稳定性和精确 save/resume；
- 完整真实 benchmark 的视觉因果增益与语言保持。

这些证据需要目标硬件。V100 阶段继续做 V1/V2 架构筛选、健康诊断和固定合同
评测；任何租机、云 GPU、付费存储或完整 0731 运行都等待用户明确授权。

## 权威文件关系

1. `configs/`、实验 `MANIFEST.json`、checkpoint manifest：机器可验证的身份和哈希。
2. `docs/qwen2.5-3b-community-eval-contract.md`：冻结的评测、预算、parser 和迁移规则。
3. 本文：唯一的 live status、证据边界和下一步队列。
4. `docs/architecture-matrix.md`：架构身份、版本差异和比较解释。
5. `HANDOFF.md` 与 `report/main.typ`：交接和长篇历史记录；更新时必须引用本文和矩阵，不能另造 live status。

## Package 15S-capacity：Qwen3.5 stripped-native 接收器先验筛选（2026-08-06）

这轮绕过 Qwen3.5 的原生视觉塔、merger 和 visual forward，只保留视觉预训练后的语言接收器，输入仍来自同一份 MoonViT-V2 → projector 的 4096 维接口；hook 显示 `native_vision_forward_calls=0`。它检验的是接收器先验，不是 Qwen3.5 原生 VLM 结果。

| 模型 | 接收宽度 | dtype / token cap | 结果 | 判定 |
|---|---:|---|---|---|
| Qwen3.5-4B | 2560，固定 grouped signed adapter | BF16 / 16 | finite，`vision−shuffle=-0.0597` | 有梯度，未出现局部因果优势 |
| Qwen3.5-4B | 2560，固定 grouped signed adapter | FP16 / full | 首次更新后 NaN/Inf | 数值失败 |
| Qwen3.5-9B | 4096，identity | BF16 / 16 | finite，`vision−shuffle=+0.6265` | 首个正的接收器先验局部信号 |

9B 使用官方 HF revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，config SHA 为 `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`，四个权重 SHA 记录在 `configs/qwen3.5-9b-hf-sha256.json`。完整 JSONL、运行配置和远端 raw 目录指针在 `experiments/qwen3b_community_eval_20260805/capacity_controls/`。

这组结果改变了问题排序：3B 的失败不能只归咎于 V2 压缩，V1 和 exact V2 都失败；9B 在相同 canonical 4096 边界出现正的正确图/打乱图局部差异，说明接收器容量与视觉预训练先验确实是可信瓶颈。证据仍然很弱：9B 只有一个样本、16 个视觉 token、一步更新，没有 ScreenSpot 或真实 VQA 能力结果，也不能宣称 projector 已经成功。Qwen3.5 诊断标记为 `transferable_with_runtime_validation`，不进入 Qwen2.5-3B 社区排行榜。

下一步固定为 9B BF16 的 32/64/128/240 token 短筛选，先确定有限梯度和正向配对信号能否随 token 数保持；随后做 Qwen2.5-7B 的同样 16/240 token 纯文本 matched control。Gate D 仍为 **NO-GO**。

### 15S-capacity 后续结果：长度与纯文本容量对照

9B BF16 stripped-native 在 32/64/128/240 个视觉 token 下全部通过 finite/input-gradient gate，`native_vision_forward_calls` 始终为 0；单样本 `vision−shuffle` 依次为 `+0.1781/-0.4574/+0.2881/-0.8842`。所以 16-token 的正 margin 可以在更长序列上保持数值稳定，却没有保持方向：它对 token 长度很敏感，不能外推为视觉能力。

Qwen2.5-7B 纯文本 matched control 使用官方 revision `a09a35458c702b33eeacc393d103063234e8bc28`、FP16 和同一个 exact V2 projector。16/240 token 都 finite，`vision−shuffle` 为 `-1.0731/-0.8335`。这条结果反驳“只要从 3B 换到 7B，纯文本 receiver 就会自然解读外部视觉 token”。它支持继续保留视觉预训练 receiver 先验作为候选解释，但 9B 结果仍然必须扩展到多样 probe 与 random-projector 对照。

当前最短路径从“换更大纯文本模型”收敛为“冻结 9B receiver-prior 的多样 probe、token 压缩与 random-projector 小筛选；若仍无稳定正向配对，再回到 projector/输入尺度设计”。所有这些都仍是诊断，不能替代固定 ScreenSpot、TextVQA、DocVQA、OCRBench。
