# 当前工程状态与下一步

更新日期：2026-08-07

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

### 8-sample receiver-prior probe

把 9B 的正确图/确定性打乱图/blind/random-projector 扩展到固定 8 个样本后，16 token 的 `vision−shuffle` 均值为 `+0.0447 ± 0.3729`，240 token 为 `-0.0748 ± 0.4520`；两档中各有 5/8 样本为正。`vision−blind` 则为 `+0.1993 ± 0.2142` 和 `+0.6753 ± 0.3335`。这支持“接收器确实响应外部视觉 token”，没有支持“它已经能区分正确图像”。后续若要进入真实 ScreenSpot，必须先解决这个 paired image attribution gap。

同一 9B、8 个样本、240 token 条件换成 V1 projector 后，`vision−shuffle=+0.0620 ± 0.4185`、`vision−blind=+0.3780 ± 0.1962`。V2 对照是 `-0.0748 ± 0.4520`、`+0.6753 ± 0.3335`。V1 轻微为正但没有超过样本波动；版本差异暂时不能解释主要失败，下一项转向 token ordering、压缩和输入尺度。

再把 V2 的普通连续位置换成 Qwen3.5 原生 3D mRoPE，8-sample/240-token 的 `vision−shuffle` 为 `-0.0375 ± 0.4537`，`vision−blind` 为 `+0.6680 ± 0.2896`，与普通位置几乎一致。因此 Qwen-specific position rule 也没有修复 paired image attribution；该结果不具备 DeepSeek 迁移资格。

9B projector-only backward 的 V100 边界也已测到：240-token 首次尝试触发 NVML allocator assert；修复训练器的 autograd graph retention、缩到 16 tokens 后，仍在 25.88 GiB allocated / 41 MiB free 时 OOM。9B 因此只保留为 inference/input-gradient receiver-prior diagnostic；需要真实多样样本训练时，V100 主线使用 3B/7B，或等待付费硬件授权。

Qwen2.5-7B 反向训练已在 V100 完成一个匹配的 3-step CE-only screen。8 个样本、16 token 全部 finite，projector RMS 和 between-image RMS 基本稳定，梯度从 `137.56` 降到 `3.40`；CE 从 `0.2381` 降到 `0.0094`，但 `vision−shuffle` 从 `+0.0333` 变成 `-0.1027`。这与 3B 的故障方向一致：纯文本容量提高后，CE 更快吸收坐标格式/文本先验，仍没有形成图像归因。下一条正式训练目标必须把 paired image supervision 和 health guard 同时纳入，CE-only 不再进入候选。

7B 的匹配 paired margin arm（λ=`0.1`、margin=`0.1`）也完成：16 token 的 `vision−shuffle` 在 3 steps 后为 `+0.0984`，240 token 只有 `+0.0090`；两者 RMS/spread 都稳定。它支持“直接监督 image-vs-shuffle 比 CE-only 更有方向”，但短 token 的正向不能迁移到完整长度，仍不满足真实 grounding 改进规则。

## 2026-08-07 监督接口审计与 7B 结果降级

对 stripped-receiver 小工具做数据审计时发现，旧的 3B/7B/9B receiver-prior 运行只读取了
`FeatureCache` 的记录。该 manifest 只含样本 ID、图片 SHA、尺寸和 feature shape，不含真实问题与答案。
`build_inputs()` 当时静默回退到统一问题和 `click(start_box=[500, 500])`，所以这些运行的 CE、RMS、梯度和
`vision−shuffle` 只能说明不同视觉 token 会改变固定答案的 log-prob，不能说明模型依据了对应图片。
完整 raw 产物保留，`capability_claim_allowed` 统一降级为 `false`，旧的 token/容量排序不再作为能力证据。

修复后的合同要求在运行前用冻结 manifest 按 ID 连接 cache，并逐条校验 image SHA；缺少
`question/instruction` 或 `target_answer/answers` 立即失败，不再使用默认 prompt/坐标。冻结的 8 条真实答案探针为
[`qwen25_7b_real_probe_manifest.json`](../experiments/qwen3b_community_eval_20260805/capacity_controls/qwen25_7b_real_probe_manifest.json)，SHA-256 为
`9fb216e...de130`，其中包含 TextVQA、DocVQA、ShowUI click 和普通 VQA，属于 receiver-prior 诊断，不是 ScreenSpot benchmark。

修复后 Qwen2.5-7B 的真实答案 matched screen 已完成两条：

| 条件 | CE step0→3 | vision−shuffle step0→3 | 健康 | 解释 |
|---|---:|---:|---|---|
| CE-only，prefix 16 | `6.9373→4.4393` | `-0.2741→-0.8746` | finite，RMS/spread 稳定 | 真实答案监督下仍向 image-agnostic/错误图方向移动 |
| paired margin λ=0.1，prefix 16 | `6.9373→4.5825` | `-0.2741→-0.1613` | finite，RMS/spread 稳定 | margin 减少负向程度，但没有形成正的图像归因 |

这两条结果反驳“7B 只需 CE 或小 paired margin 就能读入 MoonViT token”。它们支持更窄的结论：7B 训练和反向链路可运行，当前 projector/receiver/目标组合在真实答案上仍未产生可用的视觉因果信号。240-token matched arm 正在同一合同下运行；在它完成前不启动 ScreenSpot 或继续扩大训练量。

## 机制经验与设计依据

1. CE loss 可以快速下降而视觉归因不升，必须同时看 correct-image、blind、shuffled 的 paired 指标；loss、RMS、gradient 只属于训练健康。
2. 3B exact-V2 的高学习率会在首步制造 common-direction collapse；geometry objective、LayerNorm/RMSNorm、zero-init residual 目前都未通过早期 health gate。
3. exact V2 小学习率能保住表示几何，却没有自动产生 image-vs-shuffle 优势；“表征没塌缩”和“模型看懂图像”是两道门。
4. 9B visual-pretrained receiver 能感受到外部视觉 token，但多样样本的正确图/打乱图差异接近零；它是 receiver-prior 证据，不能替代真实 VLM benchmark。纯文本 7B 的训练稳定性也没有转化为视觉能力。
5. V1/V2 与 mRoPE 对照尚未改变 paired attribution gap，当前更有判别力的变量是监督接口、token 覆盖/压缩、输入尺度和 receiver 是否具备可解码先验。

真实答案 240-token matched arm 已完成：CE-only 的 `vision−shuffle` 为 `+0.3338→+0.2444`，paired margin λ=`0.1` 为 `+0.3338→+0.3375`；两者都 finite，RMS/spread 稳定，margin 相对 control 的末步差约 `+0.093`。这是一条有限的“paired objective 没有破坏全长信号”证据，仍只有 8 条混合 VQA/ShowUI 样本，未做 bootstrap、自由生成或 ScreenSpot。

同一真实答案合同下的 16-token 单变量 screen：prefix margin `-0.2741→-0.1613`，uniform margin `-0.2421→+0.0630`，mean-pool margin `-0.2036→-0.1363`。uniform 的方向比 prefix 好，mean-pool 训练中出现 4,292 的梯度峰值；三臂均未达到真实 grounding 改进规则。它支持“token 覆盖/选择是有判别力的变量”，不支持“16-token 任意压缩都能修复 receiver”。

下一步固定顺序：先对 prefix/uniform/mean-pool 和 240 full 做逐样本 step0/step3 probe、random-projector 与 paired bootstrap；只有方向在样本层面稳定，才运行 32-sample probe 或 ScreenSpot50。之后再做 projector scale/结构 screen。任何 arm 只有在真实 ScreenSpot、TextVQA、DocVQA、OCRBench 和 paired CI 同时满足合同后，才可进入 DeepSeek 候选列表。

## 2026-08-07 32-sample token 与尺度机制筛选

固定 seed `20260805` 从 feature cache 与 `train_mix.jsonl` 的交集按四类来源各抽 8 条，冻结 32 条真实答案样本；manifest SHA-256 为
`c726ebfd...a5a629f`。四个 probe 共享同一顺序、同一循环 derangement、同一 exact step0 projector、同一 random-projector seed，执行 2,000 次 bootstrap。

| 条件 | vision−shuffle 均值 | paired bootstrap 95% CI | positive count | vision−blind 均值 |
|---|---:|---:|---:|---:|
| full/prefix 240 | `-0.22` | `[-0.64, 0.13]` | 15/32 | `+2.44` |
| prefix 16 | `-0.07` | `[-0.35, 0.21]` | 17/32 | `+0.97` |
| uniform 16 | `-0.05` | `[-0.31, 0.22]` | 18/32 | `+1.05` |
| mean-pool 16 | `+0.14` | `[-0.12, 0.39]` | 19/32 | `+1.18` |

四种条件的 `vision−blind` 都显著为正，`vision−shuffle` 的 CI 都跨 0。Qwen2.5-7B 会受到视觉 token 影响，仍然不能稳定区分正确图片和 shuffled 图片。mean-pool 的来源分层均值都略正，但每层只有 8 条；random-projector 差异也没有稳定 CI 下界。token 覆盖是有效诊断变量，尚无可晋升的 grounding 改进。

随后做了 projector scale sweep。Qwen2.5-7B 输入 embedding RMS 为 `0.01364`，当前 V2 projector 输出 RMS 为约 `0.994`，约 73 倍差异。scale=`0.01/0.03/0.1/0.25/1.0` 的 32-sample full-token probe 中，`vision−shuffle` 均值约为 `+0.01/+0.00/+0.04/-0.03/-0.22`，所有 paired CI 仍跨 0。尺度必须保留为后续训练变量，但冻结 scale screen 已排除“单纯乘一个常数即可获得可靠归因”。

32-sample mean-pool 训练也完成 matched control：CE-only 的 `vision−shuffle +0.1351→+0.2051`，paired margin λ=`0.1` 为 `+0.1351→+0.1722`；RMS/spread 稳定，梯度峰值约 3,000，margin 没有优于 CE-only。下一项只做 scale=`0.1` 的 matched training screen；若 paired CI 仍跨 0，Qwen 代理训练量暂止，转 projector 结构/辅助目标设计，不再扩展数据或长训。

## 2026-08-07 scale=0.1 matched training 结果

scale=`0.1` 是在 32-sample 预冻结 probe 中唯一值得继续做短训练的尺度。它把 projector 输出 RMS 从约 `0.994` 拉到接近 Qwen2.5-7B 文本 embedding 的数量级参考，同时保持 finite。训练使用 exact step0 projector、mean-pool 16 tokens、同一真实答案 manifest、同一循环 derangement 和冻结 7B receiver；CE-only 与 paired margin (`lambda=0.1`, `margin=0.1`) 是严格匹配的两臂。

| 条件 | CE step0→3 | vision−shuffle step0→3 | 训练健康 | 解释 |
|---|---:|---:|---|---|
| scale 0.1 CE-only | `6.9045→5.8405` | `-0.0167→+0.1297` | finite；gradient peak 约 `781` | 有限的正向点估计，未形成稳健归因 |
| scale 0.1 paired margin | `6.9045→5.9001` | `-0.0167→+0.2487` | finite；gradient peak 约 `781` | 点估计优于 CE-only，仍未过 paired CI |

训练后在同一 32 条 probe 上做 2,000 次 paired bootstrap。CE-only 的 `vision−shuffle` 为 `+0.1297`，95% CI `[-0.3042, 0.5542]`；margin 为 `+0.2487`，CI `[-0.1099, 0.6001]`。两臂的 `vision−blind` 均显著为正，margin 的 `vision−random_projector` 为 `-0.5038`，CI `[-1.0500, -0.0247]`，说明随机 projector 会降低答案概率；这仍然没有证明正确图片胜过 shuffled 图片。margin 相对 CE-only 的配对差为 `+0.1190`，CI `[-0.0429, 0.2881]`，因此改进方向尚未达到“真实 grounding 改进”合同。

这轮支持三个较窄的判断：输入尺度影响训练数值和局部归因，paired margin 比 CE-only 更接近正确方向，7B 训练/保存/回测链路可稳定运行。它反驳“把 projector 乘一个常数就能修复视觉 grounding”，也反驳“随机 projector 结果变差就等于模型已经读懂正确图像”。Qwen 代理当前仍没有可替换 `previous_best` 的 checkpoint；ScreenSpot、TextVQA、DocVQA、OCRBench 和语言保持不应被启动来包装一个尚未过 paired CI 的候选。

### 研究经验归档与下一步

目前仓库已经把以下现象和设计依据作为正式记录：projector-only 在 3B、7B 纯文本 receiver 与 9B 视觉预训练 receiver 上的差异；vision/blind/shuffled/random-projector attribution 的分离；16/32/64/128/240 token 的长度与压缩敏感性；CE 下降而正确图像归因不升；V1/V2 与 Qwen3.5 mRoPE 的版本/位置对照；projector RMS、spread、rank、Gram 和梯度的 collapse 轨迹。每条结论都标注 `capability_claim_allowed`，伪监督旧运行与真实答案运行严格分开。

下一项注册为单变量 projector/辅助目标 screen：优先测试保留 scale=`0.1` 的更强 paired objective 或轻量 residual/尺度约束，并保留同初始化、同预算的 CE-only control。若 32-sample paired CI 仍跨零，Qwen 训练量冻结，转向架构重设计和 DeepSeek runtime 代码审计；不再靠延长训练制造“最好 checkpoint”。

## Gate D 进度估计（2026-08-07）

当前 Gate D 仍是 **NO-GO**。本地已经具备 MoonViT-V2 真权重、4096 projector、placeholder 展开、冻结 receiver 的 projector backward、自动 collapse guard、checkpoint/RNG/save-resume 和固定 benchmark 工具；Qwen2.5-7B 进一步证明 V100 上的低成本训练可以稳定执行。DeepSeek-V4-Flash-0731 仍缺完整权重加载、目标 FP4/FP8 input-gradient、完整 Hash-MoE 图像 forward/backward、batch/routing/activation-checkpointing 一致性、20-step 稳定 checkpoint 与真实 benchmark。

按当前证据，V100 本地还需要约 1--3 个短实验周期（每周期数小时级，取决于远端 tmux 队列）来冻结 projector 结构/辅助目标候选、补齐独立 verifier 和文档。进入 DeepSeek 真实训练仍取决于付费硬件 Gate D；得到授权后，最小顺序是“单模块权重加载 → 单 batch forward → input DGRAD → projector-only backward → 20-step save/resume → 小规模真实评测”。在授权前不下载完整 0731、不租卡，也不把 Qwen 结果写成 DeepSeek 能力。

## 2026-08-07 λ=0.5 paired margin screen

为区分“paired 监督方向太弱”和“7B receiver 无法解码外部 token”，在已冻结的 scale=`0.1`、mean-pool 16、32-sample 真实答案合同上预注册 λ=`0.5`，并运行同初始化、同数据顺序的 CE-only control。两臂都 finite，scale 和 between-image RMS 稳定：

| 条件 | CE step0→3 | vision−shuffle step0→3 | gradient peak |
|---|---:|---:|---:|
| λ=0 CE-only | `6.9045→5.8405` | `-0.0167→+0.1297` | `约 781` |
| λ=0.5 paired margin | `6.9045→5.9831` | `-0.0167→+0.4874` | `约 459` |

训练后 32-sample probe 的 2,000 次 bootstrap 给出：λ=0.5 的 `vision−shuffle=+0.4874`，95% CI `[+0.1423,+0.8786]`；`vision−blind=+1.9574`，CI `[+1.3909,+2.5699]`。相对同批 CE-only 的配对提升为 `+0.3577`，CI `[-0.1287,+0.8397]`；`vision−random_projector=-0.3397`，CI `[-0.7099,+0.0082]`。这是当前第一条在真实答案 32-sample receiver-prior probe 上通过 paired image attribution CI 的训练轨迹，但它仍是 teacher-forced、16 token、Qwen2.5-7B 的机制诊断，`capability_claim_allowed=false`，不能直接写成 ScreenSpot 或 VLM 成功。

这轮支持“paired image-vs-shuffle 监督强度确实能改变局部图像归因”并反驳“所有正 margin 都只是随机波动”。它没有证明 7B 能完成坐标 grounding，也没有证明该 λ 可迁移到 DeepSeek。下一项从继续长训转为把 λ=0.5 轨迹接入一个可审查的 7B formal evaluator：先复用统一 parser、blind/shuffled/random-projector 条件和固定生成配置，确认自由生成与 teacher-forced 信号方向一致，再决定是否值得把同一目标带回 Qwen2.5-3B formal contract。大模型训练量暂不扩大。
