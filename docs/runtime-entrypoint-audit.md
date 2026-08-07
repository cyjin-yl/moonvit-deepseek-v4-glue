# 真实 VLM 训练与运行入口审计（2026-08-08）

本页回答三个执行问题：Qwen2.5-7B 现在由哪个脚本训练，DeepSeek-V4-Flash-0731 在哪里切换，以及真实 `image → answer` 闭环还缺哪些钩子。审计绑定 Git commit `cbb6ceb`；机器可读源文件哈希与判定见 `runtime_entrypoint_audit_v1.json`。

## 结论先行

仓库已经有一条通过单元测试和 tiny DeepSeek 验证的软件接缝，但还没有一条生产级、同一入口覆盖 Qwen2.5-7B 与完整 DeepSeek-V4-Flash-0731 的正式训练器。

- 最近的 Qwen2.5-7B / Qwen3.5 external-MoonViT 训练实际使用 `tools/train_stripped_receiver_prior.py`。该脚本自报 `diagnostic_only`，默认 3 steps，只保存最终 projector 和 optimizer，不能直接扩成 500/2000-step 正式训练。它还绕过 `VisionCausalLM`，经 `smoke_stripped_qwen35.expanded_forward` 只传 `inputs_embeds`，因此不是可迁移到 DeepSeek Hash-MoE routing 的训练路径。
- `tools/train_overfit.py` 是目前最接近共享全循环的入口：任意纯文本 HF causal LM、MoonViT V1/V2 或 feature cache、projector-only backward、周期 checkpoint、resume 和训练统计都已存在。它仍缺固定评测合同、在线 collapse guard、绑定式 checkpoint 校验、真实批处理，以及完整 0731 所需的量化/分布式 loader。
- `tools/train_qwen3b_proxy.py` 拥有最完整的合同、健康探针、自动 stop/rollback、SHA 绑定 checkpoint 和 resume 语义，但它硬绑定 Qwen2.5-3B 合同与 2048-width receiver。它是应当抽取的安全训练骨架，不是 7B 或 DeepSeek 可以原样调用的入口。
- DeepSeek 的软件切换已经存在于 `src/moonvit_glue/model.py`：`model_type=deepseek_v4` 自动选用重复 placeholder routing IDs，并在 forward/prefill 临时替换 embedding lookup。`src/moonvit_glue/merge.py` 负责视觉 span、连续位置、attention mask 和视觉 label `-100`。
- `src/moonvit_glue/loaders.py::load_deepseek_flash_0731()` 是名义上的正式加载入口，但仓库里没有训练或评测 tool 调用它，也尚未在完整 0731 权重上验证。`tools/train_overfit.py` / `tools/eval_vlm.py` 仍采用 `from_pretrained(...).to(device)`，不能承担约 160 GB 的真实量化/多卡加载。

所以当前正确表述是：**glue 软件结构已实现；7B 诊断训练可运行；完整 DeepSeek 运行时闭环未实现到可执行状态。** Gate D 仍为 NO-GO。

## 入口矩阵

| 入口 | 当前用途 | 已有能力 | 不能宣称/主要缺口 | 判定 |
|---|---|---|---|---|
| `tools/train_stripped_receiver_prior.py` | 7B/4B receiver-prior 短筛选 | 外部特征、冻结 receiver、CE/paired-margin、简化 health、最终保存 | 绕过 `VisionCausalLM` 和 routing IDs；无 resume、无周期 checkpoint、无 probe schedule/rollback、无正式数据预算；脚本明确禁止能力声明 | 诊断入口 |
| `tools/train_overfit.py` | 通用 projector-only 早期训练 | V1/V2/cache、任意 text-only HF LM、save/resume、shuffle-loss 验证、步时/显存 | 无强制四条件生成合同、无 collapse guard、micro-batch 固定 1、无完整 DSV4 量化/多卡 loader | 共享训练骨架 |
| `tools/train_qwen3b_proxy.py` | 固定 Qwen2.5-3B 正式合同 | SHA/order/cache 验证、高频 probe、自动止损/回滚、绑定 checkpoint、精确 resume | 模型/receiver/chat template/4k budget 与 3B 合同耦合 | 安全合同参考实现 |
| `tools/eval_vlm.py` | 通用生成与 teacher-forced shuffle-loss | V1/V2/cache、greedy generation、blind、元数据 | 不是完整七条件 ScreenSpot runner；DSV4 真实 generate/KV cache 未验证 | 通用评测骨架 |
| `src/moonvit_glue/loaders.py` | DeepSeek 名义加载 | revision 参数、hidden-size/placeholder 检查、冻结 wrapper | 无权重 manifest/SHA strict gate、无本地-only 强制、无目标量化/TP runtime、未加载 MoonViT | 软件 loader seam |
| `tools/gate_d_dgrad.py` | 真实量化线性层 input-DGRAD | 可发现真实 quantized targets，逐目标结构化保存结果 | 没有完整权重/硬件时只能 reference/candidate harness；不能证明整网训练 | Gate D 单模块工具 |
| `tools/gate_d_tiny_deepseek_e2e.py` | tiny DeepSeek 20-step 闭环 | forward/backward、projector-only optimizer、save/resume、generate | tiny 随机配置，不是 0731 权重/43 层 Hash-MoE/FP4/FP8 | 软件阳性证据 |

## 三个问题的精确答案

### 1. 哪个脚本负责 Qwen2.5-7B？

已有 7B 结果由 `tools/train_stripped_receiver_prior.py` 产生；它不是正式长训器。若现在直接把 `--steps 3` 改成 `2000`，会失去项目已经冻结的高频探针、自动止损、最近健康 checkpoint 和精确恢复合同。

仓库中最接近正式 7B 全循环的是 `tools/train_overfit.py --text-model <Qwen2.5-7B local path>`，但必须先把 `tools/train_qwen3b_proxy.py` 的通用健康/绑定 checkpoint 能力抽到共享层，且先过 100-step causal screen，再有条件进入 500/2000 steps。

### 2. 在哪里切换到 DeepSeek-V4？

切换由三层共同完成：

1. `src/moonvit_glue/loaders.py:54-107` 固定 DeepSeek model ID、验证 `model_type=deepseek_v4`、projector width 4096 与 `<｜image｜>`。
2. `src/moonvit_glue/model.py:72-79,98-117` 自动选择 `deepseek_v4` backbone；forward 传扩展 routing IDs，并用临时 embedding hook 注入视觉向量。
3. `src/moonvit_glue/merge.py:62-173` 把一个 placeholder 展开成多个视觉 token，同时生成 routing IDs、attention/position IDs 和 loss mask。

这三层已经通过真实 Transformers tiny `DeepseekV4ForCausalLM` 测试。它们尚未通过完整 0731 权重、真实 `tid2eid`、FP4/FP8 kernel、目标多卡布局和真实 KV-cache generate。

### 3. 缟少哪些 checkpoint / inference hooks？

不是所有 hook 都缺；应区分“已有但未验证”和“尚未接入正式入口”。

已有但未在完整 0731 验证：

- placeholder 展开、routing IDs、连续 position IDs、视觉 label mask；
- projector → receiver input gradient；
- DeepSeek prefill embedding override 与生成分支；
- projector/optimizer/RNG 的基础 save/resume；
- tiny 20-step exact-resume 与 generate。

正式入口仍缺：

1. 本地 resolved revision、48 个真实权重分片和 tokenizer/config 的加载前 SHA manifest gate；
2. 与目标 NVFP4/FP8 runtime 匹配的多卡 loader、device map/TP 和可微 input-DGRAD；projector、cache feature 和 input IDs 必须显式放到 input embedding 所在设备，不能整模 `.to(device)`；
3. 完整 43 层 Hash-MoE 下的图像 forward/backward、routing 一致性与 frozen-backbone 断言；
4. 真实训练入口中的 activation checkpointing、显存/吞吐、micro/global batch 与 answer-token 记账；
5. 把通用在线 health、probe、critical stop/rollback 和 failure artifact 接入 7B/DSV4，而不是只留在 3B runner；
6. checkpoint 与模型 revision、数据顺序、feature cache、projector 初始化、runtime 源码 SHA 的绑定验证；
7. 完整 0731 的 prefill/decode/KV-cache 生成回归，以及 vision/blind/shuffled/random 四条件同序评测；
8. 面向用户的 `image + prompt → answer` demo CLI；它只能在真实生成路径通过后标记为 DeepSeek demo。

## 冻结后的执行顺序

### V100 本地

1. 把 3B runner 的通用 health/checkpoint/rollback 抽成 receiver-agnostic 组件，接到共享 7B 训练入口。
2. 在同一 7B 初始化、数据顺序、16-token/scale 0.1 和 CE/paired control 下运行 100-step formal screen；按 step 0/1/2/5/10/20/30/50/75/100 保存健康快照。
3. 每个候选先跑 ScreenSpot50 vision/blind/shuffled/random。任一 causal CI 下界不大于 0，或 critical health guard 触发，就停止；不自动进入 500/2000。
4. 只有 100-step 健康且通过 causal screen，才在相同 examples-seen 合同下推进 500，再推进 2000；每一阶段都必须保留匹配 control。
5. 通过候选才扩展完整 ScreenSpot、TextVQA、DocVQA、OCRBench 和语言保持。

### DeepSeek Gate D（需合适硬件与明确授权）

1. 真实权重 manifest/strict load；
2. text-only forward；
3. step0 vision/blind/shuffled/random receiver-prior；
4. 单模块和整网 input-DGRAD；
5. 单 batch projector-only backward；
6. 20-step health/throughput/save-resume；
7. 真实 greedy generate 与 checkpoint round-trip；
8. ScreenSpot50 四条件。

八项全部通过后，才开始 DeepSeek 小规模真实图文训练。租卡、下载完整 0731 或产生费用仍需用户明确授权。

## 对当前希望程度的判断

项目不是“没接上”：软件接缝、真实 MoonViT cache、projector backward、checkpoint 和评测器都存在。项目也不是“已经成为 VLM”：7B 完整 ScreenSpot 的 vision click `3.30%` 低于 blind `3.46%`，Qwen3.5 external MoonViT V1/V2 都没有同时通过 vision-minus-blind 与 vision-minus-shuffled 因果门。

因此希望来自 DeepSeek 预留多模态接口和真实 receiver 先验尚未测试，而不是来自当前 Qwen benchmark。下一次能改变总体判断的结果只有两类：一个通过固定因果门的 7B formal screen，或真实 0731 权重上的 step0/短训练四条件结果。
