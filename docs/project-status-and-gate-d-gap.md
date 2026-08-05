# 工程主线、证据边界与 Gate D 缺口

更新日期：2026-08-05。

## 1. 项目主目标

最终交付路径固定为：

`MoonViT-V2 → 4096 维 projector → deepseek-ai/DeepSeek-V4-Flash-0731`

目标是得到可训练、可保存恢复、可生成、可归因、可复现评测的 DeepSeek 多模态模型。Qwen 小主干、synthetic 六任务、replay 和 sentinel 都是本地筛选或风险控制工具。项目成功必须由完整 DeepSeek-V4-Flash-0731 在真实图像条件下显著优于 blind、shuffled image 和 random projector 来判定。

## 2. 当前端到端链路

| 环节 | 通用/小模型路径 | 完整 DeepSeek-V4-Flash-0731 | 证据边界 |
|---|---:|---:|---|
| MoonViT-V2 权重加载与图像预处理 | 已验证 | 可复用，尚未同完整主干联跑 | K3 抽取权重在 V100 完成真实图像前向，输出 `[tokens, 4, 1024]` |
| 视觉 token 映射 | 已验证 | 4096 维配置已实现，未同完整主干联跑 | `PatchMergerProjector` 支持保存、恢复和视觉侧 trunk warm-start |
| placeholder 展开、位置与 loss mask | 已验证 | DeepSeek 专用分支已有实现，完整权重未验证 | `expand_image_placeholders` 生成扩展 embedding、routing IDs、attention mask、position IDs 和 labels |
| 插入冻结语言模型 | 已验证 | tiny DeepSeek 类通过，完整权重未验证 | 通用分支走 `inputs_embeds`；DeepSeek 分支保留 placeholder routing IDs 并覆盖 embedding lookup |
| projector-only 训练与反向 | 已验证（含 3B smoke） | 未验证 | Qwen2.5-3B 真图像 backward 中 projector 六个参数张量均为 finite/nonzero，语言主干梯度张量为 0；完整 0731 的量化 input DGRAD 仍为 `hardware_pending` |
| checkpoint 保存与精确恢复 | 已验证（含 3B smoke） | 未验证 | 3B 一步 AdamW 后的 projector、optimizer、Python RNG、step、history 均精确恢复；同时保存 fp32 master 与 bf16 serving 权重 |
| 自回归生成 | 已验证（含 3B smoke） | tiny DeepSeek 类通过，完整权重未验证 | 3B step0 vision/blind 均可生成严格 click 格式，但输出相同，尚不能证明视觉能力；完整 0731 未运行 |

结论：仓库具备一条经过小模型验证的通用多模态 glue pipeline，也具备 DeepSeek 专用的代码路径。完整 DeepSeek-V4-Flash-0731 的加载、图像前向、反向、训练、保存恢复和生成尚未形成真实闭环，因此当前不能声称已经具备真实 DeepSeek 端到端多模态链路。

## 3. 真模型、真实数据与代理实验的边界

| 证据层级 | 已完成结果 | 能支持的结论 | 不能支持的结论 |
|---|---|---|---|
| 真实视觉塔 | MoonViT-V2 真实权重、真实预处理、V100 forward/backward glue | 视觉编码器接口和 projector 输入合同可运行 | 完整 DeepSeek 能利用这些视觉 token |
| 真实数据、纯文本小主干 | Qwen2.5-0.5B + MoonViT-V2 + projector-only，真实 59,198-row mix，已见 16,000 examples | 无原生 VLM 能力时可以学到非零图像依赖；训练/save/resume/eval 链路可运行 | 绝对 benchmark 上限、3B 容量、DeepSeek Hash-MoE 收敛 |
| 3B 代理固定合同、工程 smoke 与 4k 顺序 | Qwen2.5-3B 的 9 个文件 SHA、ScreenSpot50/full、严格 parser、七条件、4096→2048 fixed receiver、240 条语言保持集；真图像 load/generate/backward/一步 AdamW/save-resume 已通过；首 4,000 条记录、target 和原图已冻结并独立验签 | 3B 路径可在 V100 运行，真实图像梯度到达 exact 4096 projector，冻结语言主干无梯度，checkpoint 可精确恢复；4k 横向比较已有不可变输入顺序 | 3B 已经获得视觉能力；step0 vision 与 blind 都输出中心点，4k grounding 尚未训练和评测 |
| 原生 VLM 阳性对照 | Qwen3.5-4B 原生视觉模型在五项真实 benchmark 上运行 | 数据、processor 和 scorer 能得到强阳性结果 | MoonViT projector 对纯文本主干的能力 |
| synthetic 包 3–14 | Qwen2.5-0.5B 上的 paired preference/generation、probe、patching、replay、sentinel | 机制定位、训练干扰、固定预算保护和评测开销 | ScreenSpot/TextVQA/DocVQA/OCRBench 的真实能力 |
| DeepSeek 结构代理 | tiny `DeepseekV4ForCausalLM`、数学 DGRAD reference、三模式 harness | wrapper、routing 与 gate 工具的接口正确性 | 真实 FP4/FP8 kernel 的 input gradient 或完整 0731 稳定性 |

历史 Gate B 的 Qwen2.5-0.5B 结果正式归类为 early-alignment/interface-learnability evidence。它的低容量和约 0.27 effective epoch 会共同压低 OCR、推理、格式遵从与自由生成，不能作为能力上限。下一条容量代理固定切换到纯文本 `Qwen/Qwen2.5-3B-Instruct`。

## 4. Replay 与 sentinel 的收束结论

包 13 已证明 fixed-budget preventive replay 在相同 1,200-example 预算内改善 count/shape preference `+0.255 [0.210, 0.300]` 和 generation `+0.120 [0.050, 0.190]`，donor 合并差异接近零。包 14 进一步确定：

- Tiny 为 25 complete pairs/task；count trigger recall 0.975，精确 count-only 决策率 0.935，familywise false trigger 0.040。
- V100 teacher-only Tiny/Medium 中位用时为 22.501/43.881 秒，峰值显存 6.886 GB。
- Tiny 在模型常驻、5%/10% 评测开销下的最小间隔为 476/226 steps，操作配置向上取 512/256。
- fixed preventive replay 作为默认保护；Tiny 作为稀疏 checkpoint audit；Medium 只在 Tiny 告警后确认。

这条支线已足够为真实训练提供默认保护。后续只有会直接改变 DeepSeek 正式配方的真实合同证据，才重新开启 replay、trigger、Fisher 或 EWC 消融。

## 5. Gate D 当前判定

当前判定：**NO-GO / 未满足 Gate D**。

缺少的硬证据：

1. 固定 revision 的完整 DeepSeek-V4-Flash-0731 权重成功加载并完成文本前向。
2. MoonViT-V2 图像 token 经 4096 维 projector 插入完整主干并完成自回归生成。
3. 真实 FP4/FP8 目标模块对输入 activation 提供有限且非零的 data gradient。
4. 单 batch projector-only backward 中 projector 有梯度，语言主干和视觉塔保持冻结。
5. batch > 1、activation checkpointing 和 Hash-MoE routing 的位置与数值一致性。
6. 20-step 稳定性、峰值显存、吞吐、checkpoint 保存和一次精确恢复。
7. 固定真实数据上可重复的视觉能力增长、vision-minus-blind、vision-minus-shuffled 与语言保持证据。

其中 1–6 需要能够承载完整模型及目标量化 kernel 的硬件；当前 V100 无法关闭这些证据。任何租卡或付费资源操作继续等待用户明确授权。

## 6. 当前 V100 上的最短路径

1. **已完成（Package 15A 独立冻结）**：冻结纯文本 `Qwen/Qwen2.5-3B-Instruct` resolved revision、9 个文件 SHA-256、tokenizer bundle/chat-template SHA，并确认无 `vision_config`。
2. **已完成（Package 15A 独立冻结）**：固定 `screenspot_glm50_v1`、1,272 条完整公共测试集与 240 条 language-retention manifest，均在任何 3B 输出前生成。
3. **已完成（Package 15A 独立冻结）**：精确冻结 step0/random-projector 两份 33,564,672-parameter FP32 权重，并在 HF immutable commit `65639da5…a010` 完成 5/5 远端哈希验证。
4. **已完成（Package 15A 独立冻结）**：固定严格 `click(start_box=[x, y])` parser、七条件、完整 grounding 指标和 2,000 次 paired bootstrap。
5. **已完成（Package 15A 独立冻结）**：canonical projector 维持 4096；Qwen 使用无参数 fixed signed-pair 4096→2048 readout。readout 丢弃，代理 checkpoint 标为 `transferable_with_runtime_validation`。
6. **已完成（Package 15B）**：9/9 Qwen 文件与 MoonViT 权重先通过运行内 SHA，随后完成 Qwen3B BF16-source→FP16-runtime load、真 ScreenSpot 图像 MoonViT forward、receiver/projector 梯度、一步 AdamW 和 checkpoint round-trip。峰值显存 8,367,393,280 bytes，含约 7 GB 输入哈希的 wall time 174.476 s；step0 vision=blind，因此只算工程闭环。
7. **已完成（Package 15C）**：首 4,000 条保持源顺序、零 shuffle/holdout，固定为 500 optimizer steps。Manifest `ddca738e…c2fd` 绑定每行逻辑记录、实际 teacher target、图像 SHA/bytes/尺寸；独立 verifier 匹配 4,000/4,000 records、targets 和 images。
8. 建立强制绑定 Package-15C manifest 的内容寻址 MoonViT feature cache，运行最小 projector-only 4k baseline 与 step0/random/vision/blind/shuffled 条件。
9. 根据 4k 真实证据决定扩至 8k/16k/32k/64k；候选进入完整 ScreenSpot、TextVQA、DocVQA、OCRBench、synthetic 六任务和语言保持。
10. 所有横向比较匹配记录集合、顺序、预算、分辨率、exact step0 和生成配置；只把 `directly_transferable` 或 `transferable_with_runtime_validation` 方法纳入 DeepSeek 候选。

在完成以上本地证据后，若剩余阻塞只来自完整权重容量和量化 DGRAD，再提交最小付费 Gate D 的硬件、时价、GPU-hour、存储与止损上限，等待单独授权。
