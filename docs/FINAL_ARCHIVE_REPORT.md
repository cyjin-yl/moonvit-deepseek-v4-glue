# MoonViT → DeepSeek-V4 glue：最终归档报告

归档日期：2026-08-08
仓库：cyjin-yl/moonvit-deepseek-v4-glue
外部参考项目：[webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4](https://huggingface.co/webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4)
外部质疑记录：[WebBrain discussion #3](https://huggingface.co/webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4/discussions/3)

## 1. 归档声明

本仓库停止作为一条活跃训练主线。我们在 V100 32GB 上完成了大量工程、代理模型、机制和评测工作，却没有复现出一个 external MoonViT projector 能在固定真实 benchmark 上让纯文本 receiver 稳定地依赖正确图片。与此同时，社区 GLM-5.2V 使用 Blackwell 级硬件和不同的最终接收器，公开报道了可部署模型；我们无力租用同等级 Blackwell 卡，也没有在完整 DeepSeek-V4-Flash-0731 权重上完成 Gate D。

因此本仓库的结论是“研究线在当前资源边界下停止”，不是“删除证据”。所有原始产物、失败 checkpoint、health 日志、逐样本预测、SHA manifest、verifier、失败原因和报告继续保留，便于将来复核或由其他人接手。

我们对 WebBrain 的 projector 保持怀疑但不作无证据指控：公开权重和代码说明有人组装并声称训练过 projector；但其 manifest 明确标记 exact 0731 package 尚未 GPU 验证，model card 也没有公开训练轨迹、完整启动日志或 paired visual benchmark。因此本项目只把它标为“高价值但未核验的工程参考”，不声称对方没有训练，也不声称它已经获得可靠视觉能力。

## 2. 原始目标与固定合同

原始目标是：

MoonViT-V2 → canonical 4096 projector → DeepSeek-V4-Flash-0731

固定设计为冻结视觉塔和语言主干，只训练 projector；不修改 tokenizer，不扩展词表，使用 image placeholder；在 DeepSeek 路径中保留 Hash-MoE 所需 routing IDs；训练、保存、恢复和生成必须共享同一数据/评测合同。

真实能力门槛从一开始就不是 loss。候选必须在固定的 GLM-format ScreenSpot50 和完整公共 ScreenSpot 上报告：

- vision、blind、shuffled、random_projector 四条件；
- parse rate、Accuracy@50/100/200、click-in-box；
- vision-minus-blind、vision-minus-shuffled、trained-minus-random 的 paired bootstrap CI；
- TextVQA soft accuracy、DocVQA ANLS、OCRBench；
- language retention、checkpoint 轨迹和 failure/rollback；
- synthetic preference/generation 只能作为机制诊断。

## 3. 已经真正完成的部分

### 3.1 V100 工程和数据基础

- 真实 MoonViT-SO-400M/K2.6-lineage tower 在 V100 上完成 forward、cache、训练输入消费和多基准评测管线。
- Kimi K3/MoonViT-V2 的 vision-only 权重被从单个分片抽取，strict-load、shape、SHA 和预处理合同固定。
- 训练数据完成去重、image SHA 绑定、固定样本顺序、训练/评测分离和缓存 manifest。
- ScreenSpot GLM-format 50 条冻结子集、完整公共 ScreenSpot、严格 click parser、四条件 scorer、paired bootstrap 和独立 verifier 均已实现。
- projector、optimizer、RNG、data cursor、健康 checkpoint、failure checkpoint 和 rollback 记录已实现并在 Qwen 代理上验证。
- projector online health 已记录 RMS、between-image spread、effective rank、Gram、梯度、NaN/Inf、CE 和 geometry loss，并能在 critical collapse 时止损。

### 3.2 小主干真实信号链路

SmolLM2-135M 和 Qwen2.5-0.5B 的真实图片 caption overfit 证明了最窄意义上的信号通路：loss 下降、正确图和 shuffled 图的 teacher-forced loss 出现正差、自由生成会随图像改变。典型结果：

- SmolLM2-135M/comfy caption：shuffle delta +0.343；
- Qwen2.5-0.5B/comfy caption：+0.282；
- Qwen2.5-0.5B/flickr8k：+0.148；
- Qwen2.5-0.5B 全量 mix early alignment：shuffle delta +0.727。

这些结果证明“MoonViT → projector → 冻结纯文本 LM”的梯度和输入接口可以学习，不证明通用视觉能力。0.5B 轨迹的绝对 TextVQA、DocVQA、OCRBench、ScreenSpot 和 MMMU-Pro 分数很低，且训练量远小于社区 VLM。

### 3.3 tiny DeepSeek 软件接缝

tiny DeepseekV4ForCausalLM 完成了 placeholder 展开、routing IDs、loss mask、projector backward、冻结主干、checkpoint save/restore 和 greedy generate 的软件回归。普通 Linear input-only DGRAD 也通过。它们只证明软件接缝和数学 harness，不证明完整 0731 量化权重、Hash-MoE 多卡路由、FP4/FP8 backward 或真实图像能力。

## 4. 全部主要模型和 tower 对照

| receiver / tower | 训练条件 | 结果 | 最终解释 |
|---|---|---|---|
| SmolLM2-135M + MoonViT-SO-400M | projector-only，109 条 comfy caption，1000 steps | teacher-forced shuffle delta +0.343，生成随图变化 | 真实信号链路通过；不是可用 VLM |
| Qwen2.5-0.5B + MoonViT-V2 | projector-only，flickr8k/全量 mix | shuffle delta +0.148/+0.727；五基准低 | early alignment 通过，容量/训练量混杂 |
| Qwen2.5-3B + exact MoonViT-V2 | 4k training、grounding-enriched 500 steps、CE-only 与 paired margin | preference vision/blind/shuffled 为 52/56/54%；generation click 6/12/6%；step500 loss 4.144→1.916 | CE 下降且图像改变输出，但正确图没有胜过 shuffled |
| Qwen2.5-7B + V1 family proxy | community-scale projector-only | step33/2112 examples 触发 receiver RMS 50× critical guard | 数值/表示塌缩，未进入能力评测 |
| Qwen2.5-7B + exact V2 | 900 steps/57,600 examples | vision parse/click 6%/4%；blind 100%/10%；shuffled 6%/0%；vision-blind click CI [-16,+2]pp；vision-shuffled [0,+10]pp | 正式能力失败；teacher-forced shuffle delta 不能替代自由生成 |
| Qwen2.5-7B + V2、LR 5e-5 | 100 steps/6,400 examples | vision parse/click 2%/0%；blind 100%/10%；vision-blind [-20,-2]pp | 低 LR 修复健康，不产生视觉能力 |
| Qwen2.5-7B + V2、top-4 LoRA | 100 steps/6,400 examples，24–27 层 q/v/o rank-8 | vision/blind/shuffle/random click 2/10/0/12%；vision-blind [-16,-2]pp | 少量顶部 LoRA 不足以修复深层融合 |
| Qwen2.5-7B + Kimi K2.6 MoonViT-3d | 1152-d、community-shaped projector-only | 50 图 cache 通过；step13 spread 0.092→约0.03、RMS 0.20→4.34，自动止损 | 换成社区同源视觉塔仍塌缩，版本差异不是唯一根因 |
| Qwen3.5-4B stripped-native + V1/V2 | 绕过原生视觉塔，只接外部 MoonViT | 多条 arm 在 step1 出现 NaN projector gradient；scale 0.03 仍无 ScreenSpot 因果增益 | 有视觉预训练的 receiver 也没有自动读懂外部塔 |
| Qwen3.5-9B stripped-native + V1/V2 | receiver-prior 诊断，不用原生视觉模块 | 16/32/64/128/240 token 的局部正负 margin 不稳定；长 token OOM；正式 arm step1 NaN | 9B 规模和视觉先验能改变输出，但没有证明 grounding |
| 原生 Qwen3.5 VLM | 官方视觉塔/原生多模态层 | TextVQA 0.820、DocVQA 0.926、OCRBench 0.900、ScreenSpot accuracy 0.760 | 独立阳性对照，证明 scorer 和任务有效；绝不当作 MoonViT projector 结果 |
| no-vision control | 纯文本/无图 | parse 100%、click 10%、A@50/100/200 为 2/6/18% | 语言先验基线 |
| tiny DeepSeek-V4 | tiny Transformers Hash-MoE | 软件 forward/backward/save/resume/generate 通过 | 不代表完整 0731 |
| 完整 DeepSeek-V4-Flash-0731 | 未在当前资源边界加载 | image forward/backward、FP4/FP8 DGRAD、稳定 checkpoint、真实 benchmark 均未完成 | Gate D = NO-GO |

### 4.1 矩阵完成的准确含义

MATRIX_SUMMARY 的完成规则是“每个登记 arm 都有有效结果或不可变失败产物”，而不是“每个 receiver 都已经完成社区两 epoch 长训”。Qwen2.5-7B exact V2 是唯一完成 57,600 examples/900 steps 的稳定 external projector 主线；Qwen2.5-3B 的 exact/V1/V2 轨迹主要是 4k、500-step 和短诊断，不能写成已经完成 3B 的 57.6k 社区规模复现。Qwen3.5-4B/9B 的 NaN/OOM 臂也按失败记录收录，不能当成完整能力排行榜。

## 5. Projector 设计和结构消融经验

### 5.1 版本和形状

- exact K3/MoonViT-V2 是 1024-d K3 feature，canonical 4096 输出，精确 projector 约 33.56M；legacy V2 与 exact V2 分开保存，不能混称。
- MoonViT-SO-400M V1 是 1152-d family proxy，不能宣称与 Kimi K2.6 byte-identical。
- K26 MoonViT-3d 的真实单图输出为 3354×4×1152，FP16 RMS 约 3.04；其自身 projector 输出 7168，只适配 Kimi receiver，不能直接复用给 Qwen/GLM/DeepSeek。
- WebBrain projector 为 1152 → 4608 → 4096，约 40.119M 参数，形状与我们的 K26 legacy pre-norm projector相同；关键差异在 DeepSeek routing，不是简单的 MLP 宽度。

### 5.2 token 和接口

- full/prefix/uniform/mean-pool、16/32/64/128/240 token 的局部 probe 说明 token 数量、压缩方式和顺序会强烈改变 receiver attribution，但没有一个局部正 margin 自动变成 ScreenSpot grounding。
- mean-pool/prefix 会保留或摧毁不同程度的跨图 spread；uniform 在一轮 16-token probe 方向最好，但样本只有 32 条，不能升级为能力结论。
- 位置 ID、placeholder 展开、loss mask、separator 和 DeepSeek routing 不是形状兼容的附属细节；它们决定 receiver 是否把视觉 token 当成有效输入。
- WebBrain 使用词表外 sentinel 129280 和 prefill 期间固定 64-ID palette-cycle；本项目原先重复 placeholder routing ID，二者不等价。

### 5.3 结构和健康

- CE-only linear/MLP projector 在 step1–2 就可能把不同图片压成 common direction；loss 仍下降，RMS 却暴涨，relative spread/effective rank 下降。
- post-LayerNorm、post-RMSNorm、residual、zero-init residual、gated/scale protection 的短筛选证明：规范化和 zero-init 能改变健康轨迹，但不自动提供 grounding。
- zero-init residual 的梯度确实进入新分支（step1 grad 189.33），step2 仍触发 RMS/spread collapse；“有梯度”不等于“梯度方向正确”。
- geometry repair 的 λ=0.0102/0.04075/0.16299（ratio 0.05/0.20/0.80）被预注册为结构保护候选；目标是保 spread/rank/Gram，不能把健康通过写成能力通过。

## 6. 训练目标、数据和优化策略的全部经验

| 变量 | 观察 | 结论 |
|---|---|---|
| 纯 CE projector-only | loss 普遍下降，teacher-forced true/shuffle delta 可正；自由生成坐标常数化或不解析 | CE 是必要的语言桥接信号，不是视觉能力证据 |
| paired preference / correct-vs-shuffle margin | 在 7B 32-sample teacher-forced probe 中 λ=0.5 得到正 CI；同一 checkpoint 的自由 ScreenSpot click 仍与 shuffled 打平 | paired objective 可改善局部 attribution，但需要自由生成和真实 grounding 验证 |
| caption/short QA | 低成本 warmup 能建立图像条件差异，但空间/OCR 泛化差 | caption 是 bridge，不是完整 grounding 数据 |
| GUI/box/coordinate grounding | 3B grounding-enriched 500-step preference 52/56/54%，generation vision/blind/shuffle 6/12/6% | 少量 grounding 数据和 CE 仍不足以跨越 receiver readout gap |
| balanced six-task training | color/coordinate/count/OCR/shape/spatial 的 teacher-forced preference 全部可被拉高；自由生成只在 shape/spatial 部分出现 | 内部选择和自由生成存在裂缝，任务竞争真实存在 |
| extra projector epoch | 总体 preference/generation 提升，任务之间发生 Pareto 交换 | 延长 projector 不是全局增益 |
| top-layer LoRA | synthetic shape 特异收益，损害 count/spatial；7B formal top-4 LoRA 仍负 | 小 LoRA 不能代替 visual expert/深层融合 |
| checkpoint averaging/interpolation | alpha=.25 提高 macro/spatial，但丢 count/shape，未通过 worst-task 合同 | model soup 不能自动合并任务能力 |
| output anchoring | 能控制 projector-output MSE，但旧任务决策边界仍遗忘 | 表示距离保持不等于答案能力保持 |
| stratified batch order | step50 形成较快，step100 global 反超；分层 arm 有更多负梯度 cosine | 训练顺序改变能力交换，但不是稳定修复 |
| fixed replay | 在 synthetic 六任务同一 1,200-example 预算内恢复 count/shape，donor 总体不降 | replay 是有效保护措施，但不能替代真实视觉对齐 |
| triggered replay/sentinel | 晚触发只部分恢复 count，窗口太短；25 pairs/task 才满足预注册 power | sentinel 适合止损/回滚，不是能力目标 |
| Fisher/EWC/无限 replay 支线 | 未成为正式训练配方 | 预算固定时优先真实 grounding、receiver 融合和目标改进 |

## 7. Projector 表征塌缩的科学发现

固定 probe 上，Qwen2.5-3B grounding-enriched projector 的 step0 到 step100–500 轨迹为：

- projector relative spread 约 0.2687 → 0.037–0.039；
- effective rank 13.28 → 约 1.14；
- top-1 variance fraction 17.48% → 93–99%；
- projector sample RMS 0.124 → 35.74（step100）→ 97.31（step500）；
- receiver 边界几乎复制同样的 rank/spread collapse；
- pairwise distance/CKA 没有变成零向量，而是巨大 common direction 加近 rank-one covariance。

首个保存点 step100 已经触发 critical guard；step1–2 的 focused control/zero-init residual 进一步把 onset 缩到最早两步。CE 仍能从约 4.14 降到约 2.09，说明“loss 下降”和“视觉表示健康”可以完全背离。训练必须自动保存 failure checkpoint、optimizer/RNG、batch IDs、health/probe/log，并回滚到最近健康点；不能从 critical checkpoint 继续。

这条发现的机制解释是：未和语言空间对齐的 MoonViT 特征，被 projector 推成接近所有图片相同的 coordinate soft prompt。它能让 receiver 学会输出格式或坐标先验，却不能让正确图像在自由生成时胜过 shuffled image。

## 8. 科学结论（与 benchmark 结论分开）

1. 小模型上可以证明梯度和视觉条件进入语言模型，但不能从此推出可用 VLM。
2. teacher-forced correct-answer log-prob、shuffle loss delta 和 synthetic paired preference 可能为正，而自由生成 grounding 仍失败；它们是机制诊断，不是最终能力。
3. 正确图片相对 blind 的增益和相对 shuffled 的增益是两个不同问题。vision>blind 只说明图片比语言先验有用；vision>shuffled 才说明图片内容被正确使用。
4. frozen receiver + 输入级 soft visual prefix 很容易形成 shallow alignment。VILA、CogVLM、BLIP-2、DeepSeek-VL 和 Shikra 的文献证据都预测：需要视觉表征预对齐、深层 receiver 通道、OCR/GUI/box 监督和足够数据。
5. projector-only 在 0.5B synthetic 多任务上能建立内部选择，但自由生成和真实 ScreenSpot 会失败；这不是单纯“projector 参数量不够”。
6. 7B 能跑不等于 7B 会看；9B 的视觉预训练 receiver prior 会改变输出，但单样本/短 probe 的正确图归因仍不稳定。
7. V1 和 V2 的失败都出现后，V2 压缩不再是唯一解释。token 数、尺度、位置、placeholder/routing、目标和冻结 receiver 融合深度共同决定结果。
8. fixed replay 在 synthetic 任务上能缓解遗忘，但不能把 image-agnostic projector 变成视觉模型。
9. checkpoint averaging、anchoring、zero-init residual 和低学习率主要改善稳定性或能力分配，不能替代真实视觉监督。
10. 原生 Qwen VLM 的阳性结果证明评测链路有效；它不能证明外部 MoonViT projector 已对齐。

## 9. 社区与外部实现的证据边界

[Baseten GLM-5.2V](https://huggingface.co/baseten/GLM-5.2-Vision-NVFP4) 使用 Kimi K2.6 MoonViT-3d、冻结 GLM-5.2 和约 49.5M projector；[Baseten 文章](https://www.baseten.co/blog/glm-52-with-vision/)报告 66k image-QA、batch 64、两 epoch、约 step900 grokking 和 MMMU-Pro 55%。这是公开 VLM 的强正向参考，但没有本项目的 blind/shuffled causal table。

WebBrain 发布了 DeepSeek-V4-Flash-0731 的 vision overlay、Kimi tower、40.119M projector、SGLang processor 和 palette routing。它的 manifest 明确 exact package 的 GPU 验证为 false，README 也承认当前组装包的 fresh image smoke 尚未重跑；没有公开训练 trace 或 fixed visual benchmark。因此我们可以且已经公开要求可复现证据，但不宣称其 projector 必然无效，也不指控造假。讨论记录在 [discussion #3](https://huggingface.co/webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4/discussions/3)。

## 10. Gate D 最终状态

Gate D = NO-GO。未完成的硬证据：

1. 完整 0731 resolved revision、所有分片 SHA 和大模型 placement；
2. 真实 text-only forward；
3. 真实 image placeholder 展开、routing/palette 和 prefill/decode；
4. 真实 FP4/FP8 input gradient 和有限非零 projector gradient；
5. activation checkpointing、batch>1 多图和 Hash-MoE routing 一致性；
6. 20-step 稳定训练、NaN/Inf、显存/吞吐；
7. fresh-process checkpoint save/resume；
8. ScreenSpot50 四条件和 TextVQA/DocVQA/OCRBench；
9. 真实图像优于 blind/shuffled 的自由生成因果 CI。

这些 gate 需要 Blackwell/大显存量化环境，当前 V100 无法完成。没有获得付费硬件授权，因此本仓库没有自动租卡、没有下载运行完整 0731、没有产生外部费用。

## 11. 原始产物与后续接手

- 详细叙事报告：report/main.typ；
- 当前状态：docs/current-status.md；
- 机制账本：docs/experiment-mechanism-findings.md；
- 训练/评测合同：docs/qwen2.5-3b-community-eval-contract.md；
- runtime/Gate D：docs/runtime-entrypoint-audit.md、docs/gate-d-runbook.md、docs/deepseek-rental-training-contract.md；
- 文献证据：docs/vlm-alignment-literature.md；
- 社区矩阵：experiments/community_scale_model_ablation_20260808/MATRIX_SUMMARY.json；
- 外部审计：experiments/external_model_audits/；
- 所有失败 arm：experiments/community_scale_model_ablation_20260808/failure_artifacts/；
- V100 synthetic/机制产物：experiments/v100_perception_20260804/；
- Qwen proxy/health/benchmark：experiments/qwen3b_community_eval_20260805/。

归档不删除数据、不重写失败记录、不把 proxy 成绩冒充 DeepSeek。未来若有人继续，应从 external WebBrain 的 routing bridge 和本报告的 Gate D 清单开始，而不是从旧的 replay/trigger 消融重新开始。

## 12. Final verdict

我们完成了一个有严格证据边界的研究归档，但没有完成一个可用的 MoonViT-to-DeepSeek VLM。放弃本仓库是资源和证据约束下的项目决策，不是把失败包装成成功。外部 WebBrain 仓库应被视作待核验的参考实现；只有它或未来接手者公开并复现 exact-package 的真实图像 forward、paired benchmark 和 checkpoint/runtime 证据，才可以把“据说能工作”升级为“已证明能工作”。
