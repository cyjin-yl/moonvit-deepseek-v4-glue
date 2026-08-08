# VLM 对齐训练文献证据与本项目解释

更新：2026-08-08。本文只引用原始论文或官方技术报告；GLM-5.2V 目前没有可确认的 arXiv 原始报告，因此 GLM 部分只把 GLM-4.5V/4.1V 技术报告作为公开社区配方近似，不把它写成 GLM-5.2V 证据。

## 交叉结论

我们的现象“CE 下降、teacher-forced shuffle delta 为正、自由生成 ScreenSpot vision 不如 blind”并不矛盾。它通常说明视觉向量已经进入语言模型输入，并改变了条件答案概率，但没有完成三件更难的事：把视觉特征放进 receiver 的语言分布、让监督直接要求正确图胜过错误图、让多层解码器在自由生成时持续使用视觉信息。

| 论文/报告 | 社区做法 | 对本项目的直接含义 |
|---|---|---|
| [BLIP-2 (ICML 2023)](https://arxiv.org/abs/2301.12597) | 冻结视觉塔和 LLM 仍先训练 Q-Former；ITC/ITM hard negatives/ITG 做视觉-语言 representation alignment，再做生成对齐。跳过第一阶段会显著损害 VQA。 | 纯 projector+CE 正好缺少 representation stage；应测试 learnable query/resampler + correct-vs-shuffle 对齐，而非只延长 CE。 |
| [LLaVA (NeurIPS 2023)](https://arxiv.org/abs/2304.08485) | 线性 projector 可作为 compatible visual tokenizer，但先用约 595K caption 做 Stage-1，再用约 158K 高质量视觉指令做 Stage-2；后者不是单纯 projector warmup。 | 我们 57.6k examples 的单阶段 projector-only 只能叫 warmup/diagnostic，不是社区最终训练规模。 |
| [LLaVA-1.5](https://arxiv.org/abs/2310.03744) | MLP connector 配合 OCR、region-level VQA、academic VQA 和更高分辨率；连接器简单，数据和 token 分辨率更关键。 | V1/V2 结构差异不是唯一主因；OCR/GUI grounding 数据与图像分辨率必须进入正式能力表。 |
| [Flamingo (NeurIPS 2022)](https://arxiv.org/abs/2204.14198) | Perceiver Resampler 固定 64 tokens，并在多层插入 zero/gated cross-attention；不是只在输入端放一串 soft prefix。 | 单层 projector 可能只产生浅层 prompt；若 V1/V2 都失败，应做小型 query resampler 或 gated visual expert。 |
| [InstructBLIP (NeurIPS 2023)](https://arxiv.org/abs/2305.06500) | instruction-aware Q-Former 从指令条件下抽取 task-relevant visual features，并覆盖多数据集、多任务。 | 静态 image-only projector 不足以覆盖 click、OCR、DocVQA；需要指令条件视觉抽取或至少任务混合。 |
| [VILA: On Pre-training for VLMs](https://arxiv.org/abs/2312.07533) | 仅 projector SFT 即使加大 connector 也弱；解冻 LLM+projector 后视觉能力显著提升；interleaved image-text 优于短 caption；text-only replay 保持语言。 | 这是我们当前失败最直接的先验：必须保留 projector-only control，同时做 matched top-layer LoRA/receiver-unfrozen arm。 |
| [DeepSeek-VL](https://arxiv.org/abs/2403.05525) | Stage-1 约 1.25M caption + 2.5M OCR 只训 adaptor；后续解冻 LLM/adaptor；约 70% VL/30% text 以防语言坍塌。 | 59k 行数据和 900 steps 远小于社区完整配方；projector-only 不能代表最终 DeepSeek 训练。 |
| [DeepSeek-VL2](https://arxiv.org/abs/2412.10302) | MLP adaptor 之外还有 2×2 pixel shuffle、tile/view separator、2D 结构和高分辨率动态 tiling；先 connector，再视觉/LLM 联合阶段。 | MoonViT V1 的 1152 维可接入并不等于社区接口已复现；token 压缩、2D 位置和 separator 需要单变量消融。 |
| [CogVLM](https://arxiv.org/abs/2311.03079) | 每层 visual expert QKV+MLP 深融合；视觉输入不用时原 LM 行为保持；浅层 projector 被解释为 P-tuning 式 shallow alignment。 | 若 projector-only 失败，matched per-layer visual expert/LoRA 是最有判别力的 receiver 瓶颈测试。 |
| [Shikra](https://arxiv.org/abs/2306.15195) | 原始 LLaVA 在绝对空间 chessboard test 约等于随机；加入 RefCOCO/Visual Genome/PointQA/坐标 QA 后才学习 referential grounding，同时训练 LLM。 | ScreenSpot 是空间 grounding，不是普通 caption；没有 box/point 监督时，CE 或 shuffle delta 不能预期点击能力。 |
| [Vision-Flan](https://arxiv.org/abs/2402.11690) | 187 个任务/1.66M instances；synthetic instruction 主要改善格式，视觉能力主要来自 bridge pretraining；窄 caption 任务泛化到 OCR/grounding 很差。 | synthetic paired preference 只能作机制诊断；有限 V100 预算应优先真实 OCR/GUI/box bridge 数据。 |
| [Visual Grounding Methods for VQA are Working for the Wrong Reasons!](https://arxiv.org/abs/2004.05704) | VQA 的语言先验和数据偏差可让模型答对但理由错误，随机视觉 cue 也可能看似有效。 | vision/blind/shuffled counterfactual 配对和点击框指标是必要条件，不能用单项 accuracy 或 loss 代替归因。 |
| [GLM-4.5V/4.1V-Thinking official report](https://arxiv.org/abs/2507.01006) | ViT+AIMv2→MLP 之外，公开配方包含约 120k multimodal pretraining steps、10B+ 清洗图文、OCR/grounding/GUI/video 数据、2D 位置处理和 text mixture。 | “社区训练 100 步”是误解；我们的 100/900 steps 只能做健康筛选和 warmup，不能期待自动出现最终能力。 |

## 对当前结果的解释

1. 7B V2 的 teacher-forced shuffle delta `+2.2324` 证明图像 token 改变了已知答案前缀下的概率，但 ScreenSpot50 的 vision click `4%` 低于 blind `10%`，parse 也从 step0 `94%` 降到 `6%`。这就是浅层/格式扰动，不是 grounding。
2. Qwen3.5 4B/9B stripped V1/V2 在 step1 出现 NaN projector gradient，说明不同 receiver 的数值分布和 projector prior 也需要单独适配；换视觉塔或把 receiver 换大不能自动修复监督/接口问题。
3. 原生 Qwen VLM 阳性 control 在同一 50 条集达到 click `42%`、blind `6%`，无视觉 control 为 `10%`，所以 scorer 能分辨真实视觉链路；external MoonViT 失败不是评测器没有分辨率。

## 可验证的下一步（按判别力排序）

* 保留 Qwen2.5-7B V2 projector-only 作为 matched control，加入 top-layer LoRA 或少量 visual expert；若只有深层适配臂取得正的 vision−blind/shuffle CI，确认 frozen-receiver bottleneck。
* 在同一 projector-only 预算上加入 BLIP-2-style image-vs-shuffle contrastive/ITM hard-negative warmup，再接 CE；直接测 ScreenSpot、TextVQA、DocVQA、OCRBench，而非只看训练 loss。
* 做 32/64 learnable-query resampler、2×2 token compression、separator/2D position 的单变量消融；V1/V2 只作视觉塔变量，不再把版本差异当唯一解释。
* 增加 OCR/GUI grounded box 与 interleaved image-text 数据，保持约 70% 多模态/30% text replay 的语言保持对照；任何新方案必须先过固定 ScreenSpot50，才扩完整集和长训练。

这些建议都能在 V100 上先做最小筛选，并且不依赖 Qwen 专属原生视觉层，保留迁移到 DeepSeek-V4-Flash-0731 的可能性。DeepSeek Gate D 仍需真实权重 forward/backward、FP4/FP8 input-gradient、checkpoint round-trip 和同一套 causal benchmark。

## 预注册机制假设

* **H1：视觉未进入语言语义空间。** 做 BLIP-2 风格 ITC/ITM hard-shuffle bridge，再做 CE；若 ScreenSpot vision−shuffle CI 转正，支持 H1。
* **H2：冻结 receiver 只能形成浅层 prefix。** 比较 projector-only 与 matched top-layer LoRA/visual expert；若只有 LoRA 正，支持深层融合瓶颈。
* **H3：任务监督不足。** 比较 caption/short-QA 与 OCR+GUI/box/coordinate bridge；ScreenSpot、DocVQA、OCRBench 才能判定。
* **H4：token/空间接口不对。** 比较 V1/V2、2D position、separator 和 32/64 query compression；Shikra 的空间测试提示粗对齐接近随机。
* **H5：模型容量。** 在同一 Stage-1/2 合同下比较 7B/9B/native-prior receiver；若大 receiver 仍 vision≈blind 而 LoRA/grounding supervision 有效，就排除单纯规模解释。
