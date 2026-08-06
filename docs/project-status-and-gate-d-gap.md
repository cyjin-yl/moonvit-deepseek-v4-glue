# 工程主线、证据边界与 Gate D 缺口

更新日期：2026-08-06。

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
| projector-only 训练与反向 | 已验证（3B formal 4k） | 未验证 | Qwen2.5-3B 完成 500 steps/4,000 examples；projector 六个参数张量在首末步均 finite/nonzero，语言主干梯度张量为 0；完整 0731 的量化 input DGRAD 仍为 `hardware_pending` |
| checkpoint 保存与精确恢复 | 已验证（3B formal 4k） | 未验证 | steps 100–500 的五个 checkpoint 共 25 个 payload，经独立 verifier 重哈希并复核 optimizer/RNG/order/token accounting |
| 自回归生成与因果评测 | 已验证（3B formal GLM50） | tiny DeepSeek 类通过，完整权重未验证 | 七条件固定生成可运行；4k vision click 4%，blind 12%，step0 10%，候选因因果退化被拒绝；完整 0731 未运行 |

结论：仓库具备一条经过小模型验证的通用多模态 glue pipeline，也具备 DeepSeek 专用的代码路径。完整 DeepSeek-V4-Flash-0731 的加载、图像前向、反向、训练、保存恢复和生成尚未形成真实闭环，因此当前不能声称已经具备真实 DeepSeek 端到端多模态链路。

## 3. 真模型、真实数据与代理实验的边界

| 证据层级 | 已完成结果 | 能支持的结论 | 不能支持的结论 |
|---|---|---|---|
| 真实视觉塔 | MoonViT-V2 真实权重、真实预处理、V100 forward/backward glue | 视觉编码器接口和 projector 输入合同可运行 | 完整 DeepSeek 能利用这些视觉 token |
| 真实数据、纯文本小主干 | Qwen2.5-0.5B + MoonViT-V2 + projector-only，真实 59,198-row mix，已见 16,000 examples | 无原生 VLM 能力时可以学到非零图像依赖；训练/save/resume/eval 链路可运行 | 绝对 benchmark 上限、3B 容量、DeepSeek Hash-MoE 收敛 |
| 3B 代理固定合同、两轮 formal 4k 训练与完整 ScreenSpot | Qwen2.5-3B 的 9 个文件 SHA、ScreenSpot50/full、严格 parser、七条件、4096→2048 fixed receiver、240 条语言保持集；两套 exact order/cache、两轮 500-step 训练和 checkpoint 独立验证；首轮 50/1,272 七条件 generation/preference、grounding-enriched 50-row preference/generation 与 representation-retention screen 已完成 | 3B 路径可在 V100 运行且可恢复；grounding-enriched preference vision/blind/shuffled 为 52%/56%/54%，generation click 为 6%/12%/6%；projector effective rank 从 13.28 降到 1.14，receiver 保留同一塌缩比 | 3B 或 DeepSeek 已获得视觉能力；任一 4k checkpoint 可进入 DeepSeek 候选；TextVQA/DocVQA/OCRBench 与语言保持尚未完成 |
| 原生 VLM 阳性对照 | Qwen3.5-4B 原生视觉模型在五项真实 benchmark 上运行 | 数据、processor 和 scorer 能得到强阳性结果 | MoonViT projector 对纯文本主干的能力 |
| synthetic 包 3–14 | Qwen2.5-0.5B 上的 paired preference/generation、probe、patching、replay、sentinel | 机制定位、训练干扰、固定预算保护和评测开销 | ScreenSpot/TextVQA/DocVQA/OCRBench 的真实能力 |
| DeepSeek 结构代理 | tiny `DeepseekV4ForCausalLM`、数学 DGRAD reference、三模式 harness | wrapper、routing 与 gate 工具的接口正确性 | 真实 FP4/FP8 kernel 的 input gradient 或完整 0731 稳定性 |

历史 Gate B 的 Qwen2.5-0.5B 结果正式归类为 early-alignment/interface-learnability evidence。它的低容量和约 0.27 effective epoch 会共同压低 OCR、推理、格式遵从与自由生成，不能作为能力上限。纯文本 `Qwen/Qwen2.5-3B-Instruct` 已接替容量代理；首个 4k 结果为负，说明 0.5B 容量不是唯一瓶颈。

## 4. 首个 Qwen2.5-3B grounding 判定

`screenspot_glm50_v1` 上 trained vision 的 parse、Accuracy@50/@100/@200、click-in-box、mean distance 分别为 96%、2%/4%/16%、4%、554.53。blind click/mean 为 12%/392.59，step0 为 10%/398.59。paired bootstrap 给出：

- vision-minus-blind click `-0.08 [-0.20, 0.02]`，mean-distance improvement `-161.94 [-246.70, -89.24]`；
- current-minus-step0 click `-0.06 [-0.16, 0.04]`，mean-distance improvement `-155.94 [-246.74, -75.50]`；
- vision-minus-shuffled click `-0.02 [-0.06, 0]`。

判定为 `reject_current_candidate`；previous-best 保持 step0。结果支持 3B formal pipeline 的工程可行性，反驳 4k projector-only objective 已产生真实 grounding。

完整 1,272-row public ScreenSpot 的 trained vision parse、Accuracy@50/@100/@200、click、mean distance 为 96.46%、1.73%/4.87%/11.79%、2.67%、565.18。blind click/mean 为 3.07%/395.52，step0 为 3.30%/391.12。vision-minus-blind mean-distance improvement 为 `-169.66 [-185.68, -154.17]`；vision-minus-shuffled click 与 distance CI 均跨零。50 条负结果由完整集复现。

Teacher-forced correct-versus-counterfactual preference 进一步定位：trained vision/blind/shuffled/step0/random 为 46%/56%/52%/54%/50%。vision-minus-shuffled 为 `-0.06 [-0.14, 0]`，mean margin 为 `-0.00725 [-0.01287, -0.00186]`。训练同时把 correct-answer NLL 从 step0 的 2.50769 降到 1.22362，且正确图与 shuffled 图的 correct-answer logp 无差异。当前失败发生在 content-specific readout 形成之前，无法归因于 greedy generation 单独失效。

Package 15I–15L 把显式 grounding 从 339/4,000 提高到 2,000/4,000，同时保持 exact step0、500 steps、总 examples、分辨率、receiver 和 evaluator。新 checkpoint 的 teacher-forced vision/blind/shuffled/step0/random 为 52%/56%/54%/54%/50%；vision-minus-shuffled 为 `-0.02 [-0.06, 0]`，mean-margin 为 `-0.002378 [-0.006099, 0.001248]`。correct-answer NLL 继续降至 1.05915，相对 step0 改善 `1.44854 [1.29793, 1.60698]`，而正确图与 shuffled 图的 correct-logp 差为 `-0.001633 [-0.005786, 0.002342]`。这排除了“首轮只因 grounding 比例过低”的单因解释，并把下一项收敛为 training-only counterfactual-margin objective；在运行该新目标前仍补完本 checkpoint 的 GLM50 generation 合同。

Package 15M 已补完 generation 合同。vision/blind/shuffled 的 click-in-box 为 6%/12%/6%，mean distance 为 502.06/392.59/502.08；vision-minus-blind mean-distance improvement 为 `-109.47 [-171.64, -44.59]`，vision-minus-shuffled 为 `0.018 [-3.544, 3.213]`。vision 31/50 输出 `[125,345]`，而 2,000 个 grounding labels 有 1,066 个 unique coordinate pairs 且从未出现该点。free generation 与 preference 一致拒绝 checkpoint，并揭示非 label-mode 的窄坐标塌缩。

Package 15N 随后触发预注册的两个 gross-collapse guards。projector current/step0 relative-spread ratio 为 `0.1384`，participation-rank ratio 为 `0.0859`；effective rank 从 13.28 降到 1.14，top-1 variance fraction 从 17.48% 升到 93.46%。同时 sample RMS 从 0.124 放大到 97.31、within-image token RMS 从 0.139 放大到 18.45。绝对跨图距离没有消失，表示变成巨大、近共线的 common-direction soft prompt，跨图差异接近 rank one。receiver 的两项 ratio 为 `0.1372/0.0846`，与 projector 一致，排除 fixed receiver 是主要塌缩源。

Package 15O 把同一诊断扩到 steps 0/100/200/300/400/500。第一个保存的训练状态 step100/800 examples 已同时触发两个 guard：projector spread/rank ratio 为 `0.12985/0.07721`，sample RMS 从 0.124 放大到 35.74，top-1 variance 达 98.76%；receiver ratio 为 `0.12873/0.07596`。所有后续保存点仍塌缩。首个 100-step 窗口 loss 均值仍为 3.916，末步 loss 为 2.276，因此该失败属于早期优化动力学，继续相同 CE-only 训练没有恢复几何。精确 onset 只能限定在 steps 1–100。下一训练 screen 从首个 optimizer step 施加最小 scale/geometry-preservation treatment；counterfactual margin 暂缓。

Package 15P 已完成预注册 geometry-repair 的固定 λ 校准。冻结 step100/batch100 的 unweighted auxiliary gradient norm 为 `3.8781849597`，记录的 CE gradient norm 为 `0.7901650667`；三档 λ 为 `0.0101873051/0.0407492203/0.1629968813`，control 为零。独立 verifier 状态为 `verified`，完整 pooled tensor、config、logs、manifest 与首次 shell 编排失败均已归档。该结果只证明校准链路和梯度绑定，尚未证明视觉能力；下一步运行四臂 100-step 短筛选。

短筛选的第一次 `control` 启动在 optimizer step 1 之前被绑定检查拒绝：校准 SUMMARY 缺少 `screen_contract_file_sha256`。失败目录保留了完整 supervision records、cache verification、ATTEMPT、traceback 和日志；没有 checkpoint 或能力结果。已登记 pre-result repair，修复只补齐 SUMMARY/独立 verifier 的输入哈希和 record IDs，训练预算、顺序、目标和 λ 不变；修复提交后需重新生成校准再启动四臂。

## 5. Replay 与 sentinel 的收束结论

包 13 已证明 fixed-budget preventive replay 在相同 1,200-example 预算内改善 count/shape preference `+0.255 [0.210, 0.300]` 和 generation `+0.120 [0.050, 0.190]`，donor 合并差异接近零。包 14 进一步确定：

- Tiny 为 25 complete pairs/task；count trigger recall 0.975，精确 count-only 决策率 0.935，familywise false trigger 0.040。
- V100 teacher-only Tiny/Medium 中位用时为 22.501/43.881 秒，峰值显存 6.886 GB。
- Tiny 在模型常驻、5%/10% 评测开销下的最小间隔为 476/226 steps，操作配置向上取 512/256。
- fixed preventive replay 作为默认保护；Tiny 作为稀疏 checkpoint audit；Medium 只在 Tiny 告警后确认。

这条支线已足够为真实训练提供默认保护。后续只有会直接改变 DeepSeek 正式配方的真实合同证据，才重新开启 replay、trigger、Fisher 或 EWC 消融。

## 6. Gate D 当前判定

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

## 7. 当前 V100 上的最短路径

1. **已完成（Package 15A 独立冻结）**：冻结纯文本 `Qwen/Qwen2.5-3B-Instruct` resolved revision、9 个文件 SHA-256、tokenizer bundle/chat-template SHA，并确认无 `vision_config`。
2. **已完成（Package 15A 独立冻结）**：固定 `screenspot_glm50_v1`、1,272 条完整公共测试集与 240 条 language-retention manifest，均在任何 3B 输出前生成。
3. **已完成（Package 15A 独立冻结）**：精确冻结 step0/random-projector 两份 33,564,672-parameter FP32 权重，并在 HF immutable commit `65639da5…a010` 完成 5/5 远端哈希验证。
4. **已完成（Package 15A 独立冻结）**：固定严格 `click(start_box=[x, y])` parser、七条件、完整 grounding 指标和 2,000 次 paired bootstrap。
5. **已完成（Package 15A 独立冻结）**：canonical projector 维持 4096；Qwen 使用无参数 fixed signed-pair 4096→2048 readout。readout 丢弃，代理 checkpoint 标为 `transferable_with_runtime_validation`。
6. **已完成（Package 15B）**：9/9 Qwen 文件与 MoonViT 权重先通过运行内 SHA，随后完成 Qwen3B BF16-source→FP16-runtime load、真 ScreenSpot 图像 MoonViT forward、receiver/projector 梯度、一步 AdamW 和 checkpoint round-trip。峰值显存 8,367,393,280 bytes，含约 7 GB 输入哈希的 wall time 174.476 s；step0 vision=blind，因此只算工程闭环。
7. **已完成（Package 15C）**：首 4,000 条保持源顺序、零 shuffle/holdout，固定为 500 optimizer steps。Manifest `ddca738e…c2fd` 绑定每行逻辑记录、实际 teacher target、图像 SHA/bytes/尺寸；独立 verifier 匹配 4,000/4,000 records、targets 和 images。
8. **已完成（Package 15D）**：内容寻址 MoonViT cache 强制绑定 Package-15C manifest。4,000/4,000 记录、3,534 unique spans、466 aliases、111 shards 经独立重哈希与逐条 finite/shape/order 校验；完整根为 10,374,552,697 bytes。首个 dirty-run attempt 保留并禁止训练。
9. **已完成（Package 15E）**：fail-closed cached-feature trainer 完成 500-step/4k formal run；五个 checkpoint、exact order、answer-token count、optimizer 和 RNG 经独立 verifier 通过。
10. **已完成（Package 15F）**：GLM-format public-50 的七条件与 2,000 paired bootstrap 完成；current candidate 因 vision 弱于 blind/step0 被拒绝。
11. **已完成（Package 15G）**：完整 1,272-row public ScreenSpot 七条件生成与 2,000 paired bootstrap；GLM50 的负结果在完整集复现，所有 predictions 和逐行 scores 已保存。
12. **已完成（Package 15H）**：step0/step500 teacher-forced correct-vs-counterfactual preference；训练显著提高绝对坐标答案概率，却没有正确图相对 blind/shuffled 的选择优势。
13. **已完成（Package 15I，训练前）**：exact step0、500 steps、4,000 examples、分辨率、receiver 与 evaluator 全部固定；从冻结源 pack 分别取前 2,000 ShowUI grounding 与前 2,000 short-answer，按 grounding-first 严格交替。Manifest `d632ecc2…0bf1` 与 order `f3c3dec1…15ab` 已独立匹配 4,000 records/targets/images 和 1,255,969,179 image bytes。
14. **已完成（Package 15J）**：绑定 Package-15I 的内容寻址 MoonViT cache 为 4,000/4,000、零失败、2,013 real forwards、1,987 aliases 和 63 shards；独立 verifier 检查 2,742,976,512 logical values、全部 shard SHA 和 exact order binding。
15. **已完成（Package 15K）**：exact step0 上完成 500 steps / 4,000 examples / 36,589 answer tokens；Qwen/receiver 全冻结，五个 checkpoint、optimizer/RNG/order/token accounting 独立验证通过。最终 projector 为 `62f69393…3df4`。
16. **已完成（Package 15L）**：GLM-format public-50 teacher-forced correct-vs-counterfactual preference；vision/blind/shuffled 为 52%/56%/54%，correct-NLL 相对 step0 显著降低，图像身份依赖仍未建立，checkpoint 被拒绝。
17. **已完成（Package 15M）**：同 checkpoint 的 GLM-format public-50 七条件 generation 与 2,000 bootstrap；vision/blind/shuffled click 为 6%/12%/6%，vision 与 shuffled distance 无差异，候选不扩大 full/三 seed。
18. **已完成（Package 15N）**：projector/receiver 两个 gross-collapse guards 同时触发；projector effective rank 13.28→1.14、top-1 variance 17.48%→93.46%，receiver ratio 近似不变。pooled tensors、6,125 pair rows、50 per-sample rows及两项失败修复全部保留；17 files / 8,120,202 bytes，V100 full suite 347/347。
19. **已完成（Package 15O）**：steps 0/100/200/300/400/500 的 frozen representation trajectory；首个保存点 step100 已 gross collapse，13 个 pooled tensors、15,925 pair rows、50 per-sample rows与 500 行训练历史由独立 verifier 重算。
20. **已完成（Package 15P calibration）**：从冻结 step100/batch100 状态独立重算 geometry loss、辅助梯度和三档 λ；`INDEPENDENT_VERIFICATION.json` 为 `verified`，无能力结论。
21. **已保留失败并修复（Package 15P）**：首个 control 在 step 1 前因 calibration binding 缺字段停止；未生成结果，修复后重新校准。
22. **当前训练 screen**：运行 `control/ratio005/ratio020/ratio080` 四臂各 100 optimizer steps / 800 examples；仅按预注册 representation guards 与 final-20 CE 比例选择最小非零 arm，未通过则停止扩展并重设计。
23. **并行工程缺口**：补齐 fixed-receiver TextVQA、DocVQA、OCRBench 与 240-row language-retention evaluator；任何候选替换 previous-best 前必须跑完。

在完成以上本地证据后，若剩余阻塞只来自完整权重容量和量化 DGRAD，再提交最小付费 Gate D 的硬件、时价、GPU-hour、存储与止损上限，等待单独授权。
