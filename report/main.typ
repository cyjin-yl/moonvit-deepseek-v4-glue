#set page(paper: "a4", margin: 24mm)
#set text(font: ("Noto Sans CJK SC", "Microsoft YaHei", "Arial"), size: 10pt)
#set heading(numbering: "1.1")
#set par(justify: true, leading: 0.7em)
#set table(stroke: 0.5pt + rgb("c8c8c8"), inset: 6pt)

#align(center)[
  #text(size: 21pt, weight: "bold")[MoonViT-V2 接入 DeepSeek-V4-Flash-0731]
  #v(5pt)
  #text(size: 14pt)[真实 VLM 回归、运行入口审计与 Gate D]
  #v(8pt)
  版本 0.8 · 2026-08-08
]

#outline()
#pagebreak()

= 当前状态（2026-08-08，唯一 live 入口）

当前唯一权威状态页是 `docs/current-status.md`，运行入口审计是
`docs/runtime-entrypoint-audit.md`，架构身份矩阵是
`docs/architecture-matrix.md`。最终目标固定为
`MoonViT-V2 → 4096 维 projector → DeepSeek-V4-Flash-0731`。
Qwen2.5-3B 只承担冻结纯文本 receiver 的低成本代理角色。

当前没有可用 VLM，也没有 checkpoint 获得晋升。Qwen2.5-7B 已完成完整 1,272 条
ScreenSpot：vision/blind/shuffled click-in-box 为 `3.30%/3.46%/2.67%`，
vision−shuffle 有弱阳性，vision−blind 失败。Qwen3.5-4B external MoonViT 的
full32 V1/V2 也都没有通过 ScreenSpot50 因果门，版本单独解释被否决。

软件接缝已覆盖 placeholder expansion、routing/position、loss mask、projector
input-gradient，以及 tiny DeepSeek BF16/FP32 的 20-step save/resume/generate。
完整 0731 权重没有实际加载；真实 FP4/FP8 input-DGRAD、43 层 Hash-MoE
forward/backward/generate、真实 checkpoint round-trip 和因果 benchmark 均未完成。
Gate D 因此保持 *NO-GO*。

== 实验主线重置：社区规模模型消融优先

本报告从本节开始采用新的执行优先级：把 GPU 时间用于判断模型条件，而不是继续重复证明脚本能够运行。
pytest、strict-load、manifest、checkpoint 和独立 verifier 保留为短 preflight；projector 的
collapse/NaN/Inf/gradient/RMS/spread/rank 监控保留为在线止损与回滚。它们保障数据可信，但不能代替视觉能力。

消融矩阵固定包含 MoonViT-SO-400M/K2.6-lineage V1、K3/MoonViT-V2、无视觉与 random-projector，接收器包含
纯文本 Qwen2.5-3B/7B、去掉原生视觉模块的 Qwen3.5-4B/9B，以及独立的原生 Qwen VLM 阳性对照。原生 VLM
不得写入 external-MoonViT 排名；每个 receiver×tower 必须重新训练 projector，旧 checkpoint 只能作 step0
接口诊断。所有 arm 共享 vision/blind/shuffled/random_projector 四条件、样本顺序、图像预处理、prompt、parser
和 greedy decoding。

训练量按社区 GLM-5.2V 的公开数量级对齐：约 66,000 条短答案图文数据、global batch 64、constant learning
rate `5e-4`、约 2 epochs/2,070 optimizer steps；报告中的能力突变约在 step 900（约 57,600 examples seen）。
主评测节点为 4k/8k/16k/32k/57.6k/66k/132k examples seen。20/100 steps 仅是健康筛选，不能宣布能力成功
或失败。只有 ScreenSpot、TextVQA、DocVQA、OCRBench 和 vision−blind/shuffled 配对 CI 通过，checkpoint
才可晋升；历史 3-step、32-row、geometry/replay 结果保留为 archived mechanism evidence。

下一阶段先完成社区规模数据顺序冻结，然后跑 Qwen2.5-3B/7B 的 V1/V2 matched 对照，并同时保存无视觉、随机
projector 和原生 Qwen VLM 阳性对照。结果若显示稳定的正确图像优势，再进入更大 receiver 与 DeepSeek runtime
Gate；完整 DeepSeek-V4-Flash-0731 仍未加载，Gate D 保持 *NO-GO*。

最近 7B/Qwen3.5 训练使用的 `tools/train_stripped_receiver_prior.py` 是 3-step
`diagnostic_only` runner；`tools/train_overfit.py` 是共享全循环骨架；完整
health/stop/rollback 和绑定 checkpoint 只在 3B 专用 `tools/train_qwen3b_proxy.py`。
下一项本地工作是抽取 receiver-agnostic 安全训练组件并冻结社区规模的数据/预算，
然后跑 3B/7B 的 V1/V2 matched 消融。100-step 只作健康节点；正式能力判断要延伸到
57.6k/66k/132k examples seen，不能因短节点 CI 跨零就停止社区规模复现。

社区审计确认：公开 GLM-5.2V 页面使用 Kimi-K2.6/MoonViT-3d 家族的
1152 维视觉塔；GLM projector 在自己的 6144 维语言空间重新训练。仓库当前
注册 `local_v1_family_proxy` 与 `local_v2_exact_k3` 两条 matched control。
Package 15P–15R 测试的是 `local_v2_legacy`，其 early-collapse 结果不能外推
到 exact K3 V2。现在两条 matched control 都已完成真实 cache 和高频 health
screen；两条都在 step 2 自动止损，正式能力 benchmark 尚未启动。Gate D 仍为
*NO-GO*。

两条 matched control 的初始化已经冻结并完成 strict save/load 与确定性重建：
V1 step0/random 权重文件分别为 `f24f677f…786cf` / `a740f349…5ec0`，exact K3
V2 分别为 `bec6e8bf…54815` / `7bdfb08c…65ed`。V1 snapshot 权重集合聚合 SHA
为 `51a39391…f0ef`。训练入口同时绑定 source config、保存后的 serialized config
和 projector 权重，后续两臂不会因配置文件语义漂移形成假比较。

V1 probe-cache 的首次正式尝试在模型加载阶段发现 Transformers 5.12.1 对 HF
snapshot 符号链接的相对导入缺陷；修复后 50-row probe 与 4,000-row training
cache 均成功。V1 cache 是 3,534 次真实 tower forward、466 次同图复用、0 failures；
V2 旧 cache 通过 4,000 条记录的 ID/image/shape/order 校验后以 111 个 hard links
绑定到当前 order。失败记录、完整 raw archive、manifest 和 SHA 指针均已保留。

V1 在 step 0/1/2 的 projector/receiver rank ratio 为
1.000/1.000 → 0.562/0.451 → 0.264/0.212，触发 RMS/spread adverse-trend guard。
exact K3 V2 的 ratio 为 1.000/1.000 → 0.947/0.900 → 0.910/0.830，但
vision-minus-shuffled correct-logp 为 -0.240 → -0.204 → -0.098，连续触发
causal critical。结果削弱“V2 压缩是唯一根因”，共同瓶颈更像 projector 更新尺度、
冻结 3B receiver 的读出接口和不足的 image-vs-shuffle 监督。下一步先跑 exact V2
更小 projector learning-rate 的 matched control；只有健康且 causal 为正的轨迹才
进入完整 ScreenSpot、TextVQA、DocVQA、OCRBench 和 language-retention 合同。
没有真实 causal gain 的结果不替换 `previous_best`，也不进入 DeepSeek 付费阶段。

exact K3 V2 的小学习率探索随后把 projector learning rate 从合同默认
`5e-4` 降到 `5e-5`，其余数据、顺序、step 和 guard 保持不变。step 1/2 的
projector/receiver rank ratio 仍约为 `1.000/1.000`、`0.999/0.999`，说明
较小 LR 能避免高 LR 的几何放大；但 vision preference 仍与 shuffled 持平或
下降，vision-minus-shuffled correct-logp 为 `-0.240/-0.211/-0.285`，因果
guard 在 step 2 停止。当前结论是优化尺度参与了塌缩，监督/receiver 接口仍是
主要未解瓶颈；下一项是固定小 LR 的 image-vs-shuffle 监督 screen。

= 历史执行摘要

本项目目标是给纯文本的 DeepSeek-V4-Flash-0731 接入从 Kimi K3 抽取的 MoonViT-V2（MoonViT3d）视觉编码器。第一阶段冻结视觉塔和语言模型，只训练 Kimi 风格 PatchMerger projector；独立发布的 MoonViT-SO-400M（V1）只保留作历史对照。V2 的真实权重、预处理和 `[tokens,4,1024]` 合同均已在 V100 验证。包 3–12 依次建立 synthetic paired preference/generation、逐层 probe、activation patching、projector/LoRA 轨迹、任务干扰、checkpoint averaging、anchoring 与 batch-order 证据。包 13 在相同 1,200-example 预算内用 preventive replay 恢复 count/shape，包 14 把可靠 Tiny sentinel 固定为 25 pairs/task 并测得 V100 teacher-only 中位开销 22.501 秒。这条机制支线已收束为默认保护配方。包 15A–15D 冻结纯文本 `Qwen/Qwen2.5-3B-Instruct` 的模型、真实数据、评测、4,000-example 顺序和 MoonViT cache；包 15E 完成 500-step projector-only 训练与独立 checkpoint 验证；包 15F–15G 在 GLM-format public-50 和完整 1,272-row ScreenSpot 上一致拒绝首个 checkpoint。完整集 trained vision/blind/step0 的 click-in-box 为 2.67%/3.07%/3.30%，vision−blind 平均距离显著恶化 169.66。包 15H 的 paired preference 又显示 trained vision/blind/shuffled 为 46%/56%/52%；训练把坐标答案 NLL 从 2.51 降到 1.22，却没有正确图相对错误图的选择优势。容量切换稳定了真实链路，当前数据/目标仍只学到 image-agnostic coordinate prior。包 15I/15J 已在结果产生前冻结 2,000-grounding/2,000-short-answer 严格交替顺序与 cache；包 15K 完成相同 4,000-example/500-step 训练和 checkpoint 复核。包 15L/15M 的 preference 与 generation 均拒绝新 checkpoint：vision/blind/shuffled preference 为 52%/56%/54%，generation click 为 6%/12%/6%。包 15N 又把失败定位到 projector 输出的 scale/rank collapse：effective rank 13.28→1.14，top-1 variance 17.48%→93.46%，fixed receiver 保留相同塌缩比。包 15O 进一步发现第一个保存点 step 100/800 examples 已经塌缩，projector spread/rank 只剩 step0 的 0.1298/0.0772；包 15P 已完成从首步生效的 geometry-repair λ 校准，三档 λ 固定为 0.0101873/0.0407492/0.162997，下一项是四臂 100-step 短筛选。完整 DeepSeek-V4-Flash-0731 尚未完成图像前向、量化 input DGRAD、训练、恢复和生成闭环，Gate D 当前未通过。

MoonViT-V2 有 401.2M 参数，抽取后的 BF16 权重约 802 MB，相对于约 160 GB 级的 DeepSeek 混合精度权重很小。更大的资源变量是图像分辨率带来的视觉 token 数和冻结 LLM 反向所保留的激活，而不是视觉塔权重。

= 来源与可复现边界

- 社区 projector checkpoint：#link("https://huggingface.co/0xSero/glm-local-vision-checkpoint")[0xSero/glm-local-vision-checkpoint]。它证明了 projector-only 路线能把视觉信号接到 GLM-5.2，但公开 grounding 指标仍弱，不能等同于成熟 VLM。
- 社区复现记录：#link("https://huggingface.co/blog/0xSero/glm52-vision-on-4-gpus")[Giving a 753B Model Eyes]。
- DeepSeek 权重：#link("https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731")[deepseek-ai/DeepSeek-V4-Flash-0731]，MIT。
- 当前视觉塔：从 Kimi K3 的 MoonViT3d 权重分片抽取的 MoonViT-V2；抽取权重、配置、代码快照与哈希清单发布在项目模型仓 `vision_tower_k3/`。
- 历史对照视觉塔：#link("https://huggingface.co/moonshotai/MoonViT-SO-400M")[moonshotai/MoonViT-SO-400M]，MIT。
- Kimi-K2.5 技术报告：#link("https://arxiv.org/abs/2602.02276")[Kimi K2.5: Visual Agentic Intelligence]。
- Vast offer 搜索接口：#link("https://docs.vast.ai/api-reference/search/search-offers")[Vast.ai Search Offers API]。本文只调用搜索接口，不调用创建实例接口。

独立 MoonViT-SO-400M 来自 Kimi-VL；当前主线使用 Kimi K3 的 MoonViT3d/MoonViT-V2。两者输出合同同构不代表特征分布或通道宽度相同。视觉塔 revision 与权重哈希是训练 provenance 的组成部分；更换视觉塔必须重训 projector。

= 社区 GLM-5.2V 架构核验

2026-08-06 对公开的 #link("https://huggingface.co/baseten/GLM-5.2-Vision-NVFP4")[GLM-5.2-Vision-NVFP4] 页面、解析后的配置、远程模型代码和独立的 `mm_projector.safetensors` 做了 revision 与 SHA-256 固定。社区卡片明确说明视觉塔来自 #link("https://huggingface.co/moonshotai/Kimi-K2.6")[Kimi-K2.6] 的 MoonViT-3d：27 层、1152 维、2×2 merge；视觉塔和 GLM-5.2 文本主干冻结，只训练约 49.5M 参数的 projector。这里的“来自 K2.6”指视觉塔来源，projector 仍然为 GLM 的 6144 维语言空间重新训练，K2.6 自身的语言宽度为 7168。

公开 projector 的 tensor header 与网页配置一致：`LayerNorm(1152) → flatten(4608) → Linear(4608,4608,bias=true) → GELU → Linear(4608,6144,bias=true)`。文件大小为 99,117,136 bytes，SHA-256 为 `e7c6ce8c27424f292e708e7bbb48ade57ea9f1aaddd28bd6a1020a860d9db80c`。它是结构和尺度的可靠参考，不能直接插入 Qwen 的 2048 或 DeepSeek 的 4096 输出边界。

这次审计还修正了我们对本地 V2 的表述。当前 `PatchMergerProjector` 在 `[tokens,4,1024]` 上使用 affine pre-LayerNorm 和带 bias 的两层 MLP；它属于 V1-style PatchMerger 家族。vendored Kimi-K3/MoonViT-V2 的 `PatchMergerMLPV2` 则是 bias-free 两层 MLP 加 trainable post-RMSNorm。因而 Package 15P 的早期塌缩结论只适用于已训练的本地实现，不能写成对官方 K3 V2 结构的否定。详细来源、resolved revision、文件哈希与张量形状见 `experiments/qwen3b_community_eval_20260805/community_architecture_audit_v1/COMMUNITY_SOURCES.json`。

下一步固定为两条匹配控制：一条实现精确 K3 V2 projector，另一条使用公开 #link("https://huggingface.co/moonshotai/MoonViT-SO-400M")[MoonViT-SO-400M] V1 在 Qwen2.5-3B 上复现。两条都沿用 canonical 4096 输出、同一 frozen 4096→2048 Qwen receiver、相同样本顺序、训练预算、在线 collapse guards 和 ScreenSpot/TextVQA/DocVQA/OCRBench 合同；旧的 legacy-V2 checkpoint 不再自动成为 `previous_best`。缓存入口已支持 `--vision-tower v1` 与 pinned revision，V1 不能直接挪用 GLM-5.2V 的 6144-wide 权重。Gate D 仍为 NO-GO。

= 权重与张量合同

#table(
  columns: (1.4fr, 2.4fr, 1fr),
  [*组件*], [*合同*], [*状态*],
  [MoonViT-V2], [`[tokens, 4, 1024]`], [冻结],
  [PatchMerger], [exact K3 V2: bias-free `4096 → 4096 → 4096` + post-RMSNorm；legacy 变体单列], [训练],
  [DeepSeek embedding], [`vocab → 4096`], [冻结],
  [Placeholder], [`<｜image｜>` / ID 129279], [现有词表],
  [Hash-MoE routing], [扩展后的 placeholder IDs], [冻结],
)

exact K3 V2 projector 参数量是 33,558,528（fp32 约 134 MB）；legacy V2 为 33,564,672。备选的 V1 塔（MoonViT-SO-400M，1152 维）projector 为 40,119,040 参数，两条配置不得混用。保存格式由 `projector_config.json` 与 `projector.safetensors` 组成，加载时使用 strict state-dict 校验。MoonViT、DeepSeek 与 projector 始终保留为三个可独立哈希和升级的来源，不把大权重复制到自有仓库。

= DeepSeek-V4 特殊适配

DeepSeek-V4 的早期 Hash-MoE 通过 `tid2eid[input_ids]` 选 expert。只给 `inputs_embeds` 会丢失路由 IDs，而当前公共 forward 不允许同时给 `input_ids` 和 `inputs_embeds`。

原型保留扩展后的 placeholder IDs，并用仅在一次 forward 内生效的 embedding forward hook 替换查表输出。这样 Hash-MoE 看到合法 token IDs，Transformer 隐藏状态看到 projector embeddings，loss 仍可反向到 projector。hook 在 forward 结束后自动移除，不修改 Transformers 源码。

不得向 0731 tokenizer 增加 `<image>`：扩词表会同时要求修改输入 embedding、LM head 和 Hash-MoE 的 `tid2eid` 表。0731 已含多个视觉相关 token，因此没有必要承担该风险。

= 已完成验证

#table(
  columns: (2.2fr, 0.7fr, 2fr),
  [*测试*], [*结果*], [*意义*],
  [Placeholder 可变长度扩展和 label mask], [通过], [序列/监督合同],
  [Projector Safetensors round-trip], [通过], [权重可恢复],
  [冻结普通 GPT-2 的 backward], [通过], [通用文本主干],
  [真实 `DeepseekV4ForCausalLM` 1 层 Hash-MoE], [通过], [DeepSeek 路由合同],
  [MoonViT-V2 输出 shape/freeze 合同], [通过], [真实 K3 视觉权重],
  [真实 MoonViT-V2 + 小 LM 前向/反向], [通过], [V100 CUDA eager/sdpa],
  [eval\_vlm 生成/blind/shuffle-loss 干跑], [通过], [评测管线端到端],
  [generate()：generic 与 deepseek_v4 两种路径], [通过], [评测/推理前置],
  [指标库（VQA/ANLS/token-F1/grounding）], [通过], [纯 Python，无 torch],
  [完整测试集], [184/184], [Linux + torch 2.10.0+cu128；含包 7 masked-padding、无 random 轨迹、缓存与非单调判定回归],
)

V100 实测：真实 K3 MoonViT-V2 在 1024×1024 输入下输出 `[1369,4,1024]`，特征全部 finite、逐位确定，eager 与 sdpa 最大绝对差 3.1e-05；真实权重 strict-load、预处理与 loss/backward 合同均正常。旧 V1 路径也保留了 448px 和原生分辨率回归，但不再代表当前训练主线。评测管线的生成、blind 基线与 shuffle-loss 全部端到端通过；训练后 shuffle-loss 差值应当变正，这是对齐信号最便宜的读数。

离线 smoke 结果：输入 6 token 扩展为 8 token；projector 六组参数均获得梯度；语言模型参数梯度数为 0。同一结果在 doesworkstation（V100）上复现。

版本审计发现，公开 PyPI Transformers 4.57.6 不含 `deepseek_v4` 模块；真实 DeepSeek 类测试使用 Transformers 5.14.1。因此统一环境暂定 `transformers>=5.12,<6`，不能盲从 checkpoint config 中的历史版本字符串。

= MoonViT-V2 尺寸与适配性

MoonViT-V2 适合该任务：27 层、hidden size 1024、12 heads、401.2M 参数，原生分辨率和 NaViT packing 对 GUI、文档、截图有价值。它不会显著改变完整 0731 的权重门槛；V1 的 1152 维/16 头规格只作为对照记录。

风险来自 token 数。patch size 为 14，随后 2×2 合并。392×392 图像产生约 196 个合并 token；896×896 产生约 1024 个。第一阶段应把每图上限锁在 1024；更高分辨率必须以梯度激活显存实测决定。

= MoonViT-V2（Kimi K3 视觉塔）移植

按项目决策，视觉塔改为从 Kimi 直接抽取（K3 优先，K2.6 备选），与社区 GLM-5.2V 实验同源（他们从 K2.6 抽）。Kimi-K3 仓库全量约 1.56 TB，但其 safetensors index 显示全部 165 个 `vision_tower.*` 张量集中在单个分片 `model-00096-of-000096.safetensors`（约 802 MB）。抽取与验证全部在自有工作站完成，不依赖租用机器；租服务器时使用抽取后的独立视觉塔产物，不触碰完整 K3 仓库。

K3 的 MoonViT3d（下称 MoonViT-V2）与 MoonViT-SO-400M 的合同差异：vision width 1152 → *1024*（27 层、12 头、qkv 1536、rmsnorm、无 attn bias、divided-fixed 位置插值）；merge 方式 `sd2_tpool`，单图（t=1）时时域池化为恒等，输出仍是每图 `[tokens, 4, 1024]` 的特征组——与既有 PatchMerger 的 `[G,4,C]` 合同同构，胶水层零改动，仅 projector 的 vision_width 换为 1024（flatten 维度 4096）。输入格式为 NaViT packing 的 `[total_patches, 3, 14, 14]` + `grid_thws`，与 V1 的 flatten 格式不同，由 processor 适配器统一。

移植方式：将 K3 仓库中视觉塔所需的最小代码集 vendor 进本仓库（`src/moonvit_glue/vendor/kimi_k3/`，Apache-2.0 + Kimi K3 License，已附 LICENSE），删除文本模型依赖（`modeling_kimi_linear` 要求更新版 Transformers 的 `OutputRecorder`，与本项目 5.12 基线冲突）与条件生成类；加载不再需要 `trust_remote_code`，也不再需要下载完整 K3 仓库。`moonvit_glue.moonvit_v2` 复用既有 `MoonViTEncoder` 契约（freeze、形状校验、preprocess），新增 sdpa varlen attention（K3 代码只注册 flash-attention-2 与 eager；V100 等无 flash-attn 硬件用 eager/sdpa，数值一致性有测试）。

已验证（工作站）：vision-only 模块独立实例化（真实配置 401.2M 参数）；前向输出 `[G,4,1024]` 符合合同；eager 与 sdpa 注意力数值一致；带/不带 `vision_tower.` 前缀的 state-dict 均 strict 加载；processor 适配器产出 `pixel_values`/`image_grid_hws` 合同键；完整测试集 34/34 通过。真实 shard 的 header 元数据与模型 state_dict 逐项比对：165/165 key、形状、dtype（全 BF16）完全一致，且该 shard 不含任何非视觉权重。*服务器端同样只需部分下载*：干净房间测试（PYTHONPATH 仅含本仓库、无 K3 staging）确认 vendored 代码只依赖 stdlib/torch/transformers/numpy/PIL，租机时视觉塔侧只需 git 仓库 + 抽取产物（约 800 MB，附 sha256 MANIFEST 供下载校验），完整 K3 仓库不进入训练链路。

*真实权重回归（2026-08-03）*：shard 下载完成（802,448,352 B，sha256 `9d10c74f…`），`tools/extract_moonvit_v2.py` 抽取 165 张量 / 401.2M 参数并 strict-load 通过，产物 `moonvit_v2.safetensors`（BF16，sha256 `01436a95…`）+ MANIFEST（双哈希）。V100 上真实权重前向：1024×1024 图像 → 5476 patches（74×74 grid）→ `[1369,4,1024]`，全部 finite，特征统计健康（mean 0.0003 / std 0.0511 / absmax 0.60，RMSNorm 后典型量级）；两次前向逐位一致（确定性）；eager 与 sdpa 在真实权重上最大绝对差 3.1e-05（fp32 累加顺序的正常量级）。抽取产物上传至项目 HF 仓库 `vision_tower_k3/` 子目录，训练与评测经 `--vision-tower v2 --moonvit-v2-weights` 使用。

= 本地与云端硬件计划

#table(
  columns: (1.5fr, 1.7fr, 2.2fr),
  [*目标*], [*建议硬件*], [*判断*],
  [胶水/单元测试], [CPU 或任意 8 GB GPU], [已可完成],
  [MoonViT-V2 + 0.5B–3B LM], [12–24 GB GPU], [适合本地开发],
  [V100 32 GB], [SM70、32 GB], [可跑 MoonViT-V2/小 LM；不能装完整 0731],
  [完整 0731 推理验证], [同机 4×48 GB 起步], [上下文和 kernel 受限],
  [projector-only 正式训练], [4×H100 80 GB 或更高], [先过单 batch backward],
  [低风险首跑], [4×H200/B200], [更大激活余量],
)

消费级单卡不能纯 GPU 运行完整 0731。110–192 GB 主机内存配合 3/4-bit CPU offload 可能极慢推理，但 GGUF/llama.cpp 不能作为当前 projector 训练路径。MoonViT 单独可在普通本地机器运行；完成后的完整组合能否“本地跑”，取决于本地是否具备多卡或大内存 offload 条件。

= 分阶段门槛

== Gate A：无需大权重

1. 固定 tokenizer/image token 合同。
2. 固定 MoonViT revision、processor 和输出 shape。
3. 小型纯文本 LM 完成 overfit 与保存/恢复。
4. 真实 DeepSeek-V4 缩小配置完成 Hash-MoE backward。

== Gate B：V100 工作站

1. 在不占用现有 Qwen 优化任务的情况下盘点 CUDA/PyTorch/磁盘。✔ 已完成，并经 tmux 向 fastllm 任务留言协调。
2. 大文件仅写入 `/run/media/ezra/13D010B6FDBC1A06/`。✔ 已确认 `/home` 89% 占用，机械盘约 3.7 TB 可用。
3. 跑 MoonViT standalone 与小 LM；记录 SM70 fallback。✔ 真实 MoonViT-SO-400M 前向/反向与评测管线干跑均通过。torch 2.10.0+cu128 wheel 自带 sm\_70，V100 matmul 与 26/26 测试通过，不需要旧版 PyTorch。
4. 不尝试加载完整 0731。✔ 保持该约束。
5. 小规模 overfit 训练验证信号。✔ 通过（见下节）。

== Gate B 实测：projector overfit 实验（2026-08-02）

数据：109 条 ComfyUI 产出图的英文描述性 caption（`data/comfy_captions.jsonl`，零下载，93 训练 + 16 评测）。设置：冻结 MoonViT-SO-400M（fp32）与 SmolLM2-135M-Instruct（fp32），只训 40.1M 参数 PatchMerger projector；placeholder 用现有 token `<|endoftext|>`（id 0），不扩 vocab；`--max-image-side 448`（每图 192 token）；batch 4，AdamW。

第一次跑（200 步，lr 1e-3，约 8.6 epoch）：训练 loss 4.25 → 3.74，但 shuffle\_delta 仅 +0.007，在噪声内——模型主要学到 caption 文体先验。第二次跑（1000 步，lr 2e-3，约 43 epoch）信号确立：

#table(
  columns: (2.2fr, 1.6fr),
  [*指标*], [*数值*],
  [训练 loss（首窗 → 末窗）], [4.303 → 3.338],
  [评测真图 loss], [3.300],
  [评测打乱图 loss（16 样本 × 5 轮）], [3.642],
  [*shuffle\_delta*], [*+0.343*],
  [训练耗时（V100）], [951 秒（1000 步）],
)

生成对照（8 条评测样本，greedy 48 token）：有图输出随图变化且出现内容词（"Korean dress"、"3D model"、"beard, pastel houses"，token-F1 0.112）；同一模型的 blind 无图输出 8 条逐字节相同（"beautiful, serene landscape..."，token-F1 0.082）。三项证据一致：loss 下降、shuffle\_delta 显著为正、生成随图变化且 blind 恒定——projector 确实把图像信息送进了冻结 LM，训练合同在真实权重上成立。checkpoint 在 `checkpoints/overfit-smollm135-1k`。

注意：评测集 loss（3.30）低于训练集末窗（3.34），且 43 epoch 已过拟合，shuffle\_delta 部分来自对 93 张训练图的记忆；这一 gate 的目的是验证信号通路，不是产出可用 captioner。

第二个 backbone 复测（同一数据与超参，placeholder 自动解析为 `<|image_pad|>`）：Qwen2.5-0.5B-Instruct 1000 步后训练 loss 3.898 → 3.160，评测真图 3.310 vs 打乱图 3.592，*shuffle\_delta = +0.282*，耗时 1194 秒，checkpoint 在 `checkpoints/overfit-qwen05-1k`。两条 backbone 轨给出同量级正 delta，说明该信号来自胶水层与 projector 通路本身，与文本主干选型无关。

第三轨在更大更干净的数据上复测：flickr8k 训练分片 1100 条（1036 训练 + 64 评测；`jxie/flickr8k` 镜像，nlphuji 原仓为 gated；因代理网络反复断流，最终直接从 HF 缓存的 train parquet 离线解出，MANIFEST 记录 resolved revision）。Qwen2.5-0.5B，1500 步、batch 8、lr 2e-3（约 11.6 epoch，记忆成分远小于 comfy 小集）：训练 loss 3.091 → 2.435，评测真图 2.574 vs 打乱图 2.722（64 样本 × 5 轮），*shuffle\_delta = +0.148*，耗时 2102 秒，checkpoint 在 `checkpoints/overfit-qwen05-flickr8k`。在 10 倍数据、1/4 epoch 数下 delta 仍显著为正——信号不依赖小集记忆。

该 checkpoint 的生成对照同样干净：8 条评测样本有图输出为正常 flickr8k 风格 caption（"A boy in a red shirt and blue shorts is holding a toy"，token-F1 *0.284*）；blind 无图输出全部退化为同一句拒绝话术（"I'm sorry, but you haven't provided an image..."，token-F1 *0.0*）。

三轨汇总（判据均为真图 vs 打乱图 teacher-forced loss 差）：

#table(
  columns: (1.7fr, 1.9fr, 1.3fr, 1.1fr, 1.2fr),
  [*数据*], [*文本主干*], [*步数/epoch*], [*delta*], [*结论*],
  [comfy 109 条], [SmolLM2-135M], [1000 / 43], [+0.343], [通过],
  [comfy 109 条], [Qwen2.5-0.5B], [1000 / 43], [+0.282], [通过],
  [flickr8k 1100 条], [Qwen2.5-0.5B], [1500 / 11.6], [+0.148], [通过],
)

Gate B 结论：胶水层 + projector 训练合同在真实权重、两个文本主干、两个数据集上全部成立，可以进入 Gate D（租卡验证 0731 大权重的 Dgrad 通路）。

工作站实测注意项：

- NVML 版本不匹配（内核模块 580.159.04 vs 用户态库 580.173）：`nvidia-smi` 不可用，但 CUDA 初始化与 kernel 运行正常。不要为修复它而重载驱动或重启——同机其他任务正在使用 GPU。
- MoonViT remote code 与 Transformers 5.x 有两处不兼容：缺 `all_tied_weights_keys`（已在 `moonvit.py` shim，ViT 无 tied weights，空映射语义正确）；bf16 下 remote code 内部存在 float32/bf16 混用的 layer\_norm 调用，V100 上 MoonViT 以 fp32 运行（约 1.6 GB，可接受）。
- 小上下文文本模型装不下原生分辨率视觉 token：640×480 照片产生 1064 个合并 token，加 prompt 后超过 tiny-gpt2 的 1024 positions，表现为 scatter-gather device assert（异步报错会漂移到无关位置，需 CUDA\_LAUNCH\_BLOCKING=1 定位）。`--max-image-side` 控制 token 数；正式 0731 的 128k 上下文无此问题。
- HF 下载须走本机代理 `127.0.0.1:7890`（约 0.4–0.5 MB/s），且该仓库默认 Xet 传输在代理下会挂起，必须设 `HF_HUB_DISABLE_XET=1`。
- 复用现有 venv（torch 2.10.0+cu128 + transformers 5.12.1）；pytest 用 `pip --target` 装在独立目录，不改对方环境。
- 工作站负载高（其他 agent 编译/跑 inference server）时，机械盘上的 torch import 可达 90 秒以上；批处理脚本超时预算要放宽。

== Gate B+：租前全链路彩排（2026-08-03，正式训练的对照组）

目的不是能力验证，而是把租期闭环的每一个环节在 V100 上先炸一遍：离线取数 → 切分 → 训练（checkpoint 流式传 HF）→ trained/blind/random 三组评测 → 聚合 → 全部产物落 HF。设置：SmolLM2-135M-Instruct + 真实 K3 MoonViT-V2 权重（eager），flickr8k 1200 条（train parquet 本地直读，零下载；该数据集每行一图、只用 caption_0，行切分即图切分，无泄露），1000 训练 / 200 评测，400 步、batch 4、lr 5e-4（Baseten 配方默认值），placeholder 为现有 token `<|endoftext|>`，`--max-image-side 448`。

#table(
  columns: (2.4fr, 1.5fr, 1.5fr, 1.5fr),
  [*组别*], [*vision token-F1*], [*blind token-F1*], [*gap*],
  [训练后 projector（400 步）], [0.1565], [0.0391], [*+0.117*],
  [随机 projector（管线对照）], [0.0584], [0.0391], [+0.019],
)

训练 loss 4.30 → 3.06（前 200 步）。三点结论：(1) 训练组 gap 是对照组的 6 倍，管线对"是否训练过"可判别——评测不是摆设；(2) checkpoint 流式上传经代理真实落地（HF 上 step-000200/000400 含 projector fp32+bf16、training_state、history.json），租期中断不丢训练；(3) 全部评测原始输出（逐条预测 + 元数据）与 SUMMARY 已公开在 `eval/v100-smoke-smollm135/`。注意：400 步远低于 grokking 区间（约 900 步起），135M 主干 + caption 任务的绝对 F1 不代表正式结果；此表是正式训练的*对照组基线*。

本轮同时打通了下载通道的完整答案（此前三种路径全部失败）：hf\_transfer 在代理下 0 字节挂起、hf-mirror 限流、工作站的 datasets+dill 无法 pickle pyarrow 的 MonthDayNano（任何本地 parquet 都炸 `load_dataset`）。最终方案：aria2 `-x8 -c --max-tries=0` 预取 parquet 分片（xet-bridge CDN 随机 TLS 重置/403，无限重试磨过去）+ fetch 全程离线读本地 parquet（raw pyarrow，跳过 datasets 指纹层），新增 `tools/prefetch_parquet.py` 与 `tools/fetch_art_data.py`（0xSero art 数据集的离线复刻，schema 与 `build_train_mix` 兼容，有测试）。另修正一处数据规格错误：MMMU-Pro 仓库不存在裸 `standard` config，已固定为 `standard (10 options)`。

同日下午的带宽攻防补充了一课：(1) ModelScope 存在与 HF 逐字节一致的镜像（`lmms-lab/textvqa`、`lmms-lab/DocVQA`、`AI-ModelScope/MMMU_Pro`、`showlab/ShowUI-desktop`），境内直连可达 8.8 MB/s，但工作站 IP 在约 1.5 小时高强度拉取后被 CDN 边缘渐进限速至 0 B/s（按 IP 不按账号，token 与 IPv6 均无效；本机家庭 IP 同文件仍有 5.3 MB/s）；(2) 随即搭建的"本机 ModelScope → scp 回传工作站"中继受 Tailscale 链路限制（3.7 s RTT，多流聚合仅约 250–400 KB/s），与代理通道同速且挤占家庭带宽，应用户要求退役；(3) 最终全部十个数据源统一由工作站经 Clash 代理直下（`moondata` 八个评测/训练集 + `moonart` 的 WikiArt/fashion），代理总带宽封顶约 250 KB/s（并行不扩展，单连接约 50 KB/s 即节点拥塞），剩余约 15 GB 预计 16 小时；mihomo 核心未开 external-controller，换节点只能由用户在 GUI 操作，这是当前唯一可能提速一个数量级的杠杆。也曾评估 \$0.056/h 的香港数据盒（2–3 小时收工、总成本 < \$0.5）作为替代，用户选择免费慢磨方案。

== Gate B 全量 mix early-alignment（2026-08-04，V100，工程/信号闸门）

目的：在正式 train\_v1 mix（59,198 行 packed parquet）上验证"parquet 消费路径 + 全量数据 + early alignment + 全套评测 + 上传"的租期闭环，同时作为正式 0731 训练的本地对照组。设置：Qwen2.5-0.5B-Instruct（冻结）+ K3 MoonViT-V2（冻结，eager），只训 projector（33.6M 参数）,2000 个 optimizer step、恒定 lr 5e-4、`--max-image-side 448`。历史参数 `batch 8` 实际不是一次 8 样本 forward，而是 `micro_batch_size=1` 下串行 8 次 forward/backward 后更新，因此本轮只见过 16,000 样本，即 59,198 条 mix 的约 *0.27 epoch*；历史 answer-token 数未记录，不能精确补齐。占位 token 自动解析为 Qwen 词表已有的 `<|image_pad|>`（ID 151643，不扩词表；这是 0.5B 历史运行身份，当前 3B 合同固定使用 ID 151655）。

训练结果（train.log 实测）:

#table(
  columns: (2.2fr, 1.8fr),
  [*指标*], [*数值*],
  [loss（首窗口 → 末窗口）], [6.413 → 3.006（单步 10.19 → 3.01，最低 2.718 \@ step 1750）],
  [held-out 真实 loss（32 条）], [3.175],
  [held-out 打乱图像 loss], [3.902],
  [*shuffle\_delta*], [*+0.727*],
)

历史口径下 shuffle\_delta +0.727 是此前三条 smoke 轨（+0.343 / +0.282 / +0.148）的两倍以上：正确图片已经影响答案 loss，视觉接口可学习性成立。但留出集只是 shuffle 后末尾 32 条，乱配使用循环平移而非多组随机 derangement；加上训练量仅 0.27 epoch，该数字不能证明充分收敛，五项低分也不能写成架构能力上限。本轮同时炸出两个租前必须暴露的问题，均已定位：

1. *checkpoint 流式上传全部失败*（step 500/1000/1500/2000 共 4 次）：训练期间代理 SSL 持续重置（`UNEXPECTED_EOF_WHILE_READING` 于 S3 multipart PUT），重试 5 次仍败。训练本身无损，4 个 checkpoint 本地完好；按"上传下载串行"纪律，等评测结束带宽空出后手动一次性补传。教训：租机上 checkpoint 上传同样要与数据集预取错开。
2. *10 个评测进程全部启动即崩*：训练器默认占位 token 是 Qwen 的 `<|image_pad|>`，而评测器默认是 DeepSeek 的 `<｜image｜>`——两者默认值不一致，Qwen tokenizer 里没有后者，`resolve_placeholder_token_id` 按设计拒绝扩词表而报错。修复为候选自动探测（DeepSeek token 优先、Qwen token 兜底；显式指定仍严格报错），85/85 测试通过（commit `034be8b`）。此 bug 若不在本地炸出，租机首跑 DeepSeek 时训练器会以旧默认 `<|image_pad|>` 直接崩在 0731 tokenizer 上——本地彩排的价值正在于此。

=== 五基准评测终表（selection 半侧，1024px，逐条原始输出已公开于 HF `eval/v100-fullmix-qwen05/`）

#table(
  columns: (1.4fr, 1.3fr, 1fr, 1fr, 1fr, 1.9fr),
  [*基准*], [*指标*], [*trained*], [*random*], [*blind*], [*读法*],
  [TextVQA (250)], [soft-VQA], [*0.081*], [0.000], [0.000], [真实对齐，距成熟 VLM 远],
  [DocVQA (100)], [ANLS], [*0.039*], [0.000], [0.000], [同上，实体级读错居多],
  [OCRBench (100)], [exact], [0.000], [0.000], [0.000], [地板；小字 OCR 超出 0.5B 能力],
  [ScreenSpot (100)，域内], [解析率 / 精度\@50], [*0.51 / 0.01*], [0 / 0], [0 / 0], [格式学会一半，点位退化（常数坐标，中位误差 729/999）],
  [MMMU-Pro (150)], [exact], [*0.073*], [0.000], [0.000], [修复输入后 2.0%→7.3%；宽松提取 18%],
)

判别力三条全部成立：vision 严格大于 blind；trained 严格大于 random（随机 projector 臂五项全零——冻结 LLM 收到噪声视觉 embedding 时输出不了任何切题内容）;shuffle\_delta +0.727。Gate B 的最小结论是：*信号真实、接口可学习、评测可判别、checkpoint/上传管线可用（但易受代理抖动影响，需错峰）*。它不是充分训练后的能力评测。0.5B 冻结主干 + 33.6M projector + 仅 1.6 万样本见过的训练量只承担 early-alignment 对照角色；下一步先用等 examples-seen 的 0.5B/1.5B/3B 纯文本主干、projector/LoRA、分辨率与因果控制消融定位瓶颈，再估算 0731 Hash-MoE 实验。

=== 0.5B 主干的容量混杂

Qwen2.5-0.5B-Instruct 是纯文本 `Qwen2ForCausalLM`，没有继承原生视觉能力，但它的容量本身显著影响实验。OCR、复杂视觉推理、指令/输出格式遵从均受 0.5B 语言主干的能力上限约束，所以 Gate B 的低绝对分数更接近能力下界，不能外推完整 0731 的最终分数。反过来，小型 dense Qwen 与 DeepSeek-V4 Hash-MoE 的表示空间和优化曲面不同；它可能更易对齐，也可能因容量不足更难对齐，因此 +0.727 的收敛速度同样不能预测 0731 的训练速度。

不受这一混杂影响的最小结论只有：在一个确实纯文本的冻结 LM 上，训练后的 MoonViT-V2 projector 相对 random/blind 产生了可重复的图像依赖信号。故 Gate B 的正式定位降格为“工程与信号闸门”，不是能力代理。若在租机前追加中间尺度对照，应优先使用配置明确为 `ForCausalLM` 且无 `vision_config` 的约 3B 纯文本主干（7B 可选），保持相同 train mix、seen-record 数、分辨率、scratch 初始化和 selection benchmark；比较 loss、shuffle 统计及五项 vision−blind。该实验能量化容量敏感性，但仍不能替代完整 DeepSeek Gate D。当前工作站未缓存 3B/7B 纯文本权重，尚未运行此对照。

=== 证据边界：原生 VLM 不是 DeepSeek 代理

必须区分两个名字相近但因果意义完全不同的 Qwen 实验。Gate B 的冻结文本主干配置是 `Qwen2ForCausalLM`，且 `vision_config` 为空；它本身没有图像输入路径，视觉信号只能来自外接 K3 MoonViT-V2 与本项目训练的 projector。因此 shuffle\_delta 和 trained/random/blind 差异可以证明“纯文本 LM 接口可学习”。训练器现已加入硬防线：`--text-model` 若暴露 `vision_config` 会直接拒绝，原生 VLM 只能进入独立的 stock-eval 路径。

另一方面，Qwen3.5-4B 对照的配置是 `Qwen3_5ForConditionalGeneration` 且自带 `vision_config`；它使用官方视觉塔、processor、chat template 和既有多模态对齐，完全不经过 MoonViT-V2 或本项目 projector。它的高分只校验评测数据、图像读取、输出约束和评分器，并给出成熟小型 VLM 的参照上界；*不得据此推断 projector 能快速映射到 DeepSeek*。证据链严格分为：(1) 原生 VLM 阳性对照＝评测有效；(2) 纯文本小主干 Gate B＝接口可学；(3) tiny DeepSeek-V4 Hash-MoE＝路由与梯度合同；(4) 完整 0731 Gate D/正式训练＝目标可行性与能力，前三项均不能替代第四项。

原生 Qwen3.5-4B 的修复后阳性对照如下。它与 Gate B 使用相同的 selection 半侧、1024px 上限和 strict scorer；“vision”列是 Qwen 自带视觉塔的结果，不是本项目 projector 的结果。

#table(
  columns: (1.35fr, 1.2fr, 1.25fr, 1.05fr, 1.25fr, 1.05fr),
  [*基准*], [*指标*], [*Gate B trained*], [*Gate B blind*], [*原生 Qwen vision*], [*Qwen blind*],
  [TextVQA (250)], [soft-VQA], [0.081], [0.000], [*0.820*], [0.031],
  [DocVQA (100)], [ANLS], [0.039], [0.000], [*0.926*], [0.071],
  [OCRBench (100)], [exact], [0.000], [0.000], [*0.900*], [0.000],
  [ScreenSpot (100)], [parse / acc\@50], [0.51 / 0.01], [0 / 0], [*0.86 / 0.76*], [0.99 / 0.01],
  [MMMU-Pro (150)], [exact], [0.073], [0.000], [*0.300*], [0.280],
)

前三个感知/OCR 基准与 ScreenSpot 的 vision−blind 差距很大，证明图像读取与评分管线已恢复健康；MMMU-Pro 只有 +0.020，30% 中绝大部分来自题干、选项与语言知识先验，不能全部计作视觉能力。这正是强制报告 blind 的价值，也再次说明原生 VLM 总分不能成为 DeepSeek 映射证据。

该对照还产生了一次必须保留的完整性事故记录：首次运行中，第二个 3.99 GB safetensors 分片虽字节数正确，SHA-256 却为 `547d2f…8627`（官方 `cb544b…e188`），而模型索引恰把全部视觉塔权重放在该分片，导致模型把真实图片一致描述成空白。高分辨率样本又触发 OOM，并被工作站的 NVML 驱动/库版本不一致掩盖成 allocator assert；开放式长回答则会被严格短答案指标计零。修复措施为：逐分片 SHA-256 manifest、`--max-image-side 1024`、按指标约束短答案/选项字母/归一化坐标。损坏运行的产物从未上传；修复套件在一次完整哈希验证后运行，原始逐条输出与 suite provenance 单独发布。

=== 误判审计（应用户要求：先怀疑判别器，再怀疑模型）

对五个基准逐条重打分，用逐级放宽的提取器量化"判错"成分。结论：*判别器基本清白，最宽提取也只多 1–2 分*——textvqa 单复数漏判 0 条、子串漏判 4 条（+1.6% 封顶）；screenspot 不可解析的 49% 中 48 条是纯散文无坐标（模型在描述而非点击），仅 1 条格式边缘漏判；ocrbench 用官方 contains 式口径依然全零；docvqa 无 32-token 截断。但审计抓到两个真 *输入侧* bug:(1) MMMU-Pro 的 `options` 列在 parquet 里是*字符串装的 Python 列表字面量*，旧代码直接 `join` 字符串导致选项被逐字符拆行，prompt 不可读；(2) 渲染选项缺字母标签，而参考答案是字母（"B"），模型无从对应。修复（解析字面量 + `A./B./C.` 标签，commit `bf3413b`/`f9b94a5`，含测试）后离线重建 300 条数据并重测：2.0% → 7.3%（严格），宽松"字母出现即算"18%——0.5B 输出多为推理散文、没有落字母的习惯，严格 exact-match 只承认单字母输出。此基准的 2.0% 旧读数作废，以修复版为准。

评测补跑（5 基准 × trained/random × vision/blind）已全部完成并聚合；仓库归置（Qwen 对照产物归入 `gate_b_qwen05_v100/`、smoke 产物归入 `gate_b_smoke_smollm135_v100/`、README 标注与 DeepSeek 目标无关）与 4 个 checkpoint 补传按串行纪律排在 4B 下载之后。

= 冻结语言主干中的视觉感知涌现：V100 方向筛选实验

本章对应新的租前方向筛选阶段；约束是只使用现有 V100 32 GB，不查看 final evaluation half、不租服务器、不启动完整 DeepSeek-V4。研究问题是：(1) frozen text LM 何时开始在可测指标上依赖正确图片；(2) MoonViT 可线性解码的信息经过 projector 后保留/丢失什么；(3) 视觉属性在哪些 LM 层与 token 位置可解码并影响答案；(4) 下一轮应优先归因于训练量、projector、空间/OCR、语言容量还是冻结主干。

可证伪假设预注册如下：若 MoonViT probe 高而 projector 输出骤降，则支持 projector 信息瓶颈；若 projector/LM residual probe 高而生成与干预恢复低，则支持语言适应/解码瓶颈；若所有 checkpoint 曲线仍持续上升，则先支持训练量不足；若 0.5B 在等 examples-seen 下被 1.5B 清晰超过，才提高语言容量瓶颈证据；这些 probe/相似度只作相关性读数，因果结论必须来自 blank/wrong-image、mask、feature/activation patching 的答案 logit 变化。

== 实验基础设施与真实计量（包 1，2026-08-04）

环境快照 run `v100-perception-infra-20260804-a`：起始 commit `20c2556`；Tesla V100-PCIE-32GB（34,072,559,616 B，sm\_70）；Python 3.12.11；torch 2.10.0+cu128；CUDA build 12.8；cuDNN 91002；Transformers 5.12.1；safetensors 0.8.0。`nvidia-smi` 仍因已知 NVML 用户态/内核不匹配失败，但 PyTorch CUDA 分配与计算正常。环境 JSON、pip freeze、GPU device users 和 git dirty state 均提交于 `experiments/v100_perception_20260804/infra/environment/`。

冻结 MoonViT-V2 特征缓存采用分片 safetensors + MANIFEST：逐样本记录 ID、原图 SHA-256、原图尺寸、feature shape/dtype、shard offset；全局记录 MoonViT config/weights hash、分辨率、数据 logical-row hash、cache format version；每个 shard 自带 bytes 与 SHA-256。有效 run `v100-perception-cache-20260804-retry1` 在 448px 缓存 64/64、0 失败、14.274s、峰值 1,948,235,264 B；4 个 shard 全部二次 hash 和逐 ID 读回通过，records hash `98f81a46…55d2a`，MoonViT 权重 hash `01436a95…ced24`。第一次 foreground SSH 尝试因 stdout 断开触发 BrokenPipe，只有 3 条临时行且无 manifest，明确标记 invalid 并保留，不参与任何结论。

真实 step-time 使用同一 64 条、同一 scratch projector、相同 448px cache，5 step 中排除首个 CUDA warm-up，结果如下。三组的 `micro_batch_size` 都是 1，`actual_batched_forward=false`；所谓 batch 4/8 是串行 gradient accumulation。

#table(
  columns: (1.2fr, 1.2fr, 1.6fr, 1.5fr, 1.5fr),
  [*micro batch*], [*grad accum*], [*mean s/optimizer step*], [*examples/s*], [*peak GPU memory*],
  [1], [1], [0.0747], [13.38], [3.31 GB],
  [1], [4], [0.2720], [14.70], [3.65 GB],
  [1], [8], [0.5390], [14.84], [3.65 GB],
)

因此旧 `batch 8` 的真实成本约为 0.54s/optimizer step（在 0.5B + 已缓存视觉特征上），不是一次 8 样本 forward。线性外推到 accumulation 64 约为 4.3s/step，但该数字仍不能外推 DeepSeek；它只用于证明计量语义与 V100 小模型预算。retry2 的单步计时有效，但审查发现它仍在启动时实例化了不参与 forward 的 401M MoonViT，故其 4.92–5.25 GB peak 不作为缓存结论；修复后 retry3 完全不构造视觉塔，peak 降到 3.31–3.65 GB，且三份 report 均记录 `vision_tower_instantiated=false`。两次 step-0 无效启动也完整保留：第一次在线 HEAD 超时；第二次启用 offline 后未设置 HDD cache 路径，三臂均失败并暴露零成功行 CSV 的 driver bug。checkpoint 轨迹、probe、干预和方向判定尚未运行，当前不满足租卡前 go 条件。

== Synthetic Perception Diagnostic（包 2，2026-08-04）

生成器 `synthetic-perception-v1` 固定 seed `20260804`，以 Pillow 12.2.0 在 256×256 画布上生成 color、shape、count (1–9)、spatial（left/right、above/below、inside/outside、nearest）、OCR（2–6 位无歧义大写字母/数字）和 3×3 coordinate 六类任务。每个任务在 train 与 selection 各有 200 个基础问题；每个基础问题有 a/b 两张图，问题文本逐字节相同、答案不同，生成参数明确记录唯一变化的视觉属性。因此总计 2,400 个基础 minimal pairs、4,800 张图。train/selection 使用不同背景、边框和问题模板；图像 SHA、OCR 字符串、pair ID、template ID 的跨 split 交集均为 0。OCR 字符本身是必须读取的刺激，不另绘任何答案标签或提示文本。

#table(
  columns: (1.4fr, 1.5fr, 1.5fr, 2.5fr),
  [*任务*], [*train base/rendered*], [*selection base/rendered*], [*pair 中唯一目标变化*],
  [颜色], [200 / 400], [200 / 400], [单一图形的填充色],
  [形状], [200 / 400], [200 / 400], [图形几何形状],
  [计数], [200 / 400], [200 / 400], [可见物体数 1–9],
  [空间], [200 / 400], [200 / 400], [目标物位置/关系],
  [OCR], [200 / 400], [200 / 400], [2–6 位 glyph sequence],
  [坐标], [200 / 400], [200 / 400], [3×3 网格中的目标位置],
)

每条记录另有完整控制分配：blind 不提供图；blank 使用同 split 的纯背景；same-image 对该 split 所有样本使用同一张中性条纹图；shuffled-image 在任务内作无固定点的确定性 derangement；patch-permutation 给出逐样本 seed，在 MoonViT merged spatial-token 轴执行 `torch.randperm`，保留值与 token 数量。独立 verifier 检查了 4,800/4,800 图像 SHA、2,400/2,400 pair、4,800/4,800 控制行和全部派生文件 hash；失败数为 0，logical dataset SHA 为 `122ae820…cbaa71`。完整 PNG 在 V100 数据盘，Git 提交完整 train/selection/control JSONL、manifest/hash、计数 CSV、日志、零失败文件、验证结果与每类一组 a/b 预览。数据生成阶段没有据输出改 scorer；普通/paired/answer-flip 与控制分母由包 3 统一计算，见下一节。

== Checkpoint-wise Emergence of Visual Dependence（包 3，2026-08-05）

=== 固定口径、checkpoint 与有效性

包 3 从 commit `1cc6f631…` 的四个历史 projector checkpoint 读取 step 500/1000/1500/2000；examples seen 分别为 4,000/8,000/12,000/16,000，对应 0.0676/0.1352/0.2028/0.2704 effective epoch。`current-final` 与 step 2000 是同一权重，只作为别名，不重复推理。训练起点张量未被历史 run 保存，step 0 使用同构 projector 与 seed 0 重建，明确标为 matched random initialization control。所有 checkpoint 的配置、safetensors SHA、来源路径与别名关系由独立 manifest 交叉校验。

Teacher-forced 评测覆盖完整 authoritative synthetic selection：2,400 图、1,200 对、六任务、五个独立 checkpoint、八种条件，共 96,000 条 answer log-prob 原始记录。自由生成在运行前以 seed `20260804` 对 pair ID 做逐任务 SHA-256 排名，固定为每类 50 对/100 图，共 300 对/600 图；manifest 保存全部 ID、源 selection SHA `51ce2741…d6` 和 logical SHA `122ae820…a71`。这一子集只缩减自由生成计算；teacher-forced 分母保持完整。背景辅助集复用每个 authoritative 场景和答案，只把 selection 背景替换为 train 背景；2,400/2,400 特征缓存完成，明确 `diagnostic_only=true`、`training=false`。

有效性审计保留了三类无效产物。`preference_v1` 的数值本身与修复后 v2 在 96,000 条上逐位一致，但 36,000 条 control `visual_source_id` 写错，故整 run 标 invalid 后从头重跑；v2 独立 verifier 通过。generation v3 在 batch 2 的 1024px benchmark 筛选中仅约 0.8 sample/s，未完成 step 0；v4 在不足 1% 时确认 batch 16 仍只有约 8.3 synthetic sample/s，均保存 partial raw、逐文件 hash 与 invalidation。最终批量筛选中，synthetic/benchmark batch 64/16 为 221.3s、峰值 9.82 GB；128/32 为 215.4s、峰值 18.57 GB，仅快 2.6%，最终采用 64/16。筛选器另发现 `limit > heldout count` 的分母边界 bug；该 attempt 在生成前退出，修复加入测试后重跑。

=== Teacher-forced paired preference：shape 在 step 1500 短暂出现

最强证据集中在 shape。step 1500 的 authoritative vision strict paired-preference 为 *0.130*，mean correct margin 为 *+0.133*；matched-random 分别为 0.055/−0.021。成对 bootstrap 的 trained−random strict gap 为 *+0.075，95% CI [0.025, 0.135]*。同一 checkpoint 的 shuffled image strict preference 只有 0.015，vision−shuffle 为 *+0.115 [0.070, 0.160]*；paired-counterfactual image 把 mean margin 精确翻成 −0.133，vision−counterfactual margin 为 *+0.266 [0.219, 0.313]*。blind、blank、same-image 均为 strict 0；patch permutation 为 0.005。该组结果同时满足训练对照、样本级错图控制与单属性反事实三条因果约束。

#table(
  columns: (1.15fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  [*任务 / strict paired preference*], [*step 0*], [*step 500*], [*step 1000*], [*step 1500*], [*step 2000*],
  [color], [0.015], [0.015], [0], [0], [0],
  [shape], [0.055], [0], [0], [*0.130*], [0],
  [count], [0], [0], [0], [0], [0],
  [spatial], [0], [0], [0], [0], [0],
  [OCR], [0.030], [0], [0], [0], [0],
  [coordinate], [0.070], [0], [0.035], [0], [0],
)

表中的随机 projector 非零值说明单看“峰值”会误判；只有 shape/step 1500 同时显著超过 matched random，并在 shuffled 与 paired-image 因果控制下保留。color、count、spatial、OCR、coordinate 在 16,000 examples 内均没有 causally validated onset，现阶段的能力顺序只能写为“shape 首先出现，其余未出现”，不能给其余五类强排顺序。

该 shape 信号随后坍缩：step 2000 strict preference 回到 0，step1500−step2000 为 *+0.130 [0.085, 0.180]*；latest overall 虽保留很小的 vision mean margin +0.0081，strict preference 已为 0，vision−paired-image margin 仍有 +0.0162 [0.0121, 0.0203]。曲线因此反驳当前训练区间内的单调“多训即可”解释，支持表征/解码稳定性问题进入优先诊断。

背景匹配没有把结论翻转。shape/step 1500 的 auxiliary strict preference 为 0.105，authoritative−auxiliary 为 +0.025 [0.005, 0.050]；auxiliary mean margin反而更高（0.171 vs 0.133）。shape 信号在两种背景都存在，且 train 背景没有稳定提高两个 paired 指标，当前没有证据把主要失败归因于背景域偏移。

=== 自由生成：视觉触发存在，正确图像内容没有进入成对答案

自由生成主跑包含 37,300 条 generation 原始记录和 160 条 heldout shuffle-loss 记录，失败数为 0；总时长 2,745.9s，峰值显存 9.82 GB。独立 verifier 重算六个原始文件的 hash、精确分母、alias、固定子集 manifest 与每个控制的 `visual_source_id` 关系，确认五个独立 checkpoint、`final_half_scored=false`。step 500/1000/1500/2000 的 synthetic vision sample accuracy 分别为 0.1367/0.1550/0.1350/0.1417；严格 paired generation 与 answer-flip 在所有 checkpoint、所有任务都精确为 0。

step 2000 的 vision−blind sample-accuracy gap 为 *+0.1417 [0.1167, 0.1683]*，vision−blank 为 +0.0383 [0.0150, 0.0617]；vision−same-image 为 −0.0033 [−0.0183, 0.0100]，vision−shuffled-image 同为 −0.0033 [−0.0150, 0.0083]。视觉 token 能触发一个盲态没有的答案分布，但正确图、固定中性图与错配图之间没有可检出的内容优势。step 1500 的 shape 单样本准确率为 0.26，paired generation 仍为 0；同一数据上的 teacher-forced strict paired preference 为 0.13。这组差异直接支持“已形成可评分的内部证据、现有解码没有稳定使用它”。

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("../experiments/v100_perception_20260804/checkpoint_trajectory_v1/charts/02-paired-preference.svg", width: 100%),
    image("../experiments/v100_perception_20260804/checkpoint_trajectory_v1/charts/10b-paired-evidence-vs-paired-generation.svg", width: 100%),
  ),
  caption: [左：teacher-forced strict paired preference；右：逐任务内部 paired 证据与 paired 自由生成。shape/step 1500 是唯一经过因果控制的非零训练信号，自由生成轴仍为 0。],
)

=== 真实 heldout 与 benchmark 轨迹

32 条历史 heldout、每条十次 derangement 的 mean shuffled-minus-true loss 从 matched random 的 −0.0095 变为 step 500/1000/1500/2000 的 +1.0263/+0.6703/+0.9533/+1.1886；step 2000 repeat std 为 0.2477。projector 很早就学会让真实图像降低训练分布上的 teacher-forced loss，此后各 checkpoint 始终为正且 step 2000 最强。它测到全局图文匹配依赖，无法单独证明细粒度答案内容已进入生成。

step 2000 的 selection-half benchmark vision/blind 为：DocVQA ANLS 0.026/0，TextVQA 0.008/0，OCRBench 0/0，ScreenSpot accuracy 0.010/0（parse 0.19/0），MMMU-Pro exact 0.0067/0。blank 与 shuffled 控制也能保留部分低分，例如 TextVQA blank 0.0093、MMMU-Pro blank/shuffled 均为 0.0067；绝对 benchmark 分数因此继续只作 0.5B、0.27 epoch 条件下的诊断。ScreenSpot 的最常见坐标和 collapse rate 已逐 checkpoint 保存，未见可用 grounding 能力。

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("../experiments/v100_perception_20260804/checkpoint_trajectory_v1/charts/05-true-shuffled-loss.svg", width: 100%),
    image("../experiments/v100_perception_20260804/checkpoint_trajectory_v1/charts/08-control-checkpoint-heatmap.svg", width: 100%),
  ),
  caption: [左：heldout 真图与错配图 loss 轨迹；右：synthetic 控制条件相对 vision 的 correct-margin gap。],
)

=== 假设更新与下一项本地实验

#table(
  columns: (1.55fr, 1fr, 2.7fr),
  [*假设*], [*包 3 更新*], [*证据与动作*],
  [内部表征存在、解码使用失败], [强支持], [shape/step 1500：teacher paired 0.13，generation paired 0；先做 layerwise probe 与 upper-layer activation patching。],
  [现有训练量只需直接延长], [削弱], [shape 在 step 1500 出现、step 2000 归零，反对当前区间的单调外推；延长训练需等定位结果并加入稳定化监控。],
  [信息在 projector/冻结主干间丢失], [待定位], [shape 首先出现，OCR/coordinate 无因果 onset；需直接测 projector token 与逐层 hidden state 的属性可分性、位置池化与 patch 消融。],
  [背景域偏移是主因], [弱支持], [shape 在 authoritative 与 background-matched auxiliary 都存在，辅助背景没有一致改善。],
  [0.5B 语言容量是主瓶颈], [尚未判定], [当前结果与容量瓶颈相容，也与冻结上层不会使用视觉证据相容；机制定位后再决定 Qwen2.5-1.5B 对照或顶部 LoRA。],
)

下一包固定比较 step 1500 与 step 2000：在 projector 输出和 Qwen2.5-0.5B 全 24 层训练逐层 linear probe；用正确图激活 patch paired-counterfactual run 的图像 span，并对 upper layers 做最小筛选；加入 random-label、blind、shuffled-image 和 layer-0 控制。判别目标是定位 shape 信息何时可线性恢复、何处在 step 2000 丢失，以及少量因果替换能否恢复 correct margin。结果再决定 projector 辅助目标、顶部 LoRA、延长训练或 1.5B 容量对照的优先级。

== Shape 的逐层机制定位（包 4，2026-08-05）

=== 冻结口径与校准 null

包 4 使用 package-3 唯一通过因果控制的 shape 任务。synthetic train 与 selection 各固定 200 个完整 a/b pair、400 条记录，两者 ID 与 pair overlap 均为 0；activation patching 在运行前以 seed `20260808` 对 pair ID 做 SHA-256 排名，固定 50 pair/100 个方向。checkpoint 为 matched random、step 1500 与 step 2000。表示抽取覆盖 train vision，以及 selection 的 vision、paired-counterfactual、shuffled-image、patch-permutation；每个 cell 保存 tower/projector 三种池化、25 个 hidden-state index 的 assistant 与 image-span mean、类别 logit 和 target/source label，共 59 个 tensor key。

完整表示 run 耗时 122.3s、峰值显存 3.61 GB；30 个 safetensors/metadata 文件约 899 MB，保留在 V100 数据盘并逐文件绑定 bytes/SHA。probe 只在完全隔离的 train split 拟合 class-balanced dual ridge，固定 `alpha=1`，在 selection 上报告 raw/balanced accuracy。早期 v1 分析曾把单个 random-training-label probe 当显著性 null；该诊断方差过高，某 cell 达到 0.49。v1 已标 invalid，v2 对每个 vision cell 使用 2,000 次完整 pair 标签元组置换，最小可报告 p 值为 `1/2001=0.00050`。random-label probe 只保留作过拟合诊断。

=== projector 没丢 shape；训练后的上层把它压回 chance

tower 与 projector 在三个 checkpoint 都至少有一种预注册池化达到 *1.000 balanced accuracy*。matched-random projector 同样达到 1.000，说明高维随机映射本身能保存线性可读的 shape；这个数字不能解释成训练收益。训练 checkpoint 的差异出现在语言主干内部：step 1500 的 assistant probe 在 layer 12 达到 raw *0.790*、balanced *0.816*，pair-permutation p=0.00050、null 95% upper=0.285；step 2000 的峰值后移到 layer 14 并降为 raw *0.605*、balanced *0.632*，p=0.00050、null upper=0.278。

这两个 probe 追随实际图像来源。step1500/layer12 在 paired-counterfactual 条件下 target accuracy 降到 0.075，source accuracy 保持 0.790；shuffled-image 为 target 0.230、source 0.790。step2000/layer14 对应为 0.1125/0.605 与 0.2025/0.605。patch permutation 则把 step1500/step2000 的 target/source 同时降至 0.4475/0.445，说明空间 token 排列也参与了读出。

到 final hidden state，step 1500 与 2000 的 balanced accuracy 都精确回到 *0.250*，pair-permutation p=1；native LM-head 对 vision、paired-counterfactual 与 shuffled-image 也全部为 balanced 0.250。随机主干 final probe 仍有 balanced 0.429，进一步表明训练后的收缩是 checkpoint 特定的上层变换。证据链因此把 shape 瓶颈定位在冻结语言主干的上部使用/保留路径，projector 信息丢失解释在这个任务上被反驳。

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("../experiments/v100_perception_20260804/layerwise_mechanisms_v1/charts/01-assistant-probe.svg", width: 100%),
    image("../experiments/v100_perception_20260804/layerwise_mechanisms_v1/charts/03-tower-projector-probes.svg", width: 100%),
  ),
  caption: [左：assistant 位置的逐层 shape balanced accuracy；右：冻结 tower/projector 的池化读出。训练 checkpoint 的中层峰值在顶部消失，而 projector 始终保留完整可读信息。],
)

=== activation patching：中层存在内容特异因果通路

activation patching 对每个 checkpoint 扫描全部 24 个 decoder layer。目标 run 使用 paired-counterfactual 图，donor 来自同一样本的正确图；分别替换 image span 和最后一个 assistant token。layer 0/5/11/17/23 加入不同 pair、不同标签 donor 与 zero donor。每种干预覆盖 50 个 pair 的两个方向，以 pair 为 bootstrap 单位。最终 run 共 18,300 条原始记录、183 个 cell，耗时 306.1s、峰值显存 3.50 GB。

实现审计发现第一版把 post-final-RMSNorm 的 `hidden_states[-1]` 注入 pre-final-RMSNorm 的 layer-23 hook；旧 smoke 与 18,300 行完整 v1 均保留并标 invalid。v2 从 decoder layer forward hook 捕获精确 pre-final-norm 输出，从头重跑。最终 assistant patch 对每条样本以 `1e-6` 精度复现 clean margin；clean/counter pair margin 反对称误差为 0。

step 1500 的 correct-image-span replacement 在 layer 11 达到 *+0.3538 [0.2506, 0.4569]*；减去 wrong-label donor 的通用替换效应后仍有 *+0.2194 [0.1463, 0.2994]*。step 2000 的 raw 峰值缩为 layer 6 的 *+0.1531 [0.1125, 0.1931]*，预注册 layer 5 的 correct-minus-wrong 为 *+0.0856 [0.0519, 0.1181]*；step1500−step2000 在 layer 11 为 *+0.2219 [0.1338, 0.3094]*。这与 probe 的“后期更弱”一致，同时表明 step 2000 仍残留较早、较小的内容因果路径。

final-layer image-span effect 为 0，因为该层之后没有注意力运算把被替换的图像位置传给 assistant。final assistant patch 是正控制：step 1500 恢复 *+0.2925 [0.1875, 0.3950]*，step 2000 恢复 *+0.0950 [0.0712, 0.1200]*，数值等于各 checkpoint 的 clean−counter mean margin。

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("../experiments/v100_perception_20260804/layerwise_mechanisms_v1/charts/04-image-span-patching.svg", width: 100%),
    image("../experiments/v100_perception_20260804/layerwise_mechanisms_v1/charts/06-content-specific-patching.svg", width: 100%),
  ),
  caption: [左：正确图 image-span 的逐层 margin 恢复；右：正确 donor 减 wrong-label donor 的内容特异效应。step 1500 的 layer-11 路径最强，step 2000 明显减弱并前移。],
)

=== token-position 干预边界与假设更新

在 projector 输入 token 位置上，step 1500 的 center replacement 为 −0.0044 [−0.0119, 0.0025]，outer 为 +0.2969 [0.1963, 0.4088]，full 为 +0.2925 [0.1913, 0.4013]；outer−center 为 *+0.3013 [0.1988, 0.4188]*，full−outer 为 −0.0044 [−0.0119, 0.0025]。step 2000 的 center/outer/full 分别为 +0.0094/+0.0856/+0.0950。MoonViT 的 projector token 已经过全局 contextualization，这组结果只定位 receiver 中的 token position，不能映射成原图背景/前景像素位置。后续若要回答 region 因果问题，必须在视觉塔输入或早期 patch 网格上做遮蔽/替换。

#table(
  columns: (1.55fr, 1fr, 2.7fr),
  [*假设*], [*包 4 更新*], [*证据与动作*],
  [projector 丢失 shape], [反驳], [三个 checkpoint 的 tower/projector 最佳 balanced accuracy 均为 1.000；不再优先靠扩大 projector 容量解决 shape。],
  [冻结主干上层不会稳定使用视觉证据], [强支持], [中层 probe 与 patching 均为正，训练 checkpoint 的 final probe/native head 回到 chance；下一项直接做顶部 LoRA 诊断。],
  [继续 projector-only 训练会单调改善], [进一步削弱], [step 2000 的 mid-layer probe 与内容特异 patch effect 都弱于 step 1500；延长训练必须带 checkpoint paired 监控与 matched continuation 对照。],
  [0.5B 容量是首要瓶颈], [仍未判定], [top-LoRA 若恢复读出，先确认适配瓶颈；若失败，再以相同 examples seen 跑 Qwen2.5-1.5B 容量对照。],
  [背景像素驱动 shape 信号], [未由本实验检验], [outer projector-token 位置足够，但这些 token 已全局 contextualized；需要视觉输入/早期网格干预。],
)

下一包冻结 step-1500 projector，以小规模顶部 LoRA 只适配 Qwen2.5-0.5B 上层，并运行 projector-only continuation 与 frozen baseline。训练 checkpoint 同时计算 teacher-forced strict paired preference、paired margin、自由生成 paired/answer-flip，以及逐层 probe 是否能把中层信号传到 final。若 LoRA 有效，继续做层数/rank/辅助目标最小消融；若无效，再转 Qwen2.5-1.5B 容量对照与视觉输入 region 干预。付费 Gate D 继续暂缓。

== Shape 适配诊断（包 5，2026-08-05）

=== 等训练顺序的顶部 LoRA 与 projector 续训

两个 arm 都从 step 1500 projector 出发，使用同一 400 条 shape train、同一 shuffle 顺序、true batch 8，并在 0/50/100/200 optimizer step 保存完整 checkpoint。LoRA 在 Qwen layer 12–23 的 `q_proj/v_proj/o_proj` 注入 rank 8、alpha 16，共 442,368 个可训练参数；projector arm 继续训练全部 20,454,272 个参数。两份 `training_order.jsonl` 的 SHA-256 均为 `993f1b2e…6988`。LoRA 与 projector 的 200 步训练分别耗时 64.2s/74.6s，峰值显存 3.74/4.41 GB。

#table(
  columns: (1.8fr, 1fr, 1.15fr, 1.15fr),
  [*状态*], [*已见样本*], [*strict paired preference*], [*自由生成 paired*],
  [frozen step 1500], [0], [0.130], [0.000],
  [LoRA step 50], [400], [0.180], [0.000],
  [LoRA step 100], [800], [0.605], [0.080],
  [LoRA step 200], [1,600], [0.430], [0.080],
  [projector step 50], [400], [*1.000*], [*1.000*],
  [projector step 100], [800], [0.945], [0.880],
  [projector step 200], [1,600], [0.945], [0.880],
)

projector step 50 相对 frozen 的 strict paired preference 提升为 *+0.870 [0.825, 0.915]*，自由生成 paired 提升为 *+1.000 [1.000, 1.000]*。其 vision/shuffled strict paired 为 1.000/0.180，差值 *+0.820 [0.765, 0.870]*；vision−paired-counterfactual correct-margin 为 *+20.4459 [19.7584, 21.1797]*。这排除了只记住 answer 先验的解释。等 400 个样本下，LoRA−projector 的 strict paired 为 *−0.820 [−0.870, −0.765]*，生成 paired 为 *−1.000 [−1.000, −1.000]*。

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("../experiments/v100_perception_20260804/shape_adaptation_v1/charts/01-paired-preference.svg", width: 100%),
    image("../experiments/v100_perception_20260804/shape_adaptation_v1/charts/02-paired-generation.svg", width: 100%),
  ),
  caption: [左：strict paired preference；右：自由生成 paired accuracy。projector 续训在 400 个 shape 样本处已达到完整配对恢复，LoRA 只恢复一部分。],
)

=== 逐层复测：projector 重新建立稳定的晚层通路

最佳 LoRA step 100 与 projector step 50 都从头抽取 train/selection 表征并重跑 2,000 次 pair-label permutation probe。LoRA 的最佳 assistant 仍在 layer 12，balanced accuracy 为 0.816；layer 14–21 再度压到 chance，final assistant probe 与 native LM-head 都只有 balanced 0.500。它确实改变了自由生成，但没有建立稳定的上层线性通路。

projector step 50 的轨迹不同：layer 12 为 0.691、layer 14 为 0.750、layer 17/18 为 1.000，final assistant probe 仍有 0.945，native LM-head 达到 1.000。paired-counterfactual native target/source 为 0/1，shuffled-image native target 为 0.2375，说明最终 head 读取的是实际视觉来源。primary vision probe 的 pair-permutation p 均为 `1/2001`。

#figure(
  grid(
    columns: (1.25fr, 0.75fr),
    gutter: 10pt,
    image("../experiments/v100_perception_20260804/shape_adaptation_v1/charts/05-assistant-layerwise.svg", width: 100%),
    image("../experiments/v100_perception_20260804/shape_adaptation_v1/charts/04-vision-shuffle-control.svg", width: 100%),
  ),
  caption: [左：冻结、LoRA 与 projector 续训的 assistant 逐层 balanced accuracy；右：strict paired preference 的 vision/shuffled 控制。短程 projector 续训把中层信号延伸到 native head。],
)

=== 假设更新与下一项本地实验

#table(
  columns: (1.55fr, 1fr, 2.7fr),
  [*假设*], [*包 5 更新*], [*证据与动作*],
  [冻结主干上层不会使用视觉证据], [部分支持], [顶部 LoRA 把 strict paired 从 0.130 提到 0.605、生成 paired 从 0 提到 0.080，确认 use/decoding 瓶颈真实存在；效果显著弱于 projector 续训。],
  [当前 shape 主瓶颈是 projector 接口训练不足], [强支持], [projector step 50 在 vision 达到 preference/generation 1.000，shuffle 仅 0.180；晚层 probe 与 native head 同时恢复。],
  [需要先扩大 projector 结构], [本任务反驳], [不改结构、不改初始化，400 个额外 shape 样本已完全恢复；结构消融应等多任务迁移读数。],
  [0.5B 容量是 shape 首要瓶颈], [削弱], [同一个 0.5B 冻结主干可被短程 projector 续训驱动到 1.000；容量仍可能限制 OCR、推理与自然图像。],
  [shape 恢复会跨任务泛化], [未检验], [立即用现有六任务 selection 评估 projector step 50，并跑 balanced multi-task 最小训练；结果决定辅助目标、1.5B 对照与分辨率域偏移的顺序。],
)

独立 verifier 已重读两条训练轨迹的 8 个 checkpoint、8,400 条 preference、1,400 条 generation、63 个配对 bootstrap contrast、4,000 条 layerwise representation metadata 与 179,200 条 probe prediction。projector 大权重和两组约 286 MB 表征保留在 V100 HDD，Git 包含完整路径、bytes/SHA、聚合表、probe 权重及无损 gzip 预测。付费 Gate D 继续暂缓。

== Shape-only projector 的六任务迁移（包 6，2026-08-05）

=== 零附加训练的预注册迁移矩阵

直接把包 5 的 `shape-projector-step50` 应用于 color、coordinate、count、OCR、shape、spatial 六项 selection，不再更新参数。对照为同一 step-1500 frozen projector 与 matched random initialization。Teacher-forced 矩阵对每个 checkpoint 跑 vision、shuffled-image、paired-counterfactual-image、background-matched-aux，共 28,800 行；自由生成矩阵再加入 blind，共 9,000 行。每项 preference 有 200 个完整 counterfactual pair，generation 使用与适配训练 ID/pair 都不相交的 50 pair。所有置信区间均以完整 pair 为重采样单位，2,000 次 bootstrap。

预注册的 broad-transfer 判据要求：至少三个非 shape 任务同时满足 checkpoint 相对 step 1500 的 strict paired preference 下界大于零，且 vision 相对 shuffled-image 的下界大于零。这样，单纯改变答案先验或解码习惯无法被计作视觉迁移。

#table(
  columns: (1.1fr, 0.85fr, 0.85fr, 1.55fr, 1.45fr),
  [*任务*], [*step 1500*], [*shape projector*], [*checkpoint 增益（95% CI）*], [*vision−shuffle（95% CI）*],
  [color], [0.000], [0.010], [+0.010 [0.000, 0.025]], [−0.015 [−0.040, 0.010]],
  [coordinate], [0.000], [0.000], [0.000 [0.000, 0.000]], [0.000 [0.000, 0.000]],
  [count], [0.000], [0.005], [+0.005 [0.000, 0.015]], [−0.005 [−0.025, 0.010]],
  [OCR], [0.000], [0.015], [+0.015 [0.000, 0.035]], [+0.015 [0.000, 0.035]],
  [shape], [0.130], [*1.000*], [*+0.870 [0.820, 0.915]*], [*+0.820 [0.765, 0.875]*],
  [spatial], [0.000], [0.000], [0.000 [0.000, 0.000]], [0.000 [0.000, 0.000]],
)

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("../experiments/v100_perception_20260804/multitask_transfer_v1/charts/01-paired-preference.svg", width: 100%),
    image("../experiments/v100_perception_20260804/multitask_transfer_v1/charts/02-paired-generation.svg", width: 100%),
  ),
  caption: [左：六任务 strict paired preference；右：自由生成 paired accuracy。shape-only continuation 的完整恢复没有扩散到其余五项。],
)

=== Teacher forcing 与自由生成给出同一边界

shape 的自由生成 paired accuracy 从 step 1500 的 0 提升到 *1.000 [1.000, 1.000]*，vision−shuffled 为 *+0.980 [0.940, 1.000]*；其余五项的 checkpoint paired-generation 增益全为 0。color、count、spatial 在 sample accuracy 或 prediction flip 上出现零散变化，但始终无法同时答对完整 pair，也没有稳定跟随正确图像来源，因此不构成内容能力迁移。

#figure(
  image("../experiments/v100_perception_20260804/multitask_transfer_v1/charts/03-vision-minus-shuffle.svg", width: 72%),
  caption: [shape-projector-step50 的图像因果效应。只有 shape 在 teacher-forced 与自由生成两种口径上同时形成大幅正差。],
)

=== 假设更新与下一项本地实验

#table(
  columns: (1.65fr, 1fr, 2.65fr),
  [*假设*], [*包 6 更新*], [*证据与动作*],
  [shape 续训修复了可跨任务复用的通用视觉接口], [反驳], [五个非 shape 任务均未通过 checkpoint 增益与 visual-causality 双门槛；shape 的恢复是窄任务映射。],
  [shape 的 1.000 只是语言答案先验], [反驳], [vision−shuffled preference +0.820、generation +0.980，输出随真实图像中的配对属性变化。],
  [短程 projector continuation 足以学习新内容映射], [支持，范围待测], [40 个 shape pair 已足够；下一项给六任务相同配额，直接检验一轮 balanced supervision 的能力轨迹。],
  [应立即扩大语言主干], [暂不优先], [0.5B 在得到任务监督后可完整解决 shape；先测 balanced projector 能覆盖多少任务，再把仍失败项交给 1.5B 容量对照。],
)

下一包从 step 1500 开始，以 true batch 24 每步固定放入每任务 4 条记录；checkpoint 0/25/50/100 对应已见 0/600/1,200/2,400 样本，step 100 恰好完成 synthetic train 一轮。先构建完整 feature cache 并做 OOM smoke；随后复用本包全矩阵，按任务观察 paired preference、correct margin、paired generation 与 visual controls。若多任务广泛恢复，再做辅助目标和初始化消融；若 OCR/coordinate/spatial 仍失败，再进入 Qwen2.5-1.5B 容量对照与 projector loss/region 诊断。

独立校验重读 28,800 条 preference、9,000 条 generation、567 行聚合指标与 525 个 contrast，并核对源摘要和分析文件 SHA-256。两条 canonical run 均为零失败，完整测试集 *178/178* 通过，final odd halves 未评分。旧分析器因硬编码 `blind` teacher-forced 条件而中止，其空结果目录已显式作废并随包保留。付费 Gate D 继续暂缓。

== 六任务均衡 projector continuation（包 7，2026-08-05）

=== 可审计缓存与真实 batch

完整 synthetic train 的 2,400 条记录先在 256 px 编码为 75 个 float32 shard。独立 verifier 重哈希 3,932,170,800 bytes，并逐条读回 2,400 个张量、983,040,000 个数值，shape 与 finite 检查全通过。缓存耗时 251.5s，峰值显存 1.95 GB。

续训仍从 step 1500 出发，只更新 20,454,272 参数的 projector。每个 true batch 固定 24 条，每项任务恰好 4 条；100 步恰好遍历每项 400 条记录/200 pair 一轮。loss 在 step 1/25/50/100 为 2.7778/1.4495/1.4594/1.3490；总耗时 144.5s，峰值显存 11.80 GB。四个 projector/optimizer checkpoint、每步任务配额、6,577 个 answer token 与全部 tensor 均通过独立验证。

#figure(
  image("../experiments/v100_perception_20260804/balanced_multitask_adaptation_v1/charts/00-training-loss.svg", width: 78%),
  caption: [一轮六任务均衡 projector continuation 的 true-batch loss。每步六项任务配额固定，不把任务采样漂移混入能力轨迹。],
)

=== 一轮监督建立六任务 teacher-forced 视觉读出

Preference 终表含 38,400 行、16 个 checkpoint/condition cell、每格 1,200 个完整 counterfactual pair、零失败。step 25 只有 coordinate 通过预注册双门槛；step 50 增至 color 与 shape；step 100 六项全部同时满足 checkpoint 增益和 vision−shuffle 的 bootstrap 下界大于零。

#table(
  columns: (1fr, 0.75fr, 0.75fr, 1.55fr, 1.55fr),
  [*任务*], [*vision*], [*shuffle*], [*checkpoint 增益（95% CI）*], [*vision−shuffle（95% CI）*],
  [color], [0.230], [0.095], [+0.230 [0.175, 0.290]], [+0.135 [0.065, 0.205]],
  [coordinate], [0.055], [0.000], [+0.055 [0.025, 0.090]], [+0.055 [0.025, 0.090]],
  [count], [0.115], [0.060], [+0.115 [0.075, 0.160]], [+0.055 [0.005, 0.105]],
  [OCR], [0.135], [0.050], [+0.135 [0.085, 0.185]], [+0.085 [0.040, 0.140]],
  [shape], [0.560], [0.155], [+0.430 [0.330, 0.525]], [+0.405 [0.325, 0.480]],
  [spatial], [0.250], [0.000], [+0.250 [0.190, 0.310]], [+0.250 [0.190, 0.310]],
)

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("../experiments/v100_perception_20260804/balanced_multitask_adaptation_v1/charts/01-paired-preference.svg", width: 100%),
    image("../experiments/v100_perception_20260804/balanced_multitask_adaptation_v1/charts/03-vision-minus-shuffle.svg", width: 100%),
  ),
  caption: [左：strict paired preference 随均衡训练量的六任务轨迹；右：step 100 的图像因果差。六项都已摆脱 teacher-forced 地板。],
)

早期轨迹存在可复现的多任务干扰：shape 在 step 25 相对 step 1500 下降 *−0.130 [−0.180, −0.085]*，step 50 恢复到 0.435，step 100 达到 0.560。单个短 checkpoint 会把“暂时被别的任务覆盖”误判成结构性失败，因此包 7 的决策使用整条轨迹。

=== 剩余差距集中在冻结语言栈的使用与生成

Generation 终表含 12,000 行和 128 条 heldout shuffle-loss，零失败。step 100 只有 shape 与 spatial 的 paired generation 显著改善：相对 step 1500 为 *+0.160 [0.080, 0.280]* 和 *+0.220 [0.120, 0.340]*；vision−shuffle 为 *+0.140 [0.060, 0.240]* 和 *+0.220 [0.100, 0.340]*。color、coordinate、count、OCR 的 teacher-forced 双门槛已通过，自由生成 paired 仍为 0。

#figure(
  image("../experiments/v100_perception_20260804/balanced_multitask_adaptation_v1/charts/02-paired-generation.svg", width: 78%),
  caption: [自由生成 paired accuracy 的 checkpoint 轨迹。均衡 projector 已让 shape/spatial 可按图生成，另外四项仍停在“能偏好正确答案、不能稳定说出”的阶段。],
)

=== 假设更新与下一项本地实验

#table(
  columns: (1.6fr, 1fr, 2.7fr),
  [*假设*], [*包 7 更新*], [*证据与动作*],
  [五项地板来自 projector 结构或 tower 信息损失], [反驳], [结构、初始化与冻结主干均不变；仅补一轮均衡监督，六项 teacher-forced checkpoint/causal 下界全部转正。],
  [原训练覆盖不足是主要 teacher-forced 瓶颈], [强支持], [有效任务随每任务已见样本累积到六项；balanced supervision 是首个跨任务验证的改进方向。],
  [0.5B 容量立即限制视觉内容选择], [进一步削弱], [同一 0.5B 主干已在六项给出因果正确偏好；容量仍可能限制自由生成、OCR 与格式遵从。],
  [只延长 projector 即可关闭全部生成差距], [未定], [四项仍有 teacher-forced/generation 裂缝；下一项做等 record-order 的额外 projector epoch 与顶部 LoRA screen。],
)

下一包从 balanced-projector-step100 出发，让 projector-only continuation 与小规模顶部 LoRA 使用同一均衡记录顺序和 examples-seen。若 projector 继续训练同时抬高 preference/generation，优先延长训练；若 LoRA 对四项生成的改善明显更大，则先定位上层 use/decoding 瓶颈，再做 Qwen2.5-1.5B 容量对照。三次实现失败均完整保留并作废：padding/placeholder 同 ID、checkpoint provenance 缺字段、runner 强制 random control。final odd halves 未评分，付费 Gate D 继续暂缓。

== 等顺序 extra-projector 与顶部 LoRA 对照（包 8，2026-08-05）

=== 公平训练合同与优化器连续性

两臂都从 balanced-projector step 100 出发，使用相同 seed、相同 2,400 条记录顺序（SHA `a0929326…2f5`）、true batch 24 和 100 步；每步每项任务固定 4 条。projector 臂恢复 step-100 AdamW 动量，因而是连续的第二轮训练；top-12 rank-8 LoRA 从严格零 delta 初始化并冻结 projector。前者训练 20,454,272 个参数，耗时 159.1s、峰值 11.80 GB；后者训练 442,368 个参数，耗时 153.4s、峰值 9.93 GB。两臂 step 1 loss 精确同为 1.221496。step 100 时 projector loss/梯度范数为 0.9509/0.590，LoRA 为 1.3066/8.639，提示 LoRA 末端存在优化不稳或任务冲突。

#figure(
  image("../experiments/v100_perception_20260804/balanced_adaptation_compare_v1/charts/00-training-loss.svg", width: 82%),
  caption: [同一记录顺序下的六任务训练 loss。LoRA 在末段出现梯度尖峰，因此单个 endpoint 之后还需读 step 25/50 轨迹。],
)

=== 额外 projector epoch 提升总体读出与生成

canonical bf16 评测含 21,600 条 preference 与 3,600 条 generation 原始记录，3 个状态、9/6 个 cell，全部为完整 counterfactual pair、零失败；区间使用 2,000 次 pair bootstrap。总体 strict paired preference 为 base/LoRA/projector *0.224/0.247/0.511*：projector 相对 base *+0.287 [0.258, 0.318]*，LoRA *+0.023 [−0.003, 0.049]*。总体 paired generation 为 *0.063/0.080/0.257*：projector *+0.193 [0.147, 0.240]*，LoRA *+0.017 [−0.020, 0.050]*。

#table(
  columns: (1fr, 0.75fr, 0.75fr, 0.75fr, 1.55fr),
  [*任务*], [*base*], [*LoRA*], [*projector*], [*projector−base（95% CI）*],
  [color], [0.230], [0.330], [0.735], [+0.505 [0.430, 0.575]],
  [coordinate], [0.055], [0.095], [0.575], [+0.520 [0.450, 0.585]],
  [count], [0.115], [0.000], [0.100], [−0.015 [−0.070, 0.040]],
  [OCR], [0.135], [0.175], [0.220], [+0.085 [0.040, 0.130]],
  [shape], [0.560], [0.880], [0.435], [−0.125 [−0.180, −0.065]],
  [spatial], [0.250], [0.000], [1.000], [+0.750 [0.685, 0.805]],
)

#table(
  columns: (1fr, 0.75fr, 0.75fr, 0.75fr, 1.55fr),
  [*paired generation*], [*base*], [*LoRA*], [*projector*], [*projector−base（95% CI）*],
  [color], [0.000], [0.000], [0.160], [+0.160 [0.060, 0.260]],
  [coordinate], [0.000], [0.000], [0.240], [+0.240 [0.120, 0.360]],
  [count], [0.000], [0.000], [0.020], [+0.020 [0.000, 0.060]],
  [OCR], [0.000], [0.000], [0.000], [+0.000 [0.000, 0.000]],
  [shape], [0.160], [0.480], [0.120], [−0.040 [−0.100, 0.000]],
  [spatial], [0.220], [0.000], [1.000], [+0.780 [0.660, 0.880]],
)

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("../experiments/v100_perception_20260804/balanced_adaptation_compare_v1/charts/01-endpoint-paired-preference.svg", width: 100%),
    image("../experiments/v100_perception_20260804/balanced_adaptation_compare_v1/charts/02-endpoint-paired-generation.svg", width: 100%),
  ),
  caption: [左：六任务 strict paired preference；右：paired free generation。额外 projector epoch 产生广泛净增益，同时 shape 出现明确遗忘。],
)

=== LoRA 是 shape 特异修复，多任务竞争成为主问题

LoRA 对 shape 的 strict preference 与 generation 均提高 *+0.320*，区间分别为 [0.250, 0.395] 与 [0.200, 0.460]；color 只在 preference 提高 +0.100 [0.030, 0.175]，另外三项生成保持零。与此同时 count preference 下降 −0.115 [−0.160, −0.070]，spatial preference/generation 下降 −0.250 [−0.310, −0.190] 与 −0.220 [−0.340, −0.120]。额外 projector epoch 则显著提升 color、coordinate、OCR 与 spatial，shape preference 下降。两种适配都发生任务竞争，宽泛的“冻结上层统一阻止生成”解释被反驳。

#figure(
  image("../experiments/v100_perception_20260804/balanced_adaptation_compare_v1/charts/03-lora-minus-projector.svg", width: 70%),
  caption: [LoRA−projector 的任务差。正值集中在 shape，其余主要任务由额外 projector epoch 占优。],
)

=== 精度敏感性、假设更新与下一项

首个完整 endpoint run 使用训练态 fp32 projector，内部对照有效，但 package 7 canonical 评测使用 bf16。fp32 把 spatial base strict/generation 从 0.250/0.220 移到 0/0，并让其他边界值产生小幅漂移。该 v1 作为阈值敏感性诊断保留；bf16 v2 逐任务精确复现 package 7 base，承担所有跨包结论。这说明离散 paired accuracy 在 margin 接近零时对 serving dtype 敏感，后续需同步报告 mean margin 与阈值翻转数。

#table(
  columns: (1.7fr, 1fr, 2.6fr),
  [*假设*], [*包 8 更新*], [*证据与动作*],
  [只需延长 projector 即可广泛改善], [部分支持], [总体 preference/generation 下界转正，并新解锁 color/coordinate/spatial；count/OCR generation 仍未解决，shape 显著遗忘。],
  [冻结语言上层是四项生成裂缝的统一原因], [反驳], [top-12 LoRA 只提升 shape；color/coordinate/count/OCR generation 不动，并抹掉 spatial。],
  [0.5B 容量是当前首要瓶颈], [继续暂缓], [同一 0.5B 在额外 projector epoch 后能生成 color/coordinate/spatial；先解决训练轨迹与任务竞争，再做 1.5B 容量对照。],
  [单一 endpoint 足以选方案], [反驳], [LoRA 梯度尖峰、projector shape 遗忘与包 7 的短程回归共同要求 step 25/50/100 轨迹。],
)

下一项在 canonical bf16 下用每任务固定 50 个 selection pair 筛查两臂 step 25/50/100；自由生成继续用固定 50 pair/task。若 LoRA 在 step 25/50 出现跨任务早期峰值，转学习率/早停消融；若 projector 的任务峰值错位，测试 replay weighting 或抗干扰辅助目标；若 count/OCR 全轨迹仍为零，再进入 projector 辅助目标与 Qwen2.5-1.5B 容量对照。final odd halves 未评分，付费 Gate D 继续暂缓。

== 六任务适配轨迹与 step-50 全量确认（包 9，2026-08-05）

=== canonical-bf16 轨迹筛选

固定 screen 覆盖 frozen、LoRA/projector step 25/50/100 七个状态；teacher forcing 每任务取预注册的 50 个完整 pair，自由生成沿用相同规模的固定 manifest。有效产物含 12,600 条 preference、8,400 条 generation、21/14 个完整 cell 与 693 个 2,000 次 pair-bootstrap 对比，零失败且未触碰 final odd halves。projector step 50 是筛选中唯一在六项 strict paired preference 点估计都高于 frozen 的 checkpoint；LoRA 的有效峰值仍集中于 shape。

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("../experiments/v100_perception_20260804/balanced_adaptation_trajectory_v1/charts/01-trajectory-paired-preference.svg", width: 100%),
    image("../experiments/v100_perception_20260804/balanced_adaptation_trajectory_v1/charts/02-trajectory-paired-generation.svg", width: 100%),
  ),
  caption: [左：step 25/50/100 strict paired preference；右：paired generation。projector 的 count/shape 在 step 50 达峰，coordinate/spatial 继续上升到 step 100。],
)

=== 全量确认：step 50 是较均衡点，OCR 证据需降级

全量确认包含 frozen、LoRA step 50 与 projector step 50 的 21,600 条 preference 和 3,600 条 generation；耗时 1,137.6s、峰值 12.72 GB。projector step 50 的总体 strict preference 为 0.5117，相对 base 0.2242 提高 *+0.2875 [0.2583, 0.3167]*；总体 paired generation 为 0.2267，相对 0.0633 提高 *+0.1633 [0.1200, 0.2100]*。

screen 中 OCR 的 base-relative 正下界没有被全量复现：step 50−base 只有 +0.020 [−0.025, 0.065]。同一 checkpoint 的 OCR vision−shuffle 为 *+0.075 [0.020, 0.130]*，说明样本级图像已影响 OCR 答案排序；自由生成仍为 0。count 则确认 strict +0.265 [0.200, 0.330]，paired generation 只有 0.020。两项都保留显著的内部证据/自由生成裂缝，其中 OCR 结论强度更低。

#table(
  columns: (1fr, 0.7fr, 0.7fr, 0.7fr, 1.6fr),
  [*strict preference*], [*base*], [*P50*], [*P100*], [*P100−P50（95% CI）*],
  [color], [0.230], [0.635], [0.735], [+0.100 [0.025, 0.175]],
  [coordinate], [0.055], [0.415], [0.575], [+0.160 [0.095, 0.225]],
  [count], [0.115], [0.380], [0.100], [−0.280 [−0.345, −0.220]],
  [OCR], [0.135], [0.155], [0.220], [+0.065 [0.015, 0.120]],
  [shape], [0.560], [0.735], [0.435], [−0.300 [−0.360, −0.240]],
  [spatial], [0.250], [0.750], [1.000], [+0.250 [0.190, 0.315]],
)

#table(
  columns: (1fr, 0.7fr, 0.7fr, 0.7fr, 1.6fr),
  [*paired generation*], [*base*], [*P50*], [*P100*], [*P100−P50（95% CI）*],
  [color], [0.000], [0.140], [0.160], [+0.020 [−0.120, 0.160]],
  [coordinate], [0.000], [0.020], [0.240], [+0.220 [0.120, 0.340]],
  [count], [0.000], [0.020], [0.020], [+0.000 [−0.060, 0.060]],
  [OCR], [0.000], [0.000], [0.000], [+0.000 [0.000, 0.000]],
  [shape], [0.160], [0.400], [0.120], [−0.280 [−0.420, −0.160]],
  [spatial], [0.220], [0.780], [1.000], [+0.220 [0.120, 0.340]],
)

=== 相同总体均值掩盖任务 Pareto 迁移

跨 run 分析先逐条核对 `id/pair_id/pair_variant/task/condition`，再比较相同 pair。projector step 100−50 的总体 strict preference 为 *−0.0008 [−0.0283, 0.0275]*，总体 generation 为 *+0.0300 [−0.0133, 0.0767]*，两者都没有显著总体变化；任务层却同时出现上表中的大幅正负迁移。训练记录已精确做到每步六任务等量，因而简单的采样不均解释被排除。现有证据支持 projector 更新中的梯度或表示冲突。

LoRA step 50 的 shape strict/generation 相对 base 提高 +0.340 [0.275, 0.405] / +0.440 [0.300, 0.580]，同时 count strict 下降 −0.105 [−0.155, −0.060]、spatial strict/generation 下降 −0.250 [−0.310, −0.190] / −0.220 [−0.340, −0.120]。step 100 没有形成跨任务改善。顶部 LoRA 的广泛解码修复解释再次被反驳。

#figure(
  image("../experiments/v100_perception_20260804/balanced_adaptation_trajectory_v1/charts/03-trajectory-vision-minus-shuffle.svg", width: 78%),
  caption: [任务 × checkpoint 的 vision−shuffle strict preference。step 50 的六项点估计均为正；全量确认中六项下界也均高于 0。],
)

#table(
  columns: (1.8fr, 1fr, 2.7fr),
  [*假设*], [*包 9 更新*], [*证据与动作*],
  [继续训练会统一提高六项能力], [反驳], [step 50→100 总体不变，count/shape 显著下降而 coordinate/spatial 显著上升；单一 global early stop 无法兼顾。],
  [等量 replay 足以消除遗忘], [反驳], [每步六任务严格等量仍出现大幅 Pareto 迁移；下一项转 checkpoint 插值与梯度/辅助目标。],
  [OCR 完全没有进入模型], [反驳但证据较弱], [step-50 vision−shuffle +0.075 [0.020, 0.130]；checkpoint−base 区间跨零且 generation 为 0。],
  [count 主要缺视觉表示], [削弱], [step-50 strict−base +0.265 [0.200, 0.330]，generation 仅 0.020；优先处理读出/目标和后续遗忘。],
  [0.5B 容量是首要瓶颈], [继续暂缓], [相同主干已在多个 checkpoint 表现互补能力；先测试同 basin 合并与抗干扰目标，再做 1.5B。],
)

下一项先对 projector step 50/100 做固定系数权重插值，并用相同 canonical-bf16 50-pair screen 判断能否同时保留 count/shape、获得 coordinate/spatial。若某个插值点改善多任务 Pareto 前沿，再做全量确认；若所有点沿同一权衡曲线移动，则从 step 50 测试抗遗忘辅助目标或梯度冲突干预。付费 Gate D、完整 DeepSeek-V4 与任何租卡继续暂缓。

== Projector checkpoint 插值与同 basin 合并检验（包 10，2026-08-05）

=== 预注册规则与端点精确复现

包 9 的 step 50/100 总体分数相同、任务能力互补，因而先检验最低成本的线性 mode-connectivity/model-soup 假设。构造公式为 `(1−alpha) P50 + alpha P100`，固定 alpha=0/.25/.50/.75/1。候选必须同时满足：count/shape 距 alpha 0 不超过 0.05；coordinate/spatial 严格高于 alpha 0；worst-task strict preference 高于两个端点；macro strict 距最佳端点不超过 0.02。规则在读取中间点之前写入分析器。

插值器逐张量检查 key/shape/dtype，保存后重载并计算 ordered-tensor SHA。alpha 0/1 的 tensor SHA 分别精确复现 `fd7b07e6…d192` 与 `7b731cff…a76`。canonical-bf16 screen 含 frozen 与五个插值状态，10,800 条 preference、7,200 条 generation、630 个 metric 与 651 个 pair-bootstrap contrast；耗时 541.3s、峰值 12.72 GB、零失败。独立 verifier 进一步确认两端各有 1,800 条 preference 和 1,200 条 generation 与包 9 原端点逐字段完全一致，覆盖 raw logp/NLL/margin 与生成字符串。

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("../experiments/v100_perception_20260804/projector_interpolation_v1/charts/01-task-preference.svg", width: 100%),
    image("../experiments/v100_perception_20260804/projector_interpolation_v1/charts/02-task-generation.svg", width: 100%),
  ),
  caption: [左：六任务 strict paired preference；右：paired generation。spatial 在 alpha=.25 即到 1.0，count/shape 沿路径下降，未出现两端能力并集。],
)

=== 中间点提高宏平均，但没有保留 count/shape

没有插值点通过预注册合并规则。alpha=.25 是最佳 balance diagnostic：macro strict 0.5333、worst-task 0.160、macro generation 0.2700、endpoint regret 0.520。相对 alpha 0，它把 spatial strict 从 0.74 提到 1.00，差值 *+0.26 [0.14, 0.38]*，macro generation 提高 *+0.0433 [0.0100, 0.0767]*；与此同时 count 从 0.42 降到 0.26，差值 *−0.16 [−0.28, −0.04]*，shape 从 0.80 降到 0.70，差值 −0.10 [−0.20, 0.00]。alpha=.50/.75 的 count 进一步降到 0.14。

#table(
  columns: (0.7fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  [*alpha*], [*macro strict*], [*worst strict*], [*macro generation*], [*endpoint regret*], [*目标合并*],
  [0], [0.5267], [0.180], [0.2267], [0.560], [endpoint],
  [.25], [0.5333], [0.160], [0.2700], [0.520], [fail],
  [.50], [0.5267], [0.140], [0.2833], [0.560], [fail],
  [.75], [0.5233], [0.140], [0.2733], [0.620], [fail],
  [1], [0.5167], [0.120], [0.2567], [0.620], [endpoint],
)

alpha=.25 对 alpha 1 的小幅 macro 优势都不确定：strict +0.0167 [−0.0267, 0.0633]，generation +0.0133 [−0.0233, 0.0533]。因此它不进入全量确认。结果支持两个 checkpoint 处于 aggregate 平滑的连接路径，同时反驳“线性平均可恢复能力并集”。能力竞争发生在路径内部，后续需要改变目标或梯度，而非继续挑选同一直线上的权重。

#figure(
  image("../experiments/v100_perception_20260804/projector_interpolation_v1/charts/03-balance-summary.svg", width: 78%),
  caption: [macro preference、worst-task preference 与 macro generation 的插值轨迹。宏平均平滑，worst-task 没有越过端点前沿。],
)

#table(
  columns: (1.8fr, 1fr, 2.7fr),
  [*假设*], [*包 10 更新*], [*证据与动作*],
  [step 50/100 位于完全断裂的权重 basin], [反驳], [插值全程保持稳定 macro，未出现灾难性坍塌；两端评测逐字段精确复现。],
  [线性 checkpoint 合并可兼得互补任务], [反驳], [alpha=.25 已显著获得 spatial，却显著丢失 count；所有中间点都未通过预注册 retention/worst-task 规则。],
  [单看 macro 足以选 checkpoint], [反驳], [alpha=.25/.50 的 macro 更高，但 worst-task 更低，并掩盖 count/shape 遗忘。],
  [需要改变训练目标或梯度], [支持], [下一项从 step 50 沿精确剩余记录顺序加入 count/shape projector-output anchoring；失败再转 per-task gradient conflict。],
)

下一项从 step 50 恢复优化器和原 step 51–100 记录顺序，加入仅作用于 count/shape 的 frozen-step50 projector-output anchoring，并同时复现无正则 continuation 控制。先以最小 lambda screen 判断是否能保留 count/shape 且继续学习 coordinate/spatial；无效时转逐任务梯度冲突干预。付费 Gate D 继续暂缓。

== Task-conditioned projector 表示锚定（包 11，2026-08-05）

=== 精确续训复现与单一辅助目标

从包 9 的 `balanced_compare_projector_v1` step 50 出发，恢复 AdamW 状态 `57e9ddb…ac2f32`，读取原训练顺序 `a0929326…f2f5` 的 step 51–100。无正则控制逐张量复现原 step 100：六个 tensor 全部相等，tensor SHA 为 `7b731cff…a76`，序列化文件 SHA 也精确等于 `05f19079…092d`。这同时验证 checkpoint 权重、optimizer state 和 record cursor 的可恢复性。

辅助目标只在每个 batch 的 count/shape 八条样本上约束当前 projector 输出接近 frozen-step50 输出，主损失继续使用正常语言建模 loss，语言模型保持冻结。screen 固定 $lambda=10^(-4),10^(-3),10^(-2)$，所有 arm 共享相同 1,200 个 continuation examples、24 条首批 ID、训练顺序、seed 和 canonical-bf16 评测。候选必须同时满足：count/shape 距 step 50 不超过 0.05；coordinate/spatial 严格高于 step 50；macro strict 距最佳端点不超过 0.02。

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("../experiments/v100_perception_20260804/projector_retention_v1/charts/01-task-preference.svg", width: 100%),
    image("../experiments/v100_perception_20260804/projector_retention_v1/charts/02-task-generation.svg", width: 100%),
  ),
  caption: [左：六任务 strict paired preference；右：paired generation。锚定改变了能力分配，但三个强度都没有保留 step-50 的 count/shape 前沿。],
)

=== 表示距离可控，旧任务决策边界仍然遗忘

三个系数都未通过预注册规则。训练末端 count/shape projector-output MSE 随约束增强从轻锚定 26.26 降到中锚定 8.59、强锚定 2.09，说明目标确实控制了表示距离；能力保留没有随之出现。step 50 的 count/shape strict 为 0.42/0.80；最佳 count 的强锚定仅为 0.16/0.54，最佳 macro 的中锚定为 0.10/0.54。

#table(
  columns: (1.2fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  [*状态*], [*macro strict*], [*worst*], [*macro gen*], [*count*], [*shape*], [*coord / spatial*],
  [step 50], [0.5267], [0.180], [0.2333], [0.42], [0.80], [0.44 / 0.74],
  [control 100], [0.5167], [0.120], [0.2567], [0.12], [0.48], [0.54 / 1.00],
  [$lambda=10^(-4)$], [0.4967], [0.160], [0.2467], [0.16], [0.42], [0.48 / 1.00],
  [$lambda=10^(-3)$], [*0.5700*], [0.100], [*0.3833*], [0.10], [0.54], [0.66 / 1.00],
  [$lambda=10^(-2)$], [0.5433], [0.160], [0.3000], [0.16], [0.54], [0.72 / 0.76],
)

中锚定仍给出有判别力的 Pareto 改变：相对精确 control，overall strict 提高 *+0.0533 [0.0200, 0.0867]*，paired generation 提高 *+0.1267 [0.0900, 0.1633]*；相对 step 50，count 为 *−0.32 [−0.46, −0.18]*，shape 为 *−0.26 [−0.38, −0.14]*。color/coordinate/spatial generation 分别达到 0.64/0.52/1.00；count 仍为 0.02，OCR 仍为 0。中锚定 overall vision−shuffle strict 为 *+0.4300 [0.3667, 0.4933]*，但 count vision−shuffle 为 −0.04 [−0.18, 0.08]，其 count 点估计不能解释成可靠视觉计数。

#figure(
  image("../experiments/v100_perception_20260804/projector_retention_v1/charts/03-balance-summary.svg", width: 78%),
  caption: [端点与三个锚定强度的 macro、worst-task 和 macro generation。中锚定提高宏平均与生成，worst-task 仍受 count 限制。],
)

#table(
  columns: (1.9fr, 1fr, 2.6fr),
  [*假设*], [*包 11 更新*], [*证据与动作*],
  [完整 projector 输出距离足以保持旧能力], [反驳], [MSE 单调降到 2.09，count/shape 仍比 step 50 低 0.26；表示接近没有保持答案决策边界。],
  [辅助目标无法改变优化路径], [反驳], [$lambda=10^(-3)$ 相对 control 显著提高 macro strict 与 paired generation，说明 trade-off 可被目标塑形。],
  [延长训练带来的交换仅是不可复现噪声], [反驳], [control 权重与原 step 100 文件逐字节一致；两端 teacher-forced 各 1,800 行逐字段精确复现。],
  [立即放大语言主干是下一优先项], [继续暂缓], [count/shape 已在同一 0.5B 主干的 step 50 达到更高值；先完成分层 batch 对照与遗忘触发 replay。],
)

screen 含 9,000 条 preference 与 6,000 条 generation，零 failure，分析含 525 个 metric 和 399 个 paired-bootstrap contrast。重复 GPU generation 在相同端点出现文本级非确定性：frozen/control 分别 42/62 条 prediction 不同，correct flag 为 18/0 条不同；teacher-forced logp/NLL/margin 完全一致，候选选择只使用同一 run 内 strict preference。首轮因误选更早的 `balanced_multitask_projector_v1` step 50 而失效，六个相关 run 与一次过严 generation verifier 共七份 invalidation 均已保留和独立核验。

下一项执行严格匹配的六任务分层 batch 对全局随机 batch，随后运行预注册遗忘触发式 replay。只有两者仍无法缓解遗忘时，才进入 per-task gradient-conflict 方法。付费 Gate D、完整 DeepSeek-V4 与租卡继续暂缓。

== 分层能力覆盖 batch 对全局随机顺序（包 12，2026-08-05）

=== 唯一处理变量是 batch-order constraint

两臂从同一个 package-7 projector 与 AdamW state 开始，使用同一 seed、学习率、batch size、2,400 条记录和 2,400 examples seen。分层臂 100 个 true batch 都是六任务各 4 条；global arm 对全部记录做一次 seeded random permutation，100 个 batch 均不满足 4×6 分层，单任务最高占 11/24。独立 verifier 确认每条记录恰好一次、六任务各 400 条、step-0 tensor SHA 与 optimizer source SHA 一致。顺序约束是唯一处理变量。

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("../experiments/v100_perception_20260804/batch_stratification_v1/charts/01-summary-trajectories.svg", width: 100%),
    image("../experiments/v100_perception_20260804/batch_stratification_v1/charts/02-task-preference-delta.svg", width: 100%),
  ),
  caption: [左：macro/worst/generation 轨迹；右：各任务 stratified−global strict paired preference。step 50 的分层加速在 step 100 转化为任务交换。],
)

#table(
  columns: (0.7fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  [*step*], [*arm*], [*macro strict*], [*worst*], [*macro gen*], [*vision−shuffle*],
  [25], [stratified], [0.2017], [0], [0.0367], [0.1292],
  [25], [global], [0.2408], [0], [0.0133], [0.1708],
  [50], [stratified], [*0.5117*], [*0.155*], [*0.2333*], [*0.3975*],
  [50], [global], [0.3892], [0.145], [0.1667], [0.2975],
  [100], [stratified], [0.5108], [*0.100*], [0.2567], [0.3850],
  [100], [global], [*0.5308*], [0.055], [*0.3200*], [*0.3975*],
)

分层在 step 50 更快形成 macro、generation、worst-task 与 image-causal signal；step 25 没有单调领先。到 step 100，global 的 macro strict/generation 反超。终点 stratified−global overall strict paired preference 为 *−0.020 [−0.0442, 0.0025]*，未排除 0，预注册 verdict 为 `mixed_or_underpowered`。

=== 终点不是统一收益，而是 coordinate 对 color/shape 的交换

#table(
  columns: (1fr, 1fr, 1fr, 1fr),
  [*任务*], [*stratified*], [*global*], [*stratified−global, 95% CI*],
  [color], [0.735], [0.825], [−0.090 [−0.165, −0.025]],
  [coordinate], [0.575], [0.410], [*+0.165 [0.115, 0.220]*],
  [count], [0.100], [0.055], [+0.045 [0, 0.095]],
  [OCR], [0.220], [0.215], [+0.005 [−0.050, 0.060]],
  [shape], [0.435], [0.680], [*−0.245 [−0.315, −0.175]*],
  [spatial], [1.000], [1.000], [0],
)

分层的 coordinate/count/shape trajectory AUC 相对 global 为 +0.1156/+0.0619/+0.0706，color 为 −0.0625，OCR 近零，spatial 相同。分层从 step 50 到 100 遗忘 count 0.28、shape 0.30；global 遗忘 count 0.25，shape 在终点达到自身峰值。逐 batch 覆盖没有消除后半程任务干扰。

=== 固定小批次梯度诊断与合同更新

六任务各固定 8 条 complete-pair records，在 frozen 与六个 checkpoint 上测 projector gradient。分层 step 100 的 15 个 task pairs 中 6 个 cosine 为负，最强冲突是 count–shape −0.1704；global step 100 的 15 对全部非负，平均 cosine 为 0.1185。该诊断没有 cosine CI，作为与轨迹一致的描述性机制证据。

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("../experiments/v100_perception_20260804/batch_stratification_v1/charts/03-gradient-conflict.svg", width: 100%),
    image("../experiments/v100_perception_20260804/batch_stratification_v1/charts/04-batch-imbalance.svg", width: 100%),
  ),
  caption: [左：固定任务梯度平均 cosine 与负夹角比例；右：每个 true batch 的单任务最大占用。分层严格控制组成，但终点梯度冲突更多。],
)

#table(
  columns: (1.9fr, 1fr, 2.6fr),
  [*假设*], [*包 12 更新*], [*证据与动作*],
  [逐 batch 六任务覆盖普遍优于 global random], [未支持], [终点 overall CI 跨 0，coordinate 与 color/shape 显著反向，预注册 verdict mixed。],
  [分层覆盖提高能力形成速度], [部分支持], [step 50 macro/generation 为 0.512/0.233，对 global 0.389/0.167；step 25 不领先。],
  [分层覆盖降低任务冲突], [反驳], [分层终点 6/15 个负 cosine pairs，global 为 0；count/shape 仍显著遗忘。],
  [每 batch 分层应写成 DeepSeek 硬合同], [反驳], [合同改为固定窗口领域覆盖；分层只保留为短校准候选，并由 sentinel 监测后半程交换。],
)

全包含 50,400 条 preference、8,400 条 generation、735 个 metrics、525 个 paired contrasts、42 个 gradient norms 和 105 个 pairwise cosines；全部文件 hash 与 14 项 matched-order invariant 通过，完整仓库测试 *220/220* 全绿。下一项从已知能力交换 checkpoint 运行 ordinary balanced、fixed replay、forgetting-triggered replay 三臂；在 replay 结果前不增加块状 curriculum。付费 Gate D 继续暂缓。

== 固定训练预算内的 preventive replay（包 13，2026-08-05）

=== 1,200 examples 总量不变，replay 只重分配槽位

三条策略共享 package-12 分层臂 step 50 的 projector、AdamW state 和后续训练顺序。ordinary 完成原 steps 51–100；fixed replay 每个 25-step 窗口给 count/shape 各重放 10 个历史 complete pairs，并等量换出其他任务 pair；triggered 策略先与 ordinary 共用 steps 51–75，再由冻结 sentinel 决定 steps 76–100 的重分配。每条完整策略均为 50 steps × batch 24 = *1,200 training examples*，额外 optimizer steps 与额外训练 examples 都为 0。

#table(
  columns: (1.2fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1.2fr),
  [*策略*], [*color*], [*coord*], [*count*], [*OCR*], [*shape*], [*spatial*], [*总量*],
  [ordinary], [200], [200], [200], [200], [200], [200], [*1,200*],
  [fixed], [180], [180], [*240*], [180], [*240*], [180], [*1,200*],
  [triggered], [196], [196], [*220*], [196], [196], [196], [*1,200*],
)

fixed 重分配 80 个槽位；triggered 在后半窗只重分配 20 个 count 槽位。ordinary 逐张量精确复现历史 step 100：六个 tensor 全等，projector 文件 SHA `05f19079…092d`，tensor SHA `7b731ffc…a76`。采样账本、恢复状态和训练实现因此通过单变量检查。

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("../experiments/v100_perception_20260804/matched_replay_v1/charts/01-policy-summary-trajectories.svg", width: 100%),
    image("../experiments/v100_perception_20260804/matched_replay_v1/charts/04-fixed-budget-allocation.svg", width: 100%),
  ),
  caption: [左：ordinary、fixed 与 triggered 的 macro preference/generation 轨迹；右：同一 1,200-example 总预算内的任务分配。],
)

=== 宏平均上升时，count 仍发生可辨别坍塌

触发规则在训练前冻结：任务从 step 50 到 step 75 的 strict paired preference 绝对下降至少 0.10，且 current-minus-reference paired-bootstrap `ci95_high < 0`；最多选择下降最大的两个任务。ordinary 整体在该窗口上升 *+0.040 [0.0108, 0.0675]*，count 却从 0.380 降到 0.075，gap *−0.305 [−0.365, −0.245]*。shape 为 −0.035 [−0.090, 0.020]。机械决策只触发 count。

这组结果给正式训练一个直接约束：每个 domain/task 必须保留独立历史峰值与 paired CI；macro 改善不能覆盖局部灾难性遗忘。

=== Preventive replay 同时恢复内部选择与自由生成

#table(
  columns: (1.2fr, 1.1fr, 1fr, 1fr, 1.1fr, 1fr),
  [*策略 / step100*], [*macro pref*], [*worst*], [*count*], [*shape*], [*macro gen*],
  [ordinary], [0.5108], [0.100], [0.100], [0.435], [0.2567],
  [fixed], [*0.5983*], [*0.195*], [*0.490*], [*0.555*], [*0.3567*],
  [triggered], [0.5358], [0.155], [0.275], [0.435], [0.2600],
)

fixed 的 count+shape strict paired preference 相对 ordinary 为 *+0.255 [0.210, 0.300]*；其 donor 四任务合并为 +0.00375 [−0.0125, 0.01875]，没有可辨别的 donor 损失。目标任务 paired generation 为 *+0.120 [0.050, 0.190]*。count endpoint 达到 0.490，超过 step-50 参考 0.380；shape 从 ordinary 的 0.435 提到 0.555，仍低于 0.735±0.05 的恢复带。

triggered 对 count 仍有真实收益：相对 ordinary 为 *+0.175 [0.125, 0.230]*，donor 合并 −0.005 [−0.021, 0.011]。它只把 count 拉到 0.275，未回到 0.380±0.05；count generation +0.020 [0, 0.060] 也未形成严格正下界。fixed 相对 triggered 的 overall preference 为 *+0.0625 [0.0425, 0.0833]*。晚触发的方向有效，一个 25-step / 20-slot 修复窗强度不足。

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("../experiments/v100_perception_20260804/matched_replay_v1/charts/02-retention-task-trajectories.svg", width: 100%),
    image("../experiments/v100_perception_20260804/matched_replay_v1/charts/03-endpoint-task-deltas.svg", width: 100%),
  ),
  caption: [左：count/shape 在三策略下的 paired-preference 轨迹；右：fixed/triggered 相对 ordinary 的逐任务终点差。fixed 的主要收益集中在预注册 retention tasks，同时 donor 总体保持。],
)

#table(
  columns: (1.9fr, 1fr, 2.6fr),
  [*假设*], [*包 13 更新*], [*证据与动作*],
  [固定预算内的 replay 能缓解能力交换], [支持], [目标 preference +0.255 且 generation +0.120，二者 CI 下界均为正；总训练 examples 仍为 1,200。],
  [replay 必然牺牲 donor 任务], [未支持], [fixed donor 合并 +0.00375 [−0.0125, 0.01875]，预注册平均代价 0.05 边界未触发。],
  [宏平均足以驱动 checkpoint 选择], [反驳], [overall 上升时 count 下降 0.305；正式 sentinel 必须保存 domain-level 历史与 CI。],
  [检测后一个窗口足以完全恢复], [反驳], [triggered count 显著改善到 0.275，仍未进入 0.380±0.05 恢复带。],
  [teacher-forced 收益只反映“看见但说不出”], [反驳于 fixed 臂], [target generation +0.120 [0.050, 0.190]，内部选择与自由生成同步改善。],
)

Package verifier 重读了 21,600/3,600 条 sentinel preference/generation、50,400/8,400 条终评原始行、735 个 metrics、223 个 contrasts、18 条 trajectory 和八份 checkpoint manifest。首次 analyzer 因把字典顺序当成语义 state 顺序而退出；失败日志保留，修复为精确集合验证后重跑。完整仓库测试 *231/231* 全绿；报告共 45 页，包 13 第 33--35 页通过渲染检查。final odd half 未评分，未使用任何付费资源。

在线成本仍需压缩：三-state 全量 sentinel 用时 878.5 s，而 25 个训练 step 的纯 step wall time约 22.5 s；七-state full eval 用时 2,035.8 s。包 14 因此复用原始 rows 做 8/16/25/50/100 pair 的 trigger recall、false-trigger 与 CI 稳定性分析，再实测 teacher-only Tiny/Medium。最小可靠分母和 5%/10% 开销公式在下一节给出；此后 fixed replay 作为默认保护，机制扩展暂缓。付费 Gate D 继续暂缓。

#pagebreak()

== Sentinel 功效与 V100 开销标定（包 14，2026-08-05）

=== 25 pairs/task 是预注册护栏下的最小可靠分母

包 14 在任何 timing 结果产生前冻结源 SHA、候选分母、200 次确定性子采样、每项 2,000 次 paired bootstrap、Wilson 护栏和 trigger 规则。源数据是包 13 的 21,600 条 preference rows，仅使用 `exchange-step50` 与 `ordinary-step75` 的 `vision` 条件；final odd half 仍未评分。

#table(
  columns: (1fr, 1.1fr, 1.2fr, 1.4fr, 0.8fr),
  [*pairs/task*], [*count recall*], [*exact count-only*], [*familywise false trigger*], [*通过*],
  [8], [0.375], [0.360], [0.015], [否],
  [16], [0.760], [0.720], [0.045], [否],
  [25], [*0.975*], [*0.935*], [*0.040*], [*是*],
  [50], [1.000], [0.965], [0.035], [是],
  [100], [1.000], [1.000], [0.000], [是],
)

Tiny 因此固定为 25 pairs/task，即每个 state 300 records。其 recall Wilson 95% CI 为 `[0.943, 0.989]`，exact decision 为 `[0.892, 0.962]`，false trigger 为 `[0.020, 0.077]`；25-pair 的误触发全部来自 shape，比例 0.040。8/16-pair profile 的 recall 只有 0.375/0.760，不能进入正式合同。Medium 固定为下一档 50 pairs/task。

#figure(
  image("../experiments/v100_perception_20260804/sentinel_power_v1/charts/01-power-curves.svg", width: 86%),
  caption: [五档子样本的 count recall、精确决策率与 familywise false-trigger；虚线为预注册阈值。],
)

=== 实测 V100 时延决定稀疏 audit 频率

#table(
  columns: (1.1fr, 1fr, 1.3fr, 1.3fr, 1.2fr),
  [*profile*], [*pairs/task*], [*teacher rows/repeat*], [*teacher median*], [*end-to-end median*],
  [Tiny], [25], [600], [*22.501 s*], [31.215 s],
  [Medium], [50], [1,200], [*43.881 s*], [52.537 s],
)

六次运行均无 OOM/NaN，峰值显存 6.886 GB；同一 profile 的三次 preference rows SHA 完全一致，Tiny/Medium 的每一行又精确等于包 13 原始数据中相同 `(state,id)` 的行。以 fixed replay 的 median train-step `0.8989666 s` 代入 `t_eval / (K*t_step + t_eval) <= overhead`，模型常驻 Tiny 在 5%/10% 开销下至少间隔 476/226 steps，操作配置向上取 512/256；每次单独起进程则至少为 660/313，向上取 1024/512。DeepSeek runtime 必须用 Gate D 实测 step time 重算 K。

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("../experiments/v100_perception_20260804/sentinel_power_v1/charts/02-v100-timing.svg", width: 100%),
    image("../experiments/v100_perception_20260804/sentinel_power_v1/charts/03-required-interval.svg", width: 100%),
  ),
  caption: [左：Tiny/Medium 三次 V100 timing；右：5%/10% 开销下所需最小 checkpoint 间隔。],
)

#table(
  columns: (1.9fr, 1fr, 2.7fr),
  [*假设*], [*包 14 更新*], [*证据与动作*],
  [25 pairs/task 足以复现遗忘 trigger], [支持], [recall 0.975、exact 0.935、false trigger 0.040，三项 Wilson 护栏全部通过。],
  [8/16 pairs/task 可用于可靠在线检测], [反驳], [recall 0.375/0.760，未通过最小功效门槛。],
  [Tiny 可每 25 个小模型 step 同步执行], [反驳], [teacher compute 22.501 s，已接近一个 25-step 训练窗口。],
  [继续扩展 replay 消融是工程主线], [反驳], [fixed preventive replay 已成为默认保护；Tiny 稀疏审计，Medium 只确认告警。],
)

独立 verifier 重读 1,000 个 trial、6,000 个 task-trial 和六次 timing 的 5,400 条 raw preference rows，并核对 profile 内重复 SHA、Package-13 行级复现与全部 declared hashes。artifact manifest 的 51 个文件、3,708,513 bytes 经独立 SHA-256 重算 51/51 一致；完整仓库测试 *240/240* 全绿。报告共 49 页，包 13 收尾与包 14/Gate-D 状态第 35--38 页通过渲染目检。完整原始产物、失败边界、图表和 V100 HDD 路径均保存在 `experiments/v100_perception_20260804/sentinel_power_v1/`。未增加训练 examples，未使用付费资源。

#pagebreak()

== Qwen2.5-3B 社区可比合同冻结（包 15A，2026-08-05）

包 15A 在首个 Qwen2.5-3B 输出产生前冻结模型、数据、生成、评分、预算和迁移边界。代理主干固定为纯文本 `Qwen/Qwen2.5-3B-Instruct` revision `aa8e7253…04d1`：配置是 `Qwen2ForCausalLM`，hidden size 2048、36 层、3,085,938,688 参数，没有 `vision_config`。两份权重 shard 为 3,968,658,944 / 2,203,268,048 bytes，SHA-256 精确等于 HF LFS oid `67347b23…06c2` / `a40d941d…bfc1`；config、index 与四份 tokenizer 文件也逐文件哈希。tokenizer bundle 和 chat template SHA 分别为 `69a5cf59…93c6`、`cd8e9439…527f`。

#table(
  columns: (1.4fr, 2.1fr, 3fr),
  [*冻结项*], [*身份/规模*], [*合同动作*],
  [ScreenSpot GLM50], [50 条，10 个 platform×type strata 各 5 条], [标记为“GLM-format metric-aligned public subset”；不能声称等同社区私有 50 条。],
  [完整 ScreenSpot], [1,272 条，revision `0be08781…d5d`], [overall、text、icon/widget、Android、iOS、Windows、macOS、Web 全拆分。],
  [严格生成], [`click(start_box=[x, y])`], [只容许首尾空白；缺 canonical 空格、自然语言、浮点、越界和多坐标均 parse fail。],
  [因果条件], [vision/blind/shuffled/random/step0/previous/current], [同顺序、同生成配置；四项预注册 paired contrasts，2,000 bootstrap，seed 20260805。],
  [语言保持], [MMLU-Pro 140 + GSM8K 100], [text-only step0/previous/current；同时报告生成准确率与 teacher-forced answer NLL。],
)

公共 ScreenSpot 三份 parquet SHA 为 `ff06d312…aa8fb`、`d48b8275…0891a`、`a28a1e9f…74e5b`。Manifest 保存 fractional-xyxy/999-scale bbox、图片 SHA、source row 和类别；50/full manifest hash 为 `9583a75e…632b5` / `e556ac52…3d7cc`，derangement 无 fixed point 或同图像 SHA。

DeepSeek canonical projector 保持 33,564,672 参数与 4096 输出；Qwen 经无参数 fixed signed-pair 4096→2048 readout 接收，4096 维均有梯度路径，37,072-byte buffers SHA 为 `1cecc883…a6d47`，迁移等级为 `transferable_with_runtime_validation`。Exact step0/random 两份 134,259,248-byte FP32 权重 SHA 为 `efd942e0…b06b0` / `7bd4aacf…fc44`，重建与 save/restore 逐位一致；HF commit `65639da5…a010` 经 5/5 远端哈希验证。所有方法加载同一 step0，首个 previous-best alias step0。

官方 Qwen shard 保持逐字节 BF16 provenance；V100 计算精度在首个模型输出前单独实测冻结。目标 Tesla V100-PCIE-32GB 上，固定 4096×4096 GEMM 的 20-iteration 中位耗时为 FP16 0.03176 s、BF16 0.29078 s、FP32 0.21652 s，三组输出均 finite；BF16 为 FP16 的 9.16 倍。因此本地运行固定为 Qwen FP16、projector FP32 master weights，在 embedding splice 处做 FP16 cast，并逐步检查 loss/gradient finite。原始五次 timing 与第一次 probe 语法失败都已保留。

评分同时保存 parsed-only 与 all-sample penalized 距离、Accuracy\@50/100/200 的双分母、click-in-box、点到 bbox L2/L1 和 paired CI。无法解析的 L2 固定惩罚为 999 方形对角线 1412.7993，L1 为 1998。训练比较锁定 4k/8k/16k/32k/64k examples seen、真实 global batch 8、相同初始 projector、记录集合/顺序、分辨率和生成配置；替代 previous-best 或改变 DeepSeek 配方的结论必须三 seed。

失败记录完整保留：aria2 multi-range 生成正确长度但错误 SHA 的 ScreenSpot blob，改用 `HF_HUB_DISABLE_XET=1` 单 worker 后三 shard 全过；两次 JSON boolean 错误和首次 precision-probe 语法错误修复后重跑。仓库测试 262/262；artifact manifest 18 files / 1,967,831 bytes，独立 SHA 18/18；报告关键页完成渲染检查。本包无 Qwen3B 能力分数，只支持“合同已冻结、输入可审计”。未使用付费资源。

#pagebreak()

== Qwen2.5-3B 真图像链路 smoke（包 15B，2026-08-05）

固定合同提交后，V100 首次加载完整纯文本 `Qwen2ForCausalLM` 代理：9/9 Qwen 文件与 MoonViT 权重先在运行器内通过 SHA-256，3,085,938,688 参数全部 FP16 且冻结。冻结的 ScreenSpot 样本 `screenspot-0be08781-0472`（图像 SHA `f079f4ea…4041`）从 1080×2400 等比缩至 202×448，经真实 MoonViT-V2 得到 finite/nonzero `[128,4,1024]` 特征，再进入 exact FP32 step0 4096 projector 与固定无参数 4096→2048 receiver。官方 chat template 在渲染后按 ID 151655 精确插入一个视觉 placeholder；训练 label 只覆盖答案 token 与 `im_end`，blind 保留相同语义 prompt。

#table(
  columns: (2fr, 1.4fr, 3fr),
  [*检查*], [*结果*], [*边界*],
  [真图像前向], [`[128,4,1024]`，524,288/524,288 nonzero], [真实 MoonViT-V2 与固定 public ScreenSpot 图像；非 synthetic。],
  [projector backward], [loss 2.19208；grad norm 141.976], [六个 projector 参数张量均 present/finite/nonzero；128×4096 canonical embedding 梯度 524,236/524,288 nonzero。],
  [冻结主干], [语言梯度张量 0], [Qwen 与 MoonViT 不参与优化；receiver trainable 参数 0。],
  [一步保存恢复], [469,922,601 bytes], [FP32 projector、AdamW、Python RNG、step/history 逐值一致；另存 BF16 serving projector。],
  [资源], [8.367 GB peak；174.476 s], [Tesla V100-PCIE-32GB；wall time 含约 7 GB 冻结输入哈希；未使用付费资源。],
  [step0 生成], [vision=blind；均为中心点], [严格格式可解析；仍不建立视觉能力，不能推断模型已使用图像。],
)

首次运行已完成 load、MoonViT forward、两条件生成、backward、optimizer step 与 checkpoint 写盘，但最终验收器直接对 CPU/CUDA 两侧的 AdamW scalar 调用 `torch.equal`，在 `checkpoint_save_restore` 阶段报设备不一致。该 470,219,506-byte run 按 invalid 保留。修复仅在比较前将 restored tensor 移到 reference device，并增加跨设备相同/不同状态回归测试；retry1 通过。提交前审查又把 9 个 Qwen 文件与 MoonViT 权重 SHA 检查移入 runner，并要求六个 projector 参数张量各自 nonzero；canonical retry2 通过，checkpoint hashes 与 retry1 完全相同，generation rows 逐字节一致。独立 `sha256sum -c` 对 invalid 12/12、retry1 13/13、retry2 13/13 文件全部匹配，canonical 总量 470,235,478 bytes。完整大文件保存在 V100 HDD，Git 保存原始 manifest、prompt、generation、gradient、checkpoint 摘要、日志与失败记录。迁移判断为 `transferable_with_runtime_validation`：4096 projector、数据语义、loss mask 和 checkpoint 可复用；固定 Qwen receiver 在 DeepSeek 阶段丢弃。

本包支持“3B 真实工程链路可运行、视觉梯度到达预期模块、冻结边界与恢复格式正确”。它不支持“Qwen 3B 已获得视力”，也不改变 previous-best。Git 包 manifest 的 26 个文件、73,000 bytes 经测试逐项重算；完整仓库 269/269 tests 通过，51 页报告的包 15B 与 Gate-D 页完成渲染目检。随后包 15C 已冻结 4,000-record 顺序；能力结论仍必须来自 vision−blind、vision−shuffled、trained−random 和 ScreenSpot/TextVQA/DocVQA/OCRBench 指标。

#pagebreak()

== Qwen2.5-3B 首个 4k 训练顺序冻结（包 15C，2026-08-05）

首个 matched-budget baseline 在任何 optimizer step 产生前锁定为 Package-15A 训练包的前 4,000 行：保留源顺序、零 shuffle、零 holdout removal。micro batch 为 1、gradient accumulation 为 8、真实 global batch 为 8，因此固定为 500 optimizer steps；对 4k 子集恰好一遍，对 59,198-row 全包的 effective epochs 为 0.0675698503。源构成为 TextVQA 1,985、DocVQA 1,160、OCRBench-derived `train` 516、ShowUI desktop 339。4,000 个 ID 与路径均唯一；唯一图像内容 SHA 为 3,534，差额来自同图多问题记录。

#table(
  columns: (2.1fr, 1.4fr, 3fr),
  [*冻结项*], [*结果*], [*审计动作*],
  [顺序与预算], [4,000 examples / 500 steps], [每行保存 source-row index、record/question/answers hash；manifest self-hash `ddca738e…c2fd`。],
  [grounding target], [339/339 canonicalized], [旧训练包的 `click(start_box=[x,y])` 只在完整单动作、整数、范围合法时转为 `click(start_box=[x, y])`；自然语言与多坐标继续失败关闭。],
  [短答案 target], [1,198 passthrough；2,461 normalized majority；2 raw-majority fallback], [`(` 与 `a` 经 VQA 规范化为空，显式保留原始多数答案；每行记录实际 target 与 SHA。],
  [原图身份], [4,000/4,000 matched], [独立重读 SHA、bytes、width/height，共覆盖 1,523,324,154 image bytes。],
  [结果边界], [training=false；final=false], [无 checkpoint、无 capability score、previous-best 仍为 step0。],
)

两次失败均保留。attempt01 在第二条 ShowUI 记录发现旧 click 空格；修复后 retry1 运行到 `textvqa_train_007898`，发现多数答案 `(` 被 VQA normalization 清空。最终规则还覆盖另一条 article-only 答案 `a`。canonical retry2 生成 3,431,842-byte manifest；ordered-record SHA 为 `61fa7360…315e`。第二遍 verifier 重读全训练 JSONL，逐项匹配 4,000/4,000 records、targets、images，所有 mismatch 列表为空。Git artifact manifest 覆盖 9 files / 3,446,266 bytes；四项 order/target 与两项 artifact tests 通过，完整 V100 suite 为 268 passed、3 skipped、零失败。报告重建为 53 页，包 15C 第 40 页与 Gate-D 第 41 页通过文本提取和渲染目检。迁移判断为 `directly_transferable`：数据顺序、图像身份、target 与 examples-seen 计数不依赖 Qwen receiver，可直接供 DeepSeek 路径使用。本包不提供视觉能力证据，下一步建立内容寻址 MoonViT cache，并强制绑定该 manifest hash。

#pagebreak()

== Qwen2.5-3B 4k MoonViT 特征缓存（包 15D，2026-08-06）

Package-15C 的 4,000-row 顺序在任何 3B optimizer step 前物化为 frozen-MoonViT 内容寻址 cache。Canonical runner 来自 clean Git commit `1e4c400…4142`，运行时先校验训练顺序 self/file/record hashes、59,198-row 数据文件、每张编码图像 SHA/bytes/dimensions、MoonViT-V2 权重 `01436a95…ced24` 与 max-side 448 合同。V100 使用 eager attention 与 fp32 tower/storage；缓存特征保持 `[groups,4,1024]`，每条最多 256 groups。

#table(
  columns: (2fr, 1.5fr, 3fr),
  [*检查*], [*结果*], [*含义/边界*],
  [完成率], [4,000/4,000；0 failures], [503.5901 s wall；1,949,755,904 peak GPU bytes。],
  [内容寻址], [3,534 forwards；466 aliases], [同图多问题记录复用首个 image-byte SHA 的 tensor span，减少 11.65% tower forwards，同时保留全部记录顺序。],
  [完整存储], [111 shards；10,372,103,792 bytes], [3,534 unique spans；完整 V100 root 共 118 files / 10,374,552,697 bytes，Git 不复制 9.66 GiB tensor payload。],
  [独立 verifier], [4,000 records；111/111 shards], [重哈希并读回 2,921,816,064 logical values；2,593,021,952 unique values 均 finite，shape/order/alias/source hash 全匹配。],
  [Package-15C binding], [`ddca738e…c2fd`], [逐行匹配 ID、image SHA、width/height；3,534 unique images、466 first-occurrence aliases、max groups 256。],
  [迁移/结果边界], [`directly_transferable`], [冻结 MoonViT 表示和数据顺序不依赖 Qwen receiver；无 trained checkpoint、grounding score、paired preference 或 final-half 结果。],
)

首次 attempt 在 1,128 条后被 provenance audit 主动终止：进程执行 Package-15D 未提交源码，Git HEAD 仍为 `018b798…`，因此即使 33 个 shard 数值正常也不能作为 canonical 输入。失败根、1,128-row log、33-shard 全文件 hash 均保留，`training_use_allowed=false`；retry1 改由 clean detached worktree 和 `--require-clean-git` 在新目录重跑。随后 verifier commit `a9bd07b…a646` 独立校验 runner Git SHA、clean attestation、三份 runtime source hash 与全部 tensor 内容。Git curated package 保存 14 files / 2,728,827 bytes 加 artifact manifest，完整 shard hash 留在 `REMOTE_ARTIFACT_MANIFEST.json`。四项 verifier tests 与三项 artifact tests 通过，完整 V100 suite 为 288 passed；54 页报告的包 15D 第 41 页与 Gate-D 第 42 页完成文本和渲染检查。

结果支持“V100 能在固定顺序下低显存生成完整真实 MoonViT 训练输入”和“内容寻址能安全消除重复视觉前向”；它反驳“每条训练记录都需要独立 tower forward”。本包不触及模型是否会读图。下一步只进入严格 4,000-example projector-only 训练，再由 vision−blind、vision−shuffled、trained−random 与真实 ScreenSpot 指标作能力判定。

#pagebreak()

== Qwen2.5-3B 固定预算 4k 训练（包 15E，2026-08-06）

clean runner commit `97e9c03…a9d3a` 从 exact step0 加载 33,564,672-parameter FP32 projector，并只消费包 15C 的冻结 4,000-row 顺序与包 15D 的验签 cache。Qwen2.5-3B 的 3,085,938,688 个 FP16 参数与 4096→2048 receiver 全部冻结；AdamW 为 constant `5e-4`、weight decay 0，micro batch 1、gradient accumulation 8、真实 global batch 8。运行完成 500 optimizer steps、4,000 examples、21,532 answer tokens，即 subset 1 pass 和全量 mix 的 0.06756985 effective epochs。

#table(
  columns: (1.6fr, 1.5fr, 3fr),
  [*检查*], [*结果*], [*证据边界*],
  [训练完成度], [500/500 steps；4,000/4,000 examples], [训练 wall 532.810 s；含冻结文件哈希与加载的 total wall 905.390 s；峰值 GPU 8,979,616,768 bytes。],
  [监督计数], [21,532 answer tokens], [grounding 339、short-answer 3,661；每条 target 与包 15C canonical answer 逐条绑定。],
  [梯度冻结], [projector 6/6 tensors finite/nonzero], [step 1 与 500 均通过；Qwen gradient tensors 精确为 0，receiver trainable params 为 0。],
  [优化轨迹], [loss 4.60169 → 2.47889], [最小 1.16679、均值 3.01009；只说明 teacher-forced 优化发生，不能据此宣称视觉能力。],
  [恢复产物], [5 checkpoints；2,351,006,545 bytes], [steps 100/200/300/400/500 均含 FP32/BF16 projector、optimizer、RNG 与 history。],
  [final projector], [`566830f3…a89f`], [与 step0 `efd942e0…b06b0` 不同；下一节用固定因果评测决定是否接受。],
)

独立 verifier commit `075f3e5…acc` 重建 500 个 batch 的 exact record order 与 21,532 answer tokens，重哈希 25 个 checkpoint payload，确认 final optimizer 有 6 个 parameter states、RNG 存在、state step 精确为 500，FP32 finite 且 BF16 是 exact cast。首次 verifier attempt 因从预算字段 `RUN_CONFIG.binding` 读取 checkpoint identity 而触发 `KeyError: training_order_manifest_sha256`；失败记录保留，修复后从 contract/order/cache 与 checkpoint manifests 重建身份，训练文件未被修改。训练实现的迁移标签为 `transferable_with_runtime_validation`，能力与候选资格仍由包 15F 决定。

#pagebreak()

== Qwen2.5-3B GLM-format ScreenSpot50（包 15F，2026-08-06）

固定的 `screenspot_glm50_v1` 是 50 条公开样本、十个 platform×type strata 各 5 条的 *GLM-format metric-aligned public subset*。它不等同社区私有 50 条。50 张图像先从三个 pinned parquet 提取并逐图校验 SHA，再以 max side 1024 缓存：50 次真实 MoonViT forward、0 failure、627,596,320 feature bytes，最大 1,332 visual groups。七个角色共用固定顺序、官方 Qwen chat template、greedy 32-token generation 与 anchored exact parser；`vision=current_candidate`，`previous_best=step0`。生成 wall 240.296 s、峰值 GPU 7,245,852,672 bytes；paired bootstrap 固定 2,000 次、seed 20260805。

#table(
  columns: (1.4fr, 0.75fr, 0.75fr, 0.75fr, 0.75fr, 0.85fr, 1fr),
  [*条件*], [*parse*], [*\@50*], [*\@100*], [*\@200*], [*in-box*], [*mean dist*],
  [trained vision], [96%], [2%], [4%], [16%], [4%], [554.53],
  [blind], [100%], [6%], [6%], [16%], [12%], [392.59],
  [shuffled], [92%], [2%], [8%], [16%], [6%], [582.92],
  [step0 / previous], [100%], [4%], [6%], [14%], [10%], [398.59],
  [random projector], [92%], [4%], [6%], [12%], [8%], [468.56],
)

表内阈值与 click 使用 all-sample denominator，distance 对 unparsed 施加最大距离惩罚。trained vision 单看绝对数值达到社区参考的 parse≥92%、Accuracy\@200≥15.2% 与 mean distance≤563.7，Accuracy\@50/100 仍低于 4.3%/8.7%；更关键的因果门槛全部失败：

#table(
  columns: (2fr, 1.4fr, 2.8fr),
  [*paired comparison*], [*point estimate*], [*95% CI / 判定*],
  [vision−blind click-in-box], [−0.080], [[−0.200, 0.020]；无正增益。],
  [vision−blind mean-distance improvement], [−161.94], [[−246.70, −89.24]；显著恶化。],
  [vision−shuffled click-in-box], [−0.020], [[−0.060, 0]；无正增益。],
  [current−step0 click-in-box], [−0.060], [[−0.160, 0.040]；无正增益。],
  [current−step0 mean-distance improvement], [−155.94], [[−246.74, −75.50]；显著恶化。],
  [trained−random click-in-box], [−0.040], [[−0.140, 0.060]；无正增益。],
)

判定为 `reject_current_candidate`，previous-best 保持 exact step0，learned checkpoint 不进入 DeepSeek 候选列表。本结果反驳“0.5B 容量是唯一 grounding 瓶颈”“3B 冻结 receiver 配合 4k projector-only supervision 已足够”与“training loss 下降可证明图像被使用”。它支持更窄的结论：完整 3B train/save/resume/seven-condition eval 链路可运行；当前目标能学习输出格式，同时损伤原有的文本/中心位置先验。

完整公共 ScreenSpot 用同一 revision 的 1,272 条记录继续确认：数据已全部物化，cache 已完成 1,272/1,272、0 failure、606 real forwards 与 666 content-addressed aliases，wall 387.349 s、峰值 2,081,363,968 bytes。包 15G/15H 已完成正式生成、评分与 teacher-forced 诊断；结论见下两节。未使用付费资源，也未触碰 final halves。

#pagebreak()

== Qwen2.5-3B 完整公共 ScreenSpot（包 15G，2026-08-06）

完整 public test 固定为 1,272 条、606 张唯一图像，覆盖 Android、iOS、Windows、macOS、Web 与 text/icon-widget。图像物化共 593,342,933 bytes；max-side 1024 cache 由 606 次真实 MoonViT forward 与 666 个内容寻址 alias 组成，feature shards 共 7,609,930,976 bytes。七角色生成保持与包 15F 完全相同的模型、projector、receiver、prompt、parser、顺序和 greedy decoding，wall 2,807.658 s、峰值 7,247,035,392 V100 bytes。

#table(
  columns: (1.5fr, 0.75fr, 0.75fr, 0.75fr, 0.75fr, 0.85fr, 1fr),
  [*条件*], [*parse*], [*\@50*], [*\@100*], [*\@200*], [*in-box*], [*mean dist*],
  [trained vision], [96.46%], [1.73%], [4.87%], [11.79%], [2.67%], [565.18],
  [blind], [100%], [1.89%], [4.56%], [15.02%], [3.07%], [395.52],
  [shuffled], [96.70%], [1.42%], [5.42%], [12.03%], [2.75%], [566.26],
  [step0 / previous], [100%], [1.81%], [4.64%], [13.13%], [3.30%], [391.12],
  [random projector], [92.22%], [1.26%], [3.85%], [11.79%], [2.59%], [475.14],
)

trained vision 只达到社区 metric-aligned reference 的 parse≥92%；Accuracy\@50/100/200 低于 4.3%/8.7%/15.2%，mean distance 565.18 也略差于 563.7。该参考来自不同的社区 50 条样本，表内仅比较指标口径。因果判定更明确：

#table(
  columns: (2.2fr, 1.3fr, 2.8fr),
  [*paired comparison*], [*point estimate*], [*95% CI / 判定*],
  [vision−blind click], [−0.0039], [[−0.0165, 0.0079]；无正增益。],
  [vision−blind Accuracy\@200], [−0.0322], [[−0.0598, −0.0024]；显著恶化。],
  [vision−blind mean-distance improvement], [−169.66], [[−185.68, −154.17]；显著恶化。],
  [vision−shuffled click], [−0.0008], [[−0.0079, 0.0063]；无差异。],
  [vision−shuffled mean-distance improvement], [+1.08], [[−12.73, 14.64]；无差异。],
  [current−step0 parse], [−0.0354], [[−0.0456, −0.0259]；格式退化。],
  [current−step0 mean-distance improvement], [−174.06], [[−189.67, −157.44]；显著恶化。],
)

text/icon-widget 的 trained click 为 4.16%/0.87%，mean distance 为 516.39/624.33；macOS 的 parse 仅 77.33%、mean distance 726.97，是最大失败域。完整集复现并强化包 15F：`current_candidate` 保持拒绝，previous-best 仍是 exact step0。首次 formal 命令在新目录误加 `--resume`，在模型加载与 prediction 前 fail-closed；失败目录、全部正式 predictions、逐行 scores 与两级 SHA manifest 均已保存。

#pagebreak()

== Teacher-forced 坐标偏好诊断（包 15H，2026-08-06）

每条 ScreenSpot 样本构造两段 canonical assistant answer：正确 bbox 中心与 frozen shuffled-image derangement 所指样本的 bbox 中心。两候选在同 prompt 下组成真实 batch 2；评分为包含 `im_end` 的 token-normalized answer log probability，strict preference 只在 correct 大于 counterfactual 时记 1。该指标绕过 greedy decoding，直接测试语言头是否已把图像内容映射成正确坐标偏好。

#table(
  columns: (1.8fr, 1.1fr, 1.4fr, 1.4fr),
  [*条件*], [*preference*], [*mean margin*], [*correct NLL*],
  [trained vision], [46%], [+0.00215], [1.22362],
  [blind], [56%], [+0.00111], [2.78261],
  [trained shuffled], [52%], [+0.00940], [1.22329],
  [step0], [54%], [+0.00565], [2.50769],
  [random projector], [50%], [−0.00070], [2.47959],
)

#table(
  columns: (2.1fr, 1.25fr, 2.8fr),
  [*paired comparison*], [*point estimate*], [*95% CI / 解释*],
  [vision−blind preference], [−0.10], [[−0.22, 0.02]；无内部正确偏好。],
  [vision−shuffled preference], [−0.06], [[−0.14, 0]；错误图不更差。],
  [vision−shuffled mean margin], [−0.00725], [[−0.01287, −0.00186]；正确图显著更差。],
  [trained−random preference], [−0.04], [[−0.20, 0.12]；无选择收益。],
  [current−step0 preference], [−0.08], [[−0.26, 0.08]；无选择收益。],
  [current−step0 correct-NLL improvement], [+1.2841], [[+1.1371, +1.4389]；绝对坐标概率显著提高。],
)

训练后的 correct-image/shuffled-image correct logp 为 −1.22362/−1.22329，paired CI `[−0.03752, 0.03906]`。所以 loss 下降对应 image-agnostic coordinate-answer soft prompt；当前证据反驳“模型已在内部选对坐标，只是 greedy generation 说不出来”。step0 的 binary correct−shuffled preference 为 +0.10 `[+0.02,+0.20]`，但 mean-margin/logp CI 跨零，random projector 又是 50%/52%；它只记作初始化敏感性，不能建立能力主张。

下一项不延长同一 baseline stream，也不先调 decoding。baseline 仅有 339/4,000 条显式 click supervision；最短的单变量 screen 在相同 4,000 examples、500 steps、exact step0、分辨率、receiver 与 evaluator 下提高 ShowUI grounding 占比。若 paired preference 仍不转正，再加入训练后丢弃的 correct-vs-counterfactual margin auxiliary target，canonical 4096 projector 与 DeepSeek 迁移边界保持不变。首个 1-row 开发 run 因 alias summary 重复 ID fail-closed；commit `6c23722…09ba` 加入 role labeling 与回归测试后，clean retry/formal run 完成 2,000 bootstrap。无付费资源。

#pagebreak()

== Grounding-enriched 训练前合同（包 15I，2026-08-06）

包 15I 在任何新 checkpoint、loss 或能力分数产生前固定下一轮单变量 treatment。模型仍为纯文本 Qwen2.5-3B，projector 从 exact step0 开始，训练量仍为 4,000 examples / 500 optimizer steps，micro batch 1、accumulation 8、real global batch 8；分辨率、MoonViT、无参数 receiver、prompt、generation 和 evaluator 均不变。唯一实验变量是训练 mix 与由此显式声明的 order。

#table(
  columns: (2.2fr, 1.2fr, 2.8fr),
  [*字段*], [*值*], [*冻结规则*],
  [grounding], [2,000], [冻结源 pack 中前 2,000 条 `showui*` 记录。],
  [short answer], [2,000], [冻结源 pack 中前 2,000 条非 `showui*` 记录。],
  [merge], [4,000], [grounding-first 严格交替；每个 global batch 为 4/4。],
  [来源], [2,000 / 1,080 / 649 / 271], [ShowUI / TextVQA / DocVQA / OCRBench-derived。],
  [有效 epoch], [0.06756985], [仍以完整 59,198-row pack 为分母。],
)

Manifest 自哈希为 `d632ecc2…0bf1`，ordered-record hash 为 `f3c3dec1…15ab`。独立 verifier 从 59,198 条源记录重建 exact first-N-per-route selection，并再次读取 4,000 个 record、canonical target、图片字节和尺寸；4,000/4,000 全部匹配，覆盖 1,255,969,179 encoded-image bytes，无 mismatch。4,000 个路径对应 2,013 个 unique image hashes，反映同一 GUI screenshot 上存在多条指令。V100 完整仓库套件 317/317 通过。

这份证据只建立训练前数据身份与横向公平性，不建立视觉能力；previous-best 继续保持 exact step0。迁移标签为 `directly_transferable`，因为 source indices、targets、image identity、batch order 与 examples-seen accounting 均可直接复用于 DeepSeek。绑定 cache 已由包 15J 完成；只有 exact 500-step screen 的 GLM50 与 teacher-forced correct-image preference 同时改善，才进入完整 ScreenSpot 和三 seed。无付费资源，未评 final half。

=== Grounding-enriched MoonViT cache（包 15J）

clean runner `aa933ca…b376` 已把 Package-15I 的 4,000 条 exact order 全部缓存，zero failure。2,013 个 unique image hashes 触发真实 MoonViT forward，1,987 条后续记录复用 first-occurrence canonical span；wall 299.142 s，峰值 V100 allocation 1,947,973,120 bytes。63 个 float32 safetensors shards 共 5,943,468,912 bytes。

独立 verifier 重哈希 63 个 shards，逐条加载 2,742,976,512 logical float values / 1,485,864,960 unique values，检查 finite、shape、alias 与 Package-15I order/image binding；4,000/4,000 全部匹配，最大 visual groups 为 256。完整远端 inventory 为 70 files / 5,946,091,225 bytes。这建立可训练输入与存储/吞吐证据，不建立能力；迁移标签为 `directly_transferable`，下一步直接运行 exact step0 的 500-step projector-only screen。

#pagebreak()

== Grounding-enriched 500-step 训练（包 15K，2026-08-06）

formal run 从 exact step0 读取 Package-15I order 与 Package-15J cache。micro batch 1、accumulation/global batch 8、每个 batch 恰好 4 grounding / 4 short-answer；500 steps 覆盖 4,000 examples、36,589 answer tokens 和 0.06756985 full-pack effective epochs。Qwen 的 3,085,938,688 个 FP16 参数、MoonViT 与 fixed receiver 全冻结，只更新 33,564,672 个 FP32 projector 参数。

#table(
  columns: (2.1fr, 1.5fr, 2.6fr),
  [*字段*], [*值*], [*审计边界*],
  [loss first / last / mean], [4.14400 / 1.91563 / 2.44623], [只作 optimization evidence。],
  [training / total wall], [489.606 / 529.299 s], [同一 V100 本地 formal process。],
  [peak allocation], [8,973,374,976 bytes], [低于 32 GB hardware ceiling。],
  [checkpoints], [100 / 200 / 300 / 400 / 500], [optimizer、RNG、history、FP32/BF16 均保存。],
  [projector final SHA], [`62f69393…3df4`], [与 exact step0 `efd942e0…b06b0` 不同。],
)

step 1 与 500 的六个 projector 参数张量都存在 finite/nonzero gradient；Qwen parameter gradient tensors 始终为 0。独立 verifier 重建 500 个 batch 与 36,589 answer tokens，重哈希五个 checkpoint 的 25 个 payload，共 2,351,007,317 bytes，并确认 final training state step 500、六个 optimizer states 与 BF16 exact cast。full remote inventory 为 40 files / 2,353,629,390 bytes。

当前 `visual_ability_established=false`，previous-best 仍为 exact step0。loss 下降不能回答正确图是否优于 blind/shuffled；迁移标签只到 `transferable_with_runtime_validation`。下一步用完全相同的 GLM-format public-50、teacher-forced correct-vs-counterfactual preference、七条件与 2,000 bootstrap 评测 `62f69393…3df4`。只有两类因果证据同时改善，才进入完整 ScreenSpot/三 seed。无付费资源，未评 final half。

#pagebreak()

== Grounding-enriched paired preference（包 15L，2026-08-06）

冻结的 GLM-format public-50 先执行最有判别力的 teacher-forced gate。每条样本在同一 batch 中比较正确 bbox center 与预注册 derangement center；评分仍为包含 `im_end` 的 token-normalized assistant log probability。blind、current correct/shuffled、step0 correct/shuffled、random correct/shuffled 与四个注册 alias 全部落盘，bootstrap 固定 2,000 次、seed 20260805。formal run 用时 111.480 s，峰值 V100 allocation 为 7,652,064,768 bytes。

#table(
  columns: (2.15fr, 1.3fr, 2.85fr),
  [*条件/比较*], [*结果*], [*95% CI / 判读*],
  [vision / blind / shuffled], [52% / 56% / 54%], [正确图未超过两个因果控制。],
  [step0 / random], [54% / 50%], [current 也未超过 previous-best。],
  [vision−blind preference], [−0.04], [[−0.18, 0.10]。],
  [vision−shuffled preference], [−0.02], [[−0.06, 0]。],
  [vision−shuffled mean margin], [−0.002378], [[−0.006099, 0.001248]。],
  [trained−random preference], [+0.02], [[−0.14, 0.18]。],
  [current−step0 preference], [−0.02], [[−0.20, 0.14]。],
  [current−step0 correct-NLL improvement], [+1.44854], [[+1.29793, +1.60698]。],
)

correct-image/shuffled-image 的 correct-answer NLL 为 1.05915/1.05752；对应 correct-logp 差为 −0.001633，CI `[−0.005786, 0.002342]`。因此 2,000/4,000 grounding treatment 的新增收益仍是图像身份无关的 coordinate-answer soft prompt。它反驳“首轮失败只因 339 条 grounding 太少”和“把一半固定预算换成 ShowUI 已足以产生正确图偏好”；同时不能单独区分 frozen-LM readout、projector 表达或 CE 目标中的哪一项是主瓶颈。

paired-preference gate 判定 `reject_at_paired_preference_gate`，previous-best 保持 exact step0，不进入 full ScreenSpot/三 seed。为补齐每个 checkpoint 的固定合同，下一项仍运行相同 GLM50 自回归生成；若生成同样无 causal gain，立即筛选训练后丢弃的 correct-vs-counterfactual margin auxiliary objective。该方法不改变 canonical 4096 projector 或 tokenizer，迁移标签为 `directly_transferable`。14 个 formal 文件共 685,140 bytes 已完整归档；无付费资源，未评 final half。

#pagebreak()

== Grounding-enriched GLM50 generation（包 15M，2026-08-06）

相同 checkpoint `62f69393…3df4` 完成七条件 greedy generation 与固定 scorer。do-sample=false、temperature 0、32-token cap、chat template、strict parser 和 2,000-bootstrap seed 全部未变。formal generation 用时 121.546 s，峰值 V100 allocation 7,245,852,672 bytes；所有 prediction 与逐行 score 已保存。

#table(
  columns: (1.45fr, 0.8fr, 1.35fr, 1.0fr, 1.35fr),
  [*条件*], [*parse*], [*\@50/100/200*], [*click*], [*mean / median*],
  [vision], [100%], [2% / 2% / 14%], [6%], [502.06 / 494.94],
  [blind], [100%], [6% / 6% / 16%], [12%], [392.59 / 343.57],
  [shuffled], [100%], [2% / 2% / 10%], [6%], [502.08 / 494.94],
  [step0], [100%], [4% / 6% / 14%], [10%], [398.59 / 388.66],
  [random], [92%], [4% / 6% / 12%], [8%], [468.56 / 403.93],
)

vision−blind click 为 −0.06 `[−0.16,0.02]`，mean-distance improvement 为 −109.47 `[−171.64,−44.59]`；current−step0 mean-distance improvement 为 −103.47 `[−168.28,−38.91]`。vision−shuffled click 恰为 0 `[0,0]`，两者 mean distance 只差 0.0175，CI `[−3.5436,3.2128]`。Accuracy\@200 的 +0.04 CI 为 `[0,0.10]`，下界没有严格大于 0。parse 与 community mean-distance 门槛通过，但 Accuracy\@50/100/200 和两项图像因果门槛未通过，因此不能写成 community metric-aligned baseline。

自由生成进一步暴露坐标塌缩：vision 只有 6 种输出，31/50 为 `[125,345]`；shuffled 只有 9 种，23/50 也是 `[125,345]`，两条件 30/50 字符串完全相同。该点并非训练 label mode：2,000 个 grounding targets 有 1,066 个 unique pairs，x/y median 为 513/320，`[125,345]` 精确出现 0 次。图像身份会造成窄范围扰动，但扰动与目标位置无关。

Package 15L/15M 的 internal preference 与 free generation 结论一致，checkpoint 正式拒绝，previous-best 保持 step0。下一项先做一个分钟级 projector→fixed-receiver information-retention screen，比较 step0/current 的跨图 spread、effective rank、token 内方差、CKA 与 pairwise geometry；若信息在 projector/receiver 已塌缩，修 projector 目标或结构，若跨图多样性仍在而 LM readout 不对齐，再执行 matched counterfactual-margin auxiliary。projector 不接收文字 query，因此预注册明确禁止把 image-only target-coordinate probe 当作能力证据。该诊断直接决定下一次固定预算训练配方，不扩展 replay 支线。19 个远端 raw files 共 565,923 bytes、23 个 checked-in files 共 576,071 bytes；无付费资源，未评 final half。

#pagebreak()

== Representation-retention 结果前合同（包 15N，2026-08-06）

包 15N 在提取任何新表示结果前冻结一个短诊断。输入固定为 `screenspot_glm50_v1` 的原顺序 50 条、exact step0、grounding-enriched step 500 `62f69393…3df4` 和无参数 4096→2048 receiver；该运行不加载或改变 Qwen 权重，也不训练。

每张图先拼接 cache 中的 MoonViT groups，再在 visual-token 维做算术均值。MoonViT flattened、projector 4096 和 fixed-receiver 2048 三个边界均记录 sample RMS、between-image RMS、relative spread、participation/entropy effective rank、top-1 variance fraction、within-image token RMS、全部 pairwise RMS distance/cosine、linear CKA 和 pairwise-distance Pearson。pooled float64 tensors、全部 pair 和逐样本 norm 都必须保存并独立复算。

决策范围故意收窄：fixed-receiver 边界只有在 current/step0 relative-spread ratio < 0.25 且 participation-rank ratio < 0.5 同时成立时，才判定 gross collapse 并先修 projector/receiver representation；否则下一项执行已选定的 training-only correct-versus-counterfactual margin auxiliary。projector 没有文字 query，image-only target-coordinate probe 会混淆表征保留与 instruction-conditioned selection，本合同禁止把它当作能力证据。多样性保留也不能证明 grounding，能力门仍由包 15L/15M 的 paired preference 和 generation 决定。

本提交只冻结 analyzer、独立 verifier、metric core 与合同的逐字节哈希，不含任何 representation value、action decision 或 capability score；无付费资源，未评 final half。

=== 正式结果与分支判定

#table(
  columns: (1.45fr, 0.95fr, 0.95fr, 0.95fr, 1.0fr),
  [*边界/状态*], [*relative spread*], [*effective rank*], [*top-1 frac.*], [*sample/within RMS*],
  [projector step0], [0.2687], [13.28], [17.48%], [0.124 / 0.139],
  [projector current], [0.03719], [1.140], [93.46%], [97.31 / 18.45],
  [receiver step0], [0.2708], [13.46], [17.54%], [0.122 / 0.139],
  [receiver current], [0.03717], [1.139], [93.53%], [137.58 / 26.08],
)

两个预注册 guard 同时触发：projector current/step0 的 relative-spread 与 participation-rank ratio 为 *0.1384/0.0859*，receiver 为 *0.1372/0.0846*。projector 的 CKA 与 pairwise-distance correlation 为 0.436/0.425，receiver 为 0.428/0.416。绝对 pairwise distance 变大，并未出现零向量；更准确的机制是 projector 输出尺度暴涨、图像表示近共线、跨图 covariance 接近 rank one。receiver 几乎原样保留 ratio，反驳它是主要塌缩源。

预注册 action 因此选择 `repair_projector_or_receiver_representation_before_margin_training`。下一项先把相同 screen 扩到 steps 0/100/200/300/400/500，定位 scale/rank collapse 起点，再冻结最小 matched-budget scale/geometry-preservation treatment。counterfactual margin 暂缓，继续加同类 CE-only grounding 数据也不再是首选。

独立 verifier 重算 5 个 pooled tensors、6,125 个 pair rows、50 个 per-sample rows 与两个 action。首轮 verifier 只因 safetensors tensor-key 枚举顺序不同而失败；失败日志、冻结源码哈希与按稳定 row identity 排序的 post-result repair 均保留。首轮 full suite 又通过 348 项，只因 Windows nested manifest 计算 CRLF、Git checkout 为 LF 而失败 1 项；generic writer 已强制 LF，canonical V100 suite 为 *347/347*，17 个 package files 共 8,120,202 bytes。V100 运行前还发现 loaded kernel module 580.159.04 与 system user libraries 580.173.02 不匹配；本轮仅在 HDD 解出经 RPM Fusion official primary-metadata SHA 验证的 580.159.04 用户态库，并通过进程级 `LD_LIBRARY_PATH` 使用，没有重启、修改系统文件或停止 GPU 客户端。formal screen 用时 7.494 s，峰值显存 402,776,064 bytes；不建立视觉能力主张，无付费资源，未评 final half。

#pagebreak()

== Checkpoint-aware collapse trajectory（包 15O，2026-08-06）

本包在读取 steps 100–400 的新表示前冻结 step0/100/200/300/400/500 日程、全部 checkpoint/训练历史 SHA、相同 ScreenSpot50 顺序和包 15N 双门槛。step500 塌缩属于冻结前已知证据；真正未知的是中间四个表示与最早保存点 onset。预结果 CPU 单测曾发现 trajectory action 键与包 15N helper 不兼容，失败日志已保存；修复只增加局部占位映射，checkpoint 日程、门槛、onset 规则和训练历史绑定均未改变，修复前没有启动 GPU 分析。

#figure(
  image("../experiments/qwen3b_community_eval_20260805/representation_trajectory_v1/charts/01-collapse-trajectory.svg", width: 94%),
  caption: [Qwen2.5-3B grounding-enriched projector 的 checkpoint 表示轨迹。上图为 sample RMS 与每 100-step loss 均值；下图为相对 step0 的 spread/rank ratio 及预注册门槛。],
)

#table(
  columns: (0.65fr, 0.9fr, 1.0fr, 1.0fr, 0.9fr, 0.9fr),
  [*step*], [*sample RMS*], [*spread ratio*], [*rank ratio*], [*top-1 frac.*], [*window loss*],
  [0], [0.124], [1.000], [1.000], [17.48%], [—],
  [100], [35.74], [0.1298], [0.0772], [98.76%], [3.916],
  [200], [44.55], [0.1241], [0.0786], [97.83%], [2.069],
  [300], [59.51], [0.1226], [0.0838], [94.65%], [2.071],
  [400], [68.69], [0.1487], [0.0856], [93.62%], [2.087],
  [500], [97.31], [0.1384], [0.0859], [93.46%], [2.089],
)

projector 与 receiver 的首个保存点均同时触发两个 guard。step100 projector 的 relative spread/participation rank 为 step0 的 *0.12985/0.07721*，sample RMS 已放大 289.29 倍，top-1 variance 达 98.76%；receiver 对应 ratio 为 *0.12873/0.07596*。所有后续 checkpoint 仍处于 gross collapse。首个训练窗口 loss 均值仍有 3.916，末步 loss 2.276；塌缩先于后续 loss 平台，延长相同 CE-only 训练没有恢复几何。

预注册 action 因此选择 `apply_geometry_protection_from_initial_step_and_run_matched_lambda_screen`。证据支持 CE 梯度很早把 projector 推向巨大 common direction，反驳“只在末期过训练才塌缩”和“继续训练会自行恢复”；fixed receiver 仍被排除为主因。分辨率边界是第一个保存点为 step100，精确 onset 只能限定在 steps 1–100。下一项从 exact step0 开始做小 λ、匹配预算的 4096 输出尺度/几何保护筛选，counterfactual margin 继续暂缓。

独立 verifier 重算 13 个 pooled tensors、15,925 个 pair rows、50 个 per-sample rows、500 行训练历史、全部 geometry 与 onset；V100 全仓测试为 *361/361*，package manifest 绑定 17 个文件 / 20,683,664 bytes。formal 分析用时 56.603 s，峰值 V100 allocation 939,810,816 bytes。完整 raw tensors/rows/logs 已归档；该结果不建立视觉能力主张，不推进 previous-best，无付费资源，未评 final half，迁移标签为 `directly_transferable`。

== Geometry-repair λ calibration（包 15P，2026-08-06）

包 15P 在任何短筛选 checkpoint 产生前，使用固定 step100/batch100 和 exact step0 projector 校准预注册的 geometry objective。每个图像先将 projector token group 合并并求 token mean，再计算 log RMS、relative-spread 和 centered-Gram 三项；Gram 使用归一化 centered matrix，允许共享正交旋转。校准结果如下：

#table(
  columns: (1.2fr, 1.2fr, 1.5fr, 1.5fr),
  [*arm*], [*target auxiliary/CE norm*], [*λ*], [*status*],
  [`control`], [0.00], [0], [matched CE-only control],
  [`ratio005`], [0.05], [0.0101873051], [fixed],
  [`ratio020`], [0.20], [0.0407492203], [fixed],
  [`ratio080`], [0.80], [0.1629968813], [fixed],
)

未加权辅助梯度范数为 *3.8781849597*，记录 CE 梯度范数为 *0.7901650667*；第一次校准结果与绑定失败均完整保留。修复后的校准已重新写入 `geometry_repair_screen_v1/calibration_v2/`，SUMMARY 暴露 core/screen/order/cache/step0/step100/history/record IDs，独立 verifier 为 *verified*，且几何与 λ 数值逐位复现第一次数学结果。第一次日志管道因 `tee` 先于目录创建而返回 1，失败记录保留；第一次 control 在 step 1 前暴露 SUMMARY 绑定字段缺失，完整原始文件位于 `failures/attempt01_calibration_binding/`。修复后的 focused regression 又记录了测试-only 的 `tools/` 导入路径失败，归档于 `failures/attempt02_test_import/`，已在 GPU 重跑前修复。此处仍禁止视觉能力或 previous-best 结论；下一步使用 v2 SUMMARY 运行预注册四臂 100-step/800-example screen。

== 工程主线与 Gate D 真实缺口（2026-08-06）

当前仓库已经有真实 MoonViT-V2 编码、视觉 token 映射、placeholder 展开、loss mask、generic `inputs_embeds` 注入、DeepSeek routing-ID 保留、projector-only 训练、checkpoint 保存恢复和两分支生成实现。小文本主干与真实视觉塔已经跑通；tiny `DeepseekV4ForCausalLM` 只验证专用接口。完整 `deepseek-ai/DeepSeek-V4-Flash-0731` 权重从未完成图像 forward/backward/train/save/resume/generate，因此 Gate D 判定为 *NO-GO*。

#table(
  columns: (2fr, 1.2fr, 3fr),
  [*证据*], [*状态*], [*边界*],
  [MoonViT-V2 真实权重与预处理], [通过], [V100 输出 `[tokens,4,1024]`，可进入 projector。],
  [通用纯文本 glue 闭环], [通过（小主干）], [真实数据训练、保存、恢复和生成均有证据。],
  [Qwen2.5-0.5B 真实数据], [早期对齐证据], [16,000 examples、约 0.27 epoch；低容量混杂绝对 benchmark。],
  [Qwen2.5-3B 固定真实合同], [两轮 4k 训练 / 候选均拒绝], [首轮完整 ScreenSpot 为负；grounding-enriched preference 52%/56%/54%，generation click 6%/12%/6%。],
  [完整 DeepSeek-V4-Flash 闭环], [未通过], [完整权重未加载联跑；真实 FP4/FP8 input DGRAD 仍 `hardware_pending`。],
  [语言保持与真实视觉显著性], [两轮 preference/generation 均无 correct-image gain], [grounding-enriched vision−shuffled click 0，mean-distance CI 跨零；TextVQA/DocVQA/OCRBench、synthetic 与 language retention 尚待候选。],
)

下一条 V100 路径固定使用纯文本 `Qwen/Qwen2.5-3B-Instruct`。MoonViT-V2 保持最终视觉塔，canonical projector 输出保持 4096；Qwen 使用已冻结的无参数 4096→2048 readout，不能改写 DeepSeek 主合同。首个 matched-budget baseline 与 grounding-enriched treatment 均被 generation 和 teacher-forced 因果指标拒绝，Package 15N 又把 gross scale/rank collapse 定位到 fixed receiver 之前，Package 15O 将 onset 收紧到首个保存点 step100/800 examples，Package 15P 已把从首步施加的 geometry objective 校准为固定 λ。下一项运行四臂短 screen；只有 representation guards 与 CE 代价同时通过才训练选中 arm 的完整 500 steps。同时补齐 fixed-receiver TextVQA、DocVQA、OCRBench 与 language-retention evaluator。任何候选仍需回到 ScreenSpot50/full、通用视觉、synthetic 与语言保持合同。详细合同见 `docs/qwen2.5-3b-community-eval-contract.md`，硬阻塞与最短路径见 `docs/project-status-and-gate-d-gap.md`。付费 Gate D 继续等待明确授权，本地 3B 研究持续推进。

== Gate C：Vast 只读调研

*本节的 marketplace 价格与 offer 仅保留为 2026-08-02 历史快照，不再是可执行租机方案。* 固定 revision 源码审计发现 Transformers 量化 forward 集成尚无已确认的 autograd 注册，DeepGEMM #372 又给出 SM120/121 的 NVFP4 scale-layout weight-load blocker。当前架构建议已改为先请求单卡 SM100/B200 最小 kernel gate，只下载三个目标模块所需 shard；通过后再单独申请完整模型 Gate D。详见 `docs/dsv4-runtime-source-audit.md`、`docs/gpu-runtime-matrix.md` 与 `docs/deepseek-rental-training-contract.md`。任何付费步骤仍未获授权。

2026-08-02 12:23（UTC+8）用官方 Search Offers API 查询 verified、rentable、可靠度至少 0.98、至少 4 张卡、单卡至少 80 GB、总显存至少 320 GB 的 on-demand offer。市场是动态的，以下价格只用于预算，offer ID 不应写进自动租用脚本。

#table(
  columns: (1.7fr, 1fr, 1fr, 2fr),
  [*GPU*], [*总显存*], [*约美元/小时*], [*判断*],
  [4×A100 PCIe 80 GB], [320 GB], [\$4.00], [最便宜；无 NVLink，仅作兼容性/低价 smoke],
  [4×A100 SXM4 80 GB], [320 GB], [\$6.94], [300 GB/s NVLink；优先训练候选],
  [4×H100 PCIe 80 GB], [约 319 GB], [\$8.00], [新架构；仍需核验拓扑和 Dgrad],
  [4×H100 SXM 80 GB], [约 319 GB], [\$10.75], [性能优先候选],
  [8×A100 SXM4 80 GB], [640 GB], [\$10.37], [显存余量最大且价格低于当时 4×H100 SXM],
  [4×H200 141 GB], [约 562 GB], [\$15.74], [低 OOM 风险，成本高],
)

正式候选还应要求 NVLink/P2P、至少 512 GB RAM、至少 1.5 TB 本地盘并现场复核 CUDA/驱动。查询结果中最低价 A100 PCIe 只有约 617 GB 可用盘，不满足完整缓存计划；A100 SXM4 候选有约 10 TB 盘。当前阶段只记录 offer，明确不创建实例。

当晚第二次查询（增加 RAM ≥500 GB、盘 ≥1.5 TB 过滤，33 个 offer）要点：4×A100 SXM4 降至 \$6.41/h（961 GB RAM、2 TB 盘，Gate D 首选）；4×H100 PCIe \$6.93/h（6392 GB 盘，性价比突出）；8×A100 SXM4 \$10.30/h（11 TB 盘，情景 B 兜底）；4×B200 \$21.25/h。新出现的 4×RTX PRO 6000 96 GB（Blackwell，原生 FP4）\$4.54/h 但总线无 NVLink 且 FP4 kernel 支持未验证，仅作探索。

*最新架构修正（2026-08-05）*：A100 无原生目标量化路径，只保留完整 BF16 解量化的高成本容量参考。H100/H200 的 SM90 是 dense FP8 候选，FP4 experts 可能依赖 Triton fallback；B200 的 SM100 与固定 DeepGEMM 支持范围最吻合，成为首个单卡最小 kernel gate 的推荐架构。RTX PRO 6000/GB10 的 SM120/121 受 DeepGEMM #372 `(1,32)` NVFP4 scale-layout blocker 影响，暂不作为默认。所有型号的 weight load、forward 与 DGRAD 仍需真机逐模块验证。

=== 历史账单快照（已撤销，不用于授权）

下表是 2026-08-02 基于旧 H100/固定步数假设的敏感性计算，仅为审计保留。当前训练时长、GPU 数量和美元预算全部保持 `pending`；必须先完成用户单独授权的 SM100 单卡最小 kernel gate，再用 Gate D 实测 step time 与哨兵开销生成乐观/基准/悲观三档新预算。

流量与存储用报价接口的真实计费字段核算：权重下载 160 GB，H100 PCIe 候选流量 \$13.33/TB 约 \$2.13（重传余量加倍 + 数据集/Docker 镜像约 \$1，共约 \$4）；上传仅约 134 MB projector + 800 MB 视觉塔（首次）+ JSON，约 \$0.01。存储按 storage\_total\_cost 约 \$0.001–0.005/h，租期内几分钱——*但实例 stop 后存储继续按 \$/GB/月计费（约 \$0.107/GB/月，2 TB 停一周约 \$53），结束必须 destroy 而不是 stop*。装机与依赖调试按 +1.5 h GPU 时间计余量。

#table(
  columns: (2.6fr, 1.1fr, 1.1fr, 1.3fr),
  [*阶段（4×H100 PCIe，\$6.93/h）*], [*乐观*], [*基准*], [*悲观*],
  [装机 + Docker + 环境调试余量], [1 h], [1.5 h], [2.5 h],
  [0731 权重下载 160 GB（27 分钟理论最优；断流重传见 R4）], [0.7 h], [1 h], [2 h],
  [MoonViT-V2 视觉塔下载 800 MB（sha256 校验）], [约 0.05 h], [约 0.05 h], [0.2 h],
  [训练与评测数据约 30 GB], [0.3 h], [0.5 h], [1.5 h],
  [Gate D 判定（加载→前向→单 batch backward→20 步）], [0.5 h], [1 h], [1 h],
  [对齐训练约 3000 步（checkpoint 随传）], [3 h], [4 h], [5 h],
  [机上 benchmark 三组对照], [1.5 h], [2 h], [2.5 h],
  [权重与结果回传], [0.2 h], [0.2 h], [0.3 h],
  [*合计机时*], [*7.2 h ≈ \$50*], [*10.2 h ≈ \$71*], [*15 h ≈ \$104*],
)

流量费三档均约 \$4–5。*情景 A 最终账单：\$55（乐观）/ \$75（基准）/ \$110（悲观），按 \$120 预算。*

带宽与耗时已按报价接口的 inet\_down 字段核算：H100 PCIe 候选实测下行 2217 Mbps（约 277 MB/s），理论 160 GB 约 10 分钟；考虑 HF CDN 单连接限速（40–100 MB/s）与断流重传（本地实测 802 MB 在慢代理下耗时 2h46m），基准按 1 h、悲观按 2 h 计。checkpoint 上行流量每个约 300 MB（projector fp32+bf16+优化器）；正式频率不再固定每 500 步，由实测 save/upload/Tiny/Medium 成本满足 5%/10% 开销上限后自适应确定。

情景 A′（R3 命中，H100 无法跑 FP4 前向）：转 4×B200（\$21.25/h），同排程基准 10 h ≈ \$213；B200 有原生 FP4 Tensor Core，Dgrad 通过率也更高。决策成本为已耗的装机+下载 1.5–2 h（约 \$10–14）。情景 B（FP4 不可反传且 B200 不可用）：8×A100 SXM4 \$10.30/h，权重解量化至 bf16（568 GB）转换 1–2 h 且训练约慢 3 倍，合计 20–30 h ≈ \$210–310（该候选流量 \$1.33/TB，网络几乎免费）。*建议总预算：\$120 起步（情景 A），预留 \$220（情景 A′/B），上限 \$350；预期实际花费 \$75–110。* 决策点在租后第 2–3 小时：Gate D 不通过即 destroy 止损，损失约 \$20。

== 训练显存算术（每卡，权重张量并行切分，视觉塔/projector 每卡复制）

冻结 LLM 与冻结视觉塔意味着：LLM 侧 *没有优化器状态、没有参数梯度*——AdamW 只挂在 projector 上（`train_overfit.py` 里 `AdamW(projector.parameters())`,LLM 与视觉塔 `requires_grad_(False)`，视觉塔前向在 `no_grad` 里，梯度只需以激活形式穿过 LLM 回到 projector，不需要对视觉塔反传）。优化器全套（权重 fp32 + m/v + 梯度）只有约 0.55 GB。真正的变量是 LLM 权重本体和 Dgrad 路径的激活。

#table(
  columns: (2.4fr, 1.9fr, 1.9fr),
  [*组成（每卡）*], [*历史 SM120 4 卡算术（非候选）*], [*历史 8×A100 BF16 算术*],
  [LLM 权重], [160/4 = 40 GB], [568/8 = 71 GB],
  [LLM optimizer + 参数梯度], [0（冻结）], [0（冻结）],
  [MoonViT-V2 权重（bf16/fp32）], [0.8–1.6 GB], [0.8–1.6 GB],
  [projector 权重 + AdamW + 梯度], [约 0.55 GB], [约 0.55 GB],
  [LLM 激活（grad ckpt，seq ≤700）], [1–3 GB], [1–2 GB],
  [杂项（CUDA ctx/碎片/通信 bucket）], [5–10 GB], [4–6 GB],
  [*合计 / 可用*], [*约 47–55 / 96 GB*], [*约 77–79 / 79.1 GB*],
  [*判定*], [*余量 41–49 GB，健康*], [*贴顶，高风险*],
)

激活估算依据：开 gradient checkpointing 后只存段边界（61 层 × seq 700 × 4096 × bf16 约 0.35 GB）加段内重算临时；micro-batch 4–16 线性放大。seq 700 来自候选训练 `--max-image-side 640`（方形图最坏 529 视觉 token + 文本 ≤ 700），但 640/1024 策略须经分辨率消融后再锁定。情景 A 的生命线是 FP4/FP8 原生加载成立（R1–R3）；一旦需要 bf16，任何 4×96GB 级机器都装不下（568/4 = 142 GB > 96 GB），只剩 8 卡。情景 B 在 8×A100 上余量不足 2 GB，必须 micro-batch 1 + 更激进 checkpointing，实测 OOM 即升 4×B200（142 GB/卡，余量约 50 GB，\$21.25/h，账单已含）。

== Gate D 风险清单（租卡前预演）

不保证一次成功。下表是全部已识别坑点、概率判断与止损方案；核心原则是*失败成本锁死在租后第 2–3 小时*（Gate D 判定，损失约 \$20），任何一项失败都有明确的下一步，而不是现场想办法。

#table(
  columns: (1.9fr, 0.7fr, 3.4fr),
  [*风险*], [*概率*], [*缓解 / 止损*],
  [R1 Transformers 实际量化 module 无 input DGRAD], [高风险，真机待判], [固定源码只确认 forward dispatch，未发现 autograd 注册；底层 DeepGEMM 有 DGRAD primitive 不能证明集成路径。先用三模式脚本分别测普通 FP8、FP8 expert gate/up、FP4 expert down，任一失败即 destroy。],
  [R2 Transformers 加载原生 0731 量化权重失败], [中–高], [先做只下载目标 shards 的单卡 weight-load gate；失败不下载完整模型，也不自动切 BF16 或扩卡。],
  [R3 SM120/121 NVFP4 scale layout 缺失], [公开 blocker], [DeepGEMM #372 在审计日仍 OPEN，可在 forward 前报 `Unknown SF transformation`。RTX PRO 6000/GB10 降级；首个最小 gate 推荐 SM100/B200。],
  [R4 权重下载断流/限速], [高], [本地实测 802 MB 在慢代理下耗时 2h46m；租机标称 2217 Mbps 但 HF CDN 单连接限速 40–100 MB/s，160 GB 理论 27–67 分钟，断流可拖至 2h+。用 aria2/hf\_transfer 多连接 + 断点续传循环（已验证的做法），账单已含 0.5–2h 敏感性。],
  [R5 marketplace 实例被中断], [中], [checkpoint 含 fp32/bf16 projector、optimizer/RNG/examples/data cursor；保存与异步上传频率按实测开销自适应，队列上限为 2。中断后只在新授权内续租恢复。],
  [R6 机器环境与宣传不符（NVLink 拓扑、驱动、盘速）], [低–中], [开机 10 分钟内 `nvidia-smi topo -m` + 盘速快测，不符当场退租换机，损失不足 1h 租金。],
  [R7 情景 B 显存算术（bf16 568 GB vs 8×A100 640 GB）], [低], [冻结 LLM + activation checkpointing 下单 batch 激活很小，72 GB 余量足够；4×H200（564 GB）装不下，明确排除。],
  [R8 Hash-MoE `tid2eid` 在多卡张量并行下的分布行为], [中], [hook 方案已在真实 tiny DeepseekV4 类验证；Gate D 的单 batch backward 在大权重多卡下复验，占位位置路由一致性有断言。],
  [R9 训练数据下载（约 30 GB）经代理再耗 1–2 h], [中], [提前把训练数据镜像到项目 HF 仓库（随 checkpoint 上行通道同路），租机从 HF 直下；计入装机时间。],
  [R10 MoonViT-V2 bf16 与 fp32 参考的特征偏差], [低], [fp32 参考已锚定（eager/sdpa 差 3.1e-05）；Gate D 记录 bf16 实测差（预期约 1e-2 相对），超差则视觉塔回 fp32（仅多约 800 MB 显存）。],
  [R11 租期内时间不够闭环], [低–中], [checkpoint 流式上传保证权重永不丢；benchmark 三组对照在机上跑但数据落盘 JSON 可增量上传；最坏情况先公开 checkpoint + 部分指标。],
  [R12 多卡分布策略与互联拓扑], [中], [不再预设具体卡数或 3–5 s/step。单卡 kernel gate 通过后再按完整权重容量选择拓扑；Gate D 实测 step time、route consistency 与 sentinel 成本后生成预算，vLLM/SGLang 推理 TP 不进入反向训练。],
)

== Gate D：正式租卡前

分阶段判定（完整版见 `docs/gate-d-runbook.md`，2026-08-05 固定 revision 重写，各步独立记录不合并）：

0. 固定 revision 与配置发现；模式 A 用 BF16/FP32 frozen Linear 验证 harness，不能计作量化通过。
1. 模式 B 从 Transformers 实际加载后的模块分别测试普通 FP8Linear、FP8 expert gate/up、FP4 expert down；记录 backend、output `grad_fn`、异常与 `torch.autograd.grad`。
2. 原生失败时，模式 C 只测试预注册的 input-only DGRAD prototype；V100 reference 状态为 `hardware_pending`，不得用 BF16 替代品冒充量化通过。
3. `nvidia-smi topo -m` + 盘速快测（不符当场退租）。
4. 原生 0731 权重加载成功；文本短前向正常。
5. 单图短序列 forward（placeholder 注入）。
6. 单 batch backward，projector 梯度有限且非零，LLM/MoonViT 无梯度。
7. hook × activation checkpointing 数值一致性 + batch>1 多图位置与 Hash-MoE routing 一致。
8. 20 step 无 OOM/NaN；`--resume` 恢复一次且轨迹连续。
9. 实测 step、checkpoint save/upload、Tiny/Medium sentinel 成本，形成新的卡数、时长与美元预算。

== 自适应租期排程（价格与时长待授权）

核心约束：租期一结束就没有机器能跑动 0731 做 benchmark 或回传权重，因此训练、benchmark、上传必须在同一次租期内闭环。交付物只有 projector + 评测 JSON + 报告，与 GLM 社区只发布 projector 一致，不回传 160 GB 主干。checkpoint 发两个精度：fp32 master（约 134 MB，复现/续训用）与 bf16 serving（约 67 MB,0731 激活为 bf16)，租期内由训练产物现场转换。推理侧接入（vLLM/SGLang/llama.cpp/fastllm 补丁点、Hash-MoE 注意事项、验收检查）已写成 `docs/inference-integration.md`（2026-08-03 重写为 MoonViT-V2 版），作为后续给推理引擎提 PR 的合同文档；要点：vLLM 与 SGLang 均已 Day-0 支持 Kimi-K3（含 MoonViT3d 视觉塔）且均有 DeepSeek-V4 文本栈，patch 面只剩 projector 模块、placeholder 扩展与 Hash-MoE 路由检查；placeholder 固定为现有 `<｜image｜>`(id 129279）禁止扩 vocab，合并只替换 embedding 向量、input\_ids 保留 placeholder 供 Hash-MoE 路由。

Baseten 社区实验（baseten.co/blog/glm-52-with-vision，checkpoint baseten/GLM-5.2-Vision-NVFP4）只作为配方先验：*constant lr 5e-4*、global batch 64、约 66k 条短 QA、2 epoch ≈ 2070 optimizer steps，grokking 在第一 epoch 末附近出现。它不是本项目的时长承诺。审计发现当前训练器的历史 `batch_size=N` 是 `micro_batch_size=1` 下串行 N 次 forward/backward；若照抄 64，每个 optimizer step 会执行 64 次视觉塔和 LLM 前后向，3–6 s/step 与 2–4 h 估计均无效。新版训练器已改用 micro-batch、gradient accumulation、effective batch、examples seen 与 answer tokens 的明确计量，并暂时拒绝伪造 `micro_batch_size > 1`。正式租卡前必须实现 padded multi-example forward，在小主干上实测 micro batch 1/2/4 的吞吐与显存，再由目标 examples/token 数反推 optimizer steps 和租时。

分辨率仍由真实证据决定：先在固定数据和预算下比较训练 448/640 × 评测 448/640/1024，只有小字 OCR/grounding 收益、分布失配、视觉 token 数和吞吐都可接受时才采用更高分辨率。容量代理固定为纯文本 Qwen2.5-3B；0.5B 只保留历史 early-alignment 证据。随后在相同 ScreenSpot/TextVQA/DocVQA/OCRBench/synthetic/语言合同下筛选 projector scratch/warm-start、顶部 LoRA、blind/shuffle/random-projector、分辨率和数据配比。完整协议将由固定 3B 合同取代旧的宽泛 ablation 队列。

停训判据联合使用 macro、worst-task、vision−blind、vision−shuffle、paired generation、历史遗忘与 Pareto 前沿。loss 下降而视觉哨兵不升不能扩预算；连续多个冻结窗口无改善、达到 examples/GPU-hours/美元上限或关键任务显著退化时停止或触发预注册 replay。

#table(
  columns: (2.2fr, 1fr, 3fr),
  [*阶段*], [*时长*], [*说明*],
  [单卡最小 kernel gate], [待授权], [只下载三个目标量化模块所需 shard；weight load、forward、DGRAD；独立费用上限],
  [完整模型 Gate D], [待首次 gate 实测], [全量 hash/load、单图与 batch>1 backward、checkpointing/routing、20 steps],
  [Stage 1 短校准], [待 Gate D step time], [固定小 examples budget，高频 Tiny，测 loss/能力斜率与 LR],
  [Stage 2 受控扩展], [自适应], [只有视觉能力上升且 worst-task 可接受才扩大预算],
  [Stage 3 停止/候选评测], [自适应], [Pareto checkpoint 的 Medium/Full；final half 只对冻结候选一次],
  [*合计*], [*暂不锁定*], [由实测 examples/s、监控开销、授权时价与存储/流量公式生成三档预算],
)

若量化 DGRAD 失败，立即 destroy 并保存失败产物。完整 BF16 解量化、扩卡或更昂贵架构都需要新的用户授权，不能沿用本次 gate 预算自动执行。

= 评测与验收计划

没有 benchmark 就无法回答“接上了没有”。评测口径对标 Baseten 社区 GLM-5.2V 实验（原文 baseten.co/blog/glm-52-with-vision,0xSero/fable-glm-vision 复现）：双方视觉塔都来自 Kimi 的 MoonViT3d 家族（他们从 Kimi K2.6 抽取，我们从 Kimi K3 抽取 MoonViT-V2，当前塔宽 1024），其标志性指标是 *MMMU-Pro*（原文声称 55%，约 Claude 4.5 Haiku 水平），因此我们的评测集在 TextVQA/DocVQA/OCRBench/ScreenSpot 之外加入 *MMMU-Pro（单图子集，exact match）*。所有数字必须与 blind baseline（同一模型、无图输入）一起报告：VQA 类基准有显著语言先验，无图基线把“模型本来就会答”与“图像带来了信息”分开。

社区配方还有两个直接影响数据计划的实测结论：其一，*grokking*——batch 64 / lr 5e-4 配短答案时 loss 平台数百步后骤降（原文约 step 900）完成对齐，*长描述性答案会阻止 grok*，因此对齐数据应以短 QA 为主、长 caption 为辅，而不是只用长 caption；其二，*warm start*——从已对齐 projector 初始化可跳过大半平台期，多阶段数据混训时应复用上一阶段 checkpoint 而不是重零开始。

训练数据泄露控制（2026-08-03 定稿，GUI 修订）：正式 mix 含 TextVQA train（34.6k）、DocVQA train（25k）、0xSero art 子集（约 10k）与 ShowUI-desktop（8k）四类短答案监督；GUI 数据按用户决定纳入以建立 computer-use 基础，答案统一为 0xSero 动作格式 `click(start_box=[x,y])`（0..999 尺度，评测 parser 原生兼容）。0xSero 自带的 screenshots/multistep 行不直接复用（图像改名无法回 join；multistep 为轨迹格式），改为从同源公开数据集自取。代价与处理：ScreenSpot 自此为*域内*基准，报告中必须标注；fetch 规格机械执行 max\_answer\_words ≤ 20；组装去重为三道独立机制（2026-08-03 评审加固）：感知 aHash（hamming ≤ 6，抓缩放/重压缩重复）+ 精确像素 sha256（抓跨容器同内容）+ 归一化问题文本近重复（`--eval-jsonl` 抓跨 split 文本泄露），各机制的丢弃数分别计入 `decontamination_report.json` 随数据发布。数据产物托管于 dataset repo cyjin-yl/moonvit-dsv4-data，含来源、固定 revision 与 sha256，租机上一次下载即用（上传通道已实测：新 huggingface\_hub 分块上传经代理约 2.6 MB/s，全量数据约 1.5–2 h，见变更日志）。

已实现的评测资产：

- `moonvit_glue.metrics`：纯 Python 指标，无 torch 依赖。exact match、soft VQA（官方 min(1, 同意人数/3)）、ANLS、token-F1，以及 grounding 的 parse/Acc\@threshold/mean error。
- `tools/eval_vlm.py`：生成式评分（`--blind` 输出无图基线）与 `--shuffle-loss`（真图 vs 随机图的 teacher-forced loss 差）两种模式。shuffle-loss 是训练前最便宜的信号检验：projector 学到东西后，真图 loss 应显著低于随机图。
- `tools/fetch_eval_data.py`：固定来源拉取 TextVQA（soft VQA）、DocVQA（ANLS）、OCRBench（exact match）、ScreenSpot（in-box grounding）、MMMU-Pro（单图子集，exact match），落盘 JSONL 与 MANIFEST.json（resolved revision sha 与 JSONL sha256），沿用“信任 manifest 而不是 tag”的纪律。

#table(
  columns: (1.2fr, 2.6fr, 2fr),
  [*阶段*], [*运行内容*], [*通过判据*],
  [Gate A], [指标单元测试；小模型 shuffle-loss 管线], [测试全绿；管线可跑],
  [Gate B], [真实 MoonViT + 小 LM：生成、评分、blind 对照；overfit 训练], [端到端无报错；shuffle\_delta = +0.343（显著为正）],
  [Gate C], [训练前/后各一次：TextVQA、DocVQA、OCRBench、ScreenSpot 小子集], [训练后显著高于训练前与 blind],
)

预期锚点：0xSero 的 GLM-5.2 projector-only checkpoint 报告坐标格式解析率 92%、Accuracy\@50 4.3%、平均点击误差约 564/999。第一阶段的成功定义是 DeepSeek 路径达到同量级信号，而不是成熟 VLM 水平；grounding 在 projector-only 阶段大概率仍很弱。

本地对照臂（与租机训练并行，不花租金钱）：(1) MoonViT-V2 + 小型纯文本 LM（Gate B 各 checkpoint）跑全套 1,400 例，作为“胶水通路在小模型上的接口可学习性”读数；(2) 原生 4B VLM（官方权重、官方精度）经独立适配脚本读同一 eval JSONL、输出同格式评分，只作为评测阳性对照和成熟小 VLM 参照，不作为映射/迁移证据；(3) 小模型 projector 不能直插 0731（hidden 896/576 vs 4096），合法对照臂是“小模型 trunk 热启动（到 GELU 的 4,096 维）+ V4 重训最后一层”，与全程从零的 scratch 臂对比收敛速度。除非另有具体问题，不再追加 9B/27B 原生 VLM 对照；它们不会缩小 DeepSeek 证据缺口。

= 数据集挑选过程与构建管线

本章记录正式数据资产的挑选理由、排除项与现行构建做法（2026-08-03 定稿）。所有产物托管于 dataset repo `cyjin-yl/moonvit-dsv4-data`，来源、resolved revision 与逐片 sha256 随数据发布。

== 挑选原则

1. *对标社区实验*：评测集与训练集对齐 Baseten/0xSero 的 GLM-5.2V 配方——他们用的我们照用，保证数字横向可比；其标志指标 MMMU-Pro 必须在内。
2. *短答案优先*：社区实测长描述性答案会阻止 grokking，正式训练集全部为短答案监督（fetch 机械执行 max\_answer\_words ≤ 20）；长 caption 只用于 Gate B 信号验证，不进正式 mix。
3. *视觉塔同源*：视觉塔从 Kimi K3 抽取（与 0xSero 从 K2.6 抽取同族）；art 子集离线复刻 0xSero `build_art_dataset.py` 的 QA 模板与 schema（`tools/fetch_art_data.py`），减少配方变量。
4. *可复现优先*：每个来源固定 resolved revision；逐分片 sha256 对 LFS oid 校验；MANIFEST 随数据发布，信任 manifest 而不是 tag。

== 评测集（1,400 行）

#table(
  columns: (1.4fr, 2fr, 0.6fr, 1.1fr, 2.3fr),
  [*数据集*], [*来源（HF）*], [*行数*], [*指标*], [*挑选理由*],
  [TextVQA val], [`lmms-lab/textvqa`], [500], [soft VQA], [OCR-VQA 基本盘，社区通用口径],
  [DocVQA val], [`lmms-lab/DocVQA`], [200], [ANLS], [文档理解，与训练同族任务的 held-out],
  [OCRBench test], [`echo840/OCRBench`], [200], [exact match], [纯文字识别],
  [ScreenSpot test], [`rootsautomation/ScreenSpot`], [200], [in-box grounding], [GUI 定位；ShowUI 进训练后为*域内*基准，报告中单独标注],
  [MMMU-Pro test], [`MMMU/MMMU_Pro` `standard (10 options)` 单图子集], [300], [exact match], [社区实验标志指标（原文声称约 55%）],
)

每行一图，图像落盘，JSONL 附 MANIFEST（resolved revision + 逐分片 sha256 + JSONL sha256）。评测纪律：分 selection/final 两半——checkpoint 选择只看 selection 半，final 半只对最终 checkpoint 跑一次，避免把 test 当训练集；域内（ScreenSpot）/跨域/零样本三组分开报告，不混入一张表。

== 训练集（四类短答案监督）

#table(
  columns: (1.9fr, 1fr, 1.7fr, 2.3fr),
  [*来源*], [*行数*], [*答案形态*], [*理由*],
  [`lmms-lab/textvqa` train], [34,602], [短答案], [与评测同任务不同 split；组装时对 eval 做文本去重],
  [`lmms-lab/DocVQA` train], [约 25,000], [短答案], [文档 QA；12 个官方 train 分片离线抽取],
  [`showlab/ShowUI-desktop` train], [7,496], [GUI 动作#linebreak()#text(size: 8pt)[`click(start_box=[x,y])`]#linebreak()（0..999 尺度）], [用户决定纳入，建立 computer-use 基础；答案格式与评测 parser 原生兼容],
  [`Artificio/WikiArt_Full` + `benitomartin/fashion-product-images-small-384x512`], [池 71,780 + 2,220 val；mix 取约 10,000], [短描述 QA], [与五个评测集零交集；复刻 0xSero schema],
)

== 排除项与理由

- *长 caption 数据*（flickr8k/COCO 类）：grokking 实证表明长答案阻止对齐；flickr8k 仅用于 Gate B 信号轨，不进正式 mix。
- *0xSero screenshots / multistep 行*：screenshots 图像经改名无法回 join；multistep 为轨迹格式，与单步 QA 合同不符。GUI 监督改从同源公开集 ShowUI-desktop 自取。
- *与评测集近重复的任何行*：见下方三道去重机制。
- *低于发布精度的量化对照*：对照组评测一律官方发布权重精度 + 官方 kernel；fp8 发布的模型升 bf16 无损可做，GGUF Q4/Q6 等降精度量化不做——量化损失会混进“模型能力”读数。本地 V100 跑不动全精度 27B 则放弃该臂，或挪入租机 benchmark 窗口。

== 下载与校验通道（现行做法）

十二种数据源、93 个正式分片全部落 staging，逐片 sha256 对 LFS oid 校验并打 `.sha256ok` 幂等标记，最终 0 mismatch。通道是实测淘汰出来的：

- hf\_transfer 在代理下 0 字节挂起；hf-mirror 限流；工作站 datasets+dill 无法 pickle pyarrow MonthDayNano（任何本地 parquet 都炸 `load_dataset`）→ fetch 全程 raw pyarrow 离线直读（`--data-files`）。
- ModelScope 镜像境内直连可达 8.8 MB/s，但约 1.5 小时高强度拉取后被 CDN 按 IP 渐进限速至 0；Tailscale 中继（RTT 3.7 s，聚合 250–400 KB/s）与代理同速且挤占家庭带宽，应用户要求退役。
- 最终方案：工作站经 Clash 代理 aria2 预取 + 离线 fetch。代理上下行共享总带宽（约 250 KB/s 封顶），*上传与下载严禁并行*，HF 上传统一串行排在下载之后。
- *关键事故*：xet-bridge 签名 URL 按 range 签发，aria2 `-x8` 分段重试拿过期 URL 会读到错 range 的字节并写入错偏移——TLS 合法、文件尺寸正确、内容静默损坏（25 片）。根因定位后改单连接 `-x1 -s1` + 每片 sha256 校验，坏片全部重拉修复；另修 aria2 不认大写 `HTTPS_PROXY` 一处。

== 组装、去重与打包

1. `tools/build_train_mix.py` 按配额组装（TextVQA train 全量 + DocVQA train 25k + ShowUI 8k + art 10k），三道独立去重：感知 aHash（hamming ≤ 6，抓缩放/重压缩）+ 精确像素 sha256（抓跨容器同内容）+ 归一化问题文本近重复（对五份 eval 全量比对，抓跨 split 文本泄露）；各机制丢弃数分别计入 `decontamination_report.json` 随数据发布。
2. `tools/pack_to_parquet.py` 打包：union-keys schema（`from_pylist` 只按首行推断的坑已修，测试 6/6）；train 按 20,000 行分片，eval 五份各打一包。图像统一为 PNG 字节内嵌 parquet——付费服务器上顺序读、免密集小文件 IO，租机下载一次即用。
3. 串行上传 `cyjin-yl/moonvit-dsv4-data`（已传文件自动跳过）；README 记录来源、revision、sha256 与复现命令，原数据集协议随产物保留。

== 当前状态（2026-08-04）

- eval\_v1：五个评测集 1,400 行 + 1,400 张图像 + MANIFEST，完成并已上传 `cyjin-yl/moonvit-dsv4-data`。
- train\_v1：59,198 行 mix（TextVQA 29,252 / DocVQA 17,351 / ShowUI 5,167 / art 7,428；三道去重合计丢弃 23%）打包 3 片约 20 GB，完成并已上传。
- sft\_art：71,780 train / 2,220 val，完成并已上传。
- Gate B full-mix early alignment：完成（2000 optimizer steps、16,000 examples、约 0.27 epoch，历史 shuffle\_delta +0.727）；五基准与 stock 4B 阳性对照均完成并发布。当前优先级转为 true batching 与容量/projector/LoRA/分辨率/因果诊断消融，租时暂不锁定。

= 变更日志

#table(
  columns: (1fr, 3fr),
  [*日期*], [*变更*],
  [2026-08-02], [建立通用 placeholder 扩展、PatchMerger、MoonViT wrapper 和 DeepSeek embedding hook。],
  [2026-08-02], [确认 0731 已有 `<｜image｜>` ID 129279，禁止扩 vocab。],
  [2026-08-02], [真实缩小 DeepSeek-V4 Hash-MoE 完成 backward。],
  [2026-08-02], [发现 Transformers 4.57.6 缺少 DeepSeek-V4；基线切换为 5.12+。],
  [2026-08-02], [完成 Vast on-demand 只读快照；未创建或租用实例。],
  [2026-08-02], [公开仓库 cyjin-yl/moonvit-deepseek-v4-glue 上线并推送。],
  [2026-08-02], [新增评测资产：metrics 指标库、eval\_vlm 评分器（含 blind 与 shuffle-loss 模式）、fetch\_eval\_data 数据清单；新增 generate() 生成路径。],
  [2026-08-02], [V100 工作站：torch 2.10.0+cu128 确认含 sm\_70；26/26 测试通过；记录 NVML mismatch 与 Xet-over-proxy 挂起（HF\_HUB\_DISABLE\_XET=1 解决）。],
  [2026-08-02], [V100 完成真实 MoonViT smoke（fp32）与评测管线端到端干跑；发现 MoonViT remote code 与 Transformers 5.x 两处不兼容并 shim；确认视觉 token 数可顶爆小模型上下文。],
  [2026-08-02], [Gate B 通过：V100 上冻结 MoonViT + SmolLM2-135M 只训 projector，1000 步后 shuffle\_delta = +0.343；生成随图变化、blind 恒定。数据路径修复为相对 JSONL。],
  [2026-08-02], [Gate B 第二 backbone 复测通过：Qwen2.5-0.5B 同条件 shuffle\_delta = +0.282，确认信号与文本主干选型无关。],
  [2026-08-02], [Gate B 第三轨通过：flickr8k 1100 条（jxie 开放镜像，离线 parquet 救援落盘），Qwen2.5-0.5B 1500 步 shuffle\_delta = +0.148；三轨全部通过。],
  [2026-08-02], [训练器支持流式 checkpoint：每 500 步保存 projector(fp32+bf16)+AdamW+RNG 的完整可续训单元，后台线程传 HF；`--resume` 可从任一 checkpoint 精确续训。],
  [2026-08-02], [对齐社区 GLM-5.2V 配方：确认其视觉塔同为 MoonViT、标志指标为 MMMU-Pro（加入评测集）；记录 grokking（短答案）与 warm-start 两条训练结论；带宽速度与数据集流量成本计入预算。],
  [2026-08-03], [MoonViT-V2 移植验证：K3 视觉代码 vision-only 抽取并 vendor 进仓库（去文本模型依赖）；随机权重前向输出 `[G,4,1024]` 符合 PatchMerger 合同；新增 sdpa varlen attention 并与 eager 数值一致；`moonvit_v2` wrapper 复用 MoonViTEncoder 契约；34/34 测试通过。权重单分片（802 MB/1.56 TB）下载中，HF 写权限已验证。],
  [2026-08-03], [真实权重回归完成：shard 96 下载（sha256 `9d10c74f…`）→ 抽取 165 张量 strict-load 通过 → V100 真实前向 `[1369,4,1024]`\@1024px、finite、逐位确定、eager/sdpa 差 3.1e-05 → 34/34 回归 → 产物（含 sha256 MANIFEST）上传 HF `vision_tower_k3/`；训练/评测支持 `--vision-tower v2`。],
  [2026-08-03], [推理集成文档重写为 MoonViT-V2 版：确认 vLLM/SGLang 均 Day-0 支持 K3（含 MoonViT3d）且均有 DeepSeek-V4 文本栈，patch 面收敛为 projector 模块 + placeholder 扩展 + Hash-MoE 路由检查；llama.cpp/fastllm patch 点记录在案但 v1 不做。新增 Gate D 风险清单（R1–R11，每条带缓解与止损）与三档最终账单（情景 A \$55/\$75/\$110，预算 \$120；情景 A′ B200 与情景 B 约 \$210–310；上限 \$350）；projector 合同更新为 V2 的 33,564,672 参数。],
  [2026-08-03], [下载通道定论：aria2 多连接预取 parquet + 离线 pyarrow fetch（`--data-files`），绕开 hf\_transfer 挂起、hf-mirror 限流与 datasets+dill 的 MonthDayNano pickle 崩溃；新增 prefetch\_parquet 与 fetch\_art\_data（0xSero art 离线复刻）；MMMU-Pro config 修正为 `standard (10 options)`；62/62 测试通过。],
  [2026-08-03], [租前全链路彩排通过（Gate B+）：V100 上 400 步训练 + checkpoint 流式上传 + 三组评测 + 聚合全部闭环并落 HF；训练组 gap +0.117 vs 随机对照 +0.019，管线可判别；作为正式训练的对照组基线。],
  [2026-08-03], [外部评审（Max）并入 runbook：Gate D 分阶段化（含 `gate\_d\_dgrad.py` 单层 Dgrad reproducer、hook×梯度检查点数值一致性、`device\_map="auto"` 步时定律）；版本固定 + pip freeze 随产物；视觉 token 预算写死（训练 640 / 评测 1024）；配方谦逊条款（LR 探针兜底）；评测纪律（selection/final 两半、域内/跨域/零样本三组、3 种子 shuffle 统计）。],
  [2026-08-03], [数据管线定稿并写入报告新章节：评测 5 集 1,400 行（TextVQA/DocVQA/OCRBench/ScreenSpot/MMMU-Pro）与训练 4 类来源（TextVQA train 34.6k、DocVQA train 25k、ShowUI-desktop 7.5k、art 池 71.8k 取 10k）的挑选理由与排除项；93 个正式分片 sha256 校验 0 mismatch；三道去重、union-keys parquet 打包与串行上传纪律；DocVQA train 抽取收尾中。],
  [2026-08-04], [数据全链路闭合：train\_v1 mix 59,198 行（去重丢 23%）+ eval\_v1 + sft\_art\_v1 全部上传 `cyjin-yl/moonvit-dsv4-data`；数据仓 README 记录来源、revision、sha256 与复现命令。],
  [2026-08-04], [Gate B 全量 mix 训练完成（V100，Qwen2.5-0.5B + MoonViT-V2，2000 步）：loss 6.41→3.01，held-out shuffle\_delta *+0.727*，全量数据上视觉对齐信号明确；修复占位 token 双默认值不一致 bug（训练 Qwen `<|image_pad|>` vs 评测 DeepSeek `<｜image｜>`）为候选自动探测，85/85 测试；4 个 checkpoint 因代理 SSL 抖动待手动补传。],
  [2026-08-04], [Gate B 五基准终表（trained/random/blind）：textvqa 0.081/0/0、docvqa 0.039/0/0、ocrbench 0/0/0（地板）、screenspot 解析 0.51 精度 0.01、MMMU-Pro 0.073/0/0——判别力三条全成立，绝对读数如实留存（含坏结果）。误判审计：判别器清白（最宽提取 +1–2% 封顶）；抓到 MMMU 输入格式化两个 bug（选项字面量被逐字符拆行、缺字母标签），修复后 2.0%→7.3%，旧读数作废。],
  [2026-08-04], [原生 Qwen3.5-4B 阳性对照修复并跑完：发现同字节数坏 shard（视觉塔全集所在分片）→ SHA-256 manifest 校验、1024px 上限与指标化短输出约束。修复后 vision/blind：TextVQA 0.820/0.031、DocVQA 0.926/0.071、OCRBench 0.900/0、ScreenSpot acc 0.760/0.010、MMMU-Pro 0.300/0.280。用户指出并纠正证据边界：该模型自带视觉塔与多模态对齐，只验证评测器，不证明 MoonViT-V2→DeepSeek；训练器新增原生 VLM 拒绝保护，Gate D 才是目标能力证据。],
  [2026-08-04], [用户指出 0.5B 容量混杂：Qwen2.5-0.5B 虽为纯文本主干，但会压低 OCR/推理/格式遵从上限，dense 小模型的收敛也不能预测 DeepSeek Hash-MoE。Gate B 正式降格为工程/信号闸门；尚未完成的干净桥接对照定义为无 `vision_config` 的约 3B 纯文本主干，在相同 mix、seen-record、分辨率、scratch projector 与 selection 评测下复测。],
  [2026-08-04], [训练计量审计：历史 `batch 8` 是 micro-batch 1 下串行累积，full-mix Gate B 共见 16,000 样本、仅约 0.27 epoch，故重命名为 early alignment，低 benchmark 不作能力上限。训练器新增 examples/answer tokens/effective epochs 与显式 batch 语义、固定分层 validation manifest、10 组 derangement mean±std、多答案监督 provenance；租前实验按容量→projector/LoRA→分辨率→因果/合成诊断排序，true batching 实测前暂停锁定租时。],
  [2026-08-05], [V100 包 3/4 完成 paired preference、checkpoint 轨迹、逐层 probe 与 activation patching：shape 在 step 1500 出现内容特异中层通路，tower/projector 保留完整线性信息，训练后的语言上层把读出压回 chance。],
  [2026-08-05], [V100 包 5 等顺序适配诊断：顶部 rank-8 LoRA 最佳 strict/generation paired 为 0.605/0.080；projector 续训仅见 400 个 shape 样本即达到 1.000/1.000，vision−shuffle strict paired +0.820，并把 final assistant probe/native head 恢复到 0.945/1.000。下一项转向六任务迁移与 balanced multi-task 最小训练。],
  [2026-08-05], [V100 包 6 零训练六任务迁移：shape strict paired preference 0.130→1.000、generation 0→1.000，且 vision−shuffle 为 +0.820/+0.980；其余五项没有通过配对 bootstrap 与视觉因果双门槛。shape-only continuation 被判定为窄任务映射，下一项锁定 balanced 六任务 projector continuation。],
  [2026-08-05], [V100 包 7 完成一轮 true-batch 六任务均衡 projector 续训：step 100 时六项 strict paired preference 与 vision−shuffle 下界全部转正；自由生成仅 shape/spatial 改善，定位出四项明确的 teacher-forced/生成裂缝。下一项做等顺序额外 projector epoch 与顶部 LoRA screen。],
  [2026-08-05], [V100 包 8 等顺序 endpoint 对照：额外 projector epoch 使总体 strict preference 0.224→0.511、paired generation 0.063→0.257，并解锁 color/coordinate/spatial 生成；top-12 LoRA 只显著强化 shape 且伤害 count/spatial。fp32/bf16 敏感性被显式保留，canonical bf16 base 精确复现包 7。],
  [2026-08-05], [V100 包 9 canonical-bf16 step 25/50/100 轨迹与 step-50 全量确认：projector step 50/100 总体 strict 几乎相同，但 count/shape 与 coordinate/spatial 出现显著反向迁移；OCR 有正 vision−shuffle 内部证据、base-relative 区间仍跨零且 generation 为零。下一项先做同 basin checkpoint 插值，再决定抗遗忘辅助目标。],
  [2026-08-05], [V100 包 10 step-50/100 projector 插值：alpha 0/1 的张量和逐条评测精确复现；alpha=.25 虽提高 macro generation，却显著损失 count，并未改善 worst-task 或保留 shape。线性权重平均无法合并两端能力，下一项转任务条件抗遗忘辅助目标。],
  [2026-08-05], [V100 包 11 从 step 50 精确恢复 optimizer/order 并测试 count/shape projector-output MSE anchoring：表示距离受控，旧答案边界仍遗忘；中锚定改变 Pareto 路径但未通过 retention 规则。],
  [2026-08-05], [V100 包 12 严格匹配分层 batch 与 global random：分层在 step 50 加快 macro/generation，step 100 总体差异 CI 跨零且任务显著交换；终点梯度冲突多于 global。正式合同改为固定窗口领域覆盖，下一项进入遗忘触发 replay。],
  [2026-08-05], [V100 包 13 fixed-budget matched replay：ordinary 精确复现历史 step 100；fixed 在同一 1,200-example 预算内重分配 80 个槽位，使 count+shape preference/generation 相对 ordinary 提升 +0.255/+0.120，donor 合并近零。late trigger 识别 count 坍塌并产生 +0.175 收益，仍未完全恢复。正式候选改为 preventive replay，下一项校准 Tiny/Medium sentinel 功效与成本。],
  [2026-08-05], [V100 包 14 sentinel 功效/成本：25 pairs/task 是 Wilson 护栏下最小可靠 Tiny，count recall 0.975、exact 0.935、familywise false trigger 0.040；V100 teacher median 22.501 s，模型常驻时 5%/10% 开销至少间隔 476/226 steps。fixed replay 冻结为默认保护，Tiny 改作稀疏 audit。],
  [2026-08-05], [工程主线回到真实 VLM：容量代理固定切换为纯文本 Qwen/Qwen2.5-3B-Instruct；所有新方法必须在预冻结 ScreenSpot50/full、TextVQA、DocVQA、OCRBench、synthetic、语言保持和四项因果控制下报告。0.5B/synthetic 只保留代理证据，完整 0731 Gate D 仍未通过。],
  [2026-08-05], [包 15A 在任何 3B 输出前冻结 Qwen2.5-3B revision 与 9 个文件 SHA、1,272 条完整 ScreenSpot 和十 strata 各 5 条的 GLM-format 50、严格 click parser、七条件/paired bootstrap、无参数 4096→2048 receiver、exact step0/random projector、240 条语言保持集及 4k–64k matched-budget 节点。此提交不含能力分数。],
  [2026-08-05], [包 15B 在 V100 完成纯文本 Qwen2.5-3B + 真 MoonViT 图像的 load/generate/backward/一步 AdamW/save-resume smoke：projector 六个参数梯度均 finite/nonzero，语言梯度张量为 0，恢复逐值一致；step0 vision=blind，因此只确认工程链路。],
  [2026-08-05], [包 15C 在训练结果前冻结首个 4,000-example prefix：500 optimizer steps、零 shuffle/holdout；4,000 条 record/target/image 经第二遍独立验签。339 条旧 click 监督规范化，2 条 normalization-empty TextVQA target 显式回退原始多数答案。],
  [2026-08-06], [包 15D 将 exact 4k 顺序物化为内容寻址 MoonViT-V2 cache：4,000/4,000、0 failure、3,534 tower forwards、466 aliases；独立 verifier 重哈希 111 shards 并逐条验证 29.218 亿 logical values、Package-15C 顺序和 clean runner provenance。首个 dirty-run attempt 保留且禁止训练。],
  [2026-08-06], [包 15E 完成 fixed-budget Qwen2.5-3B projector-only 训练：500 optimizer steps、4,000 examples、21,532 answer tokens；Qwen/receiver 全冻结，五个 checkpoint 的 optimizer/RNG/顺序/哈希经独立 verifier 复核。loss 下降只记为优化证据。],
  [2026-08-06], [包 15F 完成 GLM-format public-50 七条件评测：trained vision parse/click 为 96%/4%，blind click 12%，step0 10%；vision−blind 与 current−step0 mean distance 均显著恶化。候选拒绝、previous-best 保持 step0；完整 1,272-row ScreenSpot 已缓存并开始生成。],
  [2026-08-06], [包 15G 完成 1,272-row public ScreenSpot 七条件：trained vision/blind/step0 click 2.67%/3.07%/3.30%；vision−blind mean distance 显著恶化 169.66，vision−shuffled 无差异。GLM50 失败在完整集复现，候选继续拒绝。],
  [2026-08-06], [包 15H 完成 teacher-forced correct-vs-counterfactual preference：trained vision/blind/shuffled 为 46%/56%/52%；训练把 correct NLL 从 2.51 降到 1.22，但正确图与错误图 logp 无差异。下一项固定 4k 预算提高 grounding 数据占比。],
  [2026-08-06], [包 15I 在任何新结果前冻结 2,000-grounding/2,000-short-answer exact order：每个 global batch 为 4/4；4,000 条 records/targets/images 与 1,255,969,179 image bytes 经独立复核。下一步绑定 cache 后跑 exact 500-step screen。],
  [2026-08-06], [包 15J 完成 grounding-enriched 4k cache：4,000/4,000、零失败、2,013 real forwards、1,987 aliases；独立 verifier 检查 63 shards 与 27.43 亿 logical values，并精确绑定包 15I。],
  [2026-08-06], [包 15K 完成 grounding-enriched exact 500-step 训练：4,000 examples、36,589 answer tokens、Qwen/receiver 全冻结；五个 checkpoint / 23.51 亿 bytes 经独立恢复与哈希验证，后续能力判定见包 15L。],
  [2026-08-06], [包 15L 完成 grounding-enriched GLM50 paired preference：vision/blind/shuffled 为 52%/56%/54%，vision−shuffled 为 −0.02 `[−0.06,0]`；correct-NLL 相对 step0 改善 1.44854，却无图像身份依赖，候选在因果 gate 被拒绝。],
  [2026-08-06], [包 15M 完成同 checkpoint GLM50 generation：vision/blind/shuffled click 为 6%/12%/6%，vision−blind mean distance 显著恶化 109.47，vision−shuffled distance 无差异；vision 31/50 塌缩到 `[125,345]`，该点在 2,000 个 grounding labels 中从未出现。],
  [2026-08-06], [包 15N 在读取任何新 activation 结果前冻结 representation-retention screen：固定 50-row/step0/current/receiver，记录 spread、rank、token variance、pairwise geometry 与 CKA；gross collapse 需两个 receiver guard 同时触发，并禁止把无文字 query 的 image-only coordinate probe 当作能力证据。],
  [2026-08-06], [包 15N 正式结果触发两个 gross-collapse guards：projector effective rank 13.28→1.14、top-1 variance 17.48%→93.46%，sample RMS 0.124→97.31；receiver 保留同一 ratio。下一项改为 checkpoint-aware collapse trajectory 与 scale/geometry repair，margin 暂缓。],
  [2026-08-06], [包 15O 在读取中间表示前冻结 steps 0/100/200/300/400/500、checkpoint/训练历史 SHA 与最早 onset 规则；预结果 action-key 单测失败被保存并在 GPU 分析前修复，门槛与日程未变。],
  [2026-08-06], [包 15O 发现 step100/800 examples 已 gross collapse：projector spread/rank ratio 0.1298/0.0772，sample RMS 0.124→35.74，top-1 98.76%；receiver 同步。下一训练 screen 从首步施加小 λ 的 4096 输出尺度/几何保护。],
  [2026-08-06], [包 15P 在任何 screen checkpoint 前完成 geometry-repair λ 校准：unweighted auxiliary/CE gradient norm 为 3.87818/0.79017，三档固定 λ 为 0.0101873/0.0407492/0.162997；独立 verifier 为 verified。首轮 tee 目录编排失败已保留，不影响 GPU 产物。],
  [2026-08-06], [包 15P 的第一次 control 在 optimizer step 1 前因 calibration SUMMARY 缺少 screen-contract hash 被拒绝；完整 supervision/failure 原始文件已保存，修复只补齐绑定字段并重新校准，未产生 checkpoint 或能力结果。],
  [2026-08-06], [包 15P 修复后的 focused regression 仅因测试导入路径缺少 tools/ 而在 collection 阶段失败；失败已归档并修复，没有 GPU 或训练结果。],
  [2026-08-06], [包 15P corrected calibration_v2 重新生成并由独立 verifier 核验：所有 trainer bindings 完整，λ 与几何值逐位复现；下一步启动四臂短 screen。],
  [2026-08-05], [固定 revision 的 DeepSeek 量化 runtime 源码审计与 GPU 矩阵成稿：forward 集成缺少已确认 autograd 证据，SM120/121 受 DeepGEMM #372 weight-load blocker 影响；首个付费建议降为单卡 SM100/B200 最小 kernel gate，仍等待授权。],
)

= 历史执行摘要（不再定义 live next）

Package 15A–15P、V1/V2 matched regression、Qwen2.5-7B full-public ScreenSpot 和 Qwen3.5-4B full32 external comparison 已完成并保留在历史时间线。下一步以首页 live 状态和 `docs/runtime-entrypoint-audit.md` 为准；任何付费动作等待用户明确授权。

= Projector 表征健康合同与本科生版进度解释

项目要做的事情可以画成一条链：

```text
截图 → MoonViT-V2 → 4096 维 projector → 文本主干 → click(start_box=[x, y])
```

3B 代理阶段只回答一个低成本问题：一个完全没有视觉模块的纯文本模型，能否在冻结语言能力的情况下，靠 MoonViT 和 projector 学会看图。它通过以后，才值得把同一套封装、数据顺序和监控迁移到 DeepSeek-V4-Flash-0731。0.6B 早期实验说明接口能运行，模型容量不足以做可靠能力判断，所以它保留为工程/机制证据，3B 成为固定代理。

当前证据分成三层：

#table(
  columns: (1.5fr, 2.6fr, 2.6fr),
  [*问题*], [*已经知道什么*], [*还缺什么*],
  [链路能否运行], [真实 MoonViT 图像、4096 projector、Qwen2.5-3B、冻结 receiver、训练、checkpoint、恢复和生成都已跑通；梯度确实到 projector。], [DeepSeek-V4-Flash-0731 的完整权重与真实量化 input-gradient 尚未运行。],
  [模型是否使用图像], [目前没有。ScreenSpot 的 vision 没有稳定超过 blind/shuffled；正确图与错误图的 teacher-forced preference 也没有因果差异。], [必须在修复后重新跑完整 ScreenSpot、TextVQA、DocVQA、OCRBench。],
  [失败发生在哪里], [projector 早期输出变成几乎同一个方向：RMS 爆炸、有效秩下降、top-1 variance 接近 99%，loss 仍会下降。], [需要把 collapse onset 定位到 step 1–100，并自动停止错误轨迹。],
  [Package 15P 修复是否有效], [λ 校准和四臂短跑已完成；几何看起来有改善的臂尚未通过高频 trajectory、CE 代价和视觉因果检查，因此没有晋升。], [先补高频 health trajectory；全臂失败就重设计 projector，不追加无判别力训练量。],
)

这轮新增的 `projector-health-v1` 合同把“训练健康”和“真实视觉能力”分开。每个 optimizer step 写 RMS、跨图 spread、图内 token spread、方向集中度、梯度、CE/geometry/total loss、NaN/Inf、学习率和 examples seen。固定 50 张 probe 在前 100 步高频检查 projector 与 receiver 的 effective/participation rank、top-1/top-5 variance、相对 step0 的 spread/rank/RMS、pairwise distance correlation、centered Gram similarity，以及 8 条样本的 vision/blind/shuffled teacher-forced preference。

健康门槛触发时，训练自动保存 failure checkpoint 和最近健康 checkpoint，记录当前 batch、完整 JSONL、塌缩区间，然后回滚。这个过程只说明表示还保留了图像差异，不能单独写成“模型获得视力”。真正的能力晋升仍要求完整 ScreenSpot click-in-box 提升、vision 显著优于 blind 和 shuffled、格式没有退化，并在 TextVQA/DocVQA/OCRBench 没有未解释的严重下降。

因此当前 Gate D 状态是 *NO-GO*：工程链路已经具备，视觉能力证据和 DeepSeek 真实运行证据都还不够。最短本地路径是“冻结 probe → 跑四臂高频 trajectory → 选择或否决 geometry repair → 用固定社区合同重跑真实评测 → 再做 DeepSeek runtime Gate”。

== Package 15P 高频 control：塌缩 onset 已缩到 step 1--2

第一条满足新 health 合同的 GPU 结果来自 `lambda=0` control。真实 Qwen2.5-3B、MoonViT-V2 特征缓存和冻结 receiver 均加载成功；训练在 optimizer step 2 由预注册 guard 自动停止，并回滚到 step 1 的健康 checkpoint。独立 verifier 重算了 step 0、1、2 的三组 guards，检查了三个 checkpoint 和 22 个健康产物，结果为 `verified`。

#table(
  columns: (1.2fr, 1.4fr, 1.4fr, 1.4fr),
  [*step*], [*projector*], [*receiver*], [*训练/判定*],
  [0], [RMS 0.1235；spread/rank ratio 1.000/1.000], [RMS 0.1222；spread/rank ratio 1.000/1.000], [CE 尚未开始；health pass],
  [1], [RMS 0.3531；spread/rank ratio 0.419/0.762], [RMS 0.4551；spread/rank ratio 0.341/0.625], [CE 4.1440；趋势预警尚未止损],
  [2], [RMS 0.6598；spread/rank ratio 0.269/0.502], [RMS 0.8810；spread/rank ratio 0.225/0.362], [CE 2.4380；auto-stop + rollback],
)

这条轨迹把已知故障从“step 1--100 的某处”缩小到 `[1,2]`。CE 在两步内下降约 41%，projector 与 receiver 的 RMS 同时上升，跨图 spread 和有效秩同时下降；触发原因是 `projector_rms_rising_spread_falling` 与 `receiver_rms_rising_spread_falling`。八条 teacher-forced probe 在 step 2 的 vision/shuffled preference 为 0.625/0.375，但这只是早期小探针，且没有完整 ScreenSpot 能力证据，不能提升 checkpoint。

结果支持两个判断：早期 common-direction collapse 确实是训练主故障，在线监控能在昂贵的长跑前保存可恢复现场。结果反驳“loss 下降就足以证明模型看到了图像”，也说明 lambda-zero control 不能直接进入 500-step 扩展。当前 `previous_best` 仍为 step0，Gate D 仍为 *NO-GO*；下一项按同一预算和同一停止规则运行 `ratio005`，随后依次处理另外两条预注册臂。完整 1.1 GB raw checkpoint/optimizer 现场保存在本地 V100 归档目录，Git 只提交可审查日志、manifest 和路径绑定。

`ratio005` 的 matched 结果没有改变这个判断。它使用固定的 λ=`0.01018730507868909`，在 step 2 的 total loss 为 2.45268（CE 2.43802，geometry 1.43947），projector 与 receiver 的 spread ratio 为 0.2691 与 0.2254，receiver effective-rank ratio 为 0.3622，同样触发两条 adverse-trend guard 并回滚到 step 1。control 与 ratio005 的 onset 都是 `[1,2]`，所以“control 由于没有几何项才失败”目前被反驳；需要继续跑 λ=`0.04074922031475636` 与 `0.16299688125902545`，再按预注册规则决定是否重设计 projector。

`ratio020` 也在 step 2 停止。固定 λ=`0.04074922031475636` 将 total loss 提高到 2.49668，但 projector/receiver spread ratio 仍为 0.2692/0.2255，receiver effective-rank ratio 为 0.3623；onset 仍为 `[1,2]`。control、ratio005、ratio020 三条轨迹的共同 onset 说明当前问题不能靠调大同一 geometry objective 的剂量解决。最后的 ratio080 只用于完成预注册筛选；若它也失败，停止 500-step 扩展并转 projector 结构/尺度路径。

最后的 `ratio080`（λ=`0.16299688125902545`）也在 step 2 停止，total loss 为 2.67265，receiver spread/rank ratio 为 0.2258/0.3628。四臂的 passing set 为空，`DECISION.json` 因此取消完整 500-step expansion；这条决定遵守了预注册的“无 pass 就重设计”规则。当前证据支持“塌缩是首步更新方向/尺度的结构性问题”，反驳“继续增加同一几何损失剂量即可修复”。下一条本地实验从 projector 输出 LayerNorm/RMSNorm 与 matched CE-only control 开始，仍使用同一 3B、同一数据顺序、同一 health contract。

== Package 15Q：先改输出结构，再看是否能活过前两步

Package 15Q 已在任何新结构结果前冻结。它把一个结构变量放在 `linear_2` 之后的 canonical 4096 边界：affine-free LayerNorm 或 affine-free RMSNorm。两者不增加 projector 参数，仍输出 4096 维，所有 MLP 权重、MoonViT 特征、Qwen receiver、数据顺序和预算保持不变；`baseline_none` 是同一初始化的 CE-only control。每个 arm 都必须先通过独立 structure verifier，再用同一 `projector-health-v1` 高频探针训练。

这一步回答一个很具体的问题：如果第一步更新把输出尺度推大并让图像表示共线，固定的无参数归一化能否在不改变迁移接口的前提下阻止它。健康通过只代表表示没有立刻塌缩，完整 ScreenSpot、TextVQA、DocVQA、OCRBench 和语言保持仍是能力晋升条件。若三个 arm 都失败，下一条只测试 residual/gated-residual 结构，继续保留自动止损。

在 GPU 结果前，包内还保留了三次机械失败记录：RMSNorm 单测容差、旧配置省略 `output_norm` 导致的 verifier 误判，以及后台启动器变量未导出的 shell 错误。它们均没有产生 optimizer step；前两次修复了测试/校验默认值，第三次只修复启动过程，结构合同、预算和 guards 均未改变。

== Package 15Q control：结构筛选的匹配基线

`baseline_none` 使用新 runner 在同一 100-step/800-example 合同下重新跑 control，结果在 step 2 自动止损，collapse onset 为 `[1,2]`。CE 从 4.14400 降到 2.43802，但 projector 的 spread/rank ratio 变为 0.2690/0.5022，receiver 变为 0.2254/0.3622；两条 RMS 上升、spread 下降 critical guard 同时触发。step 0/1/2 的 3 个 probe 和 3 个 checkpoint 经独立 verifier 重算通过，完整 checkpoint/optimizer/RNG 原始目录保存在 `D:/V100-artifacts/projector_structure_screen_hf_v1/baseline_none`，Git 只保留小型日志、配置、摘要和指针。

这个 control 支持“当前故障在首两步可复现，健康合同能及时止损”，同时反驳“只要 CE 继续下降就可以进入长训”。它没有产生任何真实 grounding 证据，也没有改变 previous best；下一步继续跑 matched `post_layernorm` 和 `post_rmsnorm`。

== Package 15Q：post-LayerNorm 结果

`post_layernorm` 只在 canonical 4096 输出后加入 affine-free LayerNorm，参数量和初始化权重保持不变。它仍在 step 2 自动止损，onset `[1,2]`；step 1/2 的 CE 为 4.92825/3.60105。projector 的 spread/rank ratio 为 0.1998/0.6452，receiver 为 0.1559/0.5178，两个 RMS-rising/spread-falling critical guard 同时触发。LayerNorm 把输出尺度限制在稳定范围，却没有保持跨图像 spread，说明当前首步更新的问题包含方向共线化，单纯输出归一化不够。

该臂的 3 个 probe、3 个 checkpoint 和 22 项 health artifact 已由独立 verifier 重算通过；完整原始目录在 `D:/V100-artifacts/projector_structure_screen_hf_v1/post_layernorm`。它没有进入 ScreenSpot 或其他能力评测，也没有替代 previous best。下一步运行同预算 `post_rmsnorm`；若同样失败，取消 500-step 扩展，转向 residual/gated-residual 结构。

== Package 15Q：post-RMSNorm 与最终决定

`post_rmsnorm` 同样在 step 2 自动止损，onset `[1,2]`。step 1/2 的 CE 为 4.92061/3.71350；projector spread/rank ratio 为 0.2110/0.7540，receiver 为 0.1656/0.6285。除两条 RMS-rising/spread-falling guard 外，`vision_minus_shuffle_correct_logp` 在连续 probe 点恶化，触发 causal critical guard；step 2 值为 -0.21164。3 个 probe、3 个 checkpoint 和 22 项 artifact 经独立 verifier 通过，原始目录保存在 `D:/V100-artifacts/projector_structure_screen_hf_v1/post_rmsnorm`。

三臂的共同 onset 都是 `[1,2]`，passing set 为空。`baseline_none`、`post_layernorm` 和 `post_rmsnorm` 的 CE-only/输出归一化差异没有改变早期几何崩坏，因此按预注册规则取消 500-step expansion，也不进行能力晋升。当前最有判别力的下一步是 residual 或 gated-residual projector 加 matched CE-only control；只有健康轨迹通过，才继续 ScreenSpot、TextVQA、DocVQA 和 OCRBench。

== Package 15R：残差结构合同已冻结

15Q 的三条轨迹都在 step 2 止损后，15R 把原始 projector 保留为主通路，在 `linear_2` 后增加一条 4096 宽度残差：`zero_init_residual` 将分支权重全部置零，`gated_residual` 使用正常初始化分支并将 scalar gate 置零。两者的 step0 输出逐元素等于旧 projector，base MLP 权重 SHA 完全相同；参数量分别为 50,341,888 和 50,341,889。control 仍是同一 step0 projector，训练主干、receiver、数据顺序、学习率和 health schedule 全部固定。

15R 的首阶段只跑 100 optimizer steps/800 examples，要求 projector 与 receiver 都没有 critical guard、没有早期塌缩后恢复，且 CE 代价满足预注册上限。健康通过才进入固定 ScreenSpot/TextVQA/DocVQA/OCRBench 合同；健康本身不能作为视觉能力结论。初始化 hash、独立 verifier、失败记录和迁移边界已冻结，下一步在 V100 上按 control、zero-init、gated 顺序执行。

== Package 15R：baseline control 先验复现

启动器的第一次尝试在 GPU 初始化前失败：包装脚本预先创建了输出叶目录，训练器按“拒绝覆盖已有 run”合同退出。该次日志已归档到 `projector_residual_screen_v1/failures/attempt03_precreate_output/`，没有 optimizer step、checkpoint 或能力结果；修复只改变目录创建时机。

修复后的 `baseline_none` 使用提交 `5682265c` 在 V100 上加载真实 Qwen2.5-3B、冻结 MoonViT-V2 特征和同一 receiver。step 1 probe 尚未触发 critical guard；step 2 的 projector probe RMS 为 `0.6598`，spread/rank ratio 为 `0.2690/0.5022`，receiver 为 `0.8810` 与 `0.2254/0.3622`。CE 从 `4.14400` 降到 `2.43802`，两条 RMS-rising/spread-falling critical guard 在 `[1,2]` 区间触发，训练自动停止并回滚到 step 1。

独立 verifier 重算了 3 个 probe、3 个 checkpoint 与 22 个 health artifact，状态为 `verified`，health manifest 总字节数 `1,141,294,624`。完整 checkpoint、optimizer、RNG、batch IDs 和日志已完成本地/远端双份保存；Git 只保留可审查结果和 raw pointer。该 control 只证明健康合同能复现并止损，不能证明视觉能力，也没有进入 ScreenSpot、TextVQA、DocVQA 或 OCRBench。下一步运行零初始化残差臂；若它仍在 step 2 失败，优先检查残差分支的梯度接口，而不扩大训练预算。

`zero_init_residual` 的第一次候选启动在 GPU 初始化前被旧 runner 拒绝：它把任何非 canonical projector 都当成错误 SHA。该次失败已保存为 `projector_residual_screen_v1/failures/attempt04_variant_sha_gate/`，没有 optimizer step 或 checkpoint。修复后的通用 projector binding 会在加载语言主干前检查注册 arm 的 config/weights SHA、冻结 base 权重、参数量、共享 base tensor 和逐元素相同的 step0 输出；训练与 health 合同保持不变。修复先提交，再重启候选，避免把接口修复和实验结果混在一起。

== Package 15R：zero-init residual 结果

修复后的 `zero_init_residual` 通过 variant binding 后进入真实 Qwen2.5-3B 训练。它从同一 step0 输出开始，step 1 的 projector 梯度范数（clip 前）为 `189.33`，证明残差分支确实参与优化；然而 step 2 的 projector output RMS 已升到 `1.4244`，spread/rank ratio 降到 `0.1838/0.1799`，receiver RMS 为 `1.9779`，ratio 为 `0.1672/0.1358`。CE 从 `4.14400` 降到 `2.88565`，两条固定 RMS-rising/spread-falling critical guard 在 `[1,2]` 触发，运行自动回滚。

独立 verifier 重算 3 个 probe、3 个 checkpoint 和 22 个 health artifact，状态为 `verified`，总字节数 `1,711,724,384`；完整 raw checkpoint、optimizer、RNG 与日志已保存并做 SHA 重算。该臂支持“梯度能进入残差支路，但 receiver-facing 更新仍会压扁图像差异”，反驳“zero-init residual 单独足以修复 projector”。它没有进入任何真实能力评测；最后的 gated arm 仍按同一合同执行。

== Package 15S：V1/V2 对照与首个图像因果目标

公开的 MoonViT-SO-400M/K2.6 线 V1（1152 维输出）和 K3/MoonViT-V2 exact `PatchMergerMLPV2` 已放进同一 Qwen2.5-3B projector-only health contract。两臂使用完全相同的 4,000-row 顺序、50 张 probe、receiver 与自动止损。V1 在 step 2 的 projector/receiver effective-rank ratio 降到 `0.264/0.212`；exact V2 在合同学习率 `5e-4` 下为 `0.910/0.830`，几何明显更健康，但 vision-minus-shuffle correct-logp 仍为负。V1 因此没有救活 3B 路径，结果削弱了“V2 embedding 压缩单独导致失败”的解释。

exact V2 的 projector 学习率降到 `5e-5` 后，前两步 rank/spread 几乎保持 step0，说明更新尺度参与了 geometry collapse。视觉因果信号没有随之出现。首个 paired image-shuffle margin screen 固定 `margin=0.1`、`lambda=0.1`，hinge loss 从 `0.753` 降到 `0.440`，vision-minus-shuffle correct-logp 从 `-0.240` 移到 `-0.061`；step 2 的 vision/shuffled/blind preference 为 `0.625/0.750/0.625`。训练仍由因果 guard 自动止损，独立 verifier 为 `verified`。

这里有一条有限的好消息：exact V2 加小学习率保住了图像表示差异，配对目标也把错误方向推近零。坏消息更关键：模型仍没有更信任正确图片，训练还不能进入完整 ScreenSpot 或长预算。下一项用相同 MoonViT-V2、projector、cache 和 guards 做文本主干容量/接收器先验对照：优先纯文本 Qwen2.5-7B；9B/14B 先通过 V100 32GB 的显存与 input-gradient gate；Qwen3.5 只保留视觉预训练后的语言权重，绕过原生 vision tower、merger 和 cross-attention，由我们的 MoonViT/projector 直接输入。Qwen3.5 结果只回答“有视觉预训练的 receiver 是否更容易读外部视觉 token”，不能替代 DeepSeek-V4-Flash-0731 的真实 Gate D。

== Package 15S-capacity：Qwen3.5 stripped-native 接收器先验

本包绕过 Qwen3.5 原生视觉塔、merger 和 visual forward，只在同一 canonical 4096 projector 边界把 MoonViT-V2 特征送入视觉预训练语言接收器。hook 显示三个运行的 `native_vision_forward_calls=0`。4B BF16/16-token 运行保持 finite，但 `vision-minus-shuffle=-0.0597`；4B FP16 full-token 运行在首个更新后 NaN/Inf；9B BF16/16-token 使用 identity 4096 receiver，得到 `vision-minus-shuffle=+0.6265`，CE 从 1.5632 降到 0.9113。

9B 的官方 HF revision 是 `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，config SHA 是 `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`，权重 SHA 在 `configs/qwen3.5-9b-hf-sha256.json`。这个结果支持接收器容量和视觉预训练先验是当前 3B 失败的候选瓶颈；它没有证明 projector 获得了通用视觉能力。样本数为 1、视觉 token 数为 16、只做一步更新，不能代替固定 ScreenSpot、TextVQA、DocVQA 或 OCRBench。该方法标记为 `transferable_with_runtime_validation`，`capability_claim_allowed=false`。

下一条实验先在 9B BF16 上筛选 32/64/128/240 token 的数值和局部配对稳定性，再做 Qwen2.5-7B 纯文本 matched control。Gate D 继续保持 *NO-GO*，直到 DeepSeek-V4-Flash-0731 的真实量化 input-gradient、完整图像 forward/backward、稳定 save/resume 和固定 benchmark 全部通过。

9B 的 token-length sweep 在 32、64、128、240 个视觉 token 上均通过 finite/input-gradient gate，且原生视觉 hook 均为零；单样本 `vision-minus-shuffle` 为 `+0.1781/-0.4574/+0.2881/-0.8842`。正 margin 没有随 token 数稳定，说明 16-token 的 receiver-prior 信号对 token 序列敏感。Qwen2.5-7B 纯文本 matched control 在 FP16 的 16/240 token 也均 finite，但 `vision-minus-shuffle=-1.0731/-0.8335`，容量增加本身没有带来外部视觉因果。

这些结果把当前假设排序为：视觉预训练 receiver prior 可能有帮助，纯文本容量本身不足以解决接口问题，projector 输出的 token 数、顺序和尺度仍需要多样 probe 验证。所有运行都只有单样本一步更新，因此不进入社区排行榜，也不构成真实视觉能力。下一项先做 9B 多样 probe 与 random-projector 小筛选；Gate D 继续 *NO-GO*。

固定 8 个样本的 receiver-prior probe 给出更严格的边界。9B BF16 在 16 token 时 `vision-minus-shuffle=+0.0447 ± 0.3729`，在 240 token 时为 `-0.0748 ± 0.4520`；`vision-minus-blind` 分别为 `+0.1993 ± 0.2142` 与 `+0.6753 ± 0.3335`。有视觉 token 会改变接收器的答案分布，但正确图相对打乱图的平均优势接近零。该结果支持“接收器先验可被激活”，同时反驳“单个正 margin 已足以证明真实视觉 grounding”。

同一 8-sample/240-token 条件下，V1 projector 的 `vision-minus-shuffle=+0.0620 ± 0.4185`，V2 为 `-0.0748 ± 0.4520`；V1/V2 都有正的 vision-minus-blind，但 V2 的均值更高。V1 的轻微优势被样本方差覆盖，不能称为版本修复。当前最强解释从“V2 压缩独有故障”转向 token ordering、尺度和监督接口的共同问题。

Qwen3.5 原生 3D mRoPE 的 V2 诊断在同一 8-sample/240-token 条件下给出 `vision-minus-shuffle=-0.0375 ± 0.4537`、`vision-minus-blind=+0.6680 ± 0.2896`，与普通连续位置几乎相同。它没有修复 paired image attribution gap，且只属于 Qwen-specific diagnostic，不具备 DeepSeek 迁移资格。

9B projector-only backward 的本机边界已经实测：240-token 首次尝试触发 NVML allocator assert；修复训练器的 graph retention、缩到 16 token 后仍在 25.88 GiB allocated / 41 MiB free 时 OOM。由此把 Qwen3.5-9B 固定为 inference/input-gradient receiver-prior diagnostic；V100 projector 训练继续使用 3B/7B，Gate D 仍等待真实 DeepSeek runtime 证据。

Qwen2.5-7B 的 3-step CE-only projector screen 在 V100 上完成：8 samples、16 tokens、FP16 全 finite，projector RMS `0.99775→0.99779`、between-image RMS `0.4064→0.4055`，CE `0.2381→0.0094`，但 `vision-minus-shuffle` `+0.0333→-0.1027`。它给出了容量对照的关键答案：7B 能训练，不代表会按图像回答；下一轮训练目标必须直接优化 paired image attribution，并保留 CE-only matched control。

匹配的 paired image-vs-shuffle margin（λ=`0.1`、margin=`0.1`）在 7B 上保持 finite。16-token 3-step 后 `vision-minus-shuffle=+0.0984`，240-token 为 `+0.0090`；RMS 和 between-image spread 没有崩坏。它支持监督方向的价值，却反驳“短上下文正 margin 足以代表完整视觉能力”。

== 2026-08-07：receiver-prior 监督接口审计与结论修订

对 stripped-receiver 工具做独立数据审计时发现，旧的 3B/7B/9B capacity 运行只从
`FeatureCache` manifest 读取样本。该 manifest 不含问题、instruction 或答案；旧
`build_inputs()` 因此回退到统一 prompt 和 `click(start_box=[500,500])`。旧运行的
checkpoint、optimizer、健康日志和数值结果仍完整保留，能证明模型加载、projector backward、
保存和 finite health 链路；其 CE 与 `vision-minus-shuffle` 不能再被解释为真实 grounding
attribution，也不能用于宣称某个 token 数、receiver 或 projector 更好。

本次修订冻结了带真实问题/答案和图片 SHA 的 8-sample manifest：
`experiments/qwen3b_community_eval_20260805/capacity_controls/qwen25_7b_real_probe_manifest.json`，
SHA-256 `9fb216e...de130`。训练和 probe 现在按 ID join feature cache，逐条核对 image SHA，
缺少监督字段时 fail-closed。修复后的 Qwen2.5-7B 16-token matched screen 使用真实 TextVQA、
DocVQA、ShowUI click 和 VQA 答案：CE-only 的 CE `6.9373→4.4393`、
`vision-minus-shuffle -0.2741→-0.8746`；paired margin λ=`0.1` 的 CE `6.9373→4.5825`、
`vision-minus-shuffle -0.2741→-0.1613`。两条均 finite 且 RMS/spread 稳定，但没有正的图像因果信号。

因此当前保留两层结论：7B projector-only 训练在 V100 32GB 可运行；Qwen 代理尚未证明
MoonViT token 已被纯文本 receiver 解读。后续机制记录继续同时保存训练健康（loss、RMS、spread、rank、
gradient、NaN/Inf）和真实能力（vision/blind/shuffled、ScreenSpot、TextVQA、DocVQA、OCRBench），
任何单项下降或固定答案扰动都不能替代 paired benchmark。240-token 真实答案 matched arm 完成后，
才决定 prefix/uniform/mean-pool、输入尺度和 projector 结构的最小筛选；Gate D 仍为 NO-GO。

真实答案 240-token matched screen 随后完成：CE-only 的 `vision-minus-shuffle` 为
`+0.3338→+0.2444`，paired margin λ=`0.1` 为 `+0.3338→+0.3375`；两臂均 finite，RMS/spread
稳定，margin 相对 control 的末步差约 `+0.093`。这只能支持“paired objective 在全长 token 条件下没有立即破坏已有信号”，不能称为 grounding 改进。

同一合同的 16-token 选择 screen 给出 prefix `-0.2741→-0.1613`、uniform
`-0.2421→+0.0630`、mean-pool `-0.2036→-0.1363`。uniform 的终点方向较好，mean-pool
出现约 `4,292` 的梯度峰值；8 条混合真实答案样本和 3 steps 不足以做能力结论。下一项先为四种 token 条件生成逐样本 probe、random-projector 和 bootstrap，之后再决定是否扩大到 32 samples 或 ScreenSpot50。

32-sample frozen receiver-prior probe 已完成。固定 seed `20260805`，TextVQA、DocVQA、ShowUI 和普通 VQA 各 8 条，manifest SHA 为 `c726ebfd...a5a629f`。2,000 次 paired bootstrap 的 `vision-minus-shuffle` 结果为：full/prefix 240 `-0.22`、CI `[-0.64,0.13]`；prefix 16 `-0.07`、`[-0.35,0.21]`；uniform 16 `-0.05`、`[-0.31,0.22]`；mean-pool 16 `+0.14`、`[-0.12,0.39]`。四种条件的 `vision-minus-blind` 均有正 CI，说明 receiver 被视觉 token 激活；正确图与 shuffled 图的差异仍不足以支持 grounding。

Qwen2.5-7B 的 32-sample mean-pool 训练 matched control 随后完成：CE-only 的 `vision-minus-shuffle` 为 `+0.1351→+0.2051`，paired margin λ=`0.1` 为 `+0.1351→+0.1722`；两者 RMS/spread 稳定，margin 没有优于 CE-only，梯度峰值约 3,000。尺度 sweep 以文本 embedding RMS `0.01364` 和 projector RMS `0.994` 为参照，测试 projector scale `0.01/0.03/0.1/0.25/1.0`；所有 paired CI 仍跨 0。下一步仅保留 scale=`0.1` 的 matched training screen，失败后停止扩大 Qwen 训练量，转 projector 结构和辅助目标。

== 2026-08-07：scale=0.1 训练与机制经验归档

scale=`0.1` 的 32-sample matched training 使用同一真实答案 manifest、mean-pool 16、循环 derangement、exact step0 projector 与冻结 Qwen2.5-7B receiver。CE-only 的 CE 从 `6.9045` 降至 `5.8405`，`vision-minus-shuffle` 从 `-0.0167` 到 `+0.1297`；paired margin (`lambda=0.1, margin=0.1`) 的 CE 从 `6.9045` 到 `5.9001`，`vision-minus-shuffle` 从 `-0.0167` 到 `+0.2487`。两臂全程 finite，gradient peak 约 `781`。

训练后 32 条 probe 的 2,000 次 paired bootstrap 显示：CE-only 的 `vision-minus-shuffle` 为 `+0.1297`，95% CI `[-0.3042, 0.5542]`；margin 为 `+0.2487`，CI `[-0.1099, 0.6001]`。两臂的 `vision-minus-blind` 均有正 CI，margin 的 `vision-minus-random-projector` 为 `-0.5038`，CI `[-1.0500,-0.0247]`；margin-minus-CE 配对差为 `+0.1190`，CI `[-0.0429,0.2881]`。随机 projector 对照变差和视觉 token 激活都成立，正确图像相对 shuffled 的稳定归因仍未成立。

这轮把几类机制经验收束到同一证据表：projector-only 在不同 receiver 上的容量与视觉预训练先验差异、vision/blind/shuffled/random-projector attribution、token 数量与压缩方式、CE 下降但视觉归因不升、V1/V2 与 mRoPE 对照，以及 RMS/spread/rank/Gram/gradient collapse 轨迹。健康指标只回答表示是否保持可用，paired grounding 指标才回答模型是否依据正确图像。Qwen 代理仍没有可以替换 `previous_best` 的 checkpoint；当前下一项只注册一个 projector 结构或辅助目标变量，并保留严格 matched CE-only control。

== Gate D：当前边界

Gate D 仍为 *NO-GO*。V100 已验证 MoonViT-V2 真权重、4096 projector、placeholder 展开、冻结 receiver 的 backward、自动止损和 checkpoint/RNG/save-resume；完整 DeepSeek-V4-Flash-0731 仍缺真实权重加载、目标 FP4/FP8 input DGRAD、Hash-MoE 图像 forward/backward、batch/routing/activation-checkpointing 一致性、20-step 稳定 checkpoint 以及固定真实 benchmark。按当前节奏，本地还需约 1--3 个短实验周期来冻结候选和补 verifier；真实 DeepSeek 训练仍需用户授权付费硬件，授权前不租卡、不下载完整模型，也不把 Qwen 结果写成 DeepSeek 能力。

== 2026-08-07：lambda=0.5 paired margin screen

在预注册配置下，固定 scale=`0.1`、mean-pool 16、32 条真实答案、同一循环 derangement、同一 exact step0 projector 与冻结 Qwen2.5-7B receiver，运行 CE-only control 和 paired margin lambda=`0.5`。CE-only 的 CE `6.9045→5.8405`、`vision-minus-shuffle -0.0167→+0.1297`；lambda=0.5 的 CE `6.9045→5.9831`、`vision-minus-shuffle -0.0167→+0.4874`。两臂全程 finite，gradient peak 约 `781/459`，between-image RMS 稳定。

训练后 probe 的 2,000 次 paired bootstrap：lambda=0.5 的 `vision-minus-shuffle=+0.4874`，95% CI `[+0.1423,+0.8786]`；`vision-minus-blind=+1.9574`，CI `[+1.3909,+2.5699]`；相对同批 CE-only 的 paired 提升 `+0.3577`，CI `[-0.1287,+0.8397]`；`vision-minus-random-projector=-0.3397`，CI `[-0.7099,+0.0082]`。

这是当前第一条通过真实答案 32-sample paired image attribution CI 的训练轨迹。证据仍限于 teacher-forced、16 visual tokens、Qwen2.5-7B receiver-prior 诊断，`capability_claim_allowed=false`；它不能替代 ScreenSpot、TextVQA、DocVQA、OCRBench，也不能直接改写 DeepSeek 配方。该结果支持 paired supervision 强度能够修正局部图像归因，下一步先把 λ=0.5 轨迹接入统一 7B formal evaluator，检查 parser、blind/shuffled/random-projector 和自由生成方向；方向一致后才决定是否回到 Qwen2.5-3B 社区合同。

随后对同一 λ=0.5 checkpoint 做 8 条 ShowUI 的自由生成检查，固定社区 grounding prompt、贪心解码和 32 个新 token。vision、blind、shuffled、random-projector 的 parse rate 都为 `8/8`；到目标点的平均 L2 距离为 `491.73/514.31/493.97/499.97`，vision 相对 shuffled 的逐样本距离改善均值仅 `+2.24`。预测主要集中在窄坐标先验，teacher-forced 的正向 paired attribution 没有转化为可靠自由坐标 grounding。generic prompt 和 derangement 假设的两次实现失败均保留在实验索引与 failure artifact 中。

随后把 λ=0.5 checkpoint 接入 50 条 `screenspot_glm50_v1` 的 stripped ScreenSpot 诊断。固定 16 个 mean-pool token、scale=`0.1`、grounding prompt、贪心解码、四种条件和 2,000 次 bootstrap。四种条件 parse rate 均为 `50/50`；vision/blind/shuffled/random 的 click-in-box 均为 `10%`，`Accuracy@50/@100/@200` 均为 `2%/6%/18%`，中心距离均值为 `380.73/415.11/384.45/390.22`。vision 相对 blind 的中心距离差为 `-34.38`，CI `[-70.63,-3.30]`；vision 相对 shuffled 为 `-3.72`，CI `[-9.91,+1.54]`。视觉 token 能改变距离分布，正确图像却没有带来相对 shuffled 的 grounding 增益。Qwen7B 候选拒绝晋升，后续冻结训练量。

== Qwen3.5-9B receiver-prior diagnostic

The first 50-row stripped run is retained as a decoding-contract failure:
Qwen3.5's default reasoning template consumed the 32-token budget before a
click action, so every condition parsed 0/50. After adding
`enable_thinking=false`, an 8-row repair parsed 8/8 in every condition, but
click-in-box was 0%. Vision--blind center distance was -42.74 with CI
[-127.74,+13.77]; vision--shuffled was +88.69 with CI [+3.00,+199.15], where
positive means the correct-image prediction was farther from the target. This
receiver-prior diagnostic does not support an automatic capacity or
visual-pretraining rescue. The next variables are placeholder semantics,
projector scale, position encoding and receiver-distribution alignment; the
result remains outside formal Qwen and DeepSeek capability claims.

== DeepSeek Gate D local input-gradient preflight

The reference input-only DGRAD harness passed on the V100: a frozen ordinary
Linear produced finite, non-zero input gradients while weights stayed without
gradients. The candidate mathematical interface matched the reference and was
recorded as `hardware_pending`. No complete DeepSeek-V4-Flash-0731 weights,
real FP4/FP8 kernel or Hash-MoE routing was executed, so Gate D remains NO-GO.
The next local task is a placeholder/position/routing/save-resume verifier;
real quantized targets wait for explicitly authorized hardware.

== Tiny DeepSeek-V4 software loop

The real Transformers `DeepseekV4ForCausalLM` tiny implementation passed a
batch-2, 20-step projector-only loop on the V100. Projector gradients were
finite and non-zero, language gradients stayed `None`, greedy generation
returned shape [2, 8], and step-10 save/resume matched an uninterrupted run
with maximum absolute projector and loss deltas of 0.0. The first grouped
feature-shape failure was preserved before retry. This closes the tiny software
seam; complete 0731 weights and real FP4/FP8 input-DGRAD remain pending.

The same tiny loop also passed in bfloat16 on the V100, including batch 2,
20 steps, exact save/resume and generation. This covers the local BF16 seam;
the target 0731 FP4/FP8 kernels remain unverified.

== Overall decision and remaining gates (2026-08-07)

For a reader following the engineering goal, the result is currently a
reliable adapter prototype rather than a usable VLM. The Qwen2.5-3B V1/V2
screens, Qwen2.5-7B capacity control and Qwen3.5 receiver-prior diagnostic all
run through the same placeholder/projector interface. They show that images can
reach the receiver and change logits. The 7B full-public run has a weak
vision-minus-shuffled click interval, but vision remains below blind and the
complete causal contract fails; the other external-projector screens also fail
promotion. CE decreases and geometry can remain healthy while visual
attribution stays absent; these are recorded as mechanism evidence, not hidden
behind the final benchmark table.

Gate D therefore remains *NO-GO*. The tiny DeepSeek FP32/BF16 loop only closes
the software seam. Before a 0731 pilot, the full resolved weights and image-token
routing, real FP4/FP8 finite input gradients, full Hash-MoE image
forward/backward/generation, 20-step memory/stability, exact full-checkpoint
resume, and causal ScreenSpot/TextVQA/DocVQA/OCRBench gains must be recorded.
Local software validation is estimated at 1--2 working days; an explicitly
authorized paid-hardware pilot is roughly 2--5 working days, conditional on
kernel and weight availability.

== Exact DeepSeek placeholder-ID seam

The tiny software loop was rerun with the target `<｜image｜>` ID `129279`,
expanding the fixture vocabulary to `129280`. On the V100 in BF16, 20 batch-2
projector-only steps had finite non-zero projector gradients and no language
gradients; step-10 save/resume matched exactly and generation retained the two
expanded placeholder routing IDs. This closes the low-ID software seam concern.
The full 0731 vocabulary, 43-layer Hash-MoE, FP4/FP8 kernels and input-DGRAD
remain unverified, so this result does not change Gate D's *NO-GO* status.

== Qwen2.5-7B 完整公共 ScreenSpot

λ=`0.5` checkpoint 在完整 1,272 条公共 ScreenSpot 上完成 vision、blind、shuffled、random-projector 四条件生成，共 5,088 条输出，严格 parser 全部通过。固定配置为 scale=`0.1`、mean-pool 16 visual tokens、greedy decoding 与 32 new tokens。

#table(
  columns: 6,
  table.header([rows], [click-in-box], [Accuracy\@50], [Accuracy\@100], [Accuracy\@200], [中心距离均值]),
  [vision], [3.30%], [1.18%], [5.19%], [15.33%], [404.38],
  [blind], [3.46%], [1.02%], [5.03%], [15.09%], [409.71],
  [shuffled], [2.67%], [1.02%], [4.87%], [15.02%], [406.10],
  [random projector], [2.91%], [1.26%], [4.87%], [14.94%], [405.74],
)

vision 相对 shuffled 的 click-in-box 改善为 *+0.629 个百分点*；独立分层 verifier 的 2,000-bootstrap 95% CI 为 `[+0.157,+1.179]`。vision 相对 blind 为 `-0.157` 个百分点，CI `[-0.943,+0.629]`。这表明完整公共集上已经出现很弱的正确图像归因，同时文本/坐标先验仍然主导；Accuracy 与距离的关键 paired CI 没有共同通过。

分层结果也不均匀：iOS 的 vision-shuffled click 改善 `+1.96` 个百分点，CI `[+0.39,+3.92]`；Android 的 vision-blind click 为 `-1.62` 个百分点，CI `[-3.24,-0.40]`。社区参考中的 parse rate、Accuracy\@200 与 mean distance 表面达到或接近，Accuracy\@50/100 仍未达到，vision 也没有显著胜过 blind。因此 checkpoint 继续拒绝晋升，`capability_claim_allowed=false`，不能声称达到社区 GLM-5.2V metric-aligned baseline。

这轮支持“7B 加 paired supervision 已产生少量 correct-image click attribution”，反驳“teacher-forced 正 margin 已经等价于可用 grounding”。完整原始 summary 和 rows 保存在 V100 数据盘，Git 提交分类 summary、SHA pointer 与 verifier。receiver、V1/V2、token 数、CE/attribution 分离和失败分层集中维护于 `docs/experiment-mechanism-findings.md`。

随后完成的 240-token matched screen 没有改善 vision-shuffled click，也没有解决 blind 竞争，因此 token-count 扩展已经停止。当前下一项是共享正式训练入口与 7B 100-step causal screen。

== DeepSeek 真实训练时间与剩余 Gate

当前本地软件链路已经覆盖真实 MoonViT-V2、canonical 4096 projector、目标 placeholder ID `129279`、tiny DeepSeek FP32/BF16 20-step forward/backward、冻结主干、精确 save/resume 和 generation。候选比较、240-token 对照、独立 verifier 和运行入口审计均已完成；剩余本地工作是把正式训练安全合同推广到共享 7B/DeepSeek 入口并运行 100-step causal screen。

完整 0731 pilot 仍需：resolved 权重加载与 SHA 固定；真实 FP4/FP8 kernel finite input DGRAD；43 层 Hash-MoE 图像 forward/backward 和 routing 一致性；目标 batch、activation checkpointing、显存和吞吐；20-step 稳定 checkpoint 与精确恢复；同一 ScreenSpot/TextVQA/DocVQA/OCRBench 合同。获得付费硬件明确授权后，最小 Gate D 预计 1--2 个工作日，首个真实小规模训练和固定 benchmark 再需约 2--3 个工作日。权重与 kernel 路径顺利时，授权后 *3--5 个工作日*可以得到首轮真实训练判断。

Gate D 继续为 *NO-GO*，任何租卡或完整 0731 下载等待用户明确授权。

== 7B token-count matched screen

同一个 λ=`0.5`、scale=`0.1`、Qwen2.5-7B checkpoint 在冻结的 50 条 GLM-format subset 上改用 240-token full sequence；其余图像、prompt、parser、生成和 bootstrap 配置不变。四条件 parse 均为 `50/50`，click-in-box 为 `10%/10%/10%/8%`，Accuracy\@50 为 `0%/2%/2%/0%`，Accuracy\@100 为 `6%/6%/6%/6%`，Accuracy\@200 为 `18%/18%/20%/18%`。

独立 category verifier 重算中心距离均值为 `399.51/415.11/396.78/397.02`。vision-blind 距离改善 `+15.59`，CI `[-13.51,+47.15]`；vision-shuffled 为 `-2.74`，CI `[-13.58,+9.96]`。两个 click paired 差都为 `0`，CI `[-6,+6]` 个百分点。240-token full sequence 没有改善正确图像归因，因此停止扩大 token 数量筛选，下一项转向一个 projector/辅助目标变量并保留 matched CE-only control。

原始 evaluator 的 `center_distance` 统计完整存在；旧摘要消费者读取了不兼容的 key/schema，本节同时用独立 verifier 做分类与交叉核对，边界已在 raw pointer 中记录。运行 wall time 为 `1,310.98` 秒，其中含首次 Transformers import 和 339 个权重分片的 CPU 加载；后续真实批处理优化必须把这段冷启动与 GPU throughput 分开。

== Package 15R gated residual repair

15R 的父合同保持冻结。三类失败被分开保存：第一次是源码 SHA 漂移；第二次是在严格冻结源码上，gate=0 时 `residual.weight` 的链式法则零梯度被旧 guard 误报；第三次是修复 runner 首次混入冻结 worktree 后的 runner SHA 不匹配。修复 contract 只放宽 `residual.weight` 在精确 zero gate 的首步零梯度，gate 本身及其他 projector 参数仍需 finite/non-zero，并由真实 backward 单测覆盖。

在 clean main 上用同一 4,000-row order、MoonViT cache、Qwen2.5-3B receiver、canonical step0、100-step budget 和 health contract 重跑 matched control 与 gated arm。两者均由独立 verifier 标记 `verified`，均在 step 2 自动止损，collapse onset 为 `[1,2]`。control 的 peak GPU 为 `13,131,489,928` bytes，gated 为 `13,467,034,268` bytes；critical reason 都是 `projector_rms_rising_spread_falling` 与 `receiver_rms_rising_spread_falling`。

gated 的 step 1 只允许 `residual.weight` 零梯度，gate 和其他参数通过 finite/non-zero 检查；step 2 `vision_minus_shuffle_correct_logp=+0.1071`，但 projector/receiver relative-spread ratio 已降至 `0.2690/0.2254`，effective-rank ratio 为 `0.5008/0.3611`，仍触发几何守卫。结果支持“修复训练守卫后仍存在共同的 receiver-facing collapse”，反驳“zero-gated residual 能自动保住视觉几何”。因此 15R 不进入 500-step、ScreenSpot 或 DeepSeek 候选，raw manifest 与四个目录的 SHA 指针见 `REPAIR_RESULT_POINTER_20260807.json`；`previous_best` 不变，Gate D 仍为 *NO-GO*。

下一项只注册一个可迁移的 projector 输出尺度或辅助目标变量，并保留完全匹配的 CE-only control。若该变量仍在 step 1--2 触发相同 guard，停止继续堆训练量，转 projector 结构重设计。

== Package 15T：exact K3 causal margin lambda=0.5

正式 Qwen3B supervision 在本轮前完成审计：4,000-row frozen order 按 ID、source row 和 SHA 重建真实 `train_mix.jsonl`，包含 2,000 条 grounding 与 2,000 条 short-answer；grounding 有 1,066 个不同坐标、`click(start_box=[500,500])` 为零，循环 batch 负例没有同图 SHA。上游保存的是 ShowUI point 转成的 click 字符串，当前 pack 没有独立 bbox/raw point 字段，因此只能称 point-derived click supervision；旧 cache-only stripped-receiver 的统一 500,500 fallback 只作为历史诊断。审计 pointer 为 `SUPERVISION_PROVENANCE_AUDIT_20260807.json`。

在 exact K3/MoonViT-V2 projector 上固定 geometry-safe LR `5e-5`、step0、order、cache、receiver 和 health schedule，只把 paired correct-vs-shuffled hinge lambda 从 `0.1` 提到 `0.5`。独立 verifier 重新计算 3 个 probes 与 3 个 checkpoints；step 2 自动停止，onset `[1,2]`。projector/receiver relative-spread ratio 为 `0.9903/0.9859`，effective-rank ratio 为 `1.0002/1.0001`，表示几何保持；`vision_minus_shuffle_correct_logp` 仅从 `-0.2404` 到 `-0.0515`，vision/shuffled preference 最终 `0.625/0.625`，仍触发 causal critical。CE 为 `4.8526→5.6925`，不进入 ScreenSpot 或通用能力评测；raw pointer 为 `CAUSAL_MARGIN05_RAW_POINTER_20260807.json`。

结果支持“paired 目标能把错误方向推近零”，反驳“增大 lambda 即可让冻结纯文本 3B 获得真实 grounding”。15R 与 15T 合起来说明几何健康和视觉因果是两道独立门：gated residual 没保住几何，强 paired objective 保住几何却仍没让正确图像胜过 shuffled。停止继续扫 lambda 或做 500-step 扩展；下一项只选一个 DeepSeek 可迁移的 placeholder/位置语义或 receiver 分布对齐变量，并保留 matched CE-only control。

== DeepSeek image interface screen：软件接缝通过，Gate D 仍未通过

第一版 screen 被保留为实现失败：tiny Transformers DeepSeek-V4 的 `tid2eid` 默认表全为零，routing-ID 消融没有改变 logits；首版 runner 也没有把 routing/position 因果差异纳入总通过条件。原始摘要 SHA 为 `ca975531...8eddf8`，目录和排除理由见 `deepseek_interface_screen_v2_pointer.json`。

修复后的 v2 使用预注册的冻结非退化 tiny hash 表 `tid2eid[token,k]=(token+k) mod num_experts`，真实目标 placeholder `129279`、canonical 4096 projector 边界和冻结 DeepSeek receiver 不变。V100 上所有接口检查通过：5 个 raw token 展开为 7 个 active token；3 个视觉 label 为 `-100`；routing ID 在视觉 span 重复；position ID 为连续 `0..6`；projector gradient finite/non-zero，语言主干无梯度。保持 embedding 不变时替换 routing ID，logit 最大变化 `0.0015277863`；保持 embedding 和 routing ID 不变时替换 position ID，logit 最大变化 `0.0193590522`；loss `11.75953`，projector grad norm `0.22719674`。原始结果指针为 `deepseek_interface_screen_v2_pointer.json`。

这个结果确认当前 wrapper 的 placeholder、position、routing 和 loss-mask 接缝能被真实 Transformers tiny DeepSeek 实现消费。tiny synthetic route table 只用于验证旁路确实连通，不能代替 0731 的真实 `tid2eid`、完整 43 层 Hash-MoE、FP4/FP8 input-DGRAD、显存吞吐、恢复和能力 benchmark；Gate D 继续为 *NO-GO*。

本地还需约 `1--2` 个工作日完成候选冻结、独立 verifier 和报告整理；明确授权付费硬件后，权重/kernel Gate D 约 `1--2` 个工作日，首轮固定合同训练与评测约 `2--3` 个工作日，前提是权重和 kernel 可用。

== 2026-08-08：Qwen2.5-7B V1 family proxy matched screen

预注册 screen 固定同一个 Qwen2.5-7B-Instruct receiver、32 条真实答案、同一循环 derangement、mean-pool 16、projector scale `0.1`、BF16、LR `5e-5`、3 steps、同一 receiver adapter 和 λ=`0/0.5` 两臂，只把视觉塔和 projector 换成 pinned `moonshotai/MoonViT-SO-400M` family proxy（revision `a889d399...d3e5007`）。

两臂训练都 finite 并完成 4 个 health points。V1 CE-only 的 `vision-minus-shuffle` 为 `-0.03749→+0.00615`；λ=`0.5` 为 `-0.03749→+0.01145`。32 条 teacher-forced probe 的 2,000-bootstrap 结果如下：

#table(
  columns: 5,
  table.header([臂], [vision-shuffle], [95% CI], [vision-blind], [95% CI]),
  [V1 CE-only], [+0.00615], [-0.01760,+0.03182], [+0.83930], [+0.58501,+1.11431],
  [V1 lambda=0.5], [+0.01145], [-0.02580,+0.04766], [+0.52705], [+0.39629,+0.67691],
)

V1 λ=`0.5` 相对 V1 CE-only 的 paired 差为 `+0.00530`，CI `[-0.04882,+0.05758]`；相对匹配的 V2 λ=`0.5` 为 `-0.47600`，CI `[-0.87349,-0.13102]`。因此 V1 能让 receiver 响应视觉 token，也明显优于 random projector，但正确图和 shuffled 图仍无法区分；V1 的弱归因显著低于 V2 λ=`0.5`。这条结果把“V2 embedding 压缩是主要失败根因”降为低优先级，也不支持把 V1 family proxy 替换进正式配置。没有 ScreenSpot 晋升资格，Gate D 继续为 *NO-GO*。

完整 raw pointer、compact summaries、health/probe/bootstrap 和独立 verifier 位于 `experiments/qwen3b_community_eval_20260805/capacity_controls/qwen25_7b_v1_community_screen_20260808_POINTER.json` 及同名目录；V1 大 checkpoint 与 optimizer 仍保留在 V100 数据盘。
== 2026-08-08：Qwen3.5-4B external MoonViT matched ablation

为了把视觉预训练 receiver 的作用放进同一张表，本轮固定 Qwen3.5-4B revision
`851bf6e...b7c8d0a`，绕过原生 visual/merger，使用 Kimi-K3/MoonViT-V2、exact K3
projector 和固定 4096→2560 receiver adapter，只训练 projector。训练为 32 条真实答案、
mean-pool 16 tokens、scale `0.1`、BF16、AdamW `5e-5`、3 steps；评测为固定
`screenspot_glm50_v1` 50 条、四条件和 2,000 bootstrap。合同与 pointer 分别为
`configs/qwen35-4b-external-moonvit-ablation-v1.json` 和
`qwen35-4b-external-moonvit-ablation-pointer-20260808.json`。

#table(
  columns: 9,
  table.header([projector], [vision click], [blind], [shuffled], [random], [V A50], [V A100], [V A200], [V mean dist]),
  [step0], [2%], [2%], [2%], [4%], [0%], [6%], [24%], [469.54],
  [CE-only], [0%], [2%], [4%], [4%], [0%], [8%], [22%], [470.49],
  [paired margin λ=0.5], [4%], [2%], [4%], [4%], [2%], [10%], [20%], [484.96],
)

V−blind click 95% CIs for step0, CE-only and paired-margin were `[-6,+6]`, `[-6,0]`
and `[-4,+10]` percentage points; V−shuffled CIs were `[0,0]`, `[-10,0]` and `[0,0]`.
The margin arm moved teacher-forced V−shuffle from `-0.1361` to `+0.0162`, while CE-only
reduced CE `8.3974→7.5631`; free generation still had no reliable correct-image grounding.
All three arms remain diagnostic-only and do not alter the Qwen previous-best or DeepSeek
candidate list. Since the 50-row causal gate failed, the 1,272-row expansion was not run.

== DeepSeek residual multimodal-interface hypothesis

The public V4 Flash tokenizer retains `<｜image｜>` ID `129279`, `<｜image2｜>`,
`<｜rl_image_start｜>`, `<｜rl_image_pad｜>`, 415 `place_holder_mm_span` entries and
box/point/ref/polygon markup. The public config has hidden size 4096 but no `vision_config`,
and its HF file tree contains no visual tower/projector. This is consistent with a removed or
internal multimodal seam, but it cannot prove that the released weights learned visual features.
Gate D therefore adds a real-weight step0 receiver-prior table before any projector training;
the same ScreenSpot parser and paired CIs will be used for that table and for trained checkpoints.
#heading[DeepSeek-V4-Flash 0731 权重侧多模态接口审计]

公开 0731 revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` 保留了 415 个 multimodal span placeholder 以及 image/region 标记。对 BF16 `embed.weight` 的 range 抽样显示，预留行平均范数为 `0.3841`，普通 token 抽样为 `5.6357`，比值 `0.0682`；预留行对普通 token 均值的平均 cosine 为 `-0.00026`。这符合运行时将视觉向量写入预留槽位的设计，支持优先进行真实 DeepSeek receiver-prior gate，但不能证明公开权重已经具备视觉回路。完整 forward/backward、vision/blind/shuffle 归因和 checkpoint 恢复仍未通过，Gate D 保持 NO-GO。

公开 `inference/model.py` 的 forward 入口只接受 `input_ids`，没有公开 image-to-embedding 注入；配置也没有 `vision_config`。因此发布物可复现的是文本主干和多模态占位接口，历史私有视觉训练仍需真实权重实验验证。

#heading[Qwen3.5-4B MoonViT V1/V2 回归]

执行审计发现首次 V1 runner 只使用索引 `0–7`，而冻结合同和 V2 reference 使用 `0–31`；旧结果降级为 8-row pilot。修复合同预先提交后，V1 CE-only 与 paired-margin 用全部 32 条固定记录重跑。full32 CE-only 的 vision/blind/shuffled click-in-box 为 `2%/2%/2%`；paired-margin 为 `2%/2%/0%`。paired-margin 的 vision-minus-blind click CI 为 `[-6,+6]` 个百分点，vision-minus-shuffled 为 `[0,+6]` 个百分点，严格门槛仍未通过。匹配的 full32 V2 paired-margin reference 为 `4%/2%/4%`，也没有正的 shuffle 因果下界。V1 没有救活 external MoonViT，版本差异不再是首要故障解释。
#heading[固定 baseline matrix 与下一步]

`regression_baseline_matrix_v1.json` binds the comparable rows to one receiver/tower/evaluation contract. Qwen3.5-4B external V1 and V2 both fail the causal ScreenSpot50 gate: neither has a positive lower confidence bound for both vision-minus-blind and vision-minus-shuffled click-in-box. Qwen2.5-7B exact V2 has only a weak shuffle attribution and fails the blind comparison. The old Qwen2.5-3B full-public row is marked as a legacy V2 proxy, not exact K3 V2; the native Qwen3.5 VLM is a separate positive control.

The version-only explanation is therefore rejected. The next local experiment is one DeepSeek-transferable interface/scale/target-alignment variable with a matched CE-only control, screened on the frozen 50-row set before any full-public expansion or long training. DeepSeek-V4-Flash-0731 Gate D remains NO-GO until real weights, FP4/FP8 input gradients, full routing, checkpoint round-trip and causal gains are verified.
#heading[Qwen3.5-4B V1 projector scale 0.03 screen]

The preregistered single-variable screen changed only the projector runtime scale from `0.1` to `0.03`. Training stayed finite for three projector-only steps and the final teacher-forced vision-minus-shuffle was `+0.0306`, but ScreenSpot50 vision and shuffled generations were both unparseable (`0%` parse) while blind remained at `100%`. Click-in-box was `0%/2%/0%` for vision/blind/shuffled; paired CIs were `[-6,0]` and `[0,0]` percentage points. The arm is rejected for format collapse, not promoted, and is not expanded to the full public set.

This supports treating projector output scale as a receiver-interface constraint and rejects the idea that simply shrinking the scale restores grounding. Keep scale `0.1` as the matched control; test placeholder/position or loss-mask semantics next. Gate D remains NO-GO.

#heading[Qwen2.5-7B V2：正式生成评测与经验（2026-08-08）]

Qwen2.5-7B-Instruct + MoonViT-V2 projector-only 训练完成 900 optimizer steps、57,600
examples seen，health 全程 finite；但同一 checkpoint 的固定 `screenspot_glm50_v1`
50 条四条件自由生成没有建立视觉 grounding：

#table(
  columns: 5,
  table.header([条件], [parse rate], [click-in-box], [A@50], [A@100/A@200]),
  [vision], [6%], [4%], [0%], [2% / 2%],
  [blind], [100%], [10%], [2%], [6% / 18%],
  [shuffled], [6%], [0%], [0%], [0% / 0%],
  [random projector], [96%], [10%], [2%], [10% / 14%],
  [step0], [94%], [8%], [2%], [6% / 16%],
)

2,000-bootstrap 的 click-in-box paired CI 为 vision−blind `[-16,+2]` 个百分点，
vision−shuffled `[0,+10]`，trained−random projector `[-16,+2]`。所以“正确图片比 blind
和 shuffled 更好”两个条件都没有同时成立；训练后还出现了严重的生成格式退化。旧的
teacher-forced shuffle delta `+2.2324` 只能说明答案概率受到图像条件影响，不能替代
自由生成的正确点击。这组结果记为 `valid_result_negative`，不续训同一 V2 arm，不进入
previous-best 或 DeepSeek 候选。

本轮也固定记录了三条工程经验：cache feature、candidate projector 和 random projector
必须在同一 dtype 边界；变量初始化顺序错误应作为独立工程失败保存；只有修复后的 retry
才能进入正式 scorer。今后的能力表必须同时带上 preflight、四条件逐样本 raw rows、
step0/previous-best/current-candidate、paired bootstrap 以及 ScreenSpot/TextVQA/DocVQA/
OCRBench/language-retention 的节点曲线。健康指标、loss 或 teacher-forced attribution
都不能单独升级为视觉能力声明。
同一最终 checkpoint 的多任务 selection（TextVQA、DocVQA、OCRBench 各 8 条）也已完成。TextVQA soft VQA 为 vision/blind/shuffled/random `0.125/0/0.125/0`，DocVQA ANLS 为 `0.12/0/0.12/0`，OCRBench exact match 四条件全为 `0`。vision 与 shuffled 打平，故不能将非零分数解释为正确图片 grounding；原始报告、CSV、SVG 和 SHA pointer 为 `qwen25_7b_v2_multitask_final_limit8_POINTER.json`。
无视觉 control 已登记为 parse `100%`、click-in-box `10%`、A@50/@100/@200 `2/6/18%`；原生 Qwen VLM 阳性对照为 parse `80%`、click `42%`，blind click `6%`。两条 control 只用于区分语言先验和原生视觉上界，不得写入 external MoonViT projector 排名。
