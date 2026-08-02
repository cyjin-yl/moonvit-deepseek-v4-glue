#set page(paper: "a4", margin: 24mm)
#set text(font: ("Noto Sans CJK SC", "Microsoft YaHei", "Arial"), size: 10pt)
#set heading(numbering: "1.1")
#set par(justify: true, leading: 0.7em)
#set table(stroke: 0.5pt + rgb("c8c8c8"), inset: 6pt)

#align(center)[
  #text(size: 21pt, weight: "bold")[MoonViT 接入 DeepSeek-V4-Flash-0731]
  #v(5pt)
  #text(size: 14pt)[训练前架构审计、胶水原型与硬件计划]
  #v(8pt)
  版本 0.1 · 2026-08-02
]

#outline()
#pagebreak()

= 执行摘要

本项目目标是给纯文本的 DeepSeek-V4-Flash-0731 接入 MoonViT 视觉编码器。第一阶段不训练视觉塔和语言模型，只训练一个 Kimi 风格 PatchMerger projector。当前结论是：架构合同成立，普通 causal LM 和 Transformers 的真实 DeepSeek-V4 Hash-MoE 缩小模型都已完成 loss/backward；正式 0731 大权重的 FP4/FP8 可微 kernel 是尚未消除的主要风险。

MoonViT 约 400M 参数，BF16 权重文件约 834 MB，相对于约 160 GB 级的 DeepSeek 混合精度权重很小。更大的资源变量是图像分辨率带来的视觉 token 数和冻结 LLM 反向所保留的激活，而不是视觉塔权重。

= 来源与可复现边界

- 社区 projector checkpoint：#link("https://huggingface.co/0xSero/glm-local-vision-checkpoint")[0xSero/glm-local-vision-checkpoint]。它证明了 projector-only 路线能把视觉信号接到 GLM-5.2，但公开 grounding 指标仍弱，不能等同于成熟 VLM。
- 社区复现记录：#link("https://huggingface.co/blog/0xSero/glm52-vision-on-4-gpus")[Giving a 753B Model Eyes]。
- DeepSeek 权重：#link("https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731")[deepseek-ai/DeepSeek-V4-Flash-0731]，MIT。
- 独立视觉塔：#link("https://huggingface.co/moonshotai/MoonViT-SO-400M")[moonshotai/MoonViT-SO-400M]，MIT。
- Kimi-K2.5 技术报告：#link("https://arxiv.org/abs/2602.02276")[Kimi K2.5: Visual Agentic Intelligence]。
- Vast offer 搜索接口：#link("https://docs.vast.ai/api-reference/search/search-offers")[Vast.ai Search Offers API]。本文只调用搜索接口，不调用创建实例接口。

独立 MoonViT-SO-400M 来自 Kimi-VL；Kimi-K2.5/K2.6 使用演进后的 MoonViT-3D。两者输出形状兼容不代表特征分布相同。视觉塔 revision 是训练 provenance 的组成部分；更换视觉塔必须重训 projector。

= 权重与张量合同

#table(
  columns: (1.4fr, 2.4fr, 1fr),
  [*组件*], [*合同*], [*状态*],
  [MoonViT], [`[tokens, 4, 1152]`], [冻结],
  [PatchMerger], [`LN(1152) → flatten → 4608 → GELU → 4096`], [训练],
  [DeepSeek embedding], [`vocab → 4096`], [冻结],
  [Placeholder], [`<｜image｜>` / ID 129279], [现有词表],
  [Hash-MoE routing], [扩展后的 placeholder IDs], [冻结],
)

Projector 精确参数量是 40,119,040。保存格式由 `projector_config.json` 与 `projector.safetensors` 组成，加载时使用 strict state-dict 校验。MoonViT、DeepSeek 与 projector 始终保留为三个可独立哈希和升级的来源，不把大权重复制到自有仓库。

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
  [MoonViT 输出 shape/freeze 合同], [通过], [视觉边界],
  [真实 MoonViT-SO-400M + 小 LM 前向/反向], [通过], [V100 CUDA fp32],
  [eval\_vlm 生成/blind/shuffle-loss 干跑], [通过], [评测管线端到端],
  [generate()：generic 与 deepseek_v4 两种路径], [通过], [评测/推理前置],
  [指标库（VQA/ANLS/token-F1/grounding）], [通过], [纯 Python，无 torch],
  [完整测试集], [26/26], [Linux + torch 2.10.0+cu128],
)

V100 实测：MoonViT-SO-400M 在 448px 输入下输出 `[192,4,1152]`，原生 640×480 下 `[1064,4,1152]`，符合合同；loss/backward 正常。评测管线用随机 projector + tiny-gpt2 干跑：生成、评分、blind 基线与 shuffle-loss 全部端到端通过；shuffle-loss 在未训练 projector 上给出 `mean_delta = 0.0` 的正确零结果——训练后该值应当变正，这是对齐信号最便宜的读数。

离线 smoke 结果：输入 6 token 扩展为 8 token；projector 六组参数均获得梯度；语言模型参数梯度数为 0。同一结果在 doesworkstation（V100）上复现。

版本审计发现，公开 PyPI Transformers 4.57.6 不含 `deepseek_v4` 模块；真实 DeepSeek 类测试使用 Transformers 5.14.1。因此统一环境暂定 `transformers>=5.12,<6`，不能盲从 checkpoint config 中的历史版本字符串。

= MoonViT 尺寸与适配性

MoonViT 本身适合该任务：27 层、hidden size 1152、16 heads、约 400M 参数，原生分辨率和 NaViT packing 对 GUI、文档、截图有价值。它不会显著改变完整 0731 的权重门槛。

风险来自 token 数。patch size 为 14，随后 2×2 合并。392×392 图像产生约 196 个合并 token；896×896 产生约 1024 个。第一阶段应把每图上限锁在 1024；更高分辨率必须以梯度激活显存实测决定。

= 本地与云端硬件计划

#table(
  columns: (1.5fr, 1.7fr, 2.2fr),
  [*目标*], [*建议硬件*], [*判断*],
  [胶水/单元测试], [CPU 或任意 8 GB GPU], [已可完成],
  [MoonViT + 0.5B–3B LM], [12–24 GB GPU], [适合本地开发],
  [V100 32 GB], [SM70、32 GB], [可跑 MoonViT/小 LM；不能装完整 0731],
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

第二个 backbone 复测（同一数据与超参，placeholder 自动解析为 `<|image_pad|>`）：Qwen2.5-0.5B-Instruct 1000 步后训练 loss 3.898 → 3.160，评测真图 3.310 vs 打乱图 3.592，*shuffle\_delta = +0.282*，耗时 1194 秒，checkpoint 在 `checkpoints/overfit-qwen05-1k`。两条 backbone 轨给出同量级正 delta，说明该信号来自胶水层与 projector 通路本身，与文本主干选型无关。下一步在更大更干净的 flickr8k（1100 条）上复测同一判据。

工作站实测注意项：

- NVML 版本不匹配（内核模块 580.159.04 vs 用户态库 580.173）：`nvidia-smi` 不可用，但 CUDA 初始化与 kernel 运行正常。不要为修复它而重载驱动或重启——同机其他任务正在使用 GPU。
- MoonViT remote code 与 Transformers 5.x 有两处不兼容：缺 `all_tied_weights_keys`（已在 `moonvit.py` shim，ViT 无 tied weights，空映射语义正确）；bf16 下 remote code 内部存在 float32/bf16 混用的 layer\_norm 调用，V100 上 MoonViT 以 fp32 运行（约 1.6 GB，可接受）。
- 小上下文文本模型装不下原生分辨率视觉 token：640×480 照片产生 1064 个合并 token，加 prompt 后超过 tiny-gpt2 的 1024 positions，表现为 scatter-gather device assert（异步报错会漂移到无关位置，需 CUDA\_LAUNCH\_BLOCKING=1 定位）。`--max-image-side` 控制 token 数；正式 0731 的 128k 上下文无此问题。
- HF 下载须走本机代理 `127.0.0.1:7890`（约 0.4–0.5 MB/s），且该仓库默认 Xet 传输在代理下会挂起，必须设 `HF_HUB_DISABLE_XET=1`。
- 复用现有 venv（torch 2.10.0+cu128 + transformers 5.12.1）；pytest 用 `pip --target` 装在独立目录，不改对方环境。
- 工作站负载高（其他 agent 编译/跑 inference server）时，机械盘上的 torch import 可达 90 秒以上；批处理脚本超时预算要放宽。

== Gate C：Vast 只读调研

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

== Gate D：正式租卡前

1. 原生 0731 权重加载成功。
2. 单图短序列 forward。
3. 单 batch backward，projector 梯度有限且非零。
4. LLM/MoonViT 无梯度。
5. 20 step 无 OOM/NaN。
6. 新进程恢复 projector 后输出一致。

= 评测与验收计划

没有 benchmark 就无法回答“接上了没有”。评测口径沿用社区 GLM-5.2 视觉实验（坐标格式解析率、归一化 0–999 坐标的 Accuracy\@50、平均点击误差），并补充常规 VQA/OCR 指标。所有数字必须与 blind baseline（同一模型、无图输入）一起报告：VQA 类基准有显著语言先验，无图基线把“模型本来就会答”与“图像带来了信息”分开。

已实现的评测资产：

- `moonvit_glue.metrics`：纯 Python 指标，无 torch 依赖。exact match、soft VQA（官方 min(1, 同意人数/3)）、ANLS、token-F1，以及 grounding 的 parse/Acc\@threshold/mean error。
- `tools/eval_vlm.py`：生成式评分（`--blind` 输出无图基线）与 `--shuffle-loss`（真图 vs 随机图的 teacher-forced loss 差）两种模式。shuffle-loss 是训练前最便宜的信号检验：projector 学到东西后，真图 loss 应显著低于随机图。
- `tools/fetch_eval_data.py`：固定来源拉取 TextVQA（soft VQA）、DocVQA（ANLS）、OCRBench（exact match）、ScreenSpot（in-box grounding），落盘 JSONL 与 MANIFEST.json（resolved revision sha 与 JSONL sha256），沿用“信任 manifest 而不是 tag”的纪律。

#table(
  columns: (1.2fr, 2.6fr, 2fr),
  [*阶段*], [*运行内容*], [*通过判据*],
  [Gate A], [指标单元测试；小模型 shuffle-loss 管线], [测试全绿；管线可跑],
  [Gate B], [真实 MoonViT + 小 LM：生成、评分、blind 对照；overfit 训练], [端到端无报错；shuffle\_delta = +0.343（显著为正）],
  [Gate C], [训练前/后各一次：TextVQA、DocVQA、OCRBench、ScreenSpot 小子集], [训练后显著高于训练前与 blind],
)

预期锚点：0xSero 的 GLM-5.2 projector-only checkpoint 报告坐标格式解析率 92%、Accuracy\@50 4.3%、平均点击误差约 564/999。第一阶段的成功定义是 DeepSeek 路径达到同量级信号，而不是成熟 VLM 水平；grounding 在 projector-only 阶段大概率仍很弱。

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
)

= 下一位执行者的最短路径

先运行 `pytest` 和 `examples/smoke_tiny_text_lm.py`。然后在 V100 工作站使用机械盘作为 `HF_HOME`，验证真实 MoonViT。正式 0731 实验前不要写训练循环，先证明目标 CUDA/量化 runtime 支持 input data-gradient。若失败，优先评估 FP8 可微加载或定制 Dgrad，不要改用 GGUF 假装完成训练链路。
