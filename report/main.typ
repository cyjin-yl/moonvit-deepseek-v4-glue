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
  版本 0.2 · 2026-08-03
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
  [MoonViT-V2], [`[tokens, 4, 1024]`], [冻结],
  [PatchMerger], [`LN(1024) → flatten → 4096 → GELU → 4096`], [训练],
  [DeepSeek embedding], [`vocab → 4096`], [冻结],
  [Placeholder], [`<｜image｜>` / ID 129279], [现有词表],
  [Hash-MoE routing], [扩展后的 placeholder IDs], [冻结],
)

V2  projector 精确参数量是 33,564,672（fp32 约 134 MB，bf16 约 67 MB；配置见 `configs/deepseek-v4-flash-0731-projector-moonvit-v2.json`）。备选的 V1 塔（MoonViT-SO-400M，1152 维）projector 为 40,119,040 参数，两条配置不得混用。保存格式由 `projector_config.json` 与 `projector.safetensors` 组成，加载时使用 strict state-dict 校验。MoonViT、DeepSeek 与 projector 始终保留为三个可独立哈希和升级的来源，不把大权重复制到自有仓库。

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
  [完整测试集], [34/34], [Linux + torch 2.10.0+cu128],
)

V100 实测：MoonViT-SO-400M 在 448px 输入下输出 `[192,4,1152]`，原生 640×480 下 `[1064,4,1152]`，符合合同；loss/backward 正常。评测管线用随机 projector + tiny-gpt2 干跑：生成、评分、blind 基线与 shuffle-loss 全部端到端通过；shuffle-loss 在未训练 projector 上给出 `mean_delta = 0.0` 的正确零结果——训练后该值应当变正，这是对齐信号最便宜的读数。

离线 smoke 结果：输入 6 token 扩展为 8 token；projector 六组参数均获得梯度；语言模型参数梯度数为 0。同一结果在 doesworkstation（V100）上复现。

版本审计发现，公开 PyPI Transformers 4.57.6 不含 `deepseek_v4` 模块；真实 DeepSeek 类测试使用 Transformers 5.14.1。因此统一环境暂定 `transformers>=5.12,<6`，不能盲从 checkpoint config 中的历史版本字符串。

= MoonViT 尺寸与适配性

MoonViT 本身适合该任务：27 层、hidden size 1152、16 heads、约 400M 参数，原生分辨率和 NaViT packing 对 GUI、文档、截图有价值。它不会显著改变完整 0731 的权重门槛。

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

当晚第二次查询（增加 RAM ≥500 GB、盘 ≥1.5 TB 过滤，33 个 offer）要点：4×A100 SXM4 降至 \$6.41/h（961 GB RAM、2 TB 盘，Gate D 首选）；4×H100 PCIe \$6.93/h（6392 GB 盘，性价比突出）；8×A100 SXM4 \$10.30/h（11 TB 盘，情景 B 兜底）；4×B200 \$21.25/h。新出现的 4×RTX PRO 6000 96 GB（Blackwell，原生 FP4）\$4.54/h 但总线无 NVLink 且 FP4 kernel 支持未验证，仅作探索。

*架构修正*：A100 为 Ampere，无 FP8/FP4 Tensor Core，0731 的 NVFP4 权重在 Ampere 上没有确认的内核路径，只能解量化至 bf16（568 GB），而 4×A100 的 320 GB 装不下——4×A100 实际上无法加载原生权重，Gate D 首选改为 *4×H100 PCIe*（Hopper FP8 + FP4 专用内核）。8×A100 SXM4（640 GB）仅作为解量化情景 B 的兜底。

=== 最终账单（含下载时间、网络、存储、装机余量）

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

带宽与耗时已按报价接口的 inet\_down 字段核算：H100 PCIe 候选实测下行 2217 Mbps（约 277 MB/s），理论 160 GB 约 10 分钟；考虑 HF CDN 单连接限速（40–100 MB/s）与断流重传（本地实测 802 MB 在慢代理下耗时 2h46m），基准按 1 h、悲观按 2 h 计，租机筛选条件要求 inet\_down ≥ 1000 Mbps 并用 aria2/hf\_transfer 多连接下载。checkpoint 上行流量每个约 300 MB（projector fp32+bf16+优化器），每 500 步一次，分钟级完成，不影响训练计费。训练日志（train.log）与 history.json 随 checkpoint 一并上传，社区可见完整训练过程。

情景 A′（R3 命中，H100 无法跑 FP4 前向）：转 4×B200（\$21.25/h），同排程基准 10 h ≈ \$213；B200 有原生 FP4 Tensor Core，Dgrad 通过率也更高。决策成本为已耗的装机+下载 1.5–2 h（约 \$10–14）。情景 B（FP4 不可反传且 B200 不可用）：8×A100 SXM4 \$10.30/h，权重解量化至 bf16（568 GB）转换 1–2 h 且训练约慢 3 倍，合计 20–30 h ≈ \$210–310（该候选流量 \$1.33/TB，网络几乎免费）。*建议总预算：\$120 起步（情景 A），预留 \$220（情景 A′/B），上限 \$350；预期实际花费 \$75–110。* 决策点在租后第 2–3 小时：Gate D 不通过即 destroy 止损，损失约 \$20。

== 训练显存算术（每卡，权重张量并行切分，视觉塔/projector 每卡复制）

冻结 LLM 与冻结视觉塔意味着：LLM 侧 *没有优化器状态、没有参数梯度*——AdamW 只挂在 projector 上（`train_overfit.py` 里 `AdamW(projector.parameters())`,LLM 与视觉塔 `requires_grad_(False)`，视觉塔前向在 `no_grad` 里，梯度只需以激活形式穿过 LLM 回到 projector，不需要对视觉塔反传）。优化器全套（权重 fp32 + m/v + 梯度）只有约 0.55 GB。真正的变量是 LLM 权重本体和 Dgrad 路径的激活。

#table(
  columns: (2.4fr, 1.9fr, 1.9fr),
  [*组成（每卡）*], [*情景 A：4×H100，FP4 权重 TP=4*], [*情景 B：8×A100，bf16 权重 TP=8*],
  [LLM 权重], [160/4 = 40 GB], [568/8 = 71 GB],
  [LLM optimizer + 参数梯度], [0（冻结）], [0（冻结）],
  [MoonViT-V2 权重（bf16/fp32）], [0.8–1.6 GB], [0.8–1.6 GB],
  [projector 权重 + AdamW + 梯度], [约 0.55 GB], [约 0.55 GB],
  [LLM 激活（grad ckpt，seq 约 600）], [1–3 GB], [1–2 GB],
  [杂项（CUDA ctx/碎片/通信 bucket）], [5–10 GB], [4–6 GB],
  [*合计 / 可用*], [*约 47–55 / 80 GB*], [*约 77–79 / 79.1 GB*],
  [*判定*], [*余量 25–32 GB，健康*], [*贴顶，高风险*],
)

激活估算依据：开 gradient checkpointing 后只存段边界（61 层 × seq 600 × 4096 × bf16 约 0.3 GB）加段内重算临时；micro-batch 4–16 线性放大。情景 A 的生命线是 FP4/FP8 原生加载成立（R1–R3）；一旦需要 bf16，4×H100/H200 都装不下（568/4 = 142 GB > 80/141 GB），只剩 8 卡。情景 B 在 8×A100 上余量不足 2 GB，必须 micro-batch 1 + 更激进 checkpointing，实测 OOM 即升 4×B200（142 GB/卡，余量约 50 GB，\$21.25/h，账单已含）。

== Gate D 风险清单（租卡前预演）

不保证一次成功。下表是全部已识别坑点、概率判断与止损方案；核心原则是*失败成本锁死在租后第 2–3 小时*（Gate D 判定，损失约 \$20），任何一项失败都有明确的下一步，而不是现场想办法。

#table(
  columns: (1.9fr, 0.7fr, 3.4fr),
  [*风险*], [*概率*], [*缓解 / 止损*],
  [R1 NVFP4 权重 Dgrad 不可用（推理内核不支持对输入 embedding 求梯度）], [中–高], [核心风险。Gate D 单 batch backward 判定；失败即退租转情景 B（解量化 bf16）或情景 A′（B200 原生 FP4 内核重试）。],
  [R2 Transformers 加载原生 0731 大权重失败（quantization config 格式未被 5.x 支持）], [中], [Gate D 第一步即原生加载测试；备选为 vLLM loader 导出 bf16 权重副本（同时解决 R1）。],
  [R3 H100 上 FP4 前向本身不可用（官方 NVFP4 验证环境是 Blackwell）], [中], [情景 A 的直接死因；Gate D 前 30 分钟判定。转 4×B200（\$21.25/h）或情景 B。],
  [R4 权重下载断流/限速], [高], [本地实测 802 MB 在慢代理下耗时 2h46m；租机标称 2217 Mbps 但 HF CDN 单连接限速 40–100 MB/s，160 GB 理论 27–67 分钟，断流可拖至 2h+。用 aria2/hf\_transfer 多连接 + 断点续传循环（已验证的做法），账单已含 0.5–2h 敏感性。],
  [R5 marketplace 实例被中断], [中], [checkpoint 流式上传（每 500 步，含 optimizer/RNG，28/28 测试）；中断后换机 `--resume` 精确续训，损失不足 10 分钟训练。],
  [R6 机器环境与宣传不符（NVLink 拓扑、驱动、盘速）], [低–中], [开机 10 分钟内 `nvidia-smi topo -m` + 盘速快测，不符当场退租换机，损失不足 1h 租金。],
  [R7 情景 B 显存算术（bf16 568 GB vs 8×A100 640 GB）], [低], [冻结 LLM + activation checkpointing 下单 batch 激活很小，72 GB 余量足够；4×H200（564 GB）装不下，明确排除。],
  [R8 Hash-MoE `tid2eid` 在多卡张量并行下的分布行为], [中], [hook 方案已在真实 tiny DeepseekV4 类验证；Gate D 的单 batch backward 在大权重多卡下复验，占位位置路由一致性有断言。],
  [R9 训练数据下载（约 30 GB）经代理再耗 1–2 h], [中], [提前把训练数据镜像到项目 HF 仓库（随 checkpoint 上行通道同路），租机从 HF 直下；计入装机时间。],
  [R10 MoonViT-V2 bf16 与 fp32 参考的特征偏差], [低], [fp32 参考已锚定（eager/sdpa 差 3.1e-05）；Gate D 记录 bf16 实测差（预期约 1e-2 相对），超差则视觉塔回 fp32（仅多约 800 MB 显存）。],
  [R11 租期内时间不够闭环], [低–中], [checkpoint 流式上传保证权重永不丢；benchmark 三组对照在机上跑但数据落盘 JSON 可增量上传；最坏情况先公开 checkpoint + 部分指标。],
  [R12 多卡分布策略与 PCIe 拓扑（4×H100 PCIe 无 NVLink，TP=4 每层 all-reduce 走 PCIe）], [中], [训练吞吐的直接风险：3–5 s/步 的估计在 PCIe + 重算下可能偏乐观。Gate D 实测单步耗时，>15 s/步 即换 NVLink 机型（H100 SXM \$10.75/h）或改流水线并行（冻结 LLM 下 PP 通信最小，但需 DeepSpeed/pippy，工程复杂度上升）；账单按 5 h 悲观档已可吸收 2 倍减速。],
)

== Gate D：正式租卡前

1. 原生 0731 权重加载成功。
2. 单图短序列 forward。
3. 单 batch backward，projector 梯度有限且非零。
4. LLM/MoonViT 无梯度。
5. 20 step 无 OOM/NaN。
6. 新进程恢复 projector 后输出一致。

== 租期闭环排程（2026-08-02 定价）

核心约束：租期一结束就没有机器能跑动 0731 做 benchmark 或回传权重，因此训练、benchmark、上传必须在同一次租期内闭环。交付物只有 projector + 评测 JSON + 报告，与 GLM 社区只发布 projector 一致，不回传 160 GB 主干。checkpoint 发两个精度：fp32 master（约 134 MB，复现/续训用）与 bf16 serving（约 67 MB,0731 激活为 bf16)，租期内由训练产物现场转换。推理侧接入（vLLM/SGLang/llama.cpp/fastllm 补丁点、Hash-MoE 注意事项、验收检查）已写成 `docs/inference-integration.md`（2026-08-03 重写为 MoonViT-V2 版），作为后续给推理引擎提 PR 的合同文档；要点：vLLM 与 SGLang 均已 Day-0 支持 Kimi-K3（含 MoonViT3d 视觉塔）且均有 DeepSeek-V4 文本栈，patch 面只剩 projector 模块、placeholder 扩展与 Hash-MoE 路由检查；placeholder 固定为现有 `<｜image｜>`(id 129279）禁止扩 vocab，合并只替换 embedding 向量、input\_ids 保留 placeholder 供 Hash-MoE 路由。

训练配方直接采用 Baseten 社区实验的实测方案（baseten.co/blog/glm-52-with-vision，checkpoint baseten/GLM-5.2-Vision-NVFP4，唯一公开的同级成功案例）：*constant lr 5e-4*——原文未使用任何 LR schedule，LLaVA 式 cosine 属于未在此场景验证的外推，因此不引入调度器，checkpoint 也无需保存调度器状态（该待办关闭）；global batch 64；约 66k 条*短 QA* 配对；2 个 epoch ≈ 2070 步。grokking 预期在第一 epoch 末（约 step 900–1100）出现 loss 骤降；每 500 步存 checkpoint 并立刻后台传 HF，使我们能在 pre/post-grokking checkpoint 间择优。checkpoint 是完整可续训单元（projector fp32 + bf16、AdamW 状态、RNG、步数、loss 历史，见 `moonvit_glue.checkpointing`)：实例中断不丢成果，社区可实时看到训练曲线形成，任何 checkpoint 可用 `--resume <repo-id>` 精确续训（跨机器 GPU 数不同亦可恢复）。4×RTX PRO 6000 估 3–6 s/步（13B 激活、seq 约 300–400、开 activation checkpointing），训练段 2–4 h。若 step 约 1400 仍无 grokking 迹象，先查数据是否混入长答案，而不是盲目加步数。

停训判据不看 loss，看两条 gap：主判据是 benchmark 分数 − blind 分数的 gap 随 checkpoint 的曲线，平台即停；辅助判据是留出集 shuffle\_delta > 0.1。参考锚点：projector-only 的 TextVQA 现实预期 20–30%（blind 约 10–15%，成熟 VLM 60+）；达到 GLM 社区实验同量级（grounding parse 率 >80%、Acc\@50 个位数）即成功。

#table(
  columns: (2.2fr, 1fr, 3fr),
  [*阶段*], [*时长*], [*说明*],
  [装机 + 下载权重], [1–1.5 h], [160 GB；好主机 1–2 GB/s],
  [Gate D 判定], [0.5–1 h], [FP4 Dgrad 失败则当场退租（损失约 \$10），转情景 B],
  [Stage 1 对齐训练], [2–4 h], [约 2100 步（66k 短 QA × 2 epoch，batch 64，constant lr 5e-4）；checkpoint 随训随传；step 约 1400 无 grokking 则停训查数据],
  [benchmark 全套], [1.5–2 h], [TextVQA 500 / DocVQA 200 / OCRBench 200 / ScreenSpot 200；训练 checkpoint × blind × 随机 projector 三组对照，机上完成],
  [权重与结果回传], [0.5 h], [projector 约 134 MB + 评测 JSON 上传 HF],
  [*合计*], [*6.5–9 h*], [4×RTX PRO 6000（\$5.18/h 含 400 GB 挂盘）约 \$34–47；\$50 余额单次闭环可行，失败重试需追加],
)

若 FP4 Dgrad 不可用（情景 B）：权重解量化至 bf16（568 GB），需 8×A100 SXM（\$10.30/h），同排程时长大约 ×2–3，预算 \$300–500。

= 评测与验收计划

没有 benchmark 就无法回答“接上了没有”。评测口径对标 Baseten 社区 GLM-5.2V 实验（原文 baseten.co/blog/glm-52-with-vision,0xSero/fable-glm-vision 复现）：视觉塔同为冻结 MoonViT（他们从 Kimi K2.6 抽取，我们用官方独立仓 MoonViT-SO-400M，同构 1152 维），其标志性指标是 *MMMU-Pro*（原文声称 55%，约 Claude 4.5 Haiku 水平），因此我们的评测集在 TextVQA/DocVQA/OCRBench/ScreenSpot 之外加入 *MMMU-Pro（单图子集，exact match）*。所有数字必须与 blind baseline（同一模型、无图输入）一起报告：VQA 类基准有显著语言先验，无图基线把“模型本来就会答”与“图像带来了信息”分开。

社区配方还有两个直接影响数据计划的实测结论：其一，*grokking*——batch 64 / lr 5e-4 配短答案时 loss 平台数百步后骤降（原文约 step 900）完成对齐，*长描述性答案会阻止 grok*，因此对齐数据应以短 QA 为主、长 caption 为辅，而不是只用长 caption；其二，*warm start*——从已对齐 projector 初始化可跳过大半平台期，多阶段数据混训时应复用上一阶段 checkpoint 而不是重零开始。

训练数据泄露控制（2026-08-03 定稿，GUI 修订）：正式 mix 含 TextVQA train（34.6k）、DocVQA train（25k）、0xSero art 子集（约 10k）与 ShowUI-desktop（8k）四类短答案监督；GUI 数据按用户决定纳入以建立 computer-use 基础，答案统一为 0xSero 动作格式 `click(start\_box=[x,y])`（0..999 尺度，评测 parser 原生兼容）。0xSero 自带的 screenshots/multistep 行不直接复用（图像改名无法回 join；multistep 为轨迹格式），改为从同源公开数据集自取。代价与处理：ScreenSpot 自此为*域内*基准，报告中必须标注；fetch 规格机械执行 max\_answer\_words ≤ 20；组装时对全部训练图与五个基准的全部评测图做 average-hash 去重（hamming ≤ 6 丢弃），去重报告随数据发布。数据产物托管于 dataset repo cyjin-yl/moonvit-dsv4-data，含来源、固定 revision 与 sha256，租机上一次下载即用。

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
)

= 下一位执行者的最短路径

先运行 `pytest` 和 `examples/smoke_tiny_text_lm.py`。然后在 V100 工作站使用机械盘作为 `HF_HOME`，验证真实 MoonViT。正式 0731 实验前不要写训练循环，先证明目标 CUDA/量化 runtime 支持 input data-gradient。若失败，优先评估 FP8 可微加载或定制 Dgrad，不要改用 GGUF 假装完成训练链路。
