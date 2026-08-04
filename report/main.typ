#set page(paper: "a4", margin: 24mm)
#set text(font: ("Noto Sans CJK SC", "Microsoft YaHei", "Arial"), size: 10pt)
#set heading(numbering: "1.1")
#set par(justify: true, leading: 0.7em)
#set table(stroke: 0.5pt + rgb("c8c8c8"), inset: 6pt)

#align(center)[
  #text(size: 21pt, weight: "bold")[MoonViT-V2 接入 DeepSeek-V4-Flash-0731]
  #v(5pt)
  #text(size: 14pt)[训练前架构审计、胶水原型与硬件计划]
  #v(8pt)
  版本 0.3 · 2026-08-04
]

#outline()
#pagebreak()

= 执行摘要

本项目目标是给纯文本的 DeepSeek-V4-Flash-0731 接入从 Kimi K3 抽取的 MoonViT-V2（MoonViT3d）视觉编码器。第一阶段不训练视觉塔和语言模型，只训练一个 Kimi 风格 PatchMerger projector；独立发布的 MoonViT-SO-400M（V1）只保留作历史对照。当前结论是：V2 的真实权重、预处理和 `[tokens,4,1024]` 合同均已在 V100 验证，projector-only 全量数据训练已得到明确视觉对齐信号；正式 0731 大权重的 FP4/FP8 可微 kernel 是尚未消除的主要风险。

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
  [MoonViT-V2 输出 shape/freeze 合同], [通过], [真实 K3 视觉权重],
  [真实 MoonViT-V2 + 小 LM 前向/反向], [通过], [V100 CUDA eager/sdpa],
  [eval\_vlm 生成/blind/shuffle-loss 干跑], [通过], [评测管线端到端],
  [generate()：generic 与 deepseek_v4 两种路径], [通过], [评测/推理前置],
  [指标库（VQA/ANLS/token-F1/grounding）], [通过], [纯 Python，无 torch],
  [完整测试集], [96/96], [Linux + torch 2.10.0+cu128],
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

目的：在正式 train\_v1 mix（59,198 行 packed parquet）上验证"parquet 消费路径 + 全量数据 + early alignment + 全套评测 + 上传"的租期闭环，同时作为正式 0731 训练的本地对照组。设置：Qwen2.5-0.5B-Instruct（冻结）+ K3 MoonViT-V2（冻结，eager），只训 projector（33.6M 参数）,2000 个 optimizer step、恒定 lr 5e-4、`--max-image-side 448`。历史参数 `batch 8` 实际不是一次 8 样本 forward，而是 `micro_batch_size=1` 下串行 8 次 forward/backward 后更新，因此本轮只见过 16,000 样本，即 59,198 条 mix 的约 *0.27 epoch*；历史 answer-token 数未记录，不能精确补齐。占位 token 自动解析为 Qwen 词表已有的 `<|image_pad|>`（ID 151643，不扩词表）。

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

每条记录另有完整控制分配：blind 不提供图；blank 使用同 split 的纯背景；same-image 对该 split 所有样本使用同一张中性条纹图；shuffled-image 在任务内作无固定点的确定性 derangement；patch-permutation 给出逐样本 seed，在 MoonViT merged spatial-token 轴执行 `torch.randperm`，保留值与 token 数量。独立 verifier 检查了 4,800/4,800 图像 SHA、2,400/2,400 pair、4,800/4,800 控制行和全部派生文件 hash；失败数为 0，logical dataset SHA 为 `122ae820…cbaa71`。完整 PNG 在 V100 数据盘，Git 提交完整 train/selection/control JSONL、manifest/hash、计数 CSV、日志、零失败文件、验证结果与每类一组 a/b 预览。这里尚未报告任何模型准确率；普通/paired/answer-flip 与五种控制的分母将在包 3 固定后统一计算，避免数据生成阶段改 scorer。

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
  [*组成（每卡）*], [*情景 A：4×RTX PRO 6000，FP4 权重 TP=4*], [*情景 B：8×A100，bf16 权重 TP=8*],
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
  [R1 NVFP4 权重 Dgrad 不可用（推理内核不支持对输入 embedding 求梯度）], [中–高], [核心风险。Gate D 单 batch backward 判定；失败即退租转情景 B（解量化 bf16）或情景 A′（B200 原生 FP4 内核重试）。],
  [R2 Transformers 加载原生 0731 大权重失败（quantization config 格式未被 5.x 支持）], [中], [Gate D 第一步即原生加载测试；备选为 vLLM loader 导出 bf16 权重副本（同时解决 R1）。],
  [R3 FP4 前向内核在目标卡上不可用], [低–中], [主机器 RTX PRO 6000 即 Blackwell（sm_120，原生 NVFP4），此项已从"中"降级；但硬件支持 ≠ 内核带 Dgrad——Gate D 第 0 步用 `tools/gate_d_dgrad.py` 单层 reproducer 单独判定。失败转 4×B200（\$21.25/h）或情景 B。],
  [R4 权重下载断流/限速], [高], [本地实测 802 MB 在慢代理下耗时 2h46m；租机标称 2217 Mbps 但 HF CDN 单连接限速 40–100 MB/s，160 GB 理论 27–67 分钟，断流可拖至 2h+。用 aria2/hf\_transfer 多连接 + 断点续传循环（已验证的做法），账单已含 0.5–2h 敏感性。],
  [R5 marketplace 实例被中断], [中], [checkpoint 流式上传（每 500 步，含 optimizer/RNG，28/28 测试）；中断后换机 `--resume` 精确续训，损失不足 10 分钟训练。],
  [R6 机器环境与宣传不符（NVLink 拓扑、驱动、盘速）], [低–中], [开机 10 分钟内 `nvidia-smi topo -m` + 盘速快测，不符当场退租换机，损失不足 1h 租金。],
  [R7 情景 B 显存算术（bf16 568 GB vs 8×A100 640 GB）], [低], [冻结 LLM + activation checkpointing 下单 batch 激活很小，72 GB 余量足够；4×H200（564 GB）装不下，明确排除。],
  [R8 Hash-MoE `tid2eid` 在多卡张量并行下的分布行为], [中], [hook 方案已在真实 tiny DeepseekV4 类验证；Gate D 的单 batch backward 在大权重多卡下复验，占位位置路由一致性有断言。],
  [R9 训练数据下载（约 30 GB）经代理再耗 1–2 h], [中], [提前把训练数据镜像到项目 HF 仓库（随 checkpoint 上行通道同路），租机从 HF 直下；计入装机时间。],
  [R10 MoonViT-V2 bf16 与 fp32 参考的特征偏差], [低], [fp32 参考已锚定（eager/sdpa 差 3.1e-05）；Gate D 记录 bf16 实测差（预期约 1e-2 相对），超差则视觉塔回 fp32（仅多约 800 MB 显存）。],
  [R11 租期内时间不够闭环], [低–中], [checkpoint 流式上传保证权重永不丢；benchmark 三组对照在机上跑但数据落盘 JSON 可增量上传；最坏情况先公开 checkpoint + 部分指标。],
  [R12 多卡分布策略与 PCIe 拓扑（4×RTX PRO 6000 无 NVLink，PCIe 5.0 x16 54.2 GB/s；`device_map="auto"` 朴素模型并行的激活回传走 PCIe）], [中], [训练吞吐的直接风险：3–5 s/步 的估计在 PCIe + 重算下可能偏乐观。多卡路径已定型为单进程 `device_map="auto"`（LLM 冻结，无需权重梯度分片；vLLM/SGLang 推理 TP 不可用于反向训练）。Gate D 实测单步耗时：≤8 s 维持；8–15 s 重算步数保 benchmark；>15 s 验证 transformers 原生 `tp_plan` 或换 NVLink 机型（H100 SXM \$10.75/h）；账单按 5 h 悲观档已可吸收 2 倍减速。],
)

== Gate D：正式租卡前

分阶段判定（完整版见 `docs/gate-d-runbook.md` §7，2026-08-03 评审修订，各步独立记录不合并）：

0. 配置发现 + 最小 Dgrad reproducer（`tools/gate_d_dgrad.py`：打印量化方案；只取一层真实 quantized linear 权重切片，判定 input.grad 有限非零、weight.grad 为 None）。
1. `nvidia-smi topo -m` + 盘速快测（不符当场退租）。
2. 原生 0731 权重加载成功；文本短前向正常。
3. 单图短序列 forward（placeholder 注入）。
4. 单 batch backward，projector 梯度有限且非零，LLM/MoonViT 无梯度。
5. hook × activation checkpointing 数值一致性（开/关梯度检查点 projector 梯度 allclose）+ batch>1 多图位置一致。
6. 多卡路径定型（`device_map="auto"`）+ 实测单步耗时分档（≤8 / 8–15 / >15 s）。
7. 20 step 无 OOM/NaN；`--resume` 从流式 checkpoint 恢复一次且轨迹连续。

== 租期闭环排程（2026-08-02 定价）

核心约束：租期一结束就没有机器能跑动 0731 做 benchmark 或回传权重，因此训练、benchmark、上传必须在同一次租期内闭环。交付物只有 projector + 评测 JSON + 报告，与 GLM 社区只发布 projector 一致，不回传 160 GB 主干。checkpoint 发两个精度：fp32 master（约 134 MB，复现/续训用）与 bf16 serving（约 67 MB,0731 激活为 bf16)，租期内由训练产物现场转换。推理侧接入（vLLM/SGLang/llama.cpp/fastllm 补丁点、Hash-MoE 注意事项、验收检查）已写成 `docs/inference-integration.md`（2026-08-03 重写为 MoonViT-V2 版），作为后续给推理引擎提 PR 的合同文档；要点：vLLM 与 SGLang 均已 Day-0 支持 Kimi-K3（含 MoonViT3d 视觉塔）且均有 DeepSeek-V4 文本栈，patch 面只剩 projector 模块、placeholder 扩展与 Hash-MoE 路由检查；placeholder 固定为现有 `<｜image｜>`(id 129279）禁止扩 vocab，合并只替换 embedding 向量、input\_ids 保留 placeholder 供 Hash-MoE 路由。

Baseten 社区实验（baseten.co/blog/glm-52-with-vision，checkpoint baseten/GLM-5.2-Vision-NVFP4）只作为配方先验：*constant lr 5e-4*、global batch 64、约 66k 条短 QA、2 epoch ≈ 2070 optimizer steps，grokking 在第一 epoch 末附近出现。它不是本项目的时长承诺。审计发现当前训练器的历史 `batch_size=N` 是 `micro_batch_size=1` 下串行 N 次 forward/backward；若照抄 64，每个 optimizer step 会执行 64 次视觉塔和 LLM 前后向，3–6 s/step 与 2–4 h 估计均无效。新版训练器已改用 micro-batch、gradient accumulation、effective batch、examples seen 与 answer tokens 的明确计量，并暂时拒绝伪造 `micro_batch_size > 1`。正式租卡前必须实现 padded multi-example forward，在小主干上实测 micro batch 1/2/4 的吞吐与显存，再由目标 examples/token 数反推 optimizer steps 和租时。

分辨率也从“写死”改为待证：先跑训练 448/640 × 评测 448/640/1024 的固定子集矩阵，只有在小字 OCR 收益、分布失配、视觉 token 数和吞吐都可接受时才采用训练 640/评测 1024。容量消融按相同 examples seen 比较 Qwen2.5 0.5B/1.5B/3B 纯文本主干；随后才做 projector scratch/warm-start、顶部 LoRA、blank/fixed/shuffle/patch-permutation 与 synthetic minimal-pair 控制。完整协议在仓库 `docs/ablation-protocol.md`。

停训判据不看 loss，看两条 gap：主判据是 benchmark 分数 − blind 分数的 gap 随 checkpoint 的曲线，平台即停；辅助判据是留出集 shuffle\_delta > 0.1。参考锚点：projector-only 的 TextVQA 现实预期 20–30%（blind 约 10–15%，成熟 VLM 60+）；达到 GLM 社区实验同量级（grounding parse 率 >80%、Acc\@50 个位数）即成功。

#table(
  columns: (2.2fr, 1fr, 3fr),
  [*阶段*], [*时长*], [*说明*],
  [装机 + 下载权重], [1–1.5 h], [160 GB；好主机 1–2 GB/s],
  [Gate D 判定], [0.5–1 h], [FP4 Dgrad 失败则当场退租（损失约 \$10），转情景 B],
  [Stage 1 对齐训练], [待租前实测], [按 examples/token 预算与真实 micro-batch 吞吐反推；禁止用 64 次串行 forward 冒充 global batch 64],
  [benchmark 全套], [1.5–2 h], [TextVQA 500 / DocVQA 200 / OCRBench 200 / ScreenSpot 200；训练 checkpoint × blind × 随机 projector 三组对照，机上完成],
  [权重与结果回传], [0.5 h], [projector 约 134 MB + 评测 JSON 上传 HF],
  [*合计*], [*暂不锁定*], [装机、Gate D、benchmark 与回传有时间盒；训练段待 batching/分辨率/容量消融后重新报价],
)

若 FP4 Dgrad 不可用（情景 B）：权重解量化至 bf16（568 GB），需 8×A100 SXM（\$10.30/h），同排程时长大约 ×2–3，预算 \$300–500。

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
)

= 下一位执行者的最短路径

先运行 `pytest` 和 `examples/smoke_tiny_text_lm.py`。然后在 V100 工作站使用机械盘作为 `HF_HOME`，按 MANIFEST 校验并验证真实 MoonViT-V2。正式 0731 实验前不要写训练循环，先证明目标 CUDA/量化 runtime 支持 input data-gradient。若失败，优先评估 FP8 可微加载或定制 Dgrad，不要改用 GGUF 假装完成训练链路。
