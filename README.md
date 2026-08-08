# MoonViT projector → DeepSeek-V4 glue prototype

这是一个面向真实 VLM 的接口、训练与评测工程。最终路径固定为
`MoonViT-V2 → 4096 维 projector → DeepSeek-V4-Flash-0731`。
仓库当前同时注册两条 Qwen2.5-3B 架构控制：精确 K3/MoonViT-V2
(`kimi_k3_v2`) 和 K2.6-lineage 的 MoonViT-SO-400M V1 family proxy。
旧的 `legacy_pre_norm` V2 结果只保留为失败记录，不能当作精确 K3 V2
的结论。当前状态和下一步以 [`docs/current-status.md`](docs/current-status.md)
与 [`docs/architecture-matrix.md`](docs/architecture-matrix.md) 为准；固定评测规则见
[`docs/qwen2.5-3b-community-eval-contract.md`](docs/qwen2.5-3b-community-eval-contract.md)。
当前没有可用 VLM，也没有 checkpoint 获得晋升。Qwen2.5-7B 已完成完整
1,272 条 ScreenSpot，但 vision/blind/shuffled click 仅为
`3.30%/3.46%/2.67%`：vision 对 shuffled 有弱阳性，对 blind 失败。
Qwen3.5-4B external MoonViT 的 matched V1/V2 ScreenSpot50 也都未通过因果门。
完整 DeepSeek-V4-Flash-0731 权重尚未加载运行，Gate D 为 **NO-GO**。
运行入口与硬阻塞的权威地图见
[`docs/runtime-entrypoint-audit.md`](docs/runtime-entrypoint-audit.md)。

## 项目级完成标准：完整对比矩阵（2026-08-08 修订）

“至少一组”不再是本项目的成功条件。当前目标是完成并发布整个已注册对比矩阵：配置中的每个 active arm 都必须在同一社区规模合同下实际尝试，并留下正式结果或不可变的失败记录；不能因为某一臂表现较好就提前结束，也不能静默跳过显存不足、实现失败或因果门失败的臂。

每个外部 MoonViT arm 都按 receiver 单独重新训练 projector，并统一运行相同的数据顺序、examples-seen 节点、图像预处理、prompt、parser、greedy decoding 和 vision/blind/shuffled/random_projector 条件；同时保存 step0、previous_best、current_candidate、健康日志、checkpoint、逐样本 prediction、正式 ScreenSpot50 paired bootstrap 和 artifact manifest。原生 Qwen VLM 只作独立阳性对照，历史 0.5B/3-step/replay/geometry 结果只作 archived 机制证据。

矩阵完成的含义是：所有 active rows 都出现在 MATRIX_SUMMARY.json，每行都标注 result、causal_pass、failure_reason 或 resource_limit。只有矩阵整体完成后，才能选择 transferable candidate；DeepSeek-V4-Flash-0731 Gate D 仍是独立的最终门，不会被 Qwen 代理结果替代。

## 当前实验主线：模型消融优先（2026-08-08）

项目从“反复证明脚本不会报错”切换回“在社区训练条件下判断模型是否真的获得视觉能力”。
pytest、strict-load、manifest 和 checkpoint 检查仍是每轮的短 preflight；它们是防止产生假数据的护栏，
不是本轮研究目标。主实验必须比较模型条件，而不是只比较 loss：

| 消融因素 | 固定条件 |
|---|---|
| 视觉输入 | MoonViT-SO-400M/K2.6-lineage V1、K3/MoonViT-V2、无视觉、random projector |
| 接收器 | 纯文本 Qwen2.5-3B、Qwen2.5-7B；Qwen3.5-4B/9B stripped-native；原生 Qwen VLM 只作独立阳性对照 |
| 训练 | 每个 receiver×tower 重新初始化并重新训练 projector；不跨模型复用旧 projector |
| 评测 | vision、blind、shuffled、random_projector 四条件，固定样本顺序、预处理、prompt、parser 和 greedy decoding |

第一批 scaled reproduction 对齐社区 GLM-5.2V 的公开数量级：短答案图文数据约 66k 条，global batch 64，
constant learning rate `5e-4`，约 2 epochs、约 2,070 optimizer steps；社区报告的突变/grokking 观察点约在
step 900（约 57.6k examples seen）。因此 20/100 steps 只用于健康止损和接口烟雾检查，不能宣布能力失败或成功。
所有 arm 在 `examples_seen=4k/8k/16k/32k/57.6k/66k/132k` 保存并跑同一合同；只有真实
ScreenSpot、TextVQA、DocVQA、OCRBench 及 vision−blind/shuffled 配对 CI 才能晋升 checkpoint。

表示塌缩、NaN/Inf、梯度异常和 RMS/spread/rank 只作在线止损：发现坏轨迹就保存并回滚，不能用“健康”替代
视觉能力。旧的 3-step receiver-prior、32-row 和 geometry/replay 结果保留在文档中并标为 archived，
用于解释失败机制，不再阻塞社区规模消融主线。

项目把视觉塔、projector、文本主干保持为三个独立 checkpoint，并实现：

- Kimi 风格的 2×2 PatchMerger projector；
- 一个 image placeholder 扩展为可变数量视觉 token；
- 普通 Hugging Face causal LM 的 `inputs_embeds` 路径；
- DeepSeek-V4 Hash-MoE 所需的 token-ID 路由路径；
- projector 严格 shape 校验和 Safetensors 保存/恢复；
- 冻结 MoonViT-V2 和 LLM、只让梯度进入 projector。

统一环境需要 `transformers>=5.12,<6`。公开的 PyPI 4.57.6 尚未包含 `transformers.models.deepseek_v4`，不能仅根据 checkpoint config 中的旧版本字段选环境。

## 已锁定的权重合同

| 组件 | 来源/形状 | 是否训练 |
|---|---|---|
| MoonViT-V2 | Kimi K3 的 MoonViT3d，401.2M 参数；抽取 BF16 权重约 802 MB | 冻结 |
| MoonViT-V2 输出 | 每张图 `[视觉 token 数, 4, 1024]` | — |
| Projector | 精确 K3 V2：无 pre-norm、bias-free `4096 → 4096 → 4096`、post-RMSNorm；legacy 变体另行标识 | 训练 |
| Projector 参数 | 精确 K3 V2 为 33,558,528；legacy V2 为 33,564,672 | 训练 |
| DeepSeek | `deepseek-ai/DeepSeek-V4-Flash-0731`，hidden size 4096 | 冻结 |
| Placeholder | 已有 `<｜image｜>`，token ID 129279 | 不扩 vocab |

主线配置在 `configs/deepseek-v4-flash-0731-projector-moonvit-v2.json`。Projector 权重单独保存为 `projector.safetensors`，不会复制或修改 MoonViT-V2/DeepSeek 权重。

V2 权重由 Kimi K3 的单个视觉分片抽取并 strict-load 验证，仓库内 vendor 的 vision-only 代码不依赖完整 K3 模型。`MoonViT-SO-400M`（V1，1152 维）现在承担 K2.6-lineage family control。它与 Kimi-K2.6 的塔没有 byte-identical 证明，必须在同一 benchmark 合同下单独缓存、训练和评测；V1/V2 projector checkpoint 绝不能混用。

## 为什么 DeepSeek 需要特殊路径

DeepSeek-V4 的早期 Hash-MoE 层使用 `tid2eid[input_ids]` 选择 experts。只传 `inputs_embeds` 时，这些层没有路由 token；同时 Transformers 的公共接口不允许一起传 `input_ids` 和 `inputs_embeds`。

`VisionCausalLM` 因而传入扩展后的 placeholder IDs，并在一次 forward 期间临时替换输入 embedding 层的输出。模型获得：

- placeholder IDs：用于 Hash-MoE 路由；
- projector embeddings：用于实际隐藏状态；
- projector → loss 的完整梯度路径。

hook 在 forward 结束后自动移除，不修改 Transformers 源文件或 checkpoint。

## 安装和验证

```bash
python -m pip install -e ".[test]"
pytest
python examples/smoke_tiny_text_lm.py
```

离线 smoke test 使用随机初始化的微型 GPT-2，不下载模型。预期：projector 有梯度，语言模型没有梯度。

只下载两个 config 和 DeepSeek tokenizer、检查正式尺寸：

```bash
python tools/inspect_compatibility.py
```

用旧版 V1 MoonViT 接一个很小的纯文本模型做回归 smoke：

```bash
python examples/smoke_real_moonvit.py path/to/image.jpg
```

这会下载约 834 MB 的 V1 权重和 `sshleifer/tiny-gpt2`。随机 projector 不会产生有意义的图像描述；该命令验证的是旧合同兼容性，不代表当前 V2 主线。V2 的训练/评测入口使用 `--vision-tower v2 --moonvit-v2-weights <path>/moonvit_v2.safetensors`，加载前应按随产物发布的 `MANIFEST.json` 校验 SHA-256。

## DeepSeek 加载接口（完整 0731 尚未验证）

以下是已经实现的软件 API，不是完整 0731 权重已成功加载的证据。仓库中尚无训练或
评测 tool 调用这个 helper；它只在 tiny DeepSeek 软件回归中覆盖了相同 glue seam。
正式使用前仍需目标量化/多卡 runtime、真实 input-DGRAD、43 层 Hash-MoE
forward/backward/generate 和 checkpoint round-trip Gate。

```python
from moonvit_glue import load_deepseek_flash_0731

loaded = load_deepseek_flash_0731(
    "checkpoints/projector-step-1000",
    device_map="auto",
)
model = loaded.model
tokenizer = loaded.tokenizer
```

加载器在碰权重前检查 projector 输出宽度是否为 4096，并拒绝添加新 token。正式大权重需要额外安装 `accelerate`：

```bash
python -m pip install -e ".[large-model]"
```

## 训练前还必须通过的门槛

当前 CPU 测试已使用 Transformers 的真实 `DeepseekV4ForCausalLM` 类和 1 层 Hash-MoE 配置完成 loss/backward。正式租卡前仍需在目标 CUDA 镜像上完成：

1. 原生 0731 FP4/FP8 权重成功加载；
2. 单图、短序列前向；
3. 单 batch backward 后所有 projector 参数有有限梯度；
4. LLM 和 MoonViT-V2 参数全部无梯度；
5. 至少连续 20 step 无 OOM/NaN；
6. 保存 projector 后在新进程恢复，输出逐元素一致。

最大的剩余风险是 FP4/FP8 推理 kernel 是否实现 input-gradient，而不是 glue 的 token/shape 逻辑。若原生 kernel 不可微，需要可微的 FP8/BF16 loader 或定制 data-gradient kernel。

## 评测与验收（Benchmark）

没有 benchmark 就不知道接没接上。口径沿用社区 GLM-5.2 视觉实验：grounding 报 parse rate、归一化 0–999 坐标的 Accuracy@50 与平均点击误差；文本类报 exact match / soft VQA / ANLS。**所有能力数字必须与 blind baseline（同模型、无图输入）一起报告**，VQA 基准的语言先验不能冒充视觉能力。

证据必须分层解读：原生 Qwen3.5 VLM 使用自己的视觉塔与既有多模态对齐，只能作为“数据/processor/评分器正常”的阳性对照，不能证明本项目的 MoonViT-V2 projector 能接入纯文本 DeepSeek。真正的租前接口实验使用 `Qwen2ForCausalLM`（无 `vision_config`）等纯文本主干；训练器会默认拒绝带原生视觉配置的 `--text-model`。即使纯文本小主干上学出对齐，也仍只证明接口可学习；完整 DeepSeek-V4 权重的可微加载、训练和最终 benchmark 才是目标能力证据。

2026-08-05 起，中间尺度代理固定为纯文本 `Qwen/Qwen2.5-3B-Instruct` revision `aa8e7253…04d1`。完整合同见 [`docs/qwen2.5-3b-community-eval-contract.md`](docs/qwen2.5-3b-community-eval-contract.md)：模型 9 个文件逐 SHA 固定；公开 ScreenSpot 1,272 条与十 strata 各 5 条的 `screenspot_glm50_v1` 已在任何 3B 结果前冻结；严格输出为 `click(start_box=[x, y])`；vision/blind/shuffled/random/step0/previous/current 全部进入统一评分。Qwen 的 2048 hidden width 通过无参数 fixed receiver 读取 canonical 4096 projector；exact K3 V2、V1 和 legacy V2 的参数量分别记录在架构矩阵中，不能混写。

Qwen2.5 词表虽预留 `<|image_pad|>`（ID 151655），但该纯文本 checkpoint 的
视觉相关 special-token 行是同一类 dummy 初始化，不能当作视觉预训练证据。
DeepSeek-V4-Flash-0731 的 pinned tokenizer 预留 `<｜image｜>`（ID 129279）；
glue 通过 placeholder 注入 embedding，不扩充词表。Qwen3.5-4B 的 special-token
行已经带有原生 VLM 训练痕迹，剥离其 native visual module 后可作为 receiver-prior
诊断，结果标签固定为 `qwen_specific_not_transferable`，不能进入 Qwen3B 主排名。

横向实验还必须加载同一份 exact FP32 step0（SHA-256 `efd942e0…b06b0`）；固定 random-projector control 为 `7bd4aacf…fc44`。两份 134,259,248-byte 权重已在首个 3B 输出前发布到 HF immutable commit `65639da5…a010` 并按 LFS SHA 回查，seed 只承担 provenance，不能替代权重身份。

现有 Gate B 的 `Qwen2.5-0.5B-Instruct` 虽然是纯文本模型，但 0.5B 容量会压低 OCR、推理和格式遵从上限，也不能代表 DeepSeek Hash-MoE 的优化难度。因此其 shuffle/随机/blind 差异只作工程与信号证据，绝对分数和收敛速度都不得外推 0731。3B 代理已经接替容量对照；legacy V2、exact K3 V2 与 V1 family control 的 matched screens 均已完成且未建立能力。原生 Qwen2.5-VL/Qwen3.5 VLM 不属于 Qwen3B 主实验。

此外，full-mix Gate B 的 `2000 steps × batch 8` 实际是 `micro_batch_size=1` 下串行累积 8 个样本：共见过 16,000 个样本，只约为 59,198 条 mix 的 **0.27 epoch**。因此它应称为 early-alignment / 接口可学习性运行，而不是充分训练后的能力评测；TextVQA 8.1%、DocVQA 3.9%、OCRBench 0% 不能当作架构上限。训练器现已分别记录 optimizer steps、examples、答案 token、effective epochs、micro batch 与梯度累积，并把固定分层验证、10 组 derangement 及多答案 canonical/random 监督落盘。完整的租前归因顺序见 [`docs/ablation-protocol.md`](docs/ablation-protocol.md)。在真正的多样本 forward 与真实 step-time 基准完成前，不再把“global batch 64”直接换算成租期时长。

- `moonvit_glue.metrics`：纯 Python 指标实现，无 torch 依赖，可在任何机器上验证。
- `moonvit_glue.grounding_contract` / `grounding_evaluation`：严格 click parser、双分母距离与命中率、官方 click-in-box、paired bootstrap 和七条件评分。
- `tools/build_screenspot_contract.py`：从 SHA 锁定的三份 parquet 冻结 50/full manifests、图片 SHA、类别与错图 derangement。
- `tools/score_community_grounding.py`：拒绝顺序/ID 漂移，保存每条件逐条 scores、breakdowns、四个 paired contrasts 与 artifact manifest。
- `tools/fetch_eval_data.py`：按 pin 的 revision 拉取 TextVQA / DocVQA / OCRBench / ScreenSpot，写 JSONL 与 `MANIFEST.json`（resolved sha + 文件 sha256）。
- `tools/eval_vlm.py`：生成式评分（`--blind` 输出无图基线）；`--shuffle-loss` 模式给出真图 vs 随机图的 teacher-forced loss 差，是训练前最便宜的信号检验。
- `tools/eval_stock_vlm.py`：用官方 processor/chat template 跑未修改的小型 VLM 对照组。对本地权重可传 `--weight-manifest` 做加载前逐分片 SHA-256 校验；`--max-image-side` 必须与被比较的 projector 评测保持一致，避免原图分辨率差异或高分辨率 OOM 污染结论。

```bash
python tools/fetch_eval_data.py --dataset screenspot --limit 200
python tools/eval_vlm.py --text-model <model> --projector <dir> \
    --data data/eval/textvqa.jsonl --limit 100 --blind
python tools/eval_vlm.py --text-model <model> --random-projector \
    --placeholder-token-id <id> --data data/eval/textvqa.jsonl --shuffle-loss --limit 50
python tools/eval_stock_vlm.py --model /models/Qwen3.5-4B \
    --weight-manifest configs/qwen3.5-4b-hf-sha256.json \
    --data data/eval_v1/textvqa --record-slice even --blind --max-image-side 1024
```

2026-08-04 修复后的原生 Qwen3.5-4B 阳性对照（selection 半侧，vision / blind）：TextVQA 0.820 / 0.031，DocVQA 0.926 / 0.071，OCRBench 0.900 / 0，ScreenSpot accuracy 0.760 / 0.010，MMMU-Pro 0.300 / 0.280。前三类感知任务和 grounding 证明评测链路有效；MMMU-Pro 仅 +0.020 的图像增益显示其 raw score 大多来自语言/选项先验。再次强调：这些是原生 VLM 数字，不是 MoonViT-V2 projector 或 DeepSeek 数字。

预期锚点：0xSero 的 GLM-5.2 projector-only checkpoint 报告 parse rate 92%、Accuracy@50 4.3%、平均点击误差约 564/999。第一阶段目标是 DeepSeek 路径达到同量级信号，不是成熟 VLM。

## 本地运行边界

- 只跑 MoonViT-V2：普通 CPU 或单张消费 GPU 可以；建议至少 8 GB VRAM，CPU 建议 16 GB RAM 以上。
- MoonViT-V2 + 0.5B–3B 文本模型：单张 12–24 GB GPU 很适合开发和训练胶水。
- MoonViT-V2 + 完整 DeepSeek-V4-Flash-0731：视觉塔约 0.8 GB 的权重几乎不是问题；约 160 GB 级的 DeepSeek 权重和运行 buffer 才是限制。
- 单张 24/32 GB GPU 无法纯 GPU 跑完整 0731。高内存 CPU offload/3-bit GGUF 可以极慢推理，但 GGUF 路径不适合 projector 训练。
- 实用的完整 GPU 推理/训练验证仍应按同机 4×48 GB 起步、4×80 GB 更稳来规划。

还要限制视觉 token 数。392×392 图像约产生 196 个合并后 token；896×896 约 1024 个。视觉塔约 401M 参数，但高分辨率产生的数千视觉 token 会显著放大冻结 LLM 的激活显存。第一阶段建议上限 1024 个视觉 token/图。

2026-08-02 的 Vast 只读快照中，4×A100 PCIe 80 GB 约 $4.00/小时但无 NVLink；4×A100 SXM4 80 GB 约 $6.94/小时且报告 300 GB/s NVLink；4×H100 SXM 80 GB 约 $10.75/小时。价格与可用性会变化，详见 Typst 报告；项目没有创建或租用任何实例。

## 2026-08-08：公开 GLM-5.2V / DeepSeek-V4 Vision 核对

[Baseten GLM-5.2V](https://huggingface.co/baseten/GLM-5.2-Vision-NVFP4) 已公开一条真正可部署的 VLM 路线：Kimi K2.6 MoonViT-3d（27 层、1152-d）接入冻结 GLM-5.2，只训练约 49.5M PatchMerger projector；[训练文章](https://www.baseten.co/blog/glm-52-with-vision/) 给出 66k 图文 QA、global batch 64、constant `5e-4`、两 epoch，并报告 MMMU-Pro 55%。这些是社区 VLM 证据，但不是本仓库 ScreenSpot 的 vision/blind/shuffled paired 因果结果。

[WebBrain DeepSeek-V4-Flash-0731-Vision](https://huggingface.co/webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4) 在 DeepSeek 专用打包上领先于本仓库：公开了 Kimi tower、40,119,040 参数的 4096-wide projector 和 SGLang bridge。不过其 manifest 明确 `gpu_validated_for_this_0731_package=false`，没有公开 ScreenSpot、TextVQA、DocVQA、OCRBench 或 blind/shuffled 结果。它的 projector 形状与我们的 K26 projector 相同，真正不同的是 DeepSeek 路由：词表外 image sentinel `129280`，prefill 时循环替换成固定 64-ID palette；本仓库当前 merge 尚未实现该 palette bridge。完整审计事实和 SHA 保存在 `experiments/external_model_audits/glm52v_webbrain_deepseek_20260808.json`。

因此，K26 视觉塔已能产生有限特征，不等于我们的 Qwen projector 已识图；下一步优先实现可选 DeepSeek-only palette-cycle/OOV-sentinel bridge，并在本地 tiny Gate D 验证后再决定真实 0731 运行。不要把外部 WebBrain 权重当作 Qwen 能力结果，也不自动下载其完整模型或租用付费 GPU。
