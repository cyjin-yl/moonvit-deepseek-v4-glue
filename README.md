# MoonViT-V2 (Kimi K3) → DeepSeek-V4 glue prototype

这是一个训练前的接口与权重合同原型。当前主线视觉塔是从 Kimi K3 抽取的 MoonViT-V2（MoonViT3d）；旧的 `MoonViT-SO-400M` 路径仅保留作工程回归和历史对照。项目把视觉塔、projector、文本主干保持为三个独立 checkpoint，并实现：

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
| Projector | `LN(1024) → flatten(4×1024) → 4096 → GELU → 4096` | 训练 |
| Projector 参数 | 33,564,672 | 训练 |
| DeepSeek | `deepseek-ai/DeepSeek-V4-Flash-0731`，hidden size 4096 | 冻结 |
| Placeholder | 已有 `<｜image｜>`，token ID 129279 | 不扩 vocab |

主线配置在 `configs/deepseek-v4-flash-0731-projector-moonvit-v2.json`。Projector 权重单独保存为 `projector.safetensors`，不会复制或修改 MoonViT-V2/DeepSeek 权重。

V2 权重由 Kimi K3 的单个视觉分片抽取并 strict-load 验证，仓库内 vendor 的 vision-only 代码不依赖完整 K3 模型。`MoonViT-SO-400M`（V1，1152 维）仍可用于轻量 smoke，但它与 V2 的特征分布和 projector 形状不同；两者的 projector checkpoint 绝不能混用。

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

## 加载正式 DeepSeek 权重

训练出 projector 后：

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

现有 Gate B 的 `Qwen2.5-0.5B-Instruct` 虽然是纯文本模型，但 0.5B 容量会压低 OCR、推理和格式遵从上限，也不能代表 DeepSeek Hash-MoE 的优化难度。因此其 shuffle/随机/blind 差异只作工程与信号证据，绝对分数和收敛速度都不得外推 0731。尚未完成的中间尺度对照应使用无 `vision_config` 的纯文本约 3B 主干，在相同数据、seen-record 数、分辨率、scratch projector 和 selection 评测下复测；原生 Qwen2.5-VL/Qwen3.5 VLM 不属于该实验。

- `moonvit_glue.metrics`：纯 Python 指标实现，无 torch 依赖，可在任何机器上验证。
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
