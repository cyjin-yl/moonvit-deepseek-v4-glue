# MoonViT → DeepSeek-V4 glue prototype

这是一个训练前的接口与权重合同原型。它把 MoonViT、projector、文本主干保持为三个独立 checkpoint，并实现：

- Kimi 风格的 2×2 PatchMerger projector；
- 一个 image placeholder 扩展为可变数量视觉 token；
- 普通 Hugging Face causal LM 的 `inputs_embeds` 路径；
- DeepSeek-V4 Hash-MoE 所需的 token-ID 路由路径；
- projector 严格 shape 校验和 Safetensors 保存/恢复；
- 冻结 MoonViT 和 LLM、只让梯度进入 projector。

统一环境需要 `transformers>=5.12,<6`。公开的 PyPI 4.57.6 尚未包含 `transformers.models.deepseek_v4`，不能仅根据 checkpoint config 中的旧版本字段选环境。

## 已锁定的权重合同

| 组件 | 来源/形状 | 是否训练 |
|---|---|---|
| MoonViT | `moonshotai/MoonViT-SO-400M`，约 400M，BF16 文件约 834 MB | 冻结 |
| MoonViT 输出 | 每张图 `[视觉 token 数, 4, 1152]` | — |
| Projector | `LN(1152) → flatten(4×1152) → 4608 → GELU → 4096` | 训练 |
| Projector 参数 | 40,119,040 | 训练 |
| DeepSeek | `deepseek-ai/DeepSeek-V4-Flash-0731`，hidden size 4096 | 冻结 |
| Placeholder | 已有 `<｜image｜>`，token ID 129279 | 不扩 vocab |

配置在 `configs/deepseek-v4-flash-0731-projector.json`。Projector 权重单独保存为 `projector.safetensors`，不会复制或修改 MoonViT/DeepSeek 权重。

注意：独立发布的 `MoonViT-SO-400M` 来自 Kimi-VL，适合先跑通工程。Kimi-K2.5/K2.6 使用演进后的 MoonViT-3D。它们的输出合同兼容，但权重表示不保证相同；更换视觉塔后必须重训 projector，不能沿用旧 projector 做质量对比。

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

用真实 MoonViT 接一个很小的纯文本模型：

```bash
python examples/smoke_real_moonvit.py path/to/image.jpg
```

这会下载约 834 MB 的 MoonViT 和 `sshleifer/tiny-gpt2`。随机 projector 不会产生有意义的图像描述；该命令验证的是权重加载、预处理、真实视觉特征 shape 和反向传播。

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
4. LLM 和 MoonViT 参数全部无梯度；
5. 至少连续 20 step 无 OOM/NaN；
6. 保存 projector 后在新进程恢复，输出逐元素一致。

最大的剩余风险是 FP4/FP8 推理 kernel 是否实现 input-gradient，而不是 glue 的 token/shape 逻辑。若原生 kernel 不可微，需要可微的 FP8/BF16 loader 或定制 data-gradient kernel。

## 本地运行边界

- 只跑 MoonViT：普通 CPU 或单张消费 GPU 可以；建议至少 8 GB VRAM，CPU 建议 16 GB RAM 以上。
- MoonViT + 0.5B–3B 文本模型：单张 12–24 GB GPU 很适合开发和训练胶水。
- MoonViT + 完整 DeepSeek-V4-Flash-0731：MoonViT 的约 0.8 GB 权重几乎不是问题；约 160 GB 级的 DeepSeek 权重和运行 buffer 才是限制。
- 单张 24/32 GB GPU 无法纯 GPU 跑完整 0731。高内存 CPU offload/3-bit GGUF 可以极慢推理，但 GGUF 路径不适合 projector 训练。
- 实用的完整 GPU 推理/训练验证仍应按同机 4×48 GB 起步、4×80 GB 更稳来规划。

还要限制视觉 token 数。392×392 图像约产生 196 个合并后 token；896×896 约 1024 个。视觉塔只有 400M，但高分辨率产生的数千视觉 token 会显著放大冻结 LLM 的激活显存。第一阶段建议上限 1024 个视觉 token/图。
