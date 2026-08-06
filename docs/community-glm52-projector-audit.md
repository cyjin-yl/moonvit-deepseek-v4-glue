# GLM‑5.2V / MoonViT projector architecture audit

审计冻结时间：2026-08-06。网页资料使用 Codex 内置浏览器打开的 Hugging Face 页面；文件 revision、SHA-256 和 tensor header 另存于 `experiments/qwen3b_community_eval_20260805/community_architecture_audit_v1/COMMUNITY_SOURCES.json`。

## 先回答结论

社区 GLM‑5.2V 确实把 Kimi‑K2.6 的 MoonViT‑3d 视觉塔接到了 GLM‑5.2。网页明确写出：视觉塔 27 层、1152 维，视觉塔和语言主干冻结，唯一新训练部分是约 49.5M 参数的 PatchMerger projector。这个 projector 的公开权重文件是 `mm_projector.safetensors`，SHA‑256 为 `e7c6ce8c…d9db80c`，张量头显示：

`LayerNorm(1152) → flatten(2×2) → Linear(4608,4608) → GELU → Linear(4608,6144)`，两层 Linear 都带 bias。

“从 K2.6 中提取”需要拆成两件事理解：

1. 视觉塔来自 K2.6，并且社区卡片声称与上游塔保持 byte-identical。
2. projector 是为 GLM‑5.2 输出宽度重新训练的组件，不是从 K2.6 直接复制一份可用权重。K2.6 自己的目标宽度是 7168，GLM‑5.2 的目标宽度是 6144。

## 对我们当前代码的影响

我们现在的 `PatchMergerProjector` 在 `[tokens,4,1024]` 的 V2 塔上使用：

`pre LayerNorm(affine) → flatten(4096) → Linear(bias) → GELU → Linear(bias)`。

这和 GLM‑5.2V/K2.6 的 V1-style PatchMerger 家族相似，和 Kimi‑K3/MoonViT‑V2 公开代码中的 `PatchMergerMLPV2` 不同。后者是：

`flatten(4096) → Linear(bias=false) → GELU → Linear(bias=false) → trainable post RMSNorm`。

因此，Package 15P 的塌缩结论只适用于我们当时实际训练的 projector 实现。它证明了这份实现会在早期塌缩，不能证明官方 K3 V2 projector 也会塌缩。这个边界已经写入机器可读审计，后续报告不得混写。

## 下一条最短路径

1. 在不改变 Qwen2.5‑3B、数据顺序、图像预处理、预算和健康 guards 的前提下，加入精确的 K3 `PatchMergerMLPV2` 变体，并做 CPU state/forward 对照测试。
2. 用公开 `moonshotai/MoonViT-SO-400M`（V1，1152 维）跑一条匹配的 Qwen2.5‑3B projector-only benchmark。这个实验直接回答社区 GLM‑5.2V 所用视觉塔家族能否在我们的代理上工作。
3. 只有 V1 与精确 V2 的健康轨迹通过前 100 steps，才进入完整 ScreenSpot、TextVQA、DocVQA 和 OCRBench 合同；legacy-V2 的旧 checkpoint 不再作为 `previous_best`。
4. GLM‑5.2V 的 6144 输出 projector 权重保留为公开结构和初始化尺度参考。它不能直接插进 Qwen 2048 或 DeepSeek 4096，因此不会被误写成当前项目的可加载权重。

当前 Gate D 仍为 NO‑GO：真实工程链路和梯度测试通过，社区结构对齐与可重复视觉能力证据尚未通过。

参考页面：[GLM‑5.2‑Vision‑NVFP4](https://huggingface.co/baseten/GLM-5.2-Vision-NVFP4)、[Kimi‑K2.6](https://huggingface.co/moonshotai/Kimi-K2.6)、[MoonViT‑SO‑400M](https://huggingface.co/moonshotai/MoonViT-SO-400M)、[MoonViT‑V2 镜像](https://huggingface.co/AI4Industry/MoonViT-V2)。
