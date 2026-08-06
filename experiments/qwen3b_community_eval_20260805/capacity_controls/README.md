# Qwen3.5 stripped-native capacity controls

这些运行只把 Qwen3.5 的原生视觉塔、merger 和视觉路径绕开，冻结其已经经过视觉训练的语言接收器，用同一份 MoonViT-V2 cache 和 canonical 4096 projector 注入视觉 token。`native_vision_forward_calls=0` 是硬检查。它们是接收器先验与输入梯度诊断，不能写成 ScreenSpot、TextVQA 或 DeepSeek 视觉能力结果。

| run | receiver | dtype / visual tokens | 结果 | 解释 |
|---|---|---|---|---|
| `qwen35_4b_stripped_receiver_bf16_16tok_20260806` | 2560，固定 grouped signed adapter | BF16 / 16 | finite，`vision−shuffle=-0.0597` | 能回传梯度，但没有局部因果优势 |
| `qwen35_4b_stripped_receiver_fp16_20260806` | 2560，固定 grouped signed adapter | FP16 / full | NaN/Inf after the first update | 数值失败，不能继续用该 dtype/长度 |
| `qwen35_9b_stripped_receiver_bf16_16tok_20260806` | 4096，identity | BF16 / 16 | finite，`vision−shuffle=+0.6265` | 首个正的 receiver-prior 局部信号；仍非能力结果 |

所有成功 run 都使用同一 exact-K3 V2 projector step0、同一 4k cache manifest `09df2f2cbd502cdf84422eabd499922e7c0cda4fb3368eb0c7cbac4ac21cc023`、单样本配对（正确/打乱/盲）和一步 projector update。原始目录仍在工作站：

- 4B：`/run/media/ezra/13D010B6FDBC1A06/data/qwen3b_contract/capacity_controls/qwen35_4b_stripped_receiver_bf16_16tok_20260806`
- 9B：`/run/media/ezra/13D010B6FDBC1A06/data/qwen3b_contract/capacity_controls/qwen35_9b_stripped_receiver_bf16_16tok_20260806`
- 4B FP16 failure：`/run/media/ezra/13D010B6FDBC1A06/data/qwen3b_contract/capacity_controls/failures/qwen35_4b_stripped_receiver_fp16_20260806`
- 4B full-token BF16 timeout：`/run/media/ezra/13D010B6FDBC1A06/data/qwen3b_contract/capacity_controls/failures/attempt02_bf16_full_tokens_timeout_20260806`

下一项是 9B BF16 的 32/64/128/240 token 短筛选；只有长度稳定且 `vision−shuffle` 保持正向，才进入 7B 纯文本 matched control 或更大预算。Qwen3.5 不进入 Qwen2.5 社区排行榜。
