# Qwen3.5 stripped-native capacity controls

这些运行只把 Qwen3.5 的原生视觉塔、merger 和视觉路径绕开，冻结其已经经过视觉训练的语言接收器，用同一份 MoonViT-V2 cache 和 canonical 4096 projector 注入视觉 token。`native_vision_forward_calls=0` 是硬检查。它们是接收器先验与输入梯度诊断，不能写成 ScreenSpot、TextVQA 或 DeepSeek 视觉能力结果。

| run | receiver | dtype / visual tokens | 结果 | 解释 |
|---|---|---|---|---|
| `qwen35_4b_stripped_receiver_bf16_16tok_20260806` | 2560，固定 grouped signed adapter | BF16 / 16 | finite，`vision−shuffle=-0.0597` | 能回传梯度，但没有局部因果优势 |
| `qwen35_4b_stripped_receiver_fp16_20260806` | 2560，固定 grouped signed adapter | FP16 / full | NaN/Inf after the first update | 数值失败，不能继续用该 dtype/长度 |
| `qwen35_9b_stripped_receiver_bf16_16tok_20260806` | 4096，identity | BF16 / 16 | finite，`vision−shuffle=+0.6265` | 首个正的 receiver-prior 局部信号；仍非能力结果 |
| `qwen35_9b_stripped_receiver_bf16_{32,64,128,240}tok_20260806` | 4096，identity | BF16 / 32,64,128,240 | finite；margin `+0.1781/-0.4574/+0.2881/-0.8842` | 长度敏感，不能只凭 16 token 外推 |
| `qwen25_7b_stripped_receiver_fp16_{16,240}tok_20260806` | 3584，固定 grouped signed adapter | FP16 / 16,240 | finite；margin `-1.0731/-0.8335` | 纯文本容量增加没有自动产生局部视觉因果 |

所有成功 run 都使用同一 exact-K3 V2 projector step0、同一 4k cache manifest `09df2f2cbd502cdf84422eabd499922e7c0cda4fb3368eb0c7cbac4ac21cc023`、单样本配对（正确/打乱/盲）和一步 projector update。原始目录仍在工作站：

- 4B：`/run/media/ezra/13D010B6FDBC1A06/data/qwen3b_contract/capacity_controls/qwen35_4b_stripped_receiver_bf16_16tok_20260806`
- 9B：`/run/media/ezra/13D010B6FDBC1A06/data/qwen3b_contract/capacity_controls/qwen35_9b_stripped_receiver_bf16_16tok_20260806`
- 4B FP16 failure：`/run/media/ezra/13D010B6FDBC1A06/data/qwen3b_contract/capacity_controls/failures/qwen35_4b_stripped_receiver_fp16_20260806`
- 4B full-token BF16 timeout：`/run/media/ezra/13D010B6FDBC1A06/data/qwen3b_contract/capacity_controls/failures/attempt02_bf16_full_tokens_timeout_20260806`

9B token sweep 说明有限梯度在 V100 上可运行，但接收器 margin 对 token 长度高度敏感；7B 的 16/240 token matched control 均为负。两组都只有一个样本、一步更新，仍不进入 Qwen2.5 社区排行榜，也没有 ScreenSpot 能力结论。下一步应优先把 9B 的 token 顺序/压缩方式和多样 probe 扩展冻结，再决定是否值得进入真实 benchmark。

8-sample receiver probe（同一 derangement、同一 random-projector seed）补充了这条边界：

| 9B BF16 | vision−shuffle mean ± std | vision−blind mean ± std | 结论 |
|---|---:|---:|---|
| 16 token | `+0.0447 ± 0.3729` | `+0.1993 ± 0.2142` | 有 token 影响，正确图优势很弱 |
| 240 token | `-0.0748 ± 0.4520` | `+0.6753 ± 0.3335` | 有 token 影响，正确图与打乱图仍不可分 |

这比单样本结果更接近真实判断：9B 接收器能对外部视觉 token 产生响应，但目前没有稳定的图像归因。`probe_metrics.jsonl` 保留每个样本与 random-projector log-prob；仍然不能称为 projector capability。

作为附加诊断，V2 的 `vision−random_projector` 均值为 `-0.1097`（16 token）和 `+0.2247`（240 token）；V1 在 240 token 为 `-0.1571`。它说明 projector 的固定 step0 输出在长 token 条件下可能比随机映射更接近接收器的偏好，但没有解决正确图/打乱图的 paired attribution，不能称训练改进。

同一 8-sample/240-token screen 换成 V1（MoonViT-SO-400M，1152 维）得到 `vision−shuffle=+0.0620 ± 0.4185`、`vision−blind=+0.3780 ± 0.1962`。V2 对照为 `-0.0748 ± 0.4520`、`+0.6753 ± 0.3335`。V1 略高但差异被样本方差覆盖；目前不能写成 V1 优于 V2，也不能把版本差异当作主要故障解释。

Qwen3.5 原生 3D mRoPE 的位置诊断（V2、8 samples、240 tokens）为 `vision−shuffle=-0.0375 ± 0.4537`、`vision−blind=+0.6680 ± 0.2896`，与普通连续位置几乎相同。位置规则不是当前 paired attribution gap 的主解释；该分支标记为 `qwen_specific_not_transferable`。

9B projector-only backward 还做了两次小训练修复尝试：240-token 首次 forward 触发 NVML allocator assert；修复 health graph retention 并缩到 16 tokens 后，仍在 25.88 GiB allocated / 41 MiB free 时 OOM。9B 在本机可做 forward/input-gradient 和多 probe inference gate，当前不能做多样样本 projector 训练；本地训练主线回到 3B/7B。

Qwen2.5-7B 的 3-step CE-only projector screen 完成且 finite：CE `0.2381→0.0094`，projector RMS `0.99775→0.99779`，between-image RMS `0.4064→0.4055`，但 `vision−shuffle` `+0.0333→-0.1027`。这是一条健康但错误方向的训练轨迹，说明 7B 也会用坐标/答案先验吸收 CE，不能把 loss 降低写成视觉 grounding 改进。
