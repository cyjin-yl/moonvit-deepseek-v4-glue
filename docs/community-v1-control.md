# Community GLM-5.2V V1 architecture control

这条对照臂固定使用 `moonshotai/MoonViT-SO-400M` 的 resolved revision
`a889d399ff2306053e4e28d499d3b8f97d3e5007`。它代表社区 GLM-5.2V 采用的
Kimi-K2.6/MoonViT-3d 视觉塔家族，不使用 Qwen 原生视觉模块。公开资料没有证明
SO-400M 与 Kimi-K2.6 内嵌视觉塔 byte-identical，因此它的角色是 family proxy，
不是社区权重复刻。

社区卡片的结构是：

```text
MoonViT-3d [1152] -> 2x2 merge -> LayerNorm(1152)
  -> Linear(4608, 4608, bias=True) -> GELU
  -> Linear(4608, receiver_hidden, bias=True)
```

社区 GLM-5.2V 的 receiver width 是 6144。为了让 Qwen2.5-3B 的 V1/V2
横向比较严格可比，V1 control 的 projector 仍输出 canonical 4096，随后复用
同一份冻结的 4096 -> 2048 fixed receiver；V2 也走这条边界。公开 GLM
projector 权重不能直接装入 Qwen 或 DeepSeek，因为目标宽度不同。

缓存入口现在支持：

```bash
python tools/cache_moonvit_features.py \
  --vision-tower v1 \
  --moonvit-model moonshotai/MoonViT-SO-400M \
  --moonvit-revision a889d399ff2306053e4e28d499d3b8f97d3e5007 \
  --data ... --out ...
```

`--vision-tower v2` 保持原有 Kimi-K3/MoonViT-V2 默认行为。两种塔共享 cache
行格式和后续 projector/评测接口，但 cache、projector 初始化和结果不能混用。

V1 只作为结构对照。V1 与 V2 各自保留原生 processor；比较锁定相同原图字节、
max-side、视觉 token 上限，并在 manifest 中记录 processor hash 与 token-count
分布。只有在相同 canonical 4096 边界、相同 receiver、相同 examples-seen、
相同 prompt 和
ScreenSpot/TextVQA/DocVQA/OCRBench 合同下跑完 matched control，才会判断它是否
比当前 V2 候选更接近社区路线；V1 的高分也不能直接改写 DeepSeek 正式配置。

## Matched Qwen2.5-7B result (2026-08-08)

The preregistered V1 control was run with the same Qwen2.5-7B receiver, real
answer probe, token budget, scale, BF16 dtype, optimizer and receiver adapter as
the exact-K3 V2 control. Both CE-only and λ=`0.5` completed three steps with
finite health. V1 λ=`0.5` reached `vision−shuffle=+0.01145`, CI
`[-0.02580,+0.04766]`, and `vision−blind=+0.52705`, CI
`[+0.39629,+0.67691]`. V1 λ=`0.5` minus V2 λ=`0.5` was `-0.47600`, CI
`[-0.87349,-0.13102]`.

The family proxy therefore demonstrates receiver activation without reliable
correct-image grounding. It does not replace V2 and does not enter the formal
leaderboard. The raw pointer and independent verifier are in the capacity
controls directory named `qwen25_7b_v1_community_screen_20260808`.
