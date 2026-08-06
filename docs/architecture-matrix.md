# MoonViT projector 架构矩阵与证据边界

更新日期：2026-08-06

这份文档是架构身份索引。它回答“哪一座视觉塔、哪一种
PatchMerger、哪一个输出边界正在被比较”。固定评测规则仍以
[`qwen2.5-3b-community-eval-contract.md`](qwen2.5-3b-community-eval-contract.md)
为准，当前结果和 Gate D 状态以
[`current-status.md`](current-status.md) 为准。任何一行的权重、cache 和
checkpoint 都不能跨架构复用。

## 1. 共同边界

最终目标是：

```text
MoonViT-V2 → projector 输出 4096 → DeepSeek-V4-Flash-0731 hidden size 4096
```

Qwen2.5-3B 代理保留同一个 4096 维 projector 边界，再通过无参数、冻结的
4096 → 2048 fixed receiver 进入纯文本 Qwen。Qwen 和 MoonViT 均冻结，正式
训练只更新 projector。这个 receiver 只属于代理运行时，迁移到 DeepSeek 时
丢弃。

V1 和 V2 使用相同的样本 ID、图像字节、prompt、预算、parser、生成配置和
health schedule。两座视觉塔的原生预处理器不同；比较时锁定相同的原图、
max-side、视觉 token 上限，并在 cache manifest 中记录每座塔实际的 token
数量和预处理器哈希。V1/V2 的 feature cache、projector 初始化和结果目录
始终分开。

## 2. 架构身份

| 架构 ID | 视觉来源与输出 | Projector 结构 | 输出边界与参数量 | 当前角色/状态 | 迁移判断 |
|---|---|---|---|---|---|
| `community_glm52_reference` | `baseten/GLM-5.2-Vision-NVFP4` 页面声明来自 Kimi-K2.6 MoonViT-3d；27 层、1152 维、2×2 merge | affine pre-LayerNorm；`Linear(4608,4608,bias=True)` → GELU → `Linear(4608,6144,bias=True)`；公开 projector 约 49.5M | 6144；公开文件 `mm_projector.safetensors`，99,117,136 bytes，SHA-256 `e7c6ce8c27424f292e708e7bbb48ade57ea9f1aaddd28bd6a1020a860d9db80c` | 社区结构与尺度参考。该 HF 卡是 community repack，不代表官方 zai 发布；权重不能直接装入 Qwen 或 DeepSeek | `reference_only` |
| `local_v1_family_proxy` | `moonshotai/MoonViT-SO-400M`，resolved revision `a889d399ff2306053e4e28d499d3b8f97d3e5007`；1152 维、4-way merge | `legacy_pre_norm`：affine pre-LayerNorm；`4608 → 4608 → 4096`，两层 Linear 带 bias；输出后进入共享 fixed receiver | canonical 4096；V1 DeepSeek-width projector 40,119,040 参数；Qwen 使用同一 4096 → 2048 receiver | K2.6/MoonViT-3d 家族的可运行代理，已注册 matched Qwen3B control，benchmark 尚未完成。SO-400M 与 K2.6 视觉塔没有 byte-identical 证明 | `transferable_with_runtime_validation` |
| `local_v2_legacy` | 从 Kimi-K3 抽取的 MoonViT-V2；`[tokens,4,1024]`，vision width 1024 | `legacy_pre_norm`：affine pre-LayerNorm；带 bias 的 `4096 → 4096 → 4096` MLP | canonical 4096；33,564,672 参数；配置 `configs/deepseek-v4-flash-0731-projector-moonvit-v2.json` | Package 15E–15R 的真实训练实现。15P geometry、15Q output norm、15R residual screen 的失败结论只适用于这一行；旧 checkpoint 不得成为 `previous_best` 或 DeepSeek 候选 | `historical_failure_only` |
| `local_v2_exact_k3` | 同一 K3/MoonViT-V2 tower；`[tokens,4,1024]` | vendored `PatchMergerMLPV2`：无视觉侧 pre-norm；bias-free `4096 → 4096 → 4096` MLP；trainable post-RMSNorm | canonical 4096；33,558,528 参数；配置 `configs/deepseek-v4-flash-0731-projector-moonvit-v2-k3-exact.json` 与 `configs/qwen2.5-3b-projector-moonvit-v2-k3-exact.json` | state/forward parity 已有单测；尚无 Qwen3B 训练或能力结果。它是当前 V2 主实验候选 | `transferable_with_runtime_validation` |
| `qwen35_native_diagnostic` | 原生 Qwen3.5-4B VLM，自带视觉塔和多模态对齐 | 不使用本仓库 projector | 原生模型边界 | 仅验证数据、processor、生成器和 scorer 的阳性诊断；不进入 projector 排名，也不代表 DeepSeek 迁移能力 | `diagnostic_only` |
| `qwen35_stripped_4b` | Qwen3.5-4B 的视觉预训练语言接收器；原生 visual/merger 绕过 | 本仓库 exact V2 projector + 固定 4096→2560 grouped signed adapter | receiver 2560；BF16/16 token finite，FP16 full-token nonfinite | 接收器先验诊断；`vision−shuffle=-0.0597`，不能称能力结果 | `transferable_with_runtime_validation` |
| `qwen35_stripped_9b` | Qwen3.5-9B 的视觉预训练语言接收器；原生 visual/merger 绕过 | 本仓库 exact V2 projector + 4096 identity | receiver 4096；BF16/16/32/64/128/240 token finite | 16-token 单样本 `+0.6265`，长度 sweep `+0.1781/-0.4574/+0.2881/-0.8842`；不进入 projector leaderboard | `transferable_with_runtime_validation` |
| `qwen25_text_7b` | `Qwen/Qwen2.5-7B-Instruct` 纯文本，revision `a09a35458c702b33eeacc393d103063234e8bc28` | 本仓库 exact V2 projector + 固定 4096→3584 grouped signed adapter | receiver 3584；FP16/16、240 token finite | `vision−shuffle=-1.0731/-0.8335`；容量对照，尚无能力结果 | `transferable_with_runtime_validation` |

### 2.1 来源锁定

| 来源 | resolved revision | 关键身份 |
|---|---|---|
| GLM-5.2 community reference | `04c0a6f198afc00eff109e5cd7d0a1bbae9e9085` | `baseten/GLM-5.2-Vision-NVFP4`；projector tensor header 与网页配置已核对 |
| Kimi-K2.6 reference | `7eb5002f6aadc958aed6a9177b7ed26bb94011bb` | MoonViT-3d 27 层、1152 维；语言宽度 7168 |
| V1 family proxy | `a889d399ff2306053e4e28d499d3b8f97d3e5007` | `moonshotai/MoonViT-SO-400M`；model 文件 SHA `a375216ea19430d70c8f68d4d205fae011f1b2ad9a124238bcd7006324e1fdde` |
| K3/MoonViT-V2 mirror | `c5acb88baf02a94e68ca8a225f7eebee152c1fc6` | `AI4Industry/MoonViT-V2`；结构参考 |
| 本地 K3 抽取塔 | 见 Qwen contract 与 vision manifest | 权重 SHA `01436a95939965185bb853ddf984e09c00f597b9c2f6708ba302ffbaf75ced24` |

完整来源、文件哈希和 tensor shapes 见
`experiments/qwen3b_community_eval_20260805/community_architecture_audit_v1/COMMUNITY_SOURCES.json`。

## 3. 结果解释规则

1. `local_v2_legacy` 的塌缩记录不能证明 `local_v2_exact_k3` 也会塌缩。两者必须以独立 step0、独立 cache 和相同评测合同重跑。
2. V1 control 的目的，是检验“社区使用的 1152 维/K2.6-lineage 家族是否比当前 V2 表示更容易被纯文本 receiver 读取”。它回答版本/表示假设，不提供社区私有 50 条样本的复现声明。
3. V1 通过、exact V2 失败时，优先检查视觉塔版本、预处理、token 压缩和 projector 结构；两者都失败时，3B 纯文本 receiver、监督目标或优化动力学成为更强候选解释。
4. 两者都通过 health 但 causal benchmark 仍失败时，继续检查训练数据、answer format、解码和 receiver 接口。health 通过只说明表示没有立刻丢失图像差异。
5. Qwen3.5 native diagnostic 的高分只说明评测链路能产生阳性结果，不能替代 V1/V2 的 `vision − blind`、`vision − shuffled` 和 `trained − random_projector` 证据。
6. Qwen3.5 stripped-native 运行完全绕过原生视觉路径；9B 的正 margin 只能说明视觉预训练 receiver prior 的单样本可读性，不能归因给 projector，也不能进入 Qwen2.5 社区排行榜。
7. Qwen2.5-7B 是纯文本容量 control；它的 finite gradient 和负 margin 反驳“只扩大纯文本模型就会自然获得视觉因果”。

## 4. 下一次架构 screen

按以下顺序执行：

1. 对 `local_v2_exact_k3` 做 CPU state/forward parity、parameter-count、save/load 和 cache-shape smoke。
2. 用 pinned `MoonViT-SO-400M` 生成同一 probe/ScreenSpot cache，逐条记录 1152 维输出、processor revision、图像 SHA 和 token 数。
3. 已完成 V1/V2 高频 health screen、接收器容量审计和 Qwen3.5 4B/9B 16-token stripped gate；这些结果不能直接晋升为能力。
4. 当前下一项是 Qwen3.5-9B BF16 的 32/64/128/240 token 稳定性短筛选；随后做 Qwen2.5-7B 纯文本 matched control。14B 只保留 NF4/FP4 input-gradient gate，现有 GGUF 只可推理。
5. 只有健康和因果筛选通过的臂才运行完整 ScreenSpot、TextVQA、DocVQA、OCRBench、synthetic 和 language-retention 合同。所有候选仍需七条件生成与 paired bootstrap。
6. 结果写入独立 architecture-control manifest；V1/V2、纯文本 Qwen 和 stripped-native Qwen3.5 的 checkpoint 不能互相充当 `previous_best`。

## 5. 迁移门槛

只有完整合同下的 causal visual gain、语言保持、固定预算和可恢复 checkpoint
同时成立，架构方法才可标为 `directly_transferable`。V1 family proxy 即使在
Qwen3B 上优于 V2，也至少需要 DeepSeek runtime validation；任何只依赖 Qwen
chat template、Qwen 原生视觉模块或 direct-2048 draft 的结果都不进入 DeepSeek
候选列表。

Gate D 当前仍为 `NO-GO`。该矩阵不授权租机、不授权下载完整 0731，也不把
社区 projector 权重当作本地可加载 checkpoint。
