# Package 14：sentinel 功效与固定开销标定

## 研究问题

Package 13 证明固定预算内 preventive replay 有效，也测得完整三-state sentinel 需要 878.5 秒。Package 14 检验多小的 teacher-forced paired-preference 子集仍能稳定识别 count 遗忘，并把实测 V100 时延换算成固定 5%/10% 评测开销下的 checkpoint 间隔。

## 预注册设计

- 源数据固定为 Package 13 的 21,600 条 sentinel preference rows，SHA `0c3eec4f…11a8`；只复用 `exchange-step50` 与 `ordinary-step75` 的 `vision` 条件。
- 每任务候选为 8/16/25/50/100 个完整 counterfactual pairs；每个候选做 200 次确定性子采样，每个 task contrast 做 2,000 次 paired bootstrap。
- 触发规则保持 `drop ≥ 0.10` 且 `current-minus-reference ci95_high < 0`，最多两个任务；完整数据的已知目标是 `count`。
- Tiny 必须同时满足：count recall ≥0.95 且 Wilson 下界 ≥0.90；精确 count-only 决策率 ≥0.90 且 Wilson 下界 ≥0.85；familywise false trigger ≤0.05 且 Wilson 上界 ≤0.10。
- Medium 固定为 Tiny 之后的下一档。Tiny/Medium 均在 V100 上精确比较 step50/75 两个 state，只跑 `vision` teacher-forced，generation 关闭，batch 32，各重复三次。
- 功效判据和 timing 公式在结果前由提交 `d7eb18e…` 冻结并 push。

## 功效结果

| pairs/task | count recall | exact count-only | familywise false trigger | 通过 |
|---:|---:|---:|---:|---:|
| 8 | 0.375 | 0.360 | 0.015 | 否 |
| 16 | 0.760 | 0.720 | 0.045 | 否 |
| 25 | 0.975 | 0.935 | 0.040 | 是 |
| 50 | 1.000 | 0.965 | 0.035 | 是 |
| 100 | 1.000 | 1.000 | 0.000 | 是 |

Tiny 因此为 25 pairs/task，即每个 state 300 records；其 count recall Wilson 95% CI 为 `[0.943, 0.989]`，exact decision CI 为 `[0.892, 0.962]`，false-trigger CI 为 `[0.020, 0.077]`。25-pair 的 false trigger 全来自 shape，比例 0.04。Medium 为 50 pairs/task。

## V100 timing 与预算换算

| profile | pairs/task | teacher rows/repeat | teacher median | end-to-end median |
|---|---:|---:|---:|---:|
| Tiny | 25 | 600 | 22.501 s | 31.215 s |
| Medium | 50 | 1,200 | 43.881 s | 52.537 s |

六次运行均无 OOM/NaN，峰值显存 6.886 GB；同一 profile 的三次 preference rows SHA 完全一致。Tiny/Medium 的重算结果逐行精确等于 Package 13 完整 raw rows 的相同 `(state,id)` 子集。

以 Package 13 fixed replay 的 median step time `0.8989666 s` 代入 `t_eval / (K*t_step + t_eval) <= overhead`：

- 模型常驻、只计同步 teacher compute：Tiny 在 5%/10% 上限下至少间隔 476/226 steps，操作配置向上取 512/256。
- 每次另起进程、计入加载与 setup：Tiny 至少间隔 660/313 steps，操作配置向上取 1024/512。
- Medium 模型常驻至少间隔 928/440 steps，操作配置向上取 1024/512。

目标 DeepSeek runtime 的 step time 不同，正式配置必须用 Gate D 短校准重算 K，不直接复制 V100 的绝对步数。

## 假设更新与训练动作

- **支持：** 25 pairs/task 已能在预注册 Wilson 护栏下复现 count trigger；全量 200 pairs/task 无需进入高频在线路径。
- **反驳：** 8/16 pairs/task 不能可靠检测 count 遗忘，recall 只有 0.375/0.760。
- **反驳：** 当前 Tiny sentinel 不适合每 25 个小模型训练 step 同步执行；其 teacher cost 已接近一个 25-step 训练窗口。
- **动作：** 固定预算训练默认使用 preventive replay；Tiny sentinel 作为稀疏 checkpoint audit。模型常驻时以动态公式控制在 5% 或 10% 开销内，Medium 只用于 Tiny 告警后的确认。
- 训练 optimizer steps 与 examples seen 均未增加，final odd half 未评分，未使用付费资源。

## 独立验证

- Package verifier 重读 1,000 个 trial、6,000 个 task-trial 和六次 timing 的 5,400 条 preference rows，并核对 profile 内重复 SHA 与 Package 13 相同 `(state,id)` 的逐行复现。
- artifact manifest 声明 51 个文件、3,708,513 bytes；独立 SHA-256 重算 51/51 一致。
- 完整仓库测试 240/240 全绿。报告编译为 49 页，包 13 收尾与包 14/Gate D 状态第 35–38 页通过渲染目检。

## 下一项本地实验

replay 支线到此收束：正式候选默认采用当前已验证的 fixed-budget preventive replay，Tiny 只作稀疏 checkpoint audit，不再继续 trigger、Fisher、EWC 或 replay 剂量扩展，除非真实视觉合同显示它们会直接改变正式训练配方。

下一项改为纯文本 `Qwen/Qwen2.5-3B-Instruct` 的固定真实视觉合同。先冻结模型与数据 revision、权重/tokenizer SHA-256、ScreenSpot 50 与完整公共 ScreenSpot manifest、严格 click parser、vision/blind/shuffled/random-projector 控制、生成配置和 examples-seen 节点，再运行 MoonViT-V2 + projector-only 最小基线。0.5B/0.6B 结果只保留为容量受限的历史代理证据；3B 仍是 DeepSeek-V4-Flash-0731 的低成本代理，不是项目终点。
