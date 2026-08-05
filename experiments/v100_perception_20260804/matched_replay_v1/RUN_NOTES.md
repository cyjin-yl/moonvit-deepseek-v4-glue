# Package 13：fixed-budget matched replay

## 研究问题

Package 12 发现 balanced continuation 从 step 50 到 step 100 出现明显能力交换：count paired preference 从 0.380 降到 0.100，shape 从 0.735 降到 0.435。Package 13 检验在总训练预算固定为 50 optimizer steps、1,200 examples 时，历史样本槽位重分配能否减少遗忘。

## 冻结设计

- 共同起点：`balanced_compare_projector_v1/checkpoints/step-000050`，同时恢复 step-50 AdamW。
- ordinary：沿原训练顺序完成 steps 51–100，六任务各 200 examples。
- fixed replay：每个 25-step 窗口给 count 与 shape 各重放 10 个完整 counterfactual pair；整段分配为 `180/180/240/180/240/180`。
- triggered replay：steps 51–75 与 ordinary 共用；只有任务从 step 50 到 step 75 下降至少 0.10、且 paired-bootstrap CI 上界小于 0 时，才在 steps 76–100 重放 10 个完整 pair，最多两个任务。
- 三种策略总轨迹均为 50 steps、1,200 examples。fixed 重分配 80 个槽位，triggered 重分配 20 个槽位；额外训练 examples 为 0。
- paired bootstrap 为 2,000 次，重采样单位是完整 pair。final odd half 未评分。

## 预算与复现验证

- `PLAN_VERIFICATION.json` 的 10 项检查全部通过。
- ordinary 逐张量精确复现历史 step 100：6/6 tensors，projector 文件 SHA `05f19079…`，tensor SHA `7b731ffc…`。
- ordinary / fixed 各训练 50 steps、1,200 examples；triggered 只训练 ordinary 之后剩余的 25 steps、600 examples。
- fixed 峰值显存 11.725 GB；所有训练均无 OOM、NaN。
- 完整仓库测试 231/231 通过；报告第 33–35 页完成渲染检查。

## Sentinel

step 50→75 的 ordinary 整体 paired preference 提升 `+0.040`，CI `[+0.0108,+0.0675]`，但 count 从 `0.380` 降到 `0.075`：gap `-0.305`，CI `[-0.365,-0.245]`。只有 count 满足触发规则。shape gap 为 `-0.035`，CI `[-0.090,+0.020]`，未触发。

这一结果直接说明宏平均会掩盖单任务灾难性遗忘，sentinel 必须保留 per-domain/per-task 指标。

## 终点结果

| 策略 | macro preference | count | shape | macro generation |
|---|---:|---:|---:|---:|
| ordinary | 0.5108 | 0.100 | 0.435 | 0.2567 |
| fixed replay | 0.5983 | 0.490 | 0.555 | 0.3567 |
| triggered replay | 0.5358 | 0.275 | 0.435 | 0.2600 |

- fixed 的 count+shape paired-preference 相对 ordinary 提升 `+0.255`，95% CI `[+0.210,+0.300]`。
- fixed donor 四任务合并 gap 为 `+0.00375`，CI `[-0.0125,+0.01875]`，未观察到可辨别 donor 代价。
- fixed 的 count+shape 自由生成 paired accuracy 提升 `+0.120`，CI `[+0.050,+0.190]`。
- triggered 的 count paired preference 提升 `+0.175`，CI `[+0.125,+0.230]`，说明晚介入有真实收益；endpoint 0.275 仍低于 step-50 参考 0.380 的 0.05 恢复带。
- fixed 相对 triggered 的整体 endpoint paired preference 高 `+0.0625`，CI `[+0.0425,+0.0833]`。
- fixed count 恢复并超过 step-50 参考；shape 从 ordinary 的 0.435 提到 0.555，仍未回到 0.735±0.05。

## 假设更新

- **支持 H1：** 固定预算内的预防性、pair 保留 replay 能显著减少高风险任务遗忘。
- **支持 H2：** replay 改善同时出现在 teacher-forced preference 与自由生成，效果不只是输出措辞变化。
- **支持 H3：** per-task sentinel 能识别宏平均掩盖的 count 崩塌。
- **反驳 H4：** 只在显著下降后介入、且仅重分配 20 个槽位，无法在一个 25-step 窗口内完全恢复历史能力。
- **边界：** fixed shape 未完全恢复；当前证据支持 replay 作为训练分配策略，仍不能替代 projector 表示保持目标。

## 实现失败记录

首次分析把 state 字典顺序误当成语义顺序，因 index 工具按 ID 排序而退出。失败日志保存在 `logs/analysis.failed-state-order.log`。修复为精确 state 集合校验后重跑；原始评测未重算或修改。

## 下一项本地实验

下一项先做 sentinel 功效与成本校准：复用本包完整 raw rows，对每任务 8/16/25/50/100 pair 做多 seed 子采样，测 count trigger recall、其他任务 false-trigger 和 CI 稳定性；随后实测 teacher-only Tiny/Medium wall time。全量三-state sentinel 耗时 878.5 s，而 25 个训练 step 的纯 step wall time约 22.5 s，当前全量协议不适合在线监控。冻结最小可用 sentinel 后，再做 5/10/15 pairs/task/window 的 replay 剂量筛选；所有臂继续锁为 1,200 training examples。
