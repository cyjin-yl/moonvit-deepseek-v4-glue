# DeepSeek-V4-Flash 租机训练合同

状态：`local_prerequisites_in_progress`。本文把 V100 证据转成机械执行规则；包 12 的窗口覆盖与包 13 的 fixed-budget replay 已完成，Tiny/Medium sentinel 功效/成本、正式域 replay 配额与真实量化 DGRAD 仍是显式未完成项。只有全部本地前置项冻结、用户明确授权 GPU 架构与费用上限后，才能执行任何付费步骤。

## 1. 证据边界与固定来源

正式 runtime 基线：

- Transformers commit `ddb849abe009d1089e6c691bfc897f27211c663c`（tag `v5.12.1`）；
- `kernels==0.16.0`；
- finegrained-fp8 kernel `b77d2c71fef4ff97e22127058034c1167dca8891`；
- DeepGEMM `559d79fb6994a58b8a15b4b93bf13ccc16edf247`；
- DeepSeek-V4-Flash-0731 model revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`；
- PyTorch/CUDA build 在最小 kernel gate 创建时固定，并写入新的环境 manifest；当前本地参考为 PyTorch `2.10.0+cu128`、CUDA 12.8、Triton 3.6.0。

完整源码事实、文件 SHA-256 与 `hardware_pending` 项见 `docs/dsv4-runtime-source-audit.md`。当前源码审计给出高风险判断，尚未证明 Transformers 实际量化 module 可以把梯度传回 projector。

## 2. 主训练策略

默认且唯一的主线 arm：

- MoonViT-V2 frozen；
- DeepSeek-V4-Flash frozen；
- projector-only；
- projector 参数保留 fp32 master，实际 serving/activation 路径使用 canonical BF16；
- DeepSeek activation dtype 与正式评测保持 BF16；
- AdamW 完整 optimizer state、RNG、数据游标与 examples seen 精确恢复；
- 不默认启用 LoRA；
- 不把 checkpoint averaging 作为抗遗忘方案。

本地包 8 表明顶部 LoRA 能修复局部 shape 能力，同时损害 count/spatial；它保留为诊断 arm。包 10 在同一 training basin 内对 step 50/100 的简单线性 projector 权重平均没有合并两端能力，最佳折中点仍损害 count 与 shape。该否定结论只覆盖当前两个 checkpoint 的线性平均，不外推到所有模型合并方法。

训练开始前必须从 scratch projector 创建 step-0 产物，并立即完成 save/load 与 fp32-master→bf16-serving 对照。任何 warm-start 都属于单独、预注册的次要 arm；不得覆盖 scratch 主线。

## 3. 数据合同

### 3.1 正式 mix

固定 `train_v1` 共 59,198 条：

| 域 | 条数 | 基础采样比例 |
|---|---:|---:|
| TextVQA | 29,252 | 49.41% |
| DocVQA | 17,351 | 29.31% |
| ShowUI | 5,167 | 8.73% |
| art / ordinary visual QA | 7,428 | 12.55% |

基础比例按记录数定义。每条记录的来源、固定 revision、image/content hash、去重报告与 split 身份必须来自已发布 manifest。selection 用于 checkpoint 选择；final split 在最终候选冻结前不得读取。

### 3.2 batch 与窗口覆盖

包 12 以完全相同的 2,400 条记录、初始 projector、optimizer state、seed、batch size、examples seen 和学习率比较了“六任务分层 batch”与“全局随机顺序”。分层在 step 50 更快形成能力（macro preference 0.512 对 0.389），step 100 的 global macro 反超（0.531 对 0.511）；终点 overall stratified−global 为 −0.020，95% paired CI `[−0.0442, 0.0025]`。coordinate 的分层收益与 color/shape 的显著反向损失并存，预注册 verdict 为 `mixed_or_underpowered`。分层终点仍有 6/15 个负任务梯度夹角，global 为 0。

正式规则因此冻结为：

- **固定窗口保证总体领域覆盖**，每个窗口按基础 mix 断言来源计数、最大偏差与每域最小记录数；
- 不强制每个 true batch 六类齐全；
- 阶段 1 短校准可以把分层顺序作为预注册候选，因为它在 step 50 显示更快的 macro/generation 形成；
- 若采用分层候选，后半程必须提高 Tiny/Medium sentinel 频率，专门监测 count/shape 遗忘；
- 块状 curriculum 暂不进入主线，避免在 matched replay 前增加新的 schedule 变量。

状态：`package12_complete_fixed_window_coverage`。完整证据位于 `experiments/v100_perception_20260804/batch_stratification_v1/`。

### 3.3 synthetic、sentinel 与 replay

- 六类 synthetic perception 数据默认只进入 evaluation sentinel，不混入主训练分布。
- 正式训练预算以 optimizer steps 与 examples seen 双重锁定。replay 只允许替换下一固定窗口内的既有槽位，禁止追加 optimizer step、追加训练 example 或在窗口末尾补跑。
- synthetic 数据只有在预注册规则允许且与正式域语义一致时，才可作为小比例诊断 replay；正式主线优先重放原训练域内已审计记录。所有重分配的 source ID、被换出的 donor ID 与窗口计数必须完整记录。
- OCR、count 的本地诊断若显示内部 paired preference 可靠、自由生成仍失败，才启动 Qwen2.5-1.5B 纯文本容量桥接；不得用原生 VLM 替代。
- 包 13 在完全相同的 50 steps / 1,200 examples 下验证了 complete-pair replay：固定臂重分配 80 个槽位，count+shape preference 相对 ordinary 为 `+0.255 [0.210, 0.300]`，donor 合并为 `+0.00375 [−0.0125, 0.01875]`，目标自由生成为 `+0.120 [0.050, 0.190]`。
- 预防性 replay 是阶段 1 的主候选：每 25-step 固定窗口给已知高风险域预留小配额，并从其余域等量换出。包 13 的小模型配额是每目标任务每窗口 10 个完整 pair；该数值用于 V100 机制证据，迁移到正式域前仍需按域规模与 sentinel 功效换算。
- 遗忘触发 fallback 冻结为：任务相对参考 checkpoint 的 paired preference 绝对下降至少 0.10，且 current-minus-reference paired-bootstrap `ci95_high < 0`；最多选择下降最大的两个任务。恢复带为参考值下方 0.05，恢复后下一窗口回到基础分布。
- 包 13 的 late trigger 只给 count 重分配 20 个槽位，endpoint 从 ordinary 0.100 提到 0.275，仍未回到 0.380±0.05。该结果支持触发器的检测能力，也表明 25-step 晚介入配额不足以替代预防性 replay。
- 当前状态：`package13_complete_fixed_preventive_replay_supported`。正式 DeepSeek 域的高风险配额、Tiny/Medium sentinel 分母与成本仍由下一本地包校准。
- 不允许在看到下一窗口结果后手调 task weight。

## 4. Checkpoint 合同

不再使用固定“每 500 步”策略。频率由目标机实测训练 step、保存、上传与 sentinel 成本决定。

每个 checkpoint 必须包含：

- projector fp32 master；
- projector BF16 serving 文件；
- AdamW optimizer state；
- CPU 与全部 CUDA RNG state；
- optimizer step、examples seen、answer tokens seen；
- sampler epoch/order、数据游标、当前 replay 状态；
- config、代码 commit、runtime/model/data revision 与 SHA-256；
- Tiny/Medium sentinel 结果及历史最佳表；
- 原子写入完成标志与独立校验结果。

保留策略：

1. 最近两个完整 checkpoint；
2. macro 历史最佳；
3. worst-task 历史最佳；
4. OCR、spatial/coordinate、ordinary VQA、GUI 各自历史最佳；
5. 所有非支配 Pareto checkpoint；
6. replay 触发前与恢复后的边界 checkpoint。

不得只保留最后一步。删除任何被支配 checkpoint 前，先验证其文件 hash、评测关联与远端上传完成状态。

上传采用异步队列，最多允许两个未确认任务。队列达到上限时暂停训练并排空队列，禁止无限堆积或与训练争抢 GPU 显存。训练不保存完整 DeepSeek 权重。

## 5. 在线视觉哨兵与成本控制

### Tiny Sentinel

- 六任务各 8–16 个固定 complete pairs；
- teacher-forced vision 与 seeded shuffled-image；
- correct margin、strict paired preference、vision-minus-shuffle；
- 不做自由生成；
- 直接使用内存中的模型，不重载 DeepSeek。

### Medium Sentinel

- 六任务各 50 pairs；
- teacher-forced、少量固定自由生成、blind 与 shuffle；
- 少量固定 TextVQA、DocVQA、ScreenSpot selection records；
- 报告 strict paired preference、paired generation、vision-minus-blind、vision-minus-shuffle、margin、loss 与任务级历史遗忘。

### Full Evaluation

完整 selection benchmark 只运行在少量候选 checkpoint。final split 只对预先冻结的最终候选运行一次。

先在 V100、本地 projector 训练链路上测 20 个 optimizer steps、同步 checkpoint save、异步 upload、Tiny、Medium 与 Full 的 wall time；Gate D 再对目标硬件重复。令训练 step 中位时间为 `t_step`，某监控操作时间为 `t_op`，间隔为 `K` steps，则选择最小 `K` 使：

```text
t_tiny / (K_tiny * t_step + t_tiny) <= 0.05
(tiny + medium + checkpoint amortized time) / total training wall time <= 0.10
```

早期短校准采用满足预算的最高频率；能力斜率稳定后拉长间隔；检测到退化或 replay 触发时临时恢复早期频率。若开销超标，增大间隔，保留 sentinel 本身。评测不能在同一组 GPU 上与训练并发抢显存。

包 13 的全量 V100 sentinel 三 state 共耗时 878.5 s；同一训练链路中 25 个 optimizer steps 的纯 step wall time约 22.5 s。全量 2,400-record teacher-forced 加 600-record generation sentinel 显然不能在线高频执行。最终七 state 评测耗时 2,035.8 s。下一本地包必须从现有 raw rows 做 8/16/25/50/100 pair 功效与 false-trigger 稳定性分析，再实测 teacher-only Tiny/Medium；只有能复现 count 触发且满足 5%/10% 开销公式的最小分母可以进入正式配置。

当前 V100 与目标机成本状态：`full_cost_measured_tiny_medium_pending`。任何 Tiny/Medium 频率数字在功效与实测完成前都只是公式输入，不写入最终配置。

## 6. 停训、回滚与 replay

机械判据：

1. loss 下降但 macro、worst-task、vision-minus-blind、vision-minus-shuffle 与 generation 均无改善时，不记为能力进步。
2. 每任务保存 paired preference 与 generation 历史峰值及 paired bootstrap CI。
3. 已知高风险域在阶段 1 使用固定预算内的预防性 replay；当前任务相对历史参考超过 0.10 且 paired CI 支持退化时，下一固定窗口启用或扩大预注册 fallback 配额。
4. 恢复到参考值下方 0.05 以内后，下一窗口回到基础权重；每次触发、重分配 examples、被换出的 donor IDs 与 wall time 写入 raw log。总 optimizer steps 与 examples seen 不得变化。
5. overall 上升且关键任务明显退化时，保留对应 Pareto checkpoint，不用 overall 覆盖任务损失。
6. 连续若干 sentinel 窗口无任何预注册能力改善时提前停止；窗口数在阶段 1 校准后冻结。
7. 包 13 已支持 fixed replay，anchoring 仍失败。只有 replay 剂量/域桥接不能满足恢复带时，才允许进入 per-task gradient-conflict 方法。

最终选择基于多 checkpoint Pareto 前沿。最后一步没有默认优先级，也不做默认线性平均。

## 7. 精度合同

Gate D 与正式训练均需通过：

- fp32 master 保存、转换为 BF16 serving、重新加载后的输出对照；
- BF16 projector 与 DeepSeek BF16 activation 的真实组合；
- 普通 FP8Linear、FP8 expert gate/up、FP4 expert down 各自的 input gradient；
- activation checkpointing 开关前后 projector gradient 数值一致性；
- checkpoint save/load 与 optimizer resume 后的下一步轨迹一致性；
- train、Tiny、Medium、Full 不得隐式切换 projector dtype；
- 所有 MoonViT 与 DeepSeek 参数 `grad is None`；projector 梯度有限非零。

本地包 7–11 已观察到 fp32 与 BF16 projector 行为差异，因此只用权重 tensor 接近不能代替任务级等价验证。所有正式分数标注 activation、projector、model dtype。

## 8. Gate D 顺序

用户授权付费后仍按以下顺序逐关执行；任一关键关失败立即停止并保存产物：

1. 固定环境与 revision；校验官方 weight index、所需 shards 和所有 hash；
2. `gate_d_dgrad.py` 模式 A：BF16/FP32 数学 reference，验证 harness；
3. 模式 B：实际加载后的普通 FP8Linear、FP8 expert gate/up、FP4 expert down，逐模块测真实 quantized forward 与 `grad_input`；
4. 若模式 B 失败，模式 C 测预先实现的 candidate input-only DGRAD；不得临时改成永久全模型 BF16 解量化；
5. 完整量化权重加载；
6. 单图 forward；
7. 单 batch backward，projector 有有限非零梯度，MoonViT/DeepSeek 无权重梯度；
8. activation checkpointing 开关前后梯度一致；
9. batch > 1，逐图 placeholder、embedding 与 routing 位置匹配；
10. 20 step 无 OOM/NaN；
11. checkpoint save/load 与完整 optimizer/RNG/data-cursor resume；
12. 测真实 step time、checkpoint 与 Tiny/Medium sentinel 成本；
13. 全部通过后才可进入阶段 1 短校准训练。

最小单卡 kernel gate 与完整模型 Gate D 分成两次授权边界，见 `docs/gpu-runtime-matrix.md`。

## 9. Hash-MoE 路由观测

前 3 个 hash layers 必须记录：

- placeholder token ID；
- 每个视觉 token 的 hash expert IDs；
- 每个视觉 token 的 expert mixing weights；
- 同一图片内 mixing-weight 方差；
- 视觉 token 与文本 token 的 expert overlap。

后续非 hash layers 记录 expert histogram。batch > 1 时记录每张图对应的 token/routing 区间；checkpointing 开关前后比较路由一致性。主实验不改用多个 routing token。只有单 placeholder 的路由观测显示明确容量瓶颈，才预注册 routing-palette 消融。

## 10. 自适应训练时长

“约 2 epoch / 约 2100 steps”只保留为外部历史先验，不能作为 DeepSeek-V4 承诺。

### 阶段 0：Gate D

只验证 runtime、梯度、容量、恢复和成本，不训练能力。

### 阶段 1：短校准训练

使用固定小型 examples budget和高频 Tiny Sentinel，测真实 `t_step`、梯度尺度、loss 斜率、能力斜率、sentinel/checkpoint 开销并判断 LR 是否合理。阶段 1 结束前不承诺完整训练长度。

### 阶段 2：受控扩展

只有至少一个预注册视觉能力指标上升，且 worst-task 没有不可接受退化，才扩大 examples/GPU-hour 预算。每次扩展都生成新的预算上限，不自动跨阶段。

### 阶段 3：自适应停止

根据 macro、worst-task、vision-minus-blind、vision-minus-shuffle、generation、遗忘与 Pareto 前沿停止。达到最大 examples、最大 GPU-hours、最大美元成本，或连续冻结窗口无能力改善时立即停止。

三档时长只用 Gate D 实测值计算：

```text
optimistic_hours = optimistic_examples / measured_examples_per_second / 3600 + measured_overhead
base_hours       = base_examples       / measured_examples_per_second / 3600 + measured_overhead
pessimistic_hours= max_examples        / measured_examples_per_second / 3600 + measured_overhead
cost             = hours * authorized_hourly_price + measured_storage_and_transfer
```

`optimistic_examples`、`base_examples`、`max_examples`、最大 GPU-hours 与最大美元成本在用户授权单中填写。没有实测 step time 时不得填造时长。

## 11. 费用止损与权限边界

- 当前所有付费状态均为 `authorization_pending`。
- Gate D 失败立即 destroy；stop 后仍计费的实例不可只 stop。
- 下载、初始化、最小 DGRAD、完整 Gate D、校准训练分别设置美元与 wall-time 上限。
- 任一阶段不通过，不因机器已经租用而继续消耗。
- 不自动切换更多 GPU、更高价架构、付费存储或新的云供应商。
- 单卡 kernel gate 通过只允许提交完整模型 Gate D 的授权请求，不自动执行下一阶段。
- 所有失败保留 command、环境、stdout/stderr、exit code、已下载字节、GPU-hours 和费用记录。

## 12. 冻结前清单

| 条件 | 当前状态 | 冻结证据 |
|---|---|---|
| 分层 batch 与全局随机判定 | 已完成：窗口覆盖 | Package 12，`mixed_or_underpowered` |
| matched replay 规则与效果 | 待执行 | 后续 V100 包 |
| OCR/计数瓶颈定位 | 待完成 | synthetic 轨迹与专门诊断 |
| 1.5B 容量桥接 | 条件触发 | 仅在内部视觉偏好可靠、0.5B generation 失败时 |
| checkpoint/sentinel 成本 | 待执行 | V100 成本包；Gate D 复测 |
| 三模式 DGRAD reproducer | 待实现/测试 | `tools/gate_d_dgrad.py` |
| runtime 源码审计 | 已形成 | `docs/dsv4-runtime-source-audit.md` |
| GPU 架构矩阵 | 已形成 | `docs/gpu-runtime-matrix.md` |
| 付费授权 | 未授权 | 用户明确指令才可变更 |

上表所有本地前置项收敛后，再在 `HANDOFF.md` 顶部给出最终 go/no-go。当前结论为 **no-go for paid execution；继续 V100 本地研究**。
