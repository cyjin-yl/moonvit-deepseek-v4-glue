# Qwen2.5-3B 社区可比评测合同 v1

冻结日期：2026-08-05。机器可读版本为
`configs/qwen2.5-3b-community-eval-v1.json`。本合同在首个 Qwen2.5-3B
成绩产生前冻结；后续 projector、数据配比、replay、sentinel、分辨率、结构与
训练策略实验都在同一合同下报告。

## 1. 目标与证据边界

最终路径固定为：

`MoonViT-V2 → 4096 维 projector → deepseek-ai/DeepSeek-V4-Flash-0731`

本地代理固定为纯文本 `Qwen/Qwen2.5-3B-Instruct`，resolved revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`。官方配置是
`Qwen2ForCausalLM`，hidden size 2048，共 3,085,938,688 个参数，权重文件为 BF16，
没有 `vision_config`。MoonViT-V2 与 Qwen 全冻结，训练梯度只进入 canonical
projector。

V100 运行时把 Qwen 权重显式转为 FP16，projector 保留 FP32 master weights，
在 multimodal embedding splice 处把 projector 输出转成 FP16。目标卡上的固定
4096×4096 GEMM probe 显示 BF16 虽可调用，中位耗时为 FP16 的 9.16 倍；FP16、
BF16、FP32 三组输出均 finite。该选择只改变本地计算精度，不改变模型文件哈希，
所有训练继续执行逐步 loss/gradient finite 检查。原始 probe 保存在
`contract/hardware/v100_precision_probe.json`。

Qwen 的 2048 接收宽度通过无参数的 fixed signed-pair orthogonal readout 读取
4096 维 projector 输出。该读出固定 seed `20260805`，所有控制组共用，DeepSeek
阶段丢弃。它让主 projector 的形状、保存格式和最终输出宽度持续保持 4096；
Qwen 代理 checkpoint 的迁移等级为 `transferable_with_runtime_validation`。

Canonical `step0` 和 `random_projector` 的 FP32 权重也在首个 3B 输出前精确冻结。
两者使用同一结构与 PyTorch 初始化分布，seed 分别为 `20260805` / `20260806`；
134,259,248-byte safetensors 的 SHA-256 分别为 `efd942e0…b06b0` /
`7bd4aacf…fc44`。每份权重都通过同 seed 重新初始化、保存恢复和逐 tensor-state
哈希。五个文件已发布到 `cyjin-yl/DeepSeek-V4-Flash-0731-Vision` 的 immutable
commit `65639da5…a010`，大文件由 HF LFS SHA-256 回查，小文件从该 commit 下载
重算。后续横向比较必须加载这份 step0；random control 始终加载另一份固定权重。

Qwen 排名用于筛选可迁移配方。项目成功仍要求完整 DeepSeek-V4-Flash-0731 在
同一真实评测合同中产生非零、可重复且显著高于 blind/shuffled 的视觉能力。

## 2. 固定数据

### ScreenSpot 50

`screenspot_glm50_v1` 从完整公共 ScreenSpot test 中预先选择 50 条，seed 为
字符串 `20260805`。选择以 `platform × text/icon-widget` 十个 strata 均衡，
每格 5 条；格内和最终评测顺序均由 SHA-256 稳定排序决定。

该集合只能称为 **GLM-format metric-aligned public subset**。它与社区私有 50 条
验证样本没有身份等价关系。

### 完整 ScreenSpot

数据固定为 `bevaya/ScreenSpot` test，resolved revision
`0be08781e2e188582f6131625ae1598d443b4d5d`，共 1,272 条。三份 parquet 的
预注册 SHA-256 为：

| shard | bytes | SHA-256 |
|---|---:|---|
| `test-00000-of-00003.parquet` | 134,512,659 | `ff06d312…faa8fb` |
| `test-00001-of-00003.parquet` | 198,971,508 | `d48b8275…0891a` |
| `test-00002-of-00003.parquet` | 268,832,649 | `a28a1e9f…674e5b` |

该 revision 的 `bbox` 已由上游原始 xywh 转成 fractional xyxy。manifest 同时保存
source bbox、999-scale xyxy、图片原始编码字节 SHA-256、图片尺寸、source shard、
source row、`data_type` 和 `data_source`。`forum/gitlab/shop/tool` 统一汇总为 Web；
其余平台为 Android、iOS、Windows、macOS。50 条和完整集合各自保存确定性的
无 fixed-point、无同图像 SHA 的 shuffled-image derangement。

完整评测按 overall、text、icon/widget、Android、iOS、Windows、macOS、Web
拆分。manifest 一经提交，不因模型成绩更换样本或顺序。

### 其他固定集合

候选 checkpoint 使用既有、在 3B 结果前冻结的 selection：TextVQA 250 条、
DocVQA 100 条、OCRBench 100 条。Synthetic 六任务继续报告 paired preference
和 paired generation；这些数字只承担机制诊断，不能替代真实视觉成绩。

语言保持固定为 `language_retention_v1`：`TIGER-Lab/MMLU-Pro` 的 14 个类别
各 10 条，加 `openai/gsm8k` main test 的 100 条，共 240 条，seed 同为
`20260805`。MMLU-Pro 报严格单字母准确率，GSM8K 报最终数值 exact match，
两者同时保存 teacher-forced answer NLL。只运行 text-only 条件，比较 step0、
previous best 与 current candidate；projector-only 路径应逐字节保持语言输出，
LoRA 或解冻文本层时则承担真实遗忘告警。

## 3. 图像、prompt 与生成

训练固定 MoonViT-V2 max side 448、最多 256 个视觉 token；评测固定 max side
1024、最多 1,369 个视觉 token。横向比较必须复用相同的 Kimi-K3 预处理、图片
字节、分辨率和 token 上限。

系统提示固定为：

> You are a GUI grounding model. Return exactly one click action and no other
> text. Use integer coordinates from 0 to 999 with the top-left origin.
> Required format: click(start_box=[x, y])

用户提示固定为：

> Locate the UI element described below and click its center.
> Target: {instruction}

TextVQA、DocVQA 与 OCRBench 共用短答案 system prompt：

> Answer the visual question. Return only the shortest answer that directly
> answers the question. Do not explain.

user message 保持原始 `{question}`。所有视觉条件都把单个 image placeholder 放在
user message 第一行，下一行才是任务文本。Blind 只移除 placeholder 和视觉 token，
保留全部语义文本；shuffled 只按 manifest 更换图片。训练沿用相同 task routing，
loss mask 覆盖到 assistant prefix 结束，只监督 answer 与 `<|im_end|>`。

Qwen 使用 pinned tokenizer 的官方 chat template。图片 placeholder 使用 tokenizer
中已有的 `<|image_pad|>`（ID 151655），不增加 token；DeepSeek 阶段换成其已有
placeholder ID，语义消息与 loss mask 保持一致。

生成固定 `do_sample=false`、`temperature=0`、`max_new_tokens=32`、EOS/stop
`<|im_end|>`（ID 151645）。模型必须只输出：

`click(start_box=[x, y])`

parser 只容许首尾空白；内部语法、大小写和逗号后的单个空格必须完全一致；x、y
为 0–999 整数。自然语言前后缀、浮点、越界、缺少 canonical 空格或多个坐标全部
计为 parse failure，评分器不选择“最近”的候选。

## 4. 固定条件与 checkpoint 角色

每个正式 checkpoint 运行相同样本顺序和生成配置：

1. `vision`：正确图片；
2. `blind`：相同语义文字，不插入图片 token；
3. `shuffled`：manifest 的确定性错误图片；
4. `random_projector`：相同结构与初始化分布、未训练 projector；
5. `step0`：当前训练 run 的初始 projector；
6. `previous_best`：此前在本合同下通过的最佳 checkpoint；
7. `current_candidate`：本次候选。

首个 3B run 尚无可比的 `previous_best`，其训练前比较锚点由 `step0` 和
`random_projector` 承担；七条件输出中的 `previous_best` 明确 alias 固定 step0。
首个完成四条件的真实 baseline 提交后才建立
`previous_best`。0.5B 历史 checkpoint 不进入 3B 横向差值。

## 5. Grounding 指标

每个集合同时报告：

- `parse_count`、`total_count`、`parse_rate`；
- 到目标 bbox 中心的 mean、median、p90、minimum L2；
- `Accuracy@50/@100/@200`；
- ScreenSpot `click_in_box_accuracy`；
- 预测点到 bbox 的 L2 与 L1。

距离各有 parsed-only 和 all-sample penalized 两套汇总。无法解析时，中心与 bbox
L2 使用 999-scale 方形对角线 `sqrt(2) × 999 = 1412.7993…`，bbox L1 使用
`2 × 999 = 1998`。阈值与 click 指标同时给 parsed denominator 和 all
denominator。百分位固定使用 type-7 线性插值。

逐样本配对比较固定为 vision−blind、vision−shuffled、trained−random、
current−previous。使用 2,000 次 paired percentile bootstrap，seed 20260805，
报告 95% CI。所有 improvement 统一成正值代表前一条件更好；距离项因此使用
`second − first`。

社区公开参考只放在 GLM-format metric-aligned 表中，并持续注明数据身份不同：
parse rate 92.0%，mean/median distance 563.7/566.9，parsed-denominator
Accuracy@50/@100/@200 为 4.3%/8.7%/15.2%。

## 6. 固定训练预算与公平性

训练 pack 固定 59,198 条、SHA-256
`5727a504824592907e8d5dfd681640de4e3caea7d32832f70292f8773b72f10a`，
记录集合和顺序保持不变。基线使用 micro batch 1、gradient accumulation 8、
真实 global batch 8、AdamW、lr `5e-4`、constant scheduler、weight decay 0、
gradient clip 1.0。Qwen 运行计算为 FP16，projector master weights 为 FP32，启用
activation checkpointing；MoonViT feature cache 可以替代在线 tower forward，
但 cache manifest 必须锁定图片和 tower provenance。

在 examples seen 4k、8k、16k、32k、64k 保存并评测，对应 optimizer step
500、1000、2000、4000、8000。每个结果保存 optimizer steps、examples seen、
实际 answer tokens seen、effective epochs、micro/accum/global batch、lr/scheduler、
projector 与总 trainable 参数、训练/评测 wall time、峰值显存和恢复验证。

横向比较必须匹配初始 projector、训练记录及顺序、examples seen、optimizer
steps、分辨率、prompt、生成配置与 fixed receiver。探索先用 seed 20260805；任何
替代 `previous_best`、写入结论或改变 DeepSeek 正式配置的结果使用三个独立 seed：
20260805、20260806、20260807，并报告各 seed 原始结果、mean 和 standard
deviation。

“匹配初始 projector”在文件层执行：所有方法从 `efd942e0…b06b0` 的 exact FP32
step0 开始，不能各自重新调用随机初始化。独立 seed 负责数据顺序、dropout 或实验
明确声明的随机变量；若研究初始化本身，必须保留 exact-step0 control，并把新权重
另立实验角色。

## 7. 判定与迁移

“真实 grounding 改进”必须同时满足：完整 ScreenSpot vision click-in-box 高于
matched-budget previous best；vision−blind 与 vision−shuffled 的 paired CI
下界均大于 0；GLM-format 三个阈值至少一项提高；parse rate 没有退化换距离；
TextVQA、DocVQA、OCRBench 无未解释严重退化；训练预算完全匹配。

每个实验同时填写 Qwen 代理结论与 DeepSeek 迁移判断：
`directly_transferable`、`transferable_with_runtime_validation` 或
`qwen_specific_not_transferable`。第三类即使提高 Qwen 分数，也不能修改 DeepSeek
正式方案。

当前 Gate D 仍为 NO-GO。只有完整 0731 在相同 manifest、parser、指标和因果控制
下获得非零且可重复的 vision 增益，才可以写“DeepSeek 获得了视力”。任何租卡、
云 GPU、付费存储或网络操作继续等待用户明确授权。
