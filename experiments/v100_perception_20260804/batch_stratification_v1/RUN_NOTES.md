# Package 12：分层能力覆盖 batch 对全局随机顺序

## 问题与预注册判据

本包检验同一组 2,400 条六任务记录在完全匹配训练条件下，逐 batch 固定六任务覆盖是否优于全局随机排列。预注册主判据为 step 100 的 paired preference：overall paired bootstrap CI 排除 0，或至少两个任务的 CI 同向且没有显著反向任务。辅助判据包含 step 25/50/100 的 macro、worst-task、paired generation、vision-minus-shuffle、历史峰值/遗忘、任务梯度范数与两两 cosine。

未读取 final half；所有 JSON 中 `final_half_scored=false`。

## 严格匹配证明

Arm A 沿用 package 8/9 已验证的 `balanced_compare_projector_v1`；100 个 true batch 均为六任务各 4 条。Arm B 从同一 package-7 projector/AdamW 状态开始，只把 epoch 顺序改为 seeded global random；100 个 batch 全部不满足 4×6 分层，单一任务在一个 batch 中最高 11 条。

独立验证通过 14 项约束：

- 两臂各 2,400 条、100 steps、batch size 24；
- 六任务各 400 条，每个 record 恰好一次；
- 相同 record set、base projector SHA、step-0 tensor SHA、optimizer source SHA；
- 相同 seed、LR `5e-4`、weight decay 0、gradient clip 1；
- 训练顺序确实不同，唯一处理变量是 batch-order constraint。

关键 provenance：

- base projector file SHA-256：`98a566eb2c8414f105662b6532734a4246830b7e5a2df56ae3da14ad23737ef3`
- step-0 tensor SHA-256：`d2b413c2155d6e70ef6f2fa3a5fd8b9ba8d56d0ef163d4d31d719543c2f70e69`
- restored optimizer source SHA-256：`47ceb28f64df2573beae51480e75fe5626d32e6c134f67dc91f35cfc6ba97196`
- stratified order SHA-256：`a0929326fc8b494f85b82fc504855cd23f1ddfa74348ccc9b364a13ada39f2f5`
- global order SHA-256：`20f857ee0c2350572ea30b64a7aca3ac2884609714f7f1b6405a43267f7ad16f`

## 结果

### 轨迹摘要

| step | arm | macro preference | worst-task | macro generation | vision−shuffle |
|---:|---|---:|---:|---:|---:|
| 25 | stratified | 0.202 | 0.000 | 0.037 | 0.129 |
| 25 | global | 0.241 | 0.000 | 0.013 | 0.171 |
| 50 | stratified | **0.512** | **0.155** | **0.233** | **0.398** |
| 50 | global | 0.389 | 0.145 | 0.167 | 0.298 |
| 100 | stratified | 0.511 | **0.100** | 0.257 | 0.385 |
| 100 | global | **0.531** | 0.055 | **0.320** | **0.398** |

分层 arm 在 step 50 显著更快形成多任务能力；该优势在 step 25 尚未出现，并在 step 100 的 macro/generation 上消失。终点 stratified−global overall paired preference 为 −0.020，95% paired bootstrap CI `[−0.0442, 0.0025]`，未达到预注册总体门槛。

终点任务交换：

- coordinate：分层 +0.165，CI `[0.115, 0.220]`；
- count：分层 +0.045，CI `[0.000, 0.095]`，边界性证据；
- color：分层 −0.090，CI `[−0.165, −0.025]`；
- shape：分层 −0.245，CI `[−0.315, −0.175]`；
- OCR：+0.005，CI `[−0.050, 0.060]`；
- spatial：两臂均为 1.000。

轨迹 AUC 也呈能力交换：分层相对 global 的 coordinate/count/shape AUC 分别为 +0.1156/+0.0619/+0.0706，color 为 −0.0625，OCR +0.0013，spatial 0。分层从 step 50 到 100 遗忘 count 0.28、shape 0.30；global 遗忘 count 0.25，shape 在终点达到自身峰值。

### 梯度诊断

六任务各固定 8 条 complete-pair records，对 frozen 与六个 checkpoint 测 projector gradient，共 42 个 norm rows、105 个 task-pair cosine rows。

| state | mean task cosine | negative pairs / 15 | mean gradient norm |
|---|---:|---:|---:|
| frozen | 0.0075 | 8 | 2.5565 |
| stratified-25 | 0.0687 | 4 | 1.3716 |
| global-25 | 0.0830 | 4 | 1.2761 |
| stratified-50 | 0.0359 | 6 | 1.4372 |
| global-50 | 0.0570 | 3 | 1.6960 |
| stratified-100 | 0.0397 | 6 | 0.3319 |
| global-100 | 0.1185 | 0 | 1.0766 |

分层终点最强冲突为 count–shape cosine −0.1704；global 终点 15 对全部非负。该小型固定 batch 诊断是描述性证据，没有为 cosine 计算置信区间，但它与 count/shape 轨迹交换方向一致。

## 假设判定

- “逐 batch 六类齐全在严格匹配下普遍优于 global random”：**未支持**。预注册 verdict 为 `mixed_or_underpowered`，且存在 coordinate 对 color/shape 的显著反向交换。
- “分层覆盖提高早期能力形成速度”：**部分支持**。step 50 的 macro、generation、worst-task 与 vision−shuffle 均更高；step 25 不支持单调早期优势。
- “分层覆盖降低任务间干扰”：**当前结果反驳**。分层终点仍有 6 个负 cosine pairs，global 为 0；分层 count/shape 都发生后半程遗忘。
- “分层覆盖提高终点 worst-task”：**描述性支持**，0.100 对 0.055；这一 min-statistic 没有独立 bootstrap 门槛，不能覆盖整体 mixed verdict。
- “终点差异来自不同样本量/初始化/optimizer”：**反驳**。这些因素已逐项精确匹配，剩余差异可归于 schedule/order treatment。batch coverage 本身就是顺序约束，本实验不能再把其影响拆成独立的“覆盖”与“排列”成分。

## 对正式合同的改变

DeepSeek 主线不写“每个 batch 必须六类齐全”。采用固定窗口内的领域覆盖断言，并以 Tiny/Medium sentinel 检测任务交换；batch 分层可作为短校准候选，不成为不可修改的硬规则。可选块状 curriculum 暂不追加，因为它会增加新的 schedule 处理变量，而下一项 matched replay 直接针对本包确认的遗忘。

## 产物与已知边界

- 全量 raw：50,400 preference rows、8,400 generation rows、735 metric rows、525 paired contrasts、42 gradient norms、105 cosines。
- 八个 checkpoint 只在 Git 中保留 `MANIFEST.json` 与 projector config；82 MB projector 和 164 MB optimizer state 继续留在工作站 HDD，并由 SHA-256 定位。
- 远端长期 worktree 的 metadata `git_sha=d5944d9…` 只表示其基线 HEAD；本轮脚本是逐文件同步后运行。最终 Package 12 commit 与 `ARTIFACT_MANIFEST.json` 才是代码/产物发布锚点。
- NVML 初始化警告保留在 log；CUDA 训练/评测正常，峰值显存分别由各 SUMMARY 记录。

下一实验：从已知能力交换 checkpoint 启动 matched ordinary-balanced、fixed replay、forgetting-triggered replay；预注册阈值、倍数、窗口与额外 examples 后再运行。
