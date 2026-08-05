# DeepSeek-V4-Flash Gate D 实施手册

状态：`paid_execution_not_authorized`。

本手册替代 2026-08-03 的“4×RTX PRO 6000 / 固定 marketplace offer”方案。SM120/SM121 现有公开 weight-load blocker，RTX PRO 6000 与 GB10 不再是默认候选；训练长度也不再固定为约 2100 steps。架构判断见 `docs/gpu-runtime-matrix.md`，训练调度见 `docs/deepseek-rental-training-contract.md`。

## 1. 权限与两阶段边界

所有 V100 本地准备继续执行。以下任一动作都需要用户明确授权：创建实例、挂载付费存储、产生付费流量、下载完整 DeepSeek-V4 权重到付费机器、扩卡或切换更昂贵实例。

付费验证分两次授权：

1. **单卡最小 kernel gate**：首选 SM100/B200，只下载覆盖三个目标量化模块的必要 shard，测 weight load、forward、DGRAD，尽量在一小时内完成。
2. **完整模型 Gate D**：只有第一阶段同架构、同软件 revision 全部通过后，提交新的卡数、时价、GPU-hour 与美元上限，请求用户再次授权。

第一阶段通过不会自动触发第二阶段。

## 2. 固定 runtime

基础 pin：

| 组件 | 固定值 |
|---|---|
| Transformers | `ddb849abe009d1089e6c691bfc897f27211c663c`（`v5.12.1`） |
| `kernels` | `0.16.0` |
| finegrained-fp8 kernel | `b77d2c71fef4ff97e22127058034c1167dca8891` |
| DeepGEMM | `559d79fb6994a58b8a15b4b93bf13ccc16edf247` |
| DeepSeek model | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |

PyTorch、CUDA、driver 与 Triton 在实例创建后的前 10 分钟固定。若安装解析导致上述组件 revision 漂移，Gate D 不得继续；先保存失败环境并停止。完整文件 SHA-256 与证据边界见 `docs/dsv4-runtime-source-audit.md`。

GitHub 状态核验优先使用 `gh` CLI：

```bash
gh issue view 372 --repo deepseek-ai/DeepGEMM \
  --json number,title,state,createdAt,updatedAt,url,body
gh api repos/deepseek-ai/DeepGEMM/commits/559d79fb6994a58b8a15b4b93bf13ccc16edf247 \
  --jq '{sha:.sha,date:.commit.committer.date,message:.commit.message}'
```

若 issue #372 状态或修复 revision 变化，创建新版 source audit；不得直接把关闭状态当作 runtime 已通过。

## 3. 授权单必须填写的字段

用户授权前提供：

- GPU 架构、型号、数量与 SM；
- 供应商、实例/offer ID、时价、存储与流量价格；
- 阶段 wall-time 上限、GPU-hour 上限、美元上限；
- destroy 操作与费用核对方式；
- 目标 driver/CUDA/PyTorch build；
- 需要下载的 shard 列表、总字节数与 hash；
- 失败后的唯一允许动作。

任何空白费用字段都保持 `authorization_pending`。

## 4. 阶段 1：单卡最小 kernel gate

### 4.1 实例创建后的十分钟审计

在独立 `tmux` session 中执行并保存原始输出：

```bash
mkdir -p /root/gate-d/{env,logs,artifacts}
python -VV | tee /root/gate-d/env/python.txt
python - <<'PY' | tee /root/gate-d/env/torch_cuda.txt
import json, torch
print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device_count": torch.cuda.device_count(),
    "device_name": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
}, indent=2))
PY
nvidia-smi | tee /root/gate-d/env/nvidia-smi.txt
nvidia-smi topo -m | tee /root/gate-d/env/topology.txt
pip freeze | tee /root/gate-d/env/pip-freeze.txt
```

GPU SM、磁盘、driver 或依赖 pin 不符授权单时立即停止并 destroy；只允许授权单明确写出的单次镜像修复。

### 4.2 最小 shard 解析与下载

从固定 model revision 的 weight index 定位：

- 一个普通 FP8Linear；
- 一个 FP8 expert gate/up projection；
- 一个 FP4 expert down projection；
- 对应权重、scale、必要 config 与 kernel metadata。

生成 `MINIMAL_SHARDS.json`，记录 tensor→shard 映射、文件字节、revision 和 SHA-256。只下载这些 shard。若 index 使三个目标覆盖几乎全部权重，先向用户报告字节数，不自动扩大下载。

### 4.3 三模式 DGRAD reproducer

模式 A 验证 harness：

```bash
python tools/gate_d_dgrad.py \
  --mode reference \
  --dtype bfloat16 \
  --out /root/gate-d/artifacts/dgrad-reference.json
```

通过条件：input requires grad、weight frozen、forward/backward 成功、input gradient 有限非零、weight gradient为 `None`。该模式不计作量化路径通过。

模式 B 使用 Transformers 实际加载后的模块：

```bash
python tools/gate_d_dgrad.py \
  --mode native \
  --weights /root/weights/dsv4-minimal \
  --targets fp8_linear fp8_expert_gate_up fp4_expert_down \
  --out /root/gate-d/artifacts/dgrad-native.json
```

每个 target 单独记录：module path/class、weight/scale dtype 与 shape、backend（DeepGEMM/Triton）、output `grad_fn`、`torch.autograd.grad(output.sum(), input)` 结果、异常类与完整 traceback、input-gradient finite/nonzero/norm、全部冻结权重 `grad is None`。三类全部通过才算 native DGRAD 通过。

模式 C 只在 native 失败时运行预先实现的 input-only DGRAD 原型：

```bash
python tools/gate_d_dgrad.py \
  --mode candidate \
  --weights /root/weights/dsv4-minimal \
  --targets fp8_linear fp8_expert_gate_up fp4_expert_down \
  --compare-dequantized-reference \
  --out /root/gate-d/artifacts/dgrad-candidate.json
```

候选 forward 必须使用实际 FP8/FP4 kernel；backward 只返回 `grad_input`，weight/scale/bias 均无梯度。报告最大绝对误差、相对误差、余弦相似度与运行时间。全模型永久解量化 fallback 不属于该模式。V100 单元测试只验证接口/reference，状态写为 `hardware_pending`。

### 4.4 阶段 1 判定

阶段 1 通过需要三个目标模块都完成真实 weight load 与 forward，并由 native 或预注册 candidate 路径获得合格 `grad_input`。以下任一情况立即失败：

- scale layout / capability 在 weight load 阶段报错；
- output 无 `grad_fn` 且 candidate 不可用；
- derivative 未实现；
- gradient 包含 NaN/Inf、全零或 shape 错误；
- 冻结权重产生梯度；
- 到达 wall-time / GPU-hour /美元上限。

失败后保存全部产物并 destroy。不得因机器仍在计费而现场开展无边界 kernel 开发。

## 5. 阶段 2：完整模型 Gate D

只有获得第二次明确授权后执行。

### 5.1 下载与完整加载

1. 保存官方 weight index 并校验 SHA-256；
2. 下载全部固定 revision shards，逐文件校验 hash；
3. 完整原生量化加载；
4. 短文本 forward；
5. 记录峰值 host/GPU memory、加载 wall time 与 kernel compile log。

任何 hash、layout、OOM 或 kernel 错误立即停止。不得自动切换 BF16 完整解量化或增加 GPU。

### 5.2 视觉 forward/backward

严格按序：

1. 单图 MoonViT-V2 + scratch projector + placeholder embedding 注入 forward；
2. 单 batch loss backward；
3. projector 六组参数梯度有限非零；
4. MoonViT 与 DeepSeek 全部参数无梯度；
5. 开启 activation checkpointing 重复，比较 projector gradient；
6. batch > 1，逐样本 placeholder 数、视觉 embedding 数与位置精确匹配；
7. 20 optimizer steps 无 OOM/NaN；
8. 中间保存并 resume，下一步权重、optimizer、RNG、examples 与 data cursor 轨迹一致；
9. 测真实 step、checkpoint save/upload、Tiny/Medium sentinel 成本。

checkpointing 前后梯度必须按预注册容差比较，并同时报告最大绝对/相对误差和余弦相似度。只报告“有梯度”不够。

### 5.3 Hash-MoE 路由观测

前 3 个 hash layers 保存：

- placeholder ID；
- 每个视觉 token 的 hash expert IDs 与 mixing weights；
- 单图 mixing-weight 方差；
- 视觉 token 与文本 token expert overlap。

后续层保存 hidden-state top-k expert histogram。batch > 1 时分图记录 routing 区间；checkpointing 开关前后断言路由一致。主 Gate D 不改变 placeholder 方案，也不引入多个 routing tokens。

### 5.4 Gate D 总判定

只有以下全部成立才可标记 `gate_d_passed`：

- fixed revision 和全量 hash 通过；
- 三类真实量化 module DGRAD 通过；
- 完整权重加载、单图与 batch > 1 forward 通过；
- projector 梯度有限非零，冻结模块无权重梯度；
- checkpointing 与路由一致性通过；
- 20 step、checkpoint/resume 与成本计量通过。

Gate D 通过后只允许进入合同中的阶段 1 短校准训练。完整训练预算仍需根据实测 step time 和 sentinel 开销重新提交。

## 6. 自适应训练与哨兵

正式训练不以固定 epochs/steps 启动。阶段顺序为 runtime Gate、短校准、受控扩展、自适应停止。训练 mix、checkpoint 内容、Tiny/Medium/Full 定义、5%/10% 开销目标、Pareto 保留与 replay 规则全部以 `docs/deepseek-rental-training-contract.md` 为准。

关键约束：

- canonical BF16 projector/activation 路径；
- projector-only 主线，不默认 LoRA；
- checkpoint averaging 不作为默认抗遗忘方案；
- Tiny 直接使用内存中模型，避免完整模型反复重载；
- 评测与训练不在同一 GPU 组并发；
- 异步上传队列达到上限时暂停训练；
- loss 下降但视觉哨兵无增益不能推动预算扩展。

## 7. 产物与失败纪律

每个子步骤写独立目录，至少包含：

- `COMMAND.json`：argv、cwd、env 白名单、开始/结束时间、退出码；
- stdout、stderr、完整 traceback；
- environment、revision 与 SHA-256 manifest；
- raw per-module DGRAD rows；
- GPU memory、wall time、下载字节与费用累计；
- verdict 与触发的 stop rule。

实例销毁前先验证所有小产物已回传并可从远端重新读取。完整 DeepSeek 权重不回传。最后执行 destroy 并保存供应商账单/状态截图或 API JSON。若回传失败已触及费用上限，先 destroy，再用本地保留的供应商磁盘策略处理；不得无限续费等待上传。

## 8. 当前 go/no-go

- V100 本地机制研究：**继续**。
- 单卡最小 kernel gate：**等待用户付费授权**。
- 完整模型 Gate D：**尚未具备授权前提**。
- 完整训练：**no-go**。包 12 窗口覆盖、包 13 fixed-budget replay 与包 14 Tiny/Medium sentinel 功效/成本均已收敛。当前 V100 主线是纯文本 Qwen2.5-3B 的固定 ScreenSpot/TextVQA/DocVQA/OCRBench/synthetic/语言保持合同；真实量化三模式 DGRAD 和完整 DeepSeek Gate D 仍需目标硬件与单独付费授权。
