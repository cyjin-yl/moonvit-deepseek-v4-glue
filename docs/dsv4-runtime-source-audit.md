# DeepSeek-V4-Flash 量化反向路径源码审计

审计日期：2026-08-05。本文固定源码与模型 revision，回答的是“现有公开集成是否已经给出 projector 反传的充分证据”。真实量化模块的 `grad_input` 仍需在目标 GPU 上通过 Gate D 判定。任何依赖升级都要新建审计版本，不能沿用本文结论。

## 1. 固定 revision 与文件指纹

| 组件 | 固定版本 / revision | 说明 |
|---|---|---|
| Transformers | tag `v5.12.1`，commit `ddb849abe009d1089e6c691bfc897f27211c663c` | tag object 为 `a030302d…`；审计使用解引用后的 commit |
| PyTorch | `2.10.0+cu128` | 当前 V100 工作站实测环境；租机环境必须重新记录完整 build string |
| CUDA | `12.8` | 当前 V100 工作站环境 |
| Triton | `3.6.0` | 当前 V100 工作站环境 |
| `kernels` Python package | 本机未安装；Gate D 目标固定为 `0.16.0` | “未安装”是当前环境事实，`0.16.0` 是待租环境 pin，二者不可混写 |
| finegrained-fp8 kernel bundle | `b77d2c71fef4ff97e22127058034c1167dca8891` | HF kernel revision |
| DeepGEMM | `559d79fb6994a58b8a15b4b93bf13ccc16edf247` | 2026-07-15，`Public release 26/07 (#377)` |
| DeepSeek-V4-Flash-0731 | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` | HF model revision |

模型文件 SHA-256：

- `config.json`: `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- `model.safetensors.index.json`: `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- `README.md`: `252acafdc9204d0dba3fde1b0a93d71cd1664a4ceadfe222b60117ed0ccc56ff`

Transformers 固定源码 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `src/transformers/integrations/finegrained_fp8.py` | `a00e39cbacd23904f2a01549028a8ae87dca7f7b42743f28fed27a913810b1d3` |
| `src/transformers/integrations/deepgemm.py` | `7b2320eb0a6f0e06dc28decb11b8b5aed9827a9d49c8135559348570ccd9ae66` |
| `src/transformers/quantizers/quantizer_finegrained_fp8.py` | `f4e2f01a47a8329abb4416807e0d0d147d93cb99a968efee0ecac19c54bb8e05` |
| `src/transformers/models/deepseek_v4/modeling_deepseek_v4.py` | `3be3c5211507ddd1b37ac9dbb27f47533c39d7922779c124517e1d3a7a9c4253` |

finegrained-fp8 bundle 固定文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `build/torch-cuda/matmul.py` | `e74dea55722de704613630d7fc52eb85c9e4fe8e991ab16183ecdde507c0ff0b` |
| grouped matmul implementation | `531fb9d867ab635213afd0975996ca53e3e22a911f896ab6ac82da482a5329f0` |
| batched matmul implementation | `8164c87f…` |
| activation quantization implementation | `78ce4b3…` |
| kernel metadata | `8f04faa1…` |

后三项的完整 digest 必须由 Gate D 环境快照补齐；当前审计只把已完整复核的 digest 当作强校验项。

## 2. 模型配置事实

固定 `config.json` 声明：

- `torch_dtype = bfloat16`；
- dynamic FP8，格式 `e4m3`，scale 格式 `ue8m0`，weight block size `[128, 128]`；
- expert 权重 dtype 为 `fp4`；
- hidden size 4096、43 层、256 experts、top-k 6；
- `num_hash_layers = 3`；
- 配置元数据中的 `transformers_version = 4.57.1` 只描述导出时环境，不覆盖本项目固定的 5.12.1 runtime。

## 3. 量化模块调用链

以下均为固定 revision 的源码事实：

1. `FineGrainedFP8HfQuantizer.is_trainable` 返回 `False`；`replace_with_fp8_linear` 把普通 `Linear` 与 MoE experts 替换为量化模块。
2. `FP8Linear.forward` 调用 `fp8_linear`。`fp8_linear` 根据环境分派到 DeepGEMM 或 Triton forward。
3. `FP8Experts` 为 FP4 expert 分配 packed `int8` 权重与按行 scale-factor 布局，再在 forward 中调用量化 expert kernel。
4. Transformers 的 `deepgemm_fp8_fp4_linear` 对 activation 做 per-token cast、预分配 output、调用 forward kernel并返回 output。
5. fixed finegrained-fp8 `matmul.py` 使用 `@triton_op` 并写入预分配输出；该 revision 中未发现 `register_autograd`、自定义 `torch.autograd.Function` 或 backward 符号。
6. Transformers DeepGEMM integration 会导入 NT/NN kernel 符号；当前调用链中未发现把 NN/DGRAD primitive 注册到 PyTorch autograd 的代码。
7. DeepGEMM 仓库说明存在 DGRAD 与部分 WGRAD 能力。底层能力的存在不能替代 Transformers 实际 module 的 autograd 证明。

`is_trainable=False` 只说明量化权重训练不受支持。本项目冻结 DeepSeek 权重，真正的生死条件是完整 43 层每个相关量化线性层都能向输入返回有限非零梯度。

## 4. Hash-MoE 源码事实

`modeling_deepseek_v4.py` 的 hash router 使用 `tid2eid[input_ids]` 得到前 3 个 hash layer 的 expert IDs，并从视觉 hidden state 计算 mixing weights；后续层使用 hidden-state top-k routing。因此同一 placeholder ID 展开的视觉 token 会得到相同的 hash expert ID 集合，mixing weights 仍可随视觉 hidden state 变化。

Gate D 必须记录 placeholder ID、逐视觉 token 的 hash expert IDs 与 mixing weights、单图 mixing-weight 方差、视觉/文本 expert overlap、后续层 expert histogram、batch > 1 的逐图路由位置，以及 checkpointing 开关前后的路由一致性。主实验保持单一 placeholder；routing palette 只有在观察到明确容量瓶颈后才能作为独立消融。

## 5. 证据分层与当前判定

| 命题 | 层级 | 当前判定 |
|---|---|---|
| 官方配置包含 BF16 activation、dynamic FP8 linear、FP4 experts 与 3 个 hash layers | 源码/配置事实 | 已确认 |
| quantizer 声明 `is_trainable=False` | 源码事实 | 已确认；不单独判定 `grad_input` |
| fixed Transformers 与 finegrained kernel 调用链未发现 autograd 注册 | 源码事实 | 已确认 |
| 当前原生 Transformers 量化路径很可能无法把 loss 梯度传回 projector | 源码推断 | 高风险，仍需真机判定 |
| 普通 FP8Linear 的 input gradient 有限非零 | 硬件/runtime | `hardware_pending` |
| FP8 expert gate/up 的 input gradient 有限非零 | 硬件/runtime | `hardware_pending` |
| FP4 expert down 的 input gradient 有限非零 | 硬件/runtime | `hardware_pending` |
| 自定义只算 DGRAD 的 wrapper 与量化 reference 误差可接受 | 硬件/runtime | `hardware_pending` |

Gate D 的通过条件按三个实际模块分别给出，任何一个模块失败都不能用其余两个模块的成功覆盖。

## 6. SM120/SM121 公开 blocker

DeepGEMM issue [#372](https://github.com/deepseek-ai/DeepGEMM/issues/372) 在固定审计日仍为 OPEN：`transform_sf_into_required_layout` 对 `arch_major=12`、NVFP4 expert scales `(gran_mn=1, gran_k=32)` 缺少分支，报告的直接后果是在 forward 之前阻塞权重加载并抛出 `Unknown SF transformation`。Issue 报告环境包含 GB10 SM121；RTX PRO 6000 的 SM120 同属缺失的 major-12 分支风险范围。

该公开问题使 SM120/SM121 暂时失去默认候选资格。具体矩阵与两阶段付费 gate 见 `docs/gpu-runtime-matrix.md`。

使用 `gh` CLI 复核：

```bash
gh issue view 372 --repo deepseek-ai/DeepGEMM \
  --json number,title,state,createdAt,updatedAt,url,body
gh api repos/deepseek-ai/DeepGEMM/commits/559d79fb6994a58b8a15b4b93bf13ccc16edf247 \
  --jq '{sha:.sha,date:.commit.committer.date,message:.commit.message}'
gh api repos/huggingface/transformers/commits/ddb849abe009d1089e6c691bfc897f27211c663c \
  --jq '{sha:.sha,date:.commit.committer.date,message:.commit.message}'
```

## 7. 变更控制与 Gate D 输出

Gate D 启动时必须保存：

- `pip freeze`、Python/PyTorch/CUDA/driver/GPU capability；
- 上述所有 revision 与完整文件 SHA-256；
- `config.json` 与 weight index SHA-256；
- 三模式 DGRAD reproducer 的 raw JSON、stdout/stderr 和退出码；
- 每个实际量化模块的类名、参数 dtype/shape、backend、output `grad_fn`、异常类型、input-gradient 统计及冻结权重梯度状态；
- checkpointing 前后 projector gradient 与 Hash-MoE routing 对照。

任一 revision、kernel backend 或 GPU SM 变化都会生成新的审计 ID。未经复验，旧的 Gate D 通过状态不得迁移到新组合。
