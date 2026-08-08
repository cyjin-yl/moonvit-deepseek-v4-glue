# ARCHIVED — DeepSeek-V4-Flash GPU runtime 兼容矩阵

硬件兼容性为归档前风险评估；没有因本归档自动租用或测试 Blackwell。

审计基线：Transformers `ddb849abe009d1089e6c691bfc897f27211c663c`、DeepGEMM `559d79fb6994a58b8a15b4b93bf13ccc16edf247`、finegrained-fp8 kernel `b77d2c71fef4ff97e22127058034c1167dca8891`、模型 `7872f01b1d1fe23eabc4c98b48bffcef5a386062`。完整指纹见 `docs/dsv4-runtime-source-audit.md`。

这里的“候选”只代表值得进入最小付费 kernel gate。任何型号在真实 weight load、forward、DGRAD 和完整模型 Gate D 通过前，都没有训练可用结论。GPU 产品名称或“Blackwell”标签不构成 kernel 兼容证据。

## 当前矩阵

| GPU | SM | Dense FP8 | FP4 experts | Weight load | DGRAD | Full model | 当前判断 |
|---|---:|---|---|---|---|---|---|
| H100/H200 | SM90 | 支持候选 | DeepGEMM 固定 revision 无原生 FP4 路径，可能走 Triton fallback | 待验证 | 待验证 | 待验证 | 候选 |
| B200 | SM100 | 支持候选 | 原生 FP4 候选 | 待验证 | 待验证 | 待验证 | 首选最小 gate 架构，高成本 |
| RTX PRO 6000 | SM120 | 风险 | 风险 | 已有公开 blocker | 未验证 | 未验证 | 暂不作为默认 |
| GB10 | SM121 | 风险 | 风险 | 已有公开 blocker | 未验证 | 未验证 | 暂不作为默认 |

依据与边界：

- DeepGEMM 固定 revision 的公开支持范围聚焦 SM90 与 SM100；对 SM120/121 不作同代产品类推。
- DeepGEMM issue [#372](https://github.com/deepseek-ai/DeepGEMM/issues/372) 在 2026-08-05 仍为 OPEN，指出 major-12 对 DeepSeek-V4 NVFP4 expert scale `(1,32)` 缺少 layout transform 分支，可在 forward 之前阻塞权重加载。
- B200 的 SM100 与 fixed DeepGEMM 支持范围吻合，因此适合第一个最小 kernel gate；它仍须实测 Transformers module 的 autograd 路径。
- H100/H200 的 SM90 是 dense FP8 候选；FP4 expert 可能需要不同 fallback。只有完整三模块 DGRAD 均通过，才可提升为完整模型候选。
- RTX PRO 6000 与 GB10 在 blocker 关闭、修复 revision 固定且最小 gate 通过前，不进入默认租机方案。

用 `gh` CLI 固定 GitHub 侧证据：

```bash
gh issue view 372 --repo deepseek-ai/DeepGEMM \
  --json number,title,state,createdAt,updatedAt,url,body
gh api repos/deepseek-ai/DeepGEMM/commits/559d79fb6994a58b8a15b4b93bf13ccc16edf247 \
  --jq '{sha:.sha,date:.commit.committer.date,message:.commit.message}'
```

## 两阶段付费验证设计

以下步骤只形成执行合同。没有用户明确授权时不得创建实例、挂载付费存储、下载付费流量或启动任何云任务。

### 阶段 1：单卡最小 kernel gate

首选架构：单卡 B200。若供应或成本约束促使改用 H100/H200，必须把 FP4 expert fallback 当作新的 gate 分支，不能复用 B200 判定。

时间目标：尽量在一小时内完成；实际美元上限由用户授权时填写。执行顺序：

1. 固定驱动、CUDA、PyTorch、Transformers、`kernels`、finegrained kernel、DeepGEMM 和模型 revision；保存环境快照。
2. 从 weight index 解析普通 FP8Linear、FP8 expert gate/up、FP4 expert down 各自所需 tensor 与 scale 所在 shard。
3. 只下载覆盖这三类目标层的最小必要 shard；记录下载字节、时间与 SHA-256。不得为最小 gate 预先下载完整 160 GB 模型。
4. 测三个模块的真实 weight load；任何 layout、dtype、kernel compile 或 capability 错误立即判该架构/revision 失败。
5. 运行 `tools/gate_d_dgrad.py` 的 reference、native 与 candidate 三模式，分别保存 raw JSON。
6. native 模式逐模块验证真实 forward、output `grad_fn`、backend、有限非零 `grad_input` 与所有冻结权重 `grad is None`。
7. 若 native 失败，只允许在剩余时间盒内运行预先实现好的 candidate DGRAD 原型；现场不做无边界 kernel 开发。
8. 达到时间或费用上限立即 destroy。失败机器不因已产生租金而继续运行。

阶段 1 通过标准：三类实际量化模块都能加载、forward，并以可复现路径返回合格 `grad_input`。数学 reference 成功不能替代 native/candidate 量化路径成功。

### 阶段 2：单卡 gate 通过后的完整模型 Gate D

只有阶段 1 同一 GPU 架构、同一软件 revision 全部通过后，才可以请求新的用户授权进入完整模型 Gate D：

1. 下载并校验官方 weight index 与全部 shards；
2. 完整量化权重加载与短文本 forward；
3. 单图 MoonViT/projector 注入 forward；
4. 单 batch backward，验证 projector 有有限非零梯度、MoonViT 与 DeepSeek 权重无梯度；
5. checkpointing 开关前后的梯度与 routing 一致性；
6. batch > 1 与逐图 placeholder/routing 位置；
7. 20 step 无 OOM/NaN、精确 checkpoint/resume、真实 step time；
8. 通过后再讨论正式训练预算。

单卡内存不足导致完整模型无法加载时，该结果只说明容量不足。不得自动扩到多卡或更昂贵型号；先保存失败产物并向用户提交新的架构、GPU 数量和费用上限请求。

## 推荐结论

当前首选的是 **SM100/B200 架构的最小单卡 kernel gate**，随后才决定完整模型所需卡数。H100/H200 保留为需要明确验证 FP4 expert fallback 的候选。SM120/SM121 在公开 blocker 未解决并复验前降级。所有付费动作继续处于 `authorization_pending`。
