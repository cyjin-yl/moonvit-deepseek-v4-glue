# Gate D 实施手册：vast.ai 4×RTX PRO 6000（日本）

目标：在一次租期内闭环完成 环境验证 → Gate D（可微性判定）→ projector 对齐训练 →
机上 benchmark → 权重与评测结果回传 HF → destroy 实例。租期结束不留任何依赖该机器的事。

## 1. 机器选择（已核实，2026-08-03）

offer `45766633`（verified, reliability 0.999，当前空闲）：

- GPU：4×RTX PRO 6000 WS，96GB/卡，共 382GB；sm_120（Blackwell，原生 NVFP4，利好 R3 风险）
- 互联：PCIe 5.0 x16（54.2GB/s），无 NVLink —— 短序列 TP 通信量小，非刚需；单步 >15s 再升级
- CPU/RAM：EPYC 9554 128 核 / 515GB
- 盘：WD_BLACK SN850X，剩 6291GB，实测盘速 6758MB/s
- 网络：下行 7802Mbps / 上行 7226Mbps；静态 IP `106.185.159.136`；日本
- 驱动 595.71.05，CUDA ≤13.2
- 价格：$5.069/h；存储 $0.20/GB/月（挂盘 400GB ≈ +$0.11/h）；流量 下 $0.0026/GB、上 $0.0039/GB
  （160GB 权重下载 ≈ $0.42，可忽略）

**注意（评审修正 2026-08-03）**：Blackwell 有 FP4 Tensor Core ≠ DeepSeek 0731 的 NVFP4
kernel 在这张卡上能对输入 embedding 求梯度。Gate D 第 0 步必须用最小 reproducer 单独判定
（见 §7），不能把"FP4 支持"当成一句话的结论。

## 2. 预算与时间盒

账户余额 $50（credit）+ Visa auto top-up $60。有效时价 ≈ $5.18/h → 约 9.6h（不触发 top-up）。
排程：

| 阶段 | 预算 | 终止条件（kill criteria） |
|---|---|---|
| 创建+装机 | 0.5h | 镜像/torch 与 sm_120 不匹配 → 换镜像一次，仍不行退租 |
| 下载权重 160GB + 视觉塔 0.8GB | 1.0h | 实测 <150MB/s 且 30min 无改善 → 退租 |
| Gate D（§7 全部子步） | 1.0h | Dgrad 失败（projector 无梯度）→ 当场 destroy，转情景 A′/B |
| 对齐训练 ~2100 步（66k 短QA × 2 epoch, batch 64, constant lr 5e-4） | ≤4h | step ~1400 仍无 grokking → 先查数据再决定是否 LR 探针（§8）；单步耗时反推超预算 → 减步数保 benchmark |
| Benchmark（5 基准 × 3 对照） | 1.5h | 时间不足时砍 ScreenSpot，保留 TextVQA/DocVQA/MMMU-Pro |
| 回传 + destroy | 0.5h | 必须完成：projector fp32+bf16、eval JSON、报告 |

缓冲 ≈ 1h。每 30min 对表一次；任何阶段超时立即执行该行的终止条件。

## 3. 创建实例（仅在用户明确说"租"后执行）

账号当前 `ssh_key: null`，必须先注册公钥，否则无法 SSH：

```bash
# 一次性：注册本机公钥到 vast 账号
PUBKEY=$(cat ~/.ssh/id_ed25519.pub)
curl -X PUT "https://console.vast.ai/api/v0/users/current/" \
  -H "Authorization: Bearer $VAST_API_KEY" -H 'Content-Type: application/json' \
  -d "{\"ssh_key\": \"$PUBKEY\"}"

# 创建实例（runtype ssh_direct；镜像用 cu128，torch 需支持 sm_120）
curl -X PUT "https://console.vast.ai/api/v0/asks/45766633/" \
  -H "Authorization: Bearer $VAST_API_KEY" -H 'Content-Type: application/json' \
  -d '{
    "image": "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel",
    "disk": 400,
    "runtype": "ssh_direct",
    "label": "moonvit-gate-d"
  }'
# 返回 instance id 后查询 SSH 地址：
curl "https://console.vast.ai/api/v0/instances/" -H "Authorization: Bearer $VAST_API_KEY"
```

## 4. tmux 布局（硬性要求：所有操作在 tmux 内，用户可 attach 观察）

```bash
ssh root@106.185.159.136 -p <port>
tmux new -s gated -n setup   # 装机与环境
tmux neww -t gated -n dl     # 权重下载（aria2/hf_transfer，断点续传）
tmux neww -t gated -n gate   # Gate D 判定脚本
tmux neww -t gated -n train  # 对齐训练（checkpoint 流式传 HF）
tmux neww -t gated -n eval   # benchmark
```

用户观察：`ssh -t root@106.185.159.136 -p <port> 'tmux attach -t gated'`。

## 5. 装机（setup 窗口）—— 版本固定，禁止漂移

评审修正：模型适配依赖 transformers 内部类与量化 API，5.x 小版本漂移即可改变行为。
初始安装**精确固定**在工作站已验证的组合；任何被迫的升级只允许一次，之后立即
`pip freeze > /root/env/pip_freeze.txt` 封存，续租恢复必须逐包对齐。

```bash
apt-get update && apt-get install -y aria2 tmux git
pip install -U pip
pip install "transformers==5.12.1" "safetensors" "huggingface_hub[hf_transfer]" \
            datasets accelerate pillow numpy
git clone https://github.com/cyjin-yl/moonvit-deepseek-v4-glue /root/moonvit
cd /root/moonvit && pip install -e .
python - <<'EOF'
import torch
assert torch.cuda.get_device_capability(0) == (12, 0), "sm_120 expected"
print(torch.__version__, torch.cuda.device_count(), "GPUs OK")
EOF
mkdir -p /root/env
pip freeze > /root/env/pip_freeze.txt
python -VV >> /root/env/pip_freeze.txt 2>&1
nvidia-smi >> /root/env/pip_freeze.txt
# pip_freeze.txt 随最终产物一并上传 HF（eval/<tag>/env/）
```

环境变量（HF token 从本机 .env 带入，只写进程环境，不落盘到仓库）：

```bash
export HF_TOKEN=<from .env> HF_HUB_ENABLE_HF_TRANSFER=1
```

## 6. 下载（dl 窗口）

```bash
# 0731 主干 160GB（断点续传）
hf download deepseek-ai/DeepSeek-V4-Flash-0731 --local-dir /root/weights/dsv4f
# 视觉塔（我们自己抽取的 MoonViT-V2，含 MANIFEST 双哈希）
hf download cyjin-yl/DeepSeek-V4-Flash-0731-Vision \
  --include "vision_tower_k3/*" --local-dir /root/weights/vision
python - <<'EOF'  # sha256 校验，必须匹配 MANIFEST
import hashlib, json, pathlib
root = pathlib.Path("/root/weights/vision/vision_tower_k3")
manifest = json.loads((root / "MANIFEST.json").read_text())
for name, expected in manifest["sha256"].items():
    digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
    assert digest == expected, f"HASH MISMATCH: {name}"
print("vision tower hashes OK")
EOF
```

## 7. Gate D 判定（gate 窗口，生死点）—— 分阶段，不合并

评审修正：以下各步**分开判定、分开记录**，任何一步失败都有独立结论，
禁止用一句"FP4 支持"打包。多卡训练路径也在此定型（见第 6 步）。

0. **配置发现 + 最小 Dgrad reproducer**：`tools/gate_d_dgrad.py --weights /root/weights/dsv4f`
   - 打印 config.json 的 quantization 配置（FP4/FP8 方案、kernel 来源）；
   - 只取一层真实 quantized linear 的权重切片：input.requires_grad 前向+backward，
     判定 input.grad 有限且非零、weight.grad 为 None（冻结）——这一步不过，后面全免谈；
   - 失败 → 记录 kernel 类名与报错，直接转情景 A′/B。
1. `nvidia-smi` + `nvidia-smi topo -m` + `fio` 快测（不符当场退租）。
2. 原生加载 0731（Transformers 5.x），文本短前向正常。
3. 单图短序列前向（placeholder 注入，embedding hook）。
4. **单 batch backward：projector 6 组参数梯度有限非零、LLM 梯度为零 = Dgrad 通过**。
5. **hook × activation checkpointing 兼容性**：`gradient_checkpointing_enable()` 后重复第 4 步。
   HF 的梯度检查点按 decoder block 重算，embedding 不在重算区，hook 应只触发一次——
   必须用梯度数值一致性（checkpoint 开/关 projector 梯度 allclose）证明，不许假设。
   另测 batch>1 多图（placeholder 数与图像 embedding 数逐样本一致）。
6. **多卡启动路径定型**：默认 `device_map="auto"` 单进程朴素模型并行（LLM 冻结，无需
   权重梯度分片；反向沿设备顺序回流；train_overfit.py 循环不变）。实测单步耗时：
   - ≤8s/步：维持；
   - 8–15s/步：重算训练步数保 benchmark；
   - >15s/步：评估 transformers 原生 `tp_plan`（DeepseekV4 支持度当场验证）或换 NVLink 机型。
   vLLM/SGLang 的推理 TP **不可**用于反向训练，此条已写死。
7. 20 步无 OOM/NaN；`--resume` 从流式 checkpoint 恢复一次，确认轨迹连续（step/优化器/RNG）。
8. 判定失败 → 立即 destroy（损失 <$15），报告转情景 A′（B200）或 B（解量化）。

## 8. 训练 + benchmark + 回传（train / eval 窗口）

- 训练配方（Baseten 社区实测，唯一公开同级成功案例）：constant lr 5e-4（无调度器）、
  batch 64、约 66k 条**短 QA**、2 epoch ≈ 2100 步。**数据红线：只用短答案 QA；
  长描述性答案会阻止 grokking**（Baseten 原文结论）。
- **配方谦逊条款（评审修正）**：该配方是 GLM-5.2V 上的经验先验，不是 DeepSeek-V4 的
  排程承诺。社区 GLM projector 的视觉输入为 1152 维，不能复用到我们的 1024 维 V2；
  scratch 仍是主实验。合法的可选对照是从我们自己的 V2 + 纯文本小主干 checkpoint 仅
  warm-start `pre_norm + linear_1`，映射到 DeepSeek 4096 维的 `linear_2` 必须随机初始化并
  重训，完整小模型 projector 绝不能直插 0731。若 scratch 在 step ~1400 仍无 grokking：
  (a) 先查数据是否混入长答案；
  (b) 数据无误则用剩余预算做 200 步 LR 探针 {1e-3, 2e-4} 各一，取优续训；
  (c) 仍无 → 照常完成 benchmark 并如实报告负结果，不追加预算。
- **视觉 token 预算（写死）**：训练 `--max-image-side 640`——VQA/照片典型 640×480 →
  约 391 视觉 token，最坏方形 529；GUI 截图 640×360 → 276；加文本后序列典型 ≤700。
  评测 `--max-image-side 1024`（推理侧显存独立核算，保 benchmark 保真）。
  Gate D 第 6 步必须用真实 mix 的 token 分布重测单步耗时，此前一切 3–6s/步的
  估算都只是假设。视觉塔已冻结，特征不预缓存（每 epoch 重算的塔前向约占总步时
  <10%，换来实现简单——若实测塔前向占比 >20% 再考虑缓存）。
- 命令：`tools/train_overfit.py --text-model <DeepSeek-V4-Flash-0731-path> \
  --vision-tower v2 --moonvit-v2-weights <path> \
  --lr 5e-4 --batch-size 64 --steps 2100 --checkpoint-every 500 \
  --upload-repo cyjin-yl/DeepSeek-V4-Flash-0731-Vision`
  训练器会拒绝带 `vision_config` 的原生 VLM 文本主干；Qwen3.5-4B 等 stock VLM 只能用于
  独立的评测阳性对照，不能替代本命令中的纯文本 DeepSeek，也不进入 Gate D 通过判据。
  checkpoint（projector fp32+bf16、AdamW、RNG、history、train.log）每 500 步流式上传。
- grokking 观察：loss 平台数百步后应在 step ~900–1100（第一 epoch 末）骤降；
  骤降前后的 checkpoint 都要留，benchmark 时择优。
- 训练数据（已预生产，泄露受控）：TextVQA train 34.6k + DocVQA train 25k（官方 split，
  评测只用 validation）+ 0xSero art 子集约 10k（WikiArt/fashion，与五个基准零交集）
  + **ShowUI-desktop 8k GUI grounding**（用户决定的 computer-use 方向：答案用 0xSero
  动作格式 `click(start_box=[x,y])`，0..999 同尺度，我们的 grounding parser 原生兼容）。
  0xSero 的 screenshots/multistep 行本身不直接用（其图像为改名文件无法回 join 源数据；
  multistep 为轨迹格式跳过），改为从同源公开数据集 ShowUI-desktop 自取。
- 泄露处理：GUI 数据与 ScreenSpot **同域**——报告中 ScreenSpot 必须标注为"域内"基准；
  机械保障不变，组装时对全部训练图与全部评测图做 average-hash 去重（hamming ≤ 6 丢弃），
  丢弃数随 `decontamination_report.json` 公开。flickr8k caption 偏长，不进正式 mix。
- 机械保障：fetch 的 train 规格强制 `max_answer_words ≤ 20`（短答案红线）；
  `tools/build_train_mix.py` 组装时对全部训练图与全部评测图做 average-hash 去重
  （hamming ≤ 6 丢弃），报告落盘 `decontamination_report.json` 随数据发布。
- 产物托管：dataset repo `cyjin-yl/moonvit-dsv4-data`（train_v1/ + eval_v1/），
  含来源、固定 revision sha、行数、sha256 与 README 复现命令。
  租机时 `snapshot_download` + 解包即用，不在机上拼装。
  机上网络注意：租的盒在境外（日本），HF 直连全速，**不需要 ModelScope**；
  若个别源被 HF 限速可用 ModelScope 镜像兜底（`lmms-lab/textvqa`、`lmms-lab/DocVQA`、
  `AI-ModelScope/MMMU_Pro`、`showlab/ShowUI-desktop`，布局与 HF 一致），token 在本地
  `.env` 的 `MODELSCOPE_TOKEN`，上机时经环境变量注入、不落盘进镜像。
- **续训**：`--resume cyjin-yl/DeepSeek-V4-Flash-0731-Vision` 自动拉取 HF 上最新
  `checkpoints/step-*` 精确续训（权重+优化器动量+RNG+步数；跨机器 GPU 数不同也能恢复，
  见 `src/moonvit_glue/checkpointing.py`）。租第二台机器继续训练只需这一条。
- **评测纪律（评审修正）**：
  - 每个基准先按记录 id 交错切两半：**selection 半**用于 checkpoint 选择（gap 最大者胜），
    **final 半**只对胜出的 checkpoint 跑一次——禁止看完所有 checkpoint 的 final 分数再挑。
  - 结果表分三组：**域内**（ScreenSpot，受 ShowUI 训练域覆盖，单独标注）、
    **跨域**（TextVQA/DocVQA/OCRBench，训练集为同源 train split）、
    **零样本**（MMMU-Pro，无任何同源训练数据）。
  - shuffle-loss 对照跑 3 个种子，报告 mean±std 与相对提升比例；
    固定随机-projector 基线一并给出。单次 shuffle delta > 0.1 不构成"对齐成功"。
- Benchmark：`tools/eval_vlm.py --blind --upload-repo cyjin-yl/DeepSeek-V4-Flash-0731-Vision
  --run-tag <tag>`，TextVQA/DocVQA/OCRBench/ScreenSpot(域内，须标注)/MMMU-Pro 小子集，
  trained × blind × random-projector 三组对照。每条记录的原始预测（含 question、
  参考答案、gt_box、raw prediction、逐项分数）+ 运行元数据（模型/权重/git/时间戳）
  随 report JSON 直接传 `eval/<tag>/`。
- 聚合：`tools/aggregate_eval.py --results-dir <dir> --upload-repo ... --run-tag <tag>`
  生成 SUMMARY.json（benchmark × vision/blind/gap 矩阵，ScreenSpot 标 in_domain），
  整个结果目录（原始输出 + 汇总）一并上传——租期结束前必须完成。
- 收尾：projector fp32+bf16、eval JSON、`pip_freeze.txt`、报告更新 →
  **`destroy` 实例（不是 stop，stop 后存储仍计费）**。

## 9. 回退方案

- Gate D 失败 → 情景 A′：4×B200（原生 FP4 路径不同）或情景 B：解量化到 BF16 需 8×80GB，超出现预算，需用户追加。
- 训练中途实例被收回 → checkpoint 已在 HF，换新机器 `--resume` 继续。
