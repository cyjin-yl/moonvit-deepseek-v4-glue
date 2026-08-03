# Gate D 实施手册：vast.ai 4×RTX PRO 6000（日本）

目标：在一次租期内闭环完成 环境验证 → Gate D（可微性判定）→ projector 对齐训练 →
机上 benchmark → 权重与评测结果回传 HF → destroy 实例。租期结束不留任何依赖该机器的事。

## 1. 机器选择（已核实，2026-08-03）

offer `45766633`（verified, reliability 0.999, 当前空闲）：

- GPU：4×RTX PRO 6000 WS，96GB/卡，共 382GB；sm_120（Blackwell，原生 NVFP4，利好 R3 风险）
- 互联：PCIe 5.0 x16（54.2GB/s），无 NVLink —— 短序列 TP 通信量小，非刚需；单步 >15s 再升级
- CPU/RAM：EPYC 9554 128 核 / 515GB
- 盘：WD_BLACK SN850X，剩 6291GB，实测盘速 6758MB/s
- 网络：下行 7802Mbps / 上行 7226Mbps；静态 IP `106.185.159.136`；日本
- 驱动 595.71.05，CUDA ≤13.2
- 价格：$5.069/h；存储 $0.20/GB/月（挂盘 400GB ≈ +$0.11/h）；流量 下 $0.0026/GB、上 $0.0039/GB
  （160GB 权重下载 ≈ $0.42，可忽略）

## 2. 预算与时间盒

账户余额 $50（credit）。有效时价 ≈ $5.18/h → 约 9.6h。排程：

| 阶段 | 预算 | 终止条件（kill criteria） |
|---|---|---|
| 创建+装机 | 0.5h | 镜像/torch 与 sm_120 不匹配 → 换镜像一次，仍不行退租 |
| 下载权重 160GB + 视觉塔 0.8GB | 1.0h | 实测 <150MB/s 且 30min 无改善 → 退租 |
| Gate D（加载→前向→backward→20步） | 1.0h | Dgrad 失败（projector 无梯度）→ 当场 destroy，转情景 A′/B |
| 对齐训练 ~2100 步（66k 短QA × 2 epoch, batch 64, constant lr 5e-4） | ≤4h | step ~1400 仍无 grokking（loss 骤降）→ 停训查数据是否混入长答案；单步耗时反推超预算 → 减步数保 benchmark |
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

## 5. 装机（setup 窗口）

```bash
apt-get update && apt-get install -y aria2 tmux git
pip install -U pip
pip install "transformers>=5.12,<6" "safetensors" "huggingface_hub[hf_transfer]" \
            datasets accelerate pillow numpy
git clone https://github.com/cyjin-yl/moonvit-deepseek-v4-glue /root/moonvit
cd /root/moonvit && pip install -e .
python - <<'EOF'
import torch
assert torch.cuda.get_device_capability(0) == (12, 0), "sm_120 expected"
print(torch.__version__, torch.cuda.device_count(), "GPUs OK")
EOF
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
hf download 255doesnotexist/DeepSeek-V4-Flash-0731-Vision \
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

## 7. Gate D 判定（gate 窗口，生死点）

1. `nvidia-smi` + `nvidia-smi topo -m` + `fio` 快测（不符当场退租）
2. 原生加载 0731（Transformers 5.x，TP=4 或 device_map=auto）
3. 文本短前向正常 → 单图短序列前向（placeholder 注入）
4. **单 batch backward：projector 6 组参数梯度有限非零、LLM 梯度为零 = Dgrad 通过**
5. 20 步无 OOM/NaN，实测单步耗时；>15s/步 → 重新核算训练步数或换 NVLink 机型
6. 判定失败 → 立即 destroy（损失 <$15），报告转情景 A′（B200）或 B（解量化）

## 8. 训练 + benchmark + 回传（train / eval 窗口）

- 训练配方（Baseten 社区实测，唯一公开同级成功案例）：constant lr 5e-4（无调度器）、
  batch 64、约 66k 条**短 QA**、2 epoch ≈ 2100 步。**数据红线：只用短答案 QA；
  长描述性答案会阻止 grokking**（Baseten 原文结论）。
- 命令：`tools/train_overfit.py --vision-tower v2 --moonvit-v2-weights <path> \
  --lr 5e-4 --batch-size 64 --steps 2100 --checkpoint-every 500 \
  --upload-repo 255doesnotexist/DeepSeek-V4-Flash-0731-Vision`
  checkpoint（projector fp32+bf16、AdamW、RNG、history、train.log）每 500 步流式上传。
- grokking 观察：loss 平台数百步后应在 step ~900–1100（第一 epoch 末）骤降；
  骤降前后的 checkpoint 都要留，benchmark 时择优。
- ⚠️ 预飞行缺口：66k 规模短 QA 训练数据的抓取规格还没进 `tools/fetch_eval_data.py`
  （现有的是 flickr8k 8k caption 与各基准的 validation 评测集，不含 train split）。
  租机前需补 train-split 短 QA 源（如 TextVQA/DocVQA train），或在机上现配。
- **续训**：`--resume 255doesnotexist/DeepSeek-V4-Flash-0731-Vision` 自动拉取 HF 上最新
  `checkpoints/step-*` 精确续训（权重+优化器动量+RNG+步数；跨机器 GPU 数不同也能恢复，
  见 `src/moonvit_glue/checkpointing.py`）。租第二台机器继续训练只需这一条。
- Benchmark：`tools/eval_vlm.py`，TextVQA/DocVQA/OCRBench/ScreenSpot/MMMU-Pro 小子集，
  trained × blind（无图基线）× random-projector 三组对照，结果 JSON 一并上传。
- 收尾：projector fp32+bf16、eval JSON、报告更新 → **`destroy` 实例（不是 stop，
  stop 后存储仍计费）**。

## 9. 回退方案

- Gate D 失败 → 情景 A′：4×B200（原生 FP4 路径不同）或情景 B：解量化到 BF16 需 8×80GB，超出现预算，需用户追加。
- 训练中途实例被收回 → checkpoint 已在 HF，换新机器 `--resume` 继续。
