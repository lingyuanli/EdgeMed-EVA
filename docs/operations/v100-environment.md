# V100 运行环境与验证记录

验证日期：2026-08-31
远端路径：`/home/ubuntu/EdgeMed-EVA`

## 已验证硬件

- GPU：Tesla V100-SXM2-32GB；compute capability 7.0；
- Driver：580.173.02；
- PyTorch CUDA runtime：12.6；wheel 包含 `sm_70`；
- 硬件 BF16：不支持，所有训练/推理使用 FP16 compute；
- CPU/RAM：10 cores / 62GB；
- 环境建立完成后可用磁盘：约 97GB。

## 已验证软件

- Python 3.10.12；
- PyTorch 2.10.0+cu126；torchvision 0.25.0+cu126；
- Transformers 5.16.0.dev0，固定 commit `42ca97014c85d71a88ad60d55f08cb9fb4d26e2c`；
- Accelerate 1.14.0；PEFT 0.20.0；bitsandbytes 0.50.2；TRL 1.12.0；
- qwen-vl-utils 0.0.14。

没有安装 FlashAttention 或 vLLM。V100 首阶段使用 Transformers eager attention，优先保证正确性。

## 已执行 smoke

1. FP16 2048×2048 CUDA matmul：PASS；
2. bitsandbytes NF4 Linear4bit CUDA forward：PASS；
3. Qwen3.5 config/processor 识别：PASS；
4. Qwen3.5-4B 本地 4-bit 多模态加载：PASS；
5. 512×512 合成图像确定性生成：PASS。

本次实测 Qwen 模型 footprint 约 3081MiB，峰值 CUDA 分配约 3318MiB；加载约 7.29 秒，48-token 上限的短生成约 4.83 秒。这些数字仅为单次软件 smoke，不代表 Med-CMR 吞吐或训练显存。

## 安装

```bash
cd /home/ubuntu/EdgeMed-EVA
bash environment/setup-v100.sh
```

激活环境：

```bash
source /home/ubuntu/EdgeMed-EVA/.venv/bin/activate
```

## 模型下载

模型保存在仓库外：`/home/ubuntu/models/Qwen3.5-4B`。本机上的 huggingface-hub 1.29.0 在两个大分片并发完成时出现过临时文件竞争，因此固定 `max_workers=1`：

```bash
python scripts/download-hf-snapshot.py \
  --repo-id Qwen/Qwen3.5-4B \
  --local-dir /home/ubuntu/models/Qwen3.5-4B \
  --max-workers 1
```

如需读取 token，可用 `--env-file` 指定文件。脚本只将 `HF_TOKEN` 或 `HUGGING_FACE_HUB_TOKEN` 写入当前进程环境，不打印值。`.env` 必须保持权限 `600` 且不能提交。

## 复验

```bash
cd /home/ubuntu/EdgeMed-EVA
.venv/bin/python scripts/smoke-v100.py --cuda-only
.venv/bin/python scripts/smoke-v100.py \
  --model-path /home/ubuntu/models/Qwen3.5-4B
```

注意：新版本 PyTorch 的 `torch.cuda.is_bf16_supported()` 默认可能包含软件模拟；V100 检查必须使用 `including_emulation=False`。
