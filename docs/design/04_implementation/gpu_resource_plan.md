# GPU 资源计划：RTX 4090 / V100

## 1. 结论

对 Qwen3.5-4B 的 4-bit QLoRA 路线：

- **能起步的最低配置：1× RTX 4090 24GB**；适合推理、baseline、单图/短序列 SFT smoke 和小规模正式 QLoRA；
- **推荐研发配置：4× RTX 4090 24GB**；可并行三 seed/消融，或做多卡 preference/tool-policy 训练；
- **若做在线 GRPO：最低 4×4090，推荐 8×4090**，先以 4 卡小规模 rollout 验证收益再扩容；
- **V100 不作为首选**：它没有原生 BF16，Qwen3.5 当前 BF16 权重/新算子需走 FP16 兼容路线并实测。V100 32GB 可做 1–2 卡 QLoRA；V100 16GB 通常需 2–4 卡才能获得可用余量，在线 RL 预计需 8–16 卡且性价比差。

当前实测已证明单张 V100-SXM2-32GB 可加载 NF4 Qwen3.5-4B，并以 FP16
完成 2 个真实 optimizer steps：32,464,896 个 language-only LoRA 参数，视觉塔
和 projector 冻结，梯度有限且两步均未被 GradScaler 跳过，峰值 6,775 MiB，
adapter 可序列化。因此 **1 张 32GB V100 足以启动本项目的 answer-only QLoRA**。
该证据仍只是 4 样本/2 步 smoke，不等于完整 epoch 的吞吐、稳定性或模型收益。

## 2. 按阶段配置

| 阶段 | 4090 24GB 最低 | 4090 推荐 | V100 32GB | V100 16GB | 备注 |
|---|---:|---:|---:|---:|---|
| BF16/FP16 单样本推理 | 1 | 1–2 | 1 | 1–2 | 限制并发和多图数 |
| 4-bit 批量推理 | 1 | 2–4 | 1 | 1 | V100 需确认量化/算子兼容 |
| T1 QLoRA SFT | 1 | 2–4 | 1–2 | 2–4 | 冻结 ViT、batch 1、梯度累积 |
| T2 DPO/ORPO | 1（很紧） | 2–4 | 2 | 4 | chosen/rejected 增大激活与吞吐压力 |
| T3 离线 tool policy | 1 | 2–4 | 1–2 | 2–4 | 轨迹长度决定容量 |
| 小规模 GRPO | 4 | 4–8 | 8 | 8–16 | trainer + rollout engine；不建议 V100 |
| 三 seed/消融并行 | 3 | 4–8 | 4–8 | 8+ | 单卡单 run 最容易归因 |

V100 数量区间假设使用 4-bit/FP16 和 gradient checkpointing；如果目标环境无法稳定运行 Qwen3.5 视觉栈，卡数增加也不能解决兼容问题，应停止并改用 4090/A100/H100。

本机 V100 的 FP16 路径必须使用冻结的 GradScaler initial scale 1.0，并将
非有限梯度或 scale 下降视为硬失败。实测 initial scale 128 和框架默认值
65,536 均导致非有限梯度；这不是显存不足。

## 3. 24GB 单卡训练预算

建议初始约束：

- 4-bit NF4 base；LoRA rank 32；
- 视觉编码器冻结，projector 先冻结；
- micro batch 1，gradient accumulation 64；
- 单样本 1–2 张图，训练长边/视觉 token 设保守档；
- text context 先 2k–4k，不直接使用模型最大上下文；
- gradient checkpointing、paged optimizer；
- 禁用不兼容的 attention kernel 后先求正确，再优化吞吐。

若峰值显存 >22GB 或出现碎片 OOM，依次降低图像数/分辨率、文本长度、LoRA rank；不要先牺牲数据正确性。多视图样本可用 view-aware packing 或分阶段编码，但必须保持样本语义。

## 4. GPU-hour 粗估

以 12k–20k 训练样本、1–3 epochs、多图高分辨率、三 seed 和必要消融为范围：

| 工作包 | 4090 GPU-hour 规划范围 |
|---|---:|
| B0/B1/B2 + evaluator 调试 | 20–60 |
| T1a/T1b smoke、搜索、三 seed | 80–180 |
| T2 preference | 80–200 |
| A1/A2 工具评测与离线策略 | 50–120 |
| 统计复评、消融、量化 | 50–120 |
| 可选小到中型 GRPO | 200–500 |

- **不做在线 RL 的完整研究闭环：约 280–680 4090 GPU-hour**；
- **加入受控 GRPO：约 480–1,180 4090 GPU-hour**。

这是包含失败 smoke、三 seed 和消融的研究预算；一次成功 SFT 会远低于总预算。V100 可暂按 4090 时间的 2–3 倍排期，但必须用本地 256 样本基准替换该规划系数。

## 5. 墙钟时间示例

以无 RL 400 GPU-hour 为例：

- 1×4090 串行：理论 16.7 GPU-days，考虑空档约 3–4 周；
- 4×4090：理论 4.2 天，考虑依赖和顺序约 1–2 周；
- 8×4090：并非所有阶段可线性并行，主要价值是多 seed/消融并发。

以含 RL 800 GPU-hour 为例，4×4090 理论 8.3 天，实际建议预留 2–3 周。

## 6. Smoke 决策表

先在目标机器执行四个 256 样本 smoke：

1. BF16/FP16 推理；
2. 4-bit 推理；
3. T1b QLoRA forward/backward/保存/恢复；
4. T2 chosen/rejected forward/backward。

记录峰值显存、samples/s、image tokens、text tokens、功耗、OOM/NaN 和恢复一致性。只有 T1b 峰值 <22GB 且无 NaN 才批准单 4090 长跑；否则转 2–4 卡或降低输入预算。

## 7. 硬件事实与兼容性

Qwen 官方模型卡将 Qwen3.5-4B 描述为原生视觉语言模型；模型仓库权重规模约 9.3GB（以当前仓库文件为准）。NVIDIA 文档要求计算能力 8.0 及以上才具备 BF16 支持，而 Tesla V100 属于 Volta，因此 V100 路线使用 FP16，不应假定 BF16 可用。

软件版本必须冻结：NVIDIA driver、CUDA、PyTorch、Transformers、bitsandbytes、flash-attn/vLLM/SGLang 和 ms-swift。任何自动 fallback 都要写入日志；不能把 CPU fallback 的慢运行当作 GPU 结果。

## 8. 本机条件

当前工作机是 48GB 内存的 Mac，无 CUDA GPU。它适合数据 QA、schema/evaluator、统计和小型量化推理验证；正式 QLoRA/GRPO 应使用远程 NVIDIA GPU。不要在本机默默转 CPU 长跑。
