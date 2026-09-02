# Qwen3.5-4B 医疗 Agent M1：真实运行、失败链与 operational closure

状态：`M1 OPERATIONAL PASS / EFFICACY DEFER`

日期：2026-09-02

## 1. 本阶段的验收边界

M1 回答的是“单张 V100 能否让真实 Qwen3.5-4B 通过受控视觉工具完成可恢复、可复算、reference-blind 的推理闭环”。它不回答“Agent 是否提升准确率”或“4B 是否超过商业模型”。后两项必须留给冻结开发集上的配对干预和最终 benchmark。

验收条件预先固定为：选中样本全部完成；推理阶段无 reference；模型、输入、提示词、schema、工具和输出均有哈希绑定；失败工具数为 0；E0 schema/citation/tool-trace 三项均为 1；CUDA 峰值显存有运行收据；E3 与医学专家判断在没有相应证据时必须 `DEFER`。

## 2. 冻结运行配置

- GPU：1× Tesla V100-SXM2-32GB，SM70；
- model：`Qwen/Qwen3.5-4B`，revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`；
- model source manifest SHA-256：`a8bfc09b80581bd5d74065ca9574da513433e46b6ed117bad19c5858f2d03def`；
- 两片权重在每次运行前按字节数和 SHA-256 校验；
- backend：local-only，NF4 double quant，FP16 compute，eager attention，greedy，thinking disabled；
- decision/final 上限：192/512 new tokens；图像上限 786,432 pixels；
- tools：`inspect_overview`、`region_inspect`；max steps 2；
- 数据：PMC-VQA 冻结 inference surface 的前 8 条，只作 operational smoke/analysis；
- references：推理命令不接收，只在 `inference_completed` 后由独立 finalizer 读取。

最终运行命令：

```bash
PYTHONPATH=src .venv/bin/python -m edgemed_bench.run_medical_agent \
  --manifest /home/ubuntu/data/external/surfaces/pmc-vqa-mcq-dev-512/inference.jsonl \
  --data-root /home/ubuntu/data/external/pmc-vqa-b56ae594/extracted/figures \
  --model-path /home/ubuntu/models/Qwen3.5-4B \
  --model-source-manifest baselines/local/qwen35-4b-medcmr-b0/source_manifest.json \
  --run-dir runs/qwen35-4b-medical-agent-m1-pmc-smoke8-policy2-20260902 \
  --limit 8 --max-steps 2 \
  --decision-max-new-tokens 192 --final-max-new-tokens 512 \
  --max-image-pixels 786432 \
  --tools inspect_overview region_inspect
```

## 3. 不覆盖的失败链

| Run / commit | 边界跨越 | 新暴露失败 | 处置 |
|---|---|---|---|
| `preflight1` / `2eff865` | 1/1 inference 和 region tool 完成 | final 虽为 JSON，但字段名/类型错；E0 schema/citation 为 0 | verifier `BLOCK`；只收紧 final schema contract |
| `preflight1-schemav2` / `4cc2665` | 单样本 E0 三项全 1，verifier PASS | 规模仍不足；该题答案从 C 变 D | 不用正确率调参；放大到 8 条 |
| `smoke8-schemav2` / `dc31336` | 前 6 条原子 checkpoint 完成；显存收据可用 | 第 7 条首步声称 evidence sufficient 且 `tool_call=null`，被无证据 final 安全门阻断 | 最小复现确认单一根因；不续写旧 run |
| `smoke8-policy1` / `01a0ebf` | first-acquisition policy 使 8/8 完成 | overview evidence 错填 `[0,0,1000,1000]`；E0 schema 7/8 | verifier `BLOCK`；仅做有日志的 cross-field canonicalization |
| `smoke8-policy2` / `6de6909` | 8/8 完成且 E0 全 1 | efficacy、定位和因果仍未建立 | M1 operational closure；转 M2 配对干预 |

第 7 条的确定性控制器规则只在“当前没有任何成功视觉 trace 且 backend 试图停止”时触发。它追加 `policy_intervention=first_visual_acquisition_required` 并执行一次 `inspect_overview`，没有伪装成模型自主调用。若 overview 未在 allowlist 中，原硬阻断保持不变。

cross-field canonicalization 只把 `inspect_overview/temporal_skim` 错填的 region 改为 `null`。prediction 和 trajectory 同时保存 rule、evidence index、before 和 after；它不修改答案、观察、置信度或 trace。最终 8 条中只命中第 7 条一次。

代码变化后没有复用旧 6/8 checkpoint。runner 现对 `code_commit` 漂移硬失败，避免同一 run 混入两个实现。

## 4. 最终运行事实

- code commit：`6de6909bf98101c994e68abbe89a705024e8998a`；
- run contract SHA-256：`d312762fa00d9480c250135dcedd226016cad9cfc2fdc45c93e3fbe1302ab86d`；
- inference：8/8；工具：8/8 `completed`，其中 7 次 region、1 次 forced overview；
- model calls：16 decision + 8 final；总输入/输出 tokens 19,986/3,237；
- 总运行 241.54 秒；单样本平均 30.19 秒，中位数 30.53 秒，范围 26.19–33.32 秒；
- CUDA peak allocated/reserved：3,562.71/3,946 MiB；
- E0 schema/citation/tool-trace：1.0/1.0/1.0；
- verifier：reference isolation、source/output hashes、coverage、artifact integrity、metric recompute、quality gates 全部 PASS；
- E1：3/8，仅作为 smoke 观察；样本过小且该 surface 已用于机制分析，不能支持 efficacy；
- E2：没有 reference boxes，localized count 0；
- E3：没有 no-tool/forced/oracle/compute-matched 配对，`DEFER`；
- medical correctness：没有专家盲审，`DEFER`。

两次 greedy smoke 的两条生成文本存在措辞差异，但 8 个选项答案不变。V100 上的 greedy 不能被当作 bitwise deterministic；后续效果结论仍需固定选择集、多 seed 和配对统计。

## 5. 下一阶段 M2

PMC-512 对这一 Agent 家族只保留 analysis/smoke 角色，不再根据逐样本正确率修改 prompt、tool policy 或答案逻辑。M2 必须先冻结 source-diverse development selection，再运行同 checkpoint、同 image/token 总预算的四个最小配对臂：no-tool、autonomous tool、forced tool、compute-matched no-new-pixels。oracle region 只有在存在独立黄金框时加入。

只有 operational E0 继续全通过，且 autonomous/forced 相对 no-tool 的 E1/E2 改善跨 seed 稳定、compute-matched 不能解释，才允许写“Agent 有净收益”。否则保留失败并回到错误切片，不触碰 Med-CMR test。
