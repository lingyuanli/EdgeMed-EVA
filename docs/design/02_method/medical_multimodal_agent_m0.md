# 医疗多模态 Agent M0：工具与评测闭环

状态：`M0 IMPLEMENTED / SYNTHETIC CLOSURE PASS / REAL MODEL DEFERRED`

日期：2026-09-02

## 1. 本阶段要解决什么

M0 不是追求分数，而是先证明一条预测可以从无答案输入开始，经受控视觉工具、完整轨迹和独立 finalizer，最终由另一条只在评测阶段读取 reference 的路径得到可复算指标。

```text
inference_manifest(no answer)
        │
        ▼
decision backend ── one allowed tool/call ──► hashed visual artifact
        │                                      │
        └──────── complete message trajectory ◄┘
        │
        ▼
independent finalizer ──► predictions + tool_traces + trajectories
                                      │
references(separate, scorer-only) ────┤
                                      ▼
                         E0/E1/E2/E3 metrics ──► verifier
```

不在 M0 宣称的内容：医疗语义正确、工具带来因果增益、Qwen 已经接入、长视频吞吐已达标、MedVidBench official score。

## 2. 输入契约

单图、多图与抽帧视频共用 `media`：

```json
{
  "sample_id": "case-001",
  "question_type": "mcq",
  "question": "...",
  "options": {"A": "...", "B": "..."},
  "clinical_context": "...",
  "media": [
    {
      "media_id": "current-ct",
      "kind": "image_sequence",
      "modality": "CT",
      "view": "axial",
      "timepoint": "current",
      "frames": [
        {"path": "case-001/000.png", "timestamp": 0.0},
        {"path": "case-001/001.png", "timestamp": 1.0}
      ]
    }
  ]
}
```

运行器对 inference row 执行硬拒绝：`answer`、`ground_truth`、`reference`、`visual_description` 任一出现即停止。`benchmark_dimension` 也不应写进真实推理 manifest；它只属于 scorer 的 slice metadata。

路径必须位于 `data_root` 内。工具产物必须位于当前 run directory 内，防止把未绑定的外部文件伪装成证据。

## 3. 三个 M0 工具

### 3.1 `inspect_overview`

- 输入：可选 `sample_count`，范围 1–16；
- 单图/多图：跨 media 均匀选择；
- 长序列：先在各序列中均匀取样，再形成总览；
- 输出：contact sheet、选中帧时间、源文件哈希、产物哈希；
- 禁止：生成诊断或“图中显示某病”的工具内文本。

### 3.2 `temporal_skim`

- 输入：`media_id/start_time/end_time/sample_count`；
- 只接受 `image_sequence`；
- 在指定闭区间内确定性均匀采样；
- 空区间、倒置时间和非法采样数均失败并写 trace；
- 适合手术视频、内镜、超声 cine 和动态检查。

### 3.3 `region_inspect`

- 输入：`media_id/region_xyxy_1000/target`，序列还必须给 `timestamp`；
- 选择距离 timestamp 最近的帧；
- 坐标 clamp 到 `[0,1000]`，每轴小于 1% 的框拒绝；
- 用原图裁剪，最长边不超过 1024，固定 Lanczos；
- `target` 说明本次要区分的视觉问题，但工具不回答该问题。

所有工具调用均有 request hash。完全相同的重复调用被拒绝；失败也保存在轨迹中。长序列只哈希实际入选帧，避免每次 overview 反复读取全部图像内容。

## 4. Agent 控制器

实现位于：

- `src/edgemed_bench/medical_agent.py`
- `src/edgemed_bench/medical_agent_tools.py`

控制器行为：

1. 校验 inference boundary 与 media；
2. 只把本 run 的 allowlist schema 传给 backend；
3. 每个 decision turn 最多一个工具调用；
4. assistant decision、tool request、tool result 均追加到消息轨迹；
5. 至少一个工具调用 `completed` 且生成哈希绑定产物后，才允许 finalizer；
6. decision loop 与 finalizer 是两个显式 backend 方法；
7. 达到预算时可基于已有成功证据 final，但 finish reason 必须为 `max_steps`；
8. 不把隐藏 CoT 当成 evidence；只评估显式 observation、引用和工具产物。

后端目前是 Protocol，不包含真实 Qwen 实现。M1 将把已验证的 Qwen3.5-4B V100 推理适配器接到 `decide/finalize`，同时保持工具和 scorer 不变，以便归因。

## 5. 输出与 evidence 绑定

final output 沿用 `evidence_grounded_agent.md` 的结构。额外的静态约束是：

- 每个 evidence id 唯一；
- `answer_evidence_ids` 非空且只引用存在的 evidence；
- 带 region 的 evidence 必须能匹配一个同 sample、同 media、同 region 的成功 `region_inspect` trace；
- final 声明使用的 trace 必须是 prediction 轨迹中存在的成功调用；
- 失败 trace 可以保留在 prediction 全轨迹中，但不能作为答案证据。

这能证明“证据声明绑定到了哪次像素提取”，不能证明模型对像素的医学解释正确。

## 6. 评测闭环

实现位于：

- `src/edgemed_bench/score_agent.py`
- `src/edgemed_bench/verify_agent_run.py`
- `src/edgemed_bench/medical_agent_fixture.py`

### 6.1 指标层级

| 层级 | M0 输出 | 可支持的主张 |
|---|---|---|
| E0 结构 | schema valid、citation valid、tool-trace bound rate | 输出可解析且引用闭合 |
| E1 答案 | accuracy | 在给定 reference 上的结果正确率 |
| E2 证据 | 同 media 的 best region IoU | 定位代理指标；不是临床语义正确 |
| E3 因果 | 固定 `DEFER` | 尚无 matched intervention，不可称工具有效 |
| E4 运行 | source/output hash、sample coverage、artifact integrity、metric recompute | 本次产物可独立复核 |
| E5 专家 | `DEFER` | 尚无专家盲审 |

### 6.2 目录契约

```text
runs/<run_id>/
  run_manifest.json
  inference_manifest.jsonl
  references.jsonl
  events.jsonl
  predictions.jsonl
  tool_traces.jsonl
  trajectories.jsonl
  tool_artifacts/<sha256>.png
  metrics.json
  verifier_report.json
```

虽然 reference 与运行产物可为归档目的位于同一 run 目录，但 Agent runner 的函数签名没有 reference 参数；references 只在推理完成后传给 scorer。正式冻结测试可进一步将 reference 保存在 evaluator-only 目录，接口无需改变。

## 7. 已执行的 M0 证据

执行命令：

```bash
PYTHONPATH=src /Users/xxxiaoling/miniforge3/bin/python \
  -m edgemed_bench.medical_agent_fixture \
  --output-dir /tmp/edgemed-agent-m0.<id>/run
```

fixture 由三张合成帧、脚本化 backend 和一个完全匹配的 reference 框组成，只检验管线。实际结果：

- 6 个新增 focused tests 通过；
- 全仓 71 tests 通过；
- verifier 六项检查均为 `PASS`；
- E3 causal 与 medical correctness 明确为 `DEFER`；
- fixture 的 1.0 accuracy/IoU 是构造值，不是模型结果。

临时 smoke 产物不提交到仓库；可用上述命令随时重建。

## 8. 分步迭代路线

### M1：真实 Qwen backend 接线

冻结变量：Qwen3.5-4B checkpoint、量化、greedy decoding、图像预算、三个工具实现、scorer。先跑 8–16 条公开开发/合成样本，要求：0 泄漏、100% 完成、全产物 PASS、V100 峰值显存记录。此阶段只叫 operational smoke。

### M2：无工具与工具对照

在 source-diverse、未用于训练的开发集上做同 checkpoint、同 token/image budget 的配对比较：

- `A_no_tool`：一次直接回答；
- `A_tool`：模型自主选择工具；
- `A_forced_tool`：按预注册规则强制取得同类证据；
- `A_oracle_region`：只用于估计工具上限；
- `A_compute_matched`：不给新像素、但给相同调用/推理预算。

只有 `A_tool - A_no_tool` 的 E1/E2 改善在多 seed 下稳定，且 `A_compute_matched` 不能解释收益，才能称为 Agent 净收益。

### M3：证据正确性与干预

在有黄金框/时间段的开发集上报告 region IoU、evidence recall/precision 和 claim-evidence alignment。增加三种干预：删除被引用证据、替换为无关区域、保持像素但打乱 view/time 标签。答案应按可预测方向退化，才支持 evidence dependency。

### M4：长视频任务适配

先锁定 MedVidBench validation 数据与官方 evaluator 来源，再按任务逐项接入。先做确定性任务和 parser，后接 LLM judge；judge 不可复现时单列 proxy。视频接入必须记录总帧数、实际解码帧、每工具耗时和视觉 token，不用“视频时长”替代真实计算量。

### M5：训练选择性工具策略

只从 M2/M3 证明“oracle 有增益但自主选择不足”的失败切片构造训练数据。先做 SFT 学习何时调用/何处查看，再考虑 preference 或 RL。每次只改变一个主要变量；任何训练候选必须同时通过原任务、顺序扰动、source-diverse retention 和 evidence gate，之后才有资格触碰冻结 Med-CMR 测试。

## 9. 停止条件与下一步

立即停止某条路线的条件：

- schema/trace 绑定下降，表明闭环退化；
- 工具收益只来自额外 token/调用预算；
- oracle 工具也不改善目标失败切片；
- 真实工具定位误差吃掉 oracle 收益；
- 开发收益不能跨 source 或跨 seed；
- 需要查看 Med-CMR 逐样本 test 对错才能继续调参。

下一最小动作是 M1：实现 Qwen backend adapter 和可恢复 batch runner，在单张 V100 上跑不超过 16 条的 answer-blind operational smoke。通过前不启动 full benchmark，也不宣传专属 Agent 提升。
