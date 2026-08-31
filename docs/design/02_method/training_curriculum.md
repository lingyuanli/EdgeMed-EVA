# 训练课程：从可信基线到可选 RL

## 1. 总原则

训练按“先对齐答案，再绑定证据，再学会拒绝坏证据，最后才优化工具策略”推进。任何阶段未达到预设门槛，就停止向更昂贵阶段升级。

默认 backbone：Qwen3.5-4B；默认方式：4-bit QLoRA，冻结视觉编码器，训练语言层 LoRA，视首轮 smoke 决定是否训练 multimodal projector。保存 adapter，不覆盖基础权重。

## 2. 数据边界

### 2.1 两条互斥赛道

- **Leaderboard-clean**：Med-CMR test、源文章、答案、evaluator 反馈和其近重复不得进入训练、偏好生成或 RAG；用于与官方/公开成绩比较。
- **In-domain research**：若需使用 Med-CMR 可训练部分，必须按病例/患者/文章组切分，全部 baseline 在同一 split 重跑；结果只能称本地 in-domain track。

两个赛道使用不同 run namespace、数据 manifest 和表格，不得选择性合并。

### 2.2 数据构成建议

第一轮目标 12k–20k 条高质量样本，而不是盲目扩大：

- 35% 细粒度病灶/解剖与定位；
- 20% 跨视图/时间比较；
- 15% 临床上下文整合；
- 15% 竞争性鉴别与反证；
- 10% 医学长尾但有可靠图文证据；
- 5% 工具“无需调用/调用有害”的负例。

至少 350–700 条建立双人复核的黄金证据子集，用于 validation 和证据评测，绝不参与训练。

### 2.3 每条训练样本的 provenance

```json
{
  "record_id":"...",
  "source_dataset":"...",
  "source_version":"...",
  "license":"...",
  "patient_group_hash":"...",
  "image_sha256":"...",
  "text_hash":"...",
  "annotation_type":"human|report-derived|synthetic",
  "annotator_or_generator":"...",
  "quality_status":"accepted|quarantined",
  "benchmark_overlap":"none|suspected|blocked"
}
```

`suspected` 和 `blocked` 不进入 leaderboard-clean 训练。

## 3. 训练阶段

### T0：不训练的真实基线

- B0：官方推荐预处理 + direct answer；
- B1：相同模型 + 固定证据 JSON prompt，但无工具、无训练；
- B2：相同模型 + Agent 状态机和工具，但无训练。

目标：确定模型能力、结构提示收益/损失、工具原始净收益和主要失败切片。未完成 T0 不允许训练。

### T1a：答案域对齐 SFT

输入相同，目标只包含短答案和必要临床 observation。该分支用于证明普通医学 SFT 的贡献和代价，不作为最终模型。

建议起点：LoRA rank 32（候选 16/32/64）、alpha 64、dropout 0.05；有效 batch 64–128；1–3 epochs；学习率从 `1e-4` 小网格探索；视觉编码器冻结；最大序列/图像预算由显存 smoke 决定。

### T1b：Evidence SFT

目标为完整 evidence packet、竞争假设和答案引用。混入 10%–20% `no_tool_needed` 与 `insufficient_visual_evidence` 样本，抑制虚构框和过度调用。

损失建议：

```text
L = L_answer + 0.7 L_evidence_text + 0.5 L_region + 0.3 L_relation + 0.2 L_calibration
```

权重只是预注册起点，不能在冻结测试上调。没有可靠 box 的记录不计算 `L_region`；低质量合成证据不与人工证据同权。

首版不增加专用检测头：`L_evidence_text`、`L_region` 和 `L_relation` 通过目标 JSON 不同 token 区段的 loss mask/权重实现。只有坐标 token 学习被实验证明为瓶颈时，才考虑额外 head。

### T2：Grounded Preference Optimization

以 T1b 为初始化，使用 DPO/ORPO 一类稳定的离线偏好优化。每个 chosen/rejected 对只改变一个主要错误因素：

- 正确答案 + 正确证据 vs 正确答案 + 错误/无关证据；
- 正确证据 + 正确答案 vs 正确证据 + 错答案；
- 完整跨视图证据 vs 忽略关键视图；
- 必要 crop vs 无意义/重复 crop；
- 合理不确定性 vs 编造高置信度细节。

人工审计至少 10% 合成 pair；pair 生成模型不能同时作为唯一 evaluator。

### T3：选择性工具策略

优先用离线 trajectory preference 或 behavior cloning 学习“何时调用和在哪里调用”。正轨迹来自人工/确定性 oracle，负轨迹包含重复框、错误图、无信息区域和原本正确却被工具改错的样本。

只有满足以下条件才允许在线 GRPO：

- T2 在冻结 validation 上相对 T1b 有显著增益；
- tool-enabled 相对同 checkpoint tool-free 的净收益为正；
- Call Harm 低于预设上限；
- reward 各分量与人工抽检一致，不存在明显 reward hacking；
- 预算和可恢复运行已 smoke 验证。

候选 reward：

```text
R = 1.0 R_answer
  + 0.6 R_evidence
  + 0.3 R_consistency
  + 0.2 R_calibration
  - 0.2 R_unnecessary_tool
  - 0.5 R_tool_harm
  - 0.5 R_unsupported_claim
```

禁止给“调用工具”本身正奖励；只有调用后新增的、正确且与答案相关的证据可以得分。

## 4. 训练保持与抗遗忘

每个训练 batch 混入 10%–20% 通用视觉指令和拒绝编造样本。每个 checkpoint 评估：

- Med-CMR validation 总体和七维度；
- modality 宏平均与最差模态；
- 通用视觉保持集；
- schema success、证据幻觉率和校准；
- tool-free 与 tool-enabled 两种推理。

若通用保持集或任一关键维度相对 T0 下降超过 3 个绝对点，暂停并调整数据混合/学习率，而不是继续堆训练。

## 5. 超参数探索顺序

只按以下顺序改动，每轮一个主因子：

1. 显存/吞吐：图像分辨率、序列长度、gradient checkpointing；
2. 数据质量：证据合法率、泄漏、类别/模态平衡；
3. LoRA rank 与学习率；
4. 答案/证据损失权重；
5. preference beta 与 pair composition；
6. 工具策略阈值和预算；
7. 最后才是在线 RL。

每个因子先用 10% 训练量和固定 validation 比较，只有效应方向稳定才全量重跑。

## 6. Checkpoint 选择

主选择分数在训练前冻结：

```text
Select = 0.45 normalized_answer
       + 0.30 normalized_evidence
       + 0.15 consistency
       + 0.10 calibration
       - regression_penalty
```

工具指标不进入 T1/T2 checkpoint 选择，避免提前偏向调用。测试集只在最终两个冻结候选上各运行一次；不能用测试分数挑 checkpoint。

## 7. 停止条件

- 两轮相邻阶段在主要指标上均无可重复提升；
- 增益只来自 judge coherence，而视觉/答案不变；
- tool-enabled 不优于 compute-matched tool-free；
- 数据泄漏无法排除；
- 专家抽检显示证据 hallucination 超过安全阈值；
- 预算超过阶段上限且没有新证据支持继续。
