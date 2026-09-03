# M2：Coarse-to-Fine Active Visual Acquisition

状态：`P1 AUTONOMOUS BLOCKED / P2 FORCED-LOCALIZER SCREEN NEXT / EFFICACY UNMEASURED`

日期：2026-09-04

## 1. 为什么这是当前首选突破口

M1 的 8 条真实模型 smoke 中，7 次 `region_inspect` 都使用 `[0,0,1000,1000]`。原因不是 crop 工具损坏，而是模型在第一次工具决策前没有看到像素；它只能基于问题和媒体元数据盲选区域。继续修改最终答案提示不能解决这个信息顺序错误。

外部方法给出一致的机制先验：AdaptVision 先处理低分辨率视觉 token，再按需请求 box；DeepEyes 将图像工具嵌入交错推理；CropVLM 学习 question-conditioned zoom；医疗 VQA 的 Targeted Visual Prompting 同时呈现局部和上下文；VisualNeedle 用 crop-black 检验中间图像是否真被使用。来源：

- AdaptVision: <https://arxiv.org/abs/2512.03794>
- DeepEyes: <https://arxiv.org/abs/2505.14362>
- CropVLM: <https://arxiv.org/abs/2511.19820>
- Targeted Visual Prompting for Medical VQA: <https://arxiv.org/abs/2408.03043>
- VisualNeedle: <https://arxiv.org/abs/2605.26380>

这些工作只支持机制候选，不支持本项目已有增益。

## 2. 单一改动与冻结项

唯一方法变化：在 backend 第一次规划前，controller 确定性执行一次 `inspect_overview(sample_count=1)`。模型看到该低分辨率 artifact 后，可以停止或调用一次 question-conditioned `region_inspect`。工具总预算仍为 2。

冻结：Qwen3.5-4B revision、NF4/FP16/eager、greedy/non-thinking、图像预算、final schema、scorer、reference isolation、工具实现和答案提示。`initial_visual_policy` 必须进入 run contract；不同 code commit 不允许 resume。

策略注入必须在 trajectory 中记为 `initial_overview_required`，不能记成模型自主行为。

## 3. 数据与 split

主开发筛选来自已通过跨数据集 overlap gate 的 SLAKE English validation，不使用 Med-CMR test，也不使用已反复分析的 PMC-512 选择答案策略。

从 inference-only surface 通过以下固定规则产生二分类 MCQ 候选：

1. `evaluation_metadata.answer_type == CLOSED`；
2. 问句以前缀 `is/are/does/do/did/can/could/has/have/was/were/will/would/should` 开头；
3. 排除包含 ` or ` 的二选一开放表达；
4. 固定选项 `A=Yes, B=No`；
5. 用固定 seed 对 sample id 排序，每个 image SHA 最多一题。

选择完全由 question、metadata 和 image hash 决定。reference 只在选择完成后验证答案属于 yes/no 并映射到 A/B。冻结 full screen 为 96 个唯一图像；先取同一顺序的前 16 条做 operational screen。

## 4. 分阶段对照

### M2-S0：16 条无答案 operational screen

- `P0 autonomous`：M1 policy，无初始 overview；
- `P1 overview-first`：本方法。

只看工具和结构，不读正确率。P1 晋级条件：16/16 inference complete；E0 全 1；工具失败为 0；至少 30% 样本产生面积处于 `[0.01,0.64]` 的非全图 region；不出现重复 overview 循环。若 targeted ROI rate 未过门，先停止，不跑效果比较。

### M2-S1：96 个唯一图像 paired screen

- `D0 direct`：原图一次、直接二选一；
- `P0 autonomous`：当前 Agent；
- `P1 overview-first`：coarse-to-fine Agent。

主指标是 paired exact accuracy；同时报告 schema、targeted ROI rate、Call Gain、Call Harm、模型调用数、视觉 artifact 像素、token、延迟和峰值显存。S1 是开发筛选，不触碰 Med-CMR。

P1 只有在相对 D0 和 P0 的点估计均为正、E0 不退化、工具失败为 0 时，才进入因果臂。95% CI 跨零时只能称 candidate，不能称突破已证实。

### M2-S0b：P1 失败后的 forced-localizer 上界

P1 在冻结 16 条上若因“模型看完 overview 后全部停止”而未过 targeted ROI 子门，只允许增加一个单变量诊断臂 `P2 overview_then_region`：同一 overview 后明确要求 backend 输出一次面积 `[0.01,0.64]` 的 question-conditioned region。controller 不替模型猜框；backend 返回 null、其他工具或非法框均失败并留痕。

P2 的 16 条晋级门更严格：16/16 完成、E0 全 1、零失败工具、targeted ROI rate ≥80%。P2 通过只证明 4B 在明确约束下具有定位能力上界，不证明工具有效；通过后才允许在 96 条上与 D0/P0 比较。P2 未通过则停止 zero-shot crop policy，转入有独立 ROI 监督的 tool-policy SFT 数据设计。

### M2-S2：视觉因果与 compute matching

在 S1 冻结的同一选择上追加：

- `P1-repeat`：第二次调用返回相同 overview，不增加新像素信息；
- `P1-black`：保持 crop 尺寸/调用/token 通路，但返回黑色 crop；
- `P1-oracle`：仅在独立 region gold 可得时运行。

要求 `P1 > P1-repeat` 且 `P1 > P1-black`，Relevant ROI deletion 效应大于 irrelevant deletion。否则额外调用、提示或语言先验仍可能解释观察到的收益。

## 5. 结果表数据契约

| Row | Accuracy | Paired CI vs D0 | E0 | Target ROI | Call Gain/Harm | Input/output tokens | Latency | Peak MiB | Evidence status |
|---|---:|---|---|---:|---|---|---|---:|---|
| D0 | real only | real only | N/A | N/A | N/A | receipt | receipt | receipt | pending |
| P0 | real only | real only | receipt | receipt | paired | receipt | receipt | receipt | pending |
| P1 | real only | real only | receipt | receipt | paired | receipt | receipt | receipt | pending |
| P1-repeat | real only | real only | receipt | 0 | paired | receipt | receipt | receipt | gated |
| P1-black | real only | real only | receipt | attempted | paired | receipt | receipt | receipt | gated |

不得填写 mock 数字。每个 cell 必须绑定 run ID、commit、contract、data/model/output/evaluator hash。

## 6. 停止条件

- overview 后仍主要选择全图：定位能力不足，先构建 question-to-ROI supervision，不跑 RL；
- P1 不优于 P0：信息顺序不是当前瓶颈；
- P1 优于 P0 但不优于 repeat/black：收益不是新视觉证据导致；
- 工具收益只出现在 SLAKE 文本先验强的 yes/no 问题：不得外推到 Med-CMR；
- 任一优化需要逐题查看 Med-CMR 正误：立即停止该路线。

## 7. 资源估算

M1 实测单条 Agent 平均约 30 秒、峰值 reserved 约 3.95 GiB。16 条 operational screen 预计每臂约 8–10 分钟；96 条约 48–60 分钟/Agent 臂。单张 32GB V100 足够，当前无需增加 GPU。训练只有在 S2 表明 oracle/真实 ROI 有净收益但自主定位不足时启动。
