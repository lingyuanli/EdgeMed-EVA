# 逐步迭代实验协议

## 1. Baseline ladder

所有行都必须使用同一冻结 test 输入、图像预处理和评分器。`reported` 行与 `reproduced` 行分开。

| ID | 模型/系统 | 训练 | 工具 | 目的 |
|---|---|---|---|---|
| R0 | Med-CMR 论文/榜单模型 | 未知/各自 | 各自 | 仅作外部参考 |
| B0 | Qwen3.5-4B direct | 无 | 无 | backbone 真基线 |
| B1 | Qwen3.5-4B structured | 无 | 无 | 提示/结构影响 |
| B2 | Qwen3.5-4B agent | 无 | crop | 原始工具能力与伤害 |
| M1a | Answer-SFT | T1a | 无 | 普通垂域训练贡献 |
| M1b | Evidence-SFT | T1b | 无 | 证据监督增量 |
| M2 | Grounded preference | T2 | 无 | 视觉一致偏好增量 |
| A1 | M2 checkpoint | T2 | 可选 crop | Agent 纯推理增量 |
| A2 | Tool-policy trained | T3 | 可选 crop | 工具策略学习增量 |
| A3 | Optional GRPO | T3-RL | 可选 crop | 只在前序过门后尝试 |
| O1 | 最佳 checkpoint | 同上 | oracle crop | 工具上界，不是可部署方法 |

最重要的成对比较：`B1-B0`、`M1a-B1`、`M1b-M1a`、`M2-M1b`、`A1-M2`、`A2-A1`。每次只对应一个主要机制。

## 2. 八轮执行路线

### Round 0：数据与 evaluator 接通

1. 获取许可允许的 Med-CMR 数据与评分入口；
2. 固定版本、文件数量、样本数量和哈希；
3. 随机 20 题人工核对图像、问题、答案映射；
4. 用论文提供或自带 demo 复核 evaluator；
5. 建立 patient/article/image/text 近重复检查。

交付：access receipts、data manifest、evaluator manifest、overlap report。失败则停止官方赛道。

### Round 1：跑 B0/B1/B2

先在 50 题开发切片做解析/显存 smoke，再跑全部 validation。输出总体、七维度、模态、答案×证据四象限和工具 Call Gain/Harm。

决策：如果 B1 明显损害结果，先简化 schema/prompt；如果 B2 净工具效果为负，不训练工具，只进入 evidence SFT。

### Round 2：建立错误分类与最小训练集

对至少 300 个错误双人分类：视觉未识别、跨视图/时间遗漏、上下文冲突、长尾知识、推理关系、选项/格式、工具伤害。用分类结果调整数据配比，不按测试具体答案造训练样本。

先构建 2k 高质量 seed 数据 + 350–700 黄金 validation，跑 provenance、泄漏和 evidence schema QA；通过后再扩到 12k–20k。

### Round 3：T1a/T1b 域对齐

用相同 backbone、数据预算和训练步数比较答案 SFT 与证据 SFT。每个配置先一个 seed 做 10% 数据筛选，再对唯一胜者做三 seed 全量。

决策：T1b 必须在答案主指标不劣于 T1a 的同时，显著改善证据指标；否则优化证据数据/损失，而不是进入 DPO。

### Round 4：T2 grounded preference

从真实错误生成单因素负例，人工审计后训练。重点压低 unsupported observation、正确答案/错误证据和跨视图冲突。

决策：只有答案或证据至少一项显著改善、另一项无灾难性回退，并且校准/保持集通过，才升级 Agent。

### Round 5：A1 选择性工具解码

不改权重，给 M2 checkpoint 接入工具策略。跑 no-tool、tool、forced-tool、oracle-crop、compute-matched 五臂。

决策：若 oracle 有明显上界但 learned policy 无收益，问题在策略/定位；若 oracle 也无收益，crop 不是主要瓶颈，不应继续工具训练。

### Round 6：A2 工具策略训练与可选 A3

先离线训练必要调用、目标区域和停止策略。A2 过门后才允许小规模 GRPO：先 256–512 题、固定 200–500 optimizer steps，检查 reward hacking 和工具伤害，再决定是否扩大。

### Round 7：冻结测试与统计验证

冻结代码提交、模型、adapter、prompt、工具、evaluator 和随机种子；选择前两名 validation checkpoint，各运行一次 test。执行 paired bootstrap、McNemar/Holm、三 seed 汇总和成本分析。

### Round 8：论文级证据包

只有绑定 run manifest、checkpoint hash、per-sample outputs 和 verifier report 的结果能进入主表。对外主张逐条映射到结果 cell；负结果和失败切片进入附录。

## 3. 目标梯度而非单一“刷榜线”

以下是预注册目标，不是当前结果：

- **Level 1**：4B 最终系统显著超过 B0/B1，三 seed 方向一致；
- **Level 2**：超过最强 locally reproduced 同规模/开源基线；
- **Level 3**：超过 Med-CMR 论文中的强开源 paper-reported 参考，同时证据与成本达标；
- **Level 4**：超过商业模型 paper-reported/local-reproduced 参考，并在视觉证据维度不靠 judge 格式偏差取胜。

Med-CMR 论文报告的 GPT-5 MCQ 57.81、open 48.70，以及强开源 MCQ 49.34、open 47.88 可作为外部航标，但只有在版本和协议一致时才能称为直接超越。

## 4. 消融矩阵

主消融控制在可解释范围：

- 无 evidence SFT；
- 无 grounded preference；
- 无竞争假设，仅直接证据；
- 无工具；
- 强制工具；
- 无跨视图/时间标签；
- 无不必要工具惩罚；
- 无校准项；
- 证据删除/交换。

不做所有组件的指数全组合。先用主阶梯定位有效阶段，再针对 2–3 个关键机制做交互消融。

## 5. 失败后的下一步

| 观察 | 诊断优先级 | 允许动作 |
|---|---|---|
| 答案升、证据不升 | 答案先验/装饰性证据 | 加强黄金 evidence、四象限负例；不加工具 |
| 证据升、答案不升 | 证据到决策映射失败 | 优化 hypothesis relation/answer loss |
| tool-free 升、tool 不升 | intrinsic learning | 停止宣传工具增益；修 policy/harm |
| oracle crop 也不升 | crop 非瓶颈 | 转跨视图/上下文或长尾问题 |
| 某模态崩溃 | 数据失衡/预处理不适 | 模态配比和输入适配，保持其他变量 |
| open 升、MCQ 降 | judge/长文本偏差或决策退化 | 长度匹配、选项校准、保持集 |
| 单 seed 升、均值不稳 | 方差或过拟合 | 三 seed、减少搜索、扩大 validation |
