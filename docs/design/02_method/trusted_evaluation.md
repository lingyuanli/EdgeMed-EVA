# 可信评测：结果、证据与因果分开

## 1. 评测层级

| 层级 | 回答的问题 | 指标示例 | 能否单独支持主张 |
|---|---|---|---|
| E0 结构 | 输出是否可解析、引用是否存在 | schema success、valid citation | 否 |
| E1 结果 | 答案是否正确 | MCQ accuracy、open score | 只支持性能主张 |
| E2 证据 | 定位和观察是否正确 | IoU/pointing、evidence precision/recall | 只支持证据一致性 |
| E3 因果 | 答案是否真的依赖所引证据 | ROI deletion、shuffle、evidence swap | 支持 grounded-use 主张 |
| E4 运行 | 结果是否绑定真实运行 | config/checkpoint/artifact hash | 支持可复现性，不支持科学正确 |
| E5 专家 | 自动指标是否与临床判断一致 | 双盲专家评分、一致性 | 支持外部有效性的一部分 |

任何一层通过都不能替代其他层。例如 E0 通过不等于证据医学正确，E2 通过不等于最终答案正确。

## 2. Med-CMR 官方层

- MCQ：总体 accuracy，按七个 reasoning dimensions、模态和问题类型分层；
- Open：按 consistency、coherence、visual accuracy、ground-truth match 报告，保留各子分而非只报加权总分；
- 明确区分 paper-reported、official leaderboard 和 locally reproduced；
- 若官方 evaluator 不可得，代理实现命名为 `dev_proxy_vX`，不可写成官方 open score。

## 3. 证据层

### 3.1 空间证据

黄金子集有框时：IoU、pointing-game accuracy、Recall@K、无效框率。只有点/区域描述时使用相应弱监督指标，不把弱标注伪装成精确框。

### 3.2 语义证据

将输出拆成原子 observation：

- observation precision/recall；
- unsupported observation rate；
- 与竞争假设的 entailment/contradiction 正确率；
- 答案引用覆盖率；
- 跨视图/时间冲突发现率。

自动 NLI/LLM judge 仅是代理；350–700 条黄金集由至少两名具资质标注者独立复核，分歧仲裁。

### 3.3 答案–证据四象限

必须报告：

| | 证据正确 | 证据错误/无关 |
|---|---:|---:|
| 答案正确 | grounded success | lucky/shortcut success |
| 答案错误 | reasoning failure | complete failure |

核心卖点应提升左上象限，而不是只把右上象限变多。

## 4. 工具因果分解

每个 checkpoint 在相同样本上运行：

- `A_no_tool`：禁止工具；
- `A_tool`：允许模型选择工具；
- `A_forced_tool`：强制调用，用于暴露工具伤害；
- `A_oracle_crop`：人工/黄金区域，用于估算上界；
- `A_compute_matched`：无工具但匹配生成 token/轮次。

按每题结果定义：

```text
Call Gain = P(no-tool wrong, tool right)
Call Harm = P(no-tool right, tool wrong)
Net Tool Effect = Call Gain - Call Harm
Policy Precision = P(tool beneficial | tool called)
Policy Recall = P(tool called | oracle says beneficial)
```

同时按是否调用、模态、维度和难度分层。工具卖点至少需要 Net Tool Effect 的 bootstrap 95% CI 下界大于 0。

## 5. 反事实与压力测试

- **Relevant ROI deletion**：遮挡模型引用的关键区域；
- **Irrelevant ROI deletion**：同面积随机/匹配区域；
- **Image shuffle**：样本间交换图像，检测文本先验；
- **View/time shuffle**：交换当前/既往或正/侧位标签；
- **Evidence swap**：保留答案候选但交换证据表；
- **Option permutation**：打乱选项顺序，检测位置偏差；
- **Context ablation**：去除临床上下文，定位信息来源；
- **Resolution ladder**：原图、下采样、oracle crop，区分视觉分辨率瓶颈。

成功的 grounded agent 应在 Relevant ROI deletion 上明显下降，而 Irrelevant ROI deletion 影响小；图像 shuffle 后仍高分提示泄漏或语言捷径。

## 6. 校准与选择性预测

报告 ECE、Brier、NLL 和 accuracy–confidence 曲线。Clinical-safe 模式另外报告 risk–coverage/AURC；Benchmark 模式不得通过弃答抬高 accuracy。

置信度可从显式预测、选项 logits 或多次采样一致性获得，但主方法必须在实验前固定。不同模型的显式自报置信度不能直接当作可比概率。

## 7. 效率与端侧指标

- 首 token 延迟、端到端延迟；
- 每题视觉 token、生成 token、工具调用数；
- 峰值显存、平均功耗/可得时的 GPU-hours；
- 每 1000 题推理成本；
- tool-free 与 agent 成本增量。

“端侧”必须给出实际量化精度、设备、上下文/图像数、吞吐和内存，而不是只按参数量命名。

## 8. 统计协议

- 所有模型使用同一冻结样本顺序和预处理；
- 点估计同时给 95% paired bootstrap CI（建议 10,000 次）；
- MCQ 成对正确性用 McNemar 检验；连续 open/judge 分数用成对 bootstrap/permutation；
- 七维度与多消融使用 Holm 校正；
- 训练方法至少 3 个 seed；商业 API 若受成本限制可 1 次确定性运行，但必须披露；
- 预注册主指标和唯一主比较，避免从多个切片中挑最好结果。

主比较建议：`最终 4B agent vs 最强 locally reproduced open-source baseline`；次比较为 `最终 4B agent vs 同 checkpoint tool-free`。

## 9. Judge 治理

冻结 judge 模型版本、提示、temperature、max tokens、重试规则和解析器；保存原始 judge 输入输出。执行 10% 顺序交换、长度匹配复评和专家一致性分析。judge 不得看到模型名称、训练方法或期望胜者。

如果 judge 与专家在 Visual Accuracy/Ground-Truth Match 上的相关性或一致性不足，judge 分数降级为探索性指标。
