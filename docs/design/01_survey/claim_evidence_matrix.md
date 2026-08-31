# Claim–Evidence 矩阵

| ID | 设计主张 | 支持证据 | 反证/限制 | 当前状态 | 对应实验 |
|---|---|---|---|---|---|
| C1 | Med-CMR 的核心难点需要视觉证据与临床逻辑联合优化 | Med-CMR 七维度、模型错误分析 | 论文 evaluator 与本地实现尚未取得；公开仓库不含完整数据 | `SUPPORTED_FOR_DESIGN` | B0/B1 分维度错误切片 |
| C2 | 医学答案 SFT 是必要的域对齐起点 | ChestX-Reasoner 中 answer supervision 的收益 | 单一胸片任务不保证跨 12 模态迁移 | `SUPPORTED_FOR_ORDERING` | T1a 答案 SFT vs T1b 证据 SFT |
| C3 | 证据结构化训练优于只训答案 | MAIRA-2 grounded generation、MMedPO 视觉一致偏好 | 证据 evaluator 可能循环或不完整 | `HYPOTHESIS` | M1-answer vs M1-evidence |
| C4 | crop/zoom 可改善细粒度识别 | DeepEyes 等 active perception 结果 | MED 指出很多增益来自 intrinsic learning，工具仍可能有害 | `CONTESTED` | 同 checkpoint tool-free/tool-enabled；ROI 删除 |
| C5 | 多专科工具 Agent 能提升医学任务 | MMedAgent 展示多工具规划 | 任务/模态覆盖有限，系统复杂且工具错误会级联 | `CONDITIONAL` | 仅错误驱动引入；逐工具消融 |
| C6 | Grounded preference 能减少视觉不一致幻觉 | MMedPO 构造视觉负偏好对 | 权重来自外部模型/视觉工具，可能自我验证 | `SUPPORTED_WITH_RISK` | DPO vs SFT；人工证据审计 |
| C7 | RL 应在 SFT 后执行 | ChestX-Reasoner；MMedPO；工具 RL 稳定性证据 | 不同模型/数据可能不同 | `SUPPORTED_FOR_ORDERING` | 只有 T2 过门后才允许 T3 |
| C8 | 4B QLoRA 可在 1×4090 训练 | 参数规模、4-bit 常规预算和官方训练生态 | Qwen3.5-VL 具体版本显存需实测；长序列/多图会 OOM | `ENGINEERING_HYPOTHESIS` | 256 样本显存 smoke |
| C9 | 证据评测能提高可信度 | MAIRA-2/RadFact 和 grounded VQA 方法 | 高 evidence score 不保证诊断正确 | `HYPOTHESIS` | 答案×证据四象限、专家复核、因果删除 |
| C10 | 小模型可超过商业模型 | 尚无本项目证据 | Med-CMR 论文中商业模型仍强；4B 容量受限 | `ASPIRATIONAL` | 冻结测试；必须报告 CI 与成本 |

状态词只描述 survey 对设计的支持程度。任何 `HYPOTHESIS` 或 `ASPIRATIONAL` 必须由绑定运行的实验产物升级，不能由更多文字升级。
