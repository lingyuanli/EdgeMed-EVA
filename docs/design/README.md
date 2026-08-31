# Med-CMR 小模型多模态医学推理：设计文档入口

状态：`DESIGN-READY`（设计已收敛；尚无本项目实测主结果）
更新：2026-08-31

## 1. 文档权威顺序

发生冲突时，按以下顺序解析：

1. 本文件给出的范围、状态和术语；
2. [系统蓝图](00_overview/system_blueprint.md)；
3. [方法设计](02_method/evidence_grounded_agent.md)、[训练课程](02_method/training_curriculum.md)、[可信评测](02_method/trusted_evaluation.md)；
4. [实验协议](03_experiments/experiment_protocol.md)与[阶段门](03_experiments/stage_gates.md)；
5. [运行与产物契约](04_implementation/runtime_and_artifact_contract.md)和[GPU 资源计划](04_implementation/gpu_resource_plan.md)；
6. Survey 文档仅提供设计依据，不能覆盖后续实验事实。

## 2. 当前边界

本项目以 Med-CMR 为 benchmark，以 Qwen3.5-4B 为主要端侧 backbone，目标是形成三条可单独验证的贡献：

- **小模型垂域能力**：4B 模型在公平、无泄漏条件下超过强开源基线，并向商业模型逼近；
- **专属多模态 Agent**：针对细粒度视觉证据、跨视图/时间整合和长尾临床逻辑进行受控工具调用；
- **证据 + 结果多维评测**：答案、证据、因果依赖、校准、工具净收益和成本同时报告。

当前已经完成的是“探索 + 方法与实验设计”。下列内容仍是待执行事项，不能写成结果：

- Qwen3.5-4B 在 Med-CMR 上的真实零样本和结构化提示基线；
- Med-CMR 数据许可、正式测试集和官方 evaluator 的可获得性核验；
- 任何超过论文/排行榜 baseline 的数字；
- 任何关于工具、训练或 Agent 的因果增益结论。

官方 Med-CMR GitHub 当前主要公开 README 和 leaderboard；因此，真正开跑前必须通过数据与 evaluator 可得性门。无法获得官方测试包时，只能建立明确标注的开发代理集，不能冒充官方成绩。

## 3. 文档地图

| 区域 | 目的 | 核心输出 |
|---|---|---|
| `00_overview` | 固定问题、系统分层和最小闭环 | 系统蓝图、贡献假设、AlphaResearcher 适配 |
| `01_survey` | 宽搜、深读、矛盾和研究空白 | 文献台账、证据矩阵、怀疑者报告 |
| `02_method` | 定义 Agent、训练和评测 | 可实现接口、损失、对照与指标 |
| `03_experiments` | 定义迭代顺序和停止条件 | baseline ladder、消融、阶段门 |
| `04_implementation` | 定义可复现运行与资源 | 产物 schema、精确恢复、GPU 估算 |
| `sources` | 保存外部材料入口 | [Survey 来源台账](../../sources/survey_sources.md) |

## 4. 从 AlphaResearcher-Bench 借用什么

本设计参考 AlphaResearcher-Bench 的成熟做法，但只移植与当前规模匹配的最小部分：

- Survey 采用“宽搜 → 深读 → claim/evidence/counterevidence/gap 综合”；
- 设计、执行、写作产物只有研究主流程可以改写，评测器只返回证据状态；
- 结构合法不等于科学正确；只有绑定运行、检查点和产物哈希的验证结果才能支持结论；
- baseline 来源快照、适配器和本地复现分开保存；
- 运行采用单一状态所有者、append-only 事件和精确恢复。

当前不引入常驻 supervisor、分布式调度器、完整 CEC 服务或新的账本框架。第一版用静态 schema、确定性 verifier 和少量脚本实现同一原则；只有出现至少两个真实复用路径后才升级为服务。

逐项映射和未采用项见 [AlphaResearcher-Bench 适配说明](00_overview/alpha_researcher_adaptation.md)。

## 5. 一句话执行路线

`数据/评测可得性门 → B0/B1 可信基线 → 错误切片 → 证据 SFT → grounded preference → 选择性工具策略 → 可选 GRPO → 冻结测试 → 统计验证 → 论文主张`。
