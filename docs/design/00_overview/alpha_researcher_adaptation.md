# AlphaResearcher-Bench 适配说明

## 1. 本轮参考范围

本轮只把 AlphaResearcher-Bench 当作“研究流程与证据治理参考实现”，并检查了其当前设计入口、bottleneck-oriented survey、Student open-loop ownership、structured claim-evidence verification、Stage 1–9 runtime、shared campaign runtime 和 baseline case study。

没有把 AlphaResearcher-Bench 自身实验结果当作医学方法证据，也没有复制其业务状态或历史文档。其工作树在探索时存在未提交运行状态且落后远端；本项目又尚无可信 Qwen3.5-4B 主基线，因此未启动 formal campaign。当前产物是 survey 与 executable design，不是 campaign 结果。

## 2. 机制映射

| AlphaResearcher-Bench 成熟机制 | 本项目最小适配 | 产生的文件/行为 |
|---|---|---|
| `docs/design/README.md` 权威入口 | 单一设计入口与冲突顺序 | `docs/design/README.md` |
| 宽搜→深读→结构化综合 | 20 个候选、7 个深读、claim/evidence/counterevidence/gap | `01_survey/` |
| Student 是研究产物唯一作者 | runner/trainer 产生候选；verifier 只读并给状态 | runtime contract |
| CEC typed evidence obligations | 静态、运行、证据、因果、专家五类检查分离 | trusted evaluation + verifier report |
| PASS/DEFER/BLOCK | 只有已执行且绑定产物的检查可 PASS | stage gates |
| Stage 1–9 单状态所有者 | 单 runner、阶段状态、样本 chunk 精确恢复 | runtime contract |
| shared campaign append-only lifecycle | `events.jsonl` 和不可改写的 run manifest | runtime contract |
| baseline source/adapter/reproduction 分离 | paper-reported、official、local reproduction 分层 | experiment protocol |
| frozen evidence 与结果 cell 追踪 | claim → experiment → run → hashes | traceability table |

## 3. 没有移植的部分

- 完整的多角色 Mentor/Student 编排；
- 常驻 CEC 服务、repair agent 和 supervisor feedback loop；
- 共享 GPU 调度器、远程 slot 管理和 campaign dashboard；
- 面向多课题并发的数据库/账本 schema；
- 自动论文生成和恢复流程。

理由：当前只有一个 benchmark、一个 4B 主干和一个最小 Agent 路径。上述抽象尚无两个真实复用场景，过早加入会让训练/工具因果更加难以追踪。

## 4. 何时升级为正式 campaign

满足以下条件后，才创建正式实验 PLAN/CHECKLIST 和 campaign ledger：

1. Gate 0 数据/evaluator 可得；
2. B0/B1 至少一个完整、可复算主运行；
3. 主要 failure slice 和唯一主指标冻结；
4. 至少两个模型阶段需要多 seed/消融并发；
5. 运行恢复和 verifier 已通过 smoke。

此时 campaign 的目标是系统化验证主结果，不是用更多实验搜索一个尚不存在的主结果。
