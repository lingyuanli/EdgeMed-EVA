# 阶段门与停止规则

## Gate 0：数据/评测可得

通过条件：许可明确；样本与论文规模可解释；20 题人工核验通过；evaluator 能复现示例；泄漏检查可执行。

失败后：仅允许使用明确命名的代理集做工程开发，不报告官方 Med-CMR 成绩。

## Gate 1：基线可信

通过条件：B0/B1 全量 validation 成功率 ≥99%；无静默图像缺失；配置、输出、评分和日志可绑定；重复运行差异在预期范围。

失败后：修输入/评测，不得进入训练。基础设施问题不计科学 revision。

## Gate 2：证据数据可信

通过条件：provenance 100%；blocked overlap 为 0；schema 合法率 ≥99%；黄金子集双标一致性达到预注册阈值；按模态/维度覆盖无明显空洞。

失败后：隔离问题数据，重新 QA；不得用 evaluator 分数反向补标签。

## Gate 3：SFT 有效

通过条件：T1b 相对 B1 的主答案指标 CI 下界 >0；证据主指标提高；关键维度/保持集回退不超过 3 个绝对点；三 seed 至少 2/3 同方向。

失败后：停在数据/监督设计，禁止直接上 RL。

## Gate 4：偏好优化有效

通过条件：T2 相对 T1b 至少改善答案或证据主指标，另一项无显著回退；unsupported observation 降低；人工审计未发现系统性 reward/judge 偏差。

失败后：保留 T1b 为候选最终模型。

## Gate 5：Agent 真增益

通过条件：同一 checkpoint 的 Net Tool Effect 95% CI 下界 >0；compute-matched 对照不能解释全部增益；Call Harm 在预注册阈值内；Relevant ROI deletion 效应显著大于 Irrelevant ROI deletion。

进入 Gate 5 效果评测前先过 operational 子门：overview-first 16/16 完成、E0 全 1、零失败工具，且至少 30% 样本产生面积 `[0.01,0.64]` 的非全图 ROI。该子门只检查策略是否实际执行，不得读取参考答案。

失败后：Agent 降级为可选工程特性，不能作为核心科研卖点。

## Gate 6：在线 RL 可启动

通过条件：Gate 0–5 均通过；reward 各分量和人工标签一致；256–512 题 smoke 无 OOM/死循环/奖励黑客；精确恢复验证通过；GPU 预算已批准。

失败后：最终方法停在离线 preference/tool policy。

## Gate 7：冻结测试可发布

通过条件：test 仅按预注册运行；所有结果有 commit/config/checkpoint/data/evaluator 哈希；统计脚本通过；主张—证据映射无越界；临床非适用声明存在。

## 状态语义

| 状态 | 含义 |
|---|---|
| `DESIGNED` | 文档和接口已定义 |
| `IMPLEMENTED` | 代码存在但未证明可运行 |
| `SMOKE_VERIFIED` | 小样本真实运行通过 |
| `MEASURED` | 冻结数据上生成指标 |
| `STATISTICALLY_VERIFIED` | 统计与复现实验通过 |
| `BLOCKED` | 前置条件不满足 |

不得从 `DESIGNED` 直接写成“已有效”。
