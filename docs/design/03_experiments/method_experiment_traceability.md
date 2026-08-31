# 方法—实验—主张追踪表

| 方法组件 | 预期机制 | 必需对照 | 主指标 | 反证信号 | 可支持主张 |
|---|---|---|---|---|---|
| 结构化 evidence packet | 降低无依据观察，使证据可查 | B1 vs B0；M1b vs M1a | grounded success、unsupported rate | 仅 schema success 上升 | 结构可审计/证据一致性 |
| 竞争假设 | 显式使用反证，减少单一路径确认偏差 | 去除 hypotheses | CR/TP、contradiction accuracy | 生成更长但答案不变 | 临床逻辑整合 |
| 跨视图/时间绑定 | 避免视图和时点混淆 | 去标签/打乱标签 | MSI/TP、shuffle sensitivity | 打乱不影响答案 | 多源证据整合 |
| Grounded preference | 压制看似合理的视觉幻觉 | T2 vs T1b | unsupported rate、VA、GT | judge 分升、人工不升 | 视觉一致性优化 |
| 选择性 crop | 获取关键局部细节 | 同 checkpoint no-tool/forced/oracle | Net Tool Effect、FDD | Call Harm 高、oracle 无收益 | Agent 工具贡献 |
| 校准训练 | 让置信度反映错误风险 | 去 calibration loss | ECE/Brier/AURC | accuracy 不变且 ECE 变坏 | 可选择性预测 |
| 4-bit 端侧部署 | 降低资源成本 | BF16/8-bit/4-bit | accuracy drop、latency、memory | 量化导致维度崩溃 | 端侧效率 |

## 主表结果 cell 的最小证据包

每个结果 cell 必须能解析到：

```text
claim_id
  -> experiment_id
     -> run_ids (all seeds)
        -> code_commit
        -> config_hash
        -> data_manifest_hash
        -> checkpoint_hash
        -> evaluator_hash
        -> per_sample_output_hash
        -> metrics_hash
```

缺任一关键绑定时，表格可保留为 `unverified`，不得用于“超过 baseline”的摘要结论。
