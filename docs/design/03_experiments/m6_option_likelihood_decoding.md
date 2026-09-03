# M6：候选答案语义条件似然解码

## 1. 突破口

Med-CMR official MCQ 只奖励答案 accuracy；bbox IoU 只能作为独立 E2 指标。既有生成式 B0 在 PMC 外部开发集为 `57.62%`，并存在 invalid parse 与选项顺序敏感；字母 SFT 在 Med-CMR 又发生显著负迁移。M6 不改模型权重，而是把决策从“生成 A–E”改为“分别计算每个完整选项文本在图像+问题后的条件似然”。

## 2. 单一变化

- user prompt 只包含图像和问题，不包含选项字母或候选列表；
- 每个选项文本分别作为 assistant completion 前向计算；
- score 为 completion（含消息结束 token）的平均 token log-probability；
- 选择最高 score，对完全相等值按字母排序仅作确定性 tie-break；
- 不生成自由文本，因此 invalid parse 按构造为 0；
- base 模型、NF4/FP16/eager、图像预算与数据顺序不变。

这一路径约需每题 4 次 forward，牺牲吞吐换取无字母捷径的语义决策。它不读取 reference，原始分数和每个 option-text SHA 均写入 prediction。

## 3. 分阶段门

1. 4 条 operational smoke：必须全部完成、prefix 对齐、finite scores、保存/reload 后无需状态恢复。
2. 冻结 PMC dev 前 64 条 pilot：与已有 direct B0 相同 64 条配对比较。
3. 只有 accuracy 相对 direct 提升至少 `+3.0` points、invalid 为 0，才运行完整 512 条。
4. 完整 512 必须同时满足：相对 direct paired CI 下界大于 0；在 answer-preserving rotate-1 上语义内容选择 100% 一致。由于 user prompt 不含选项列表，rotate-1 只能重映射同一组内容分数，不新增模型调用。

M6 在外部 PMC 开发集通过也不授权直接重跑 Med-CMR；需另一个 source-diverse MCQ 保持集或冻结一次新的正式 milestone。不得根据逐题 PMC 正误修改归一化、prompt 或候选集合。

## 4. 结果：64 条门失败

4 条 operational smoke 全部完成，20 个候选分数 finite，invalid=0，prefix 严格对齐。随后同一合同在前 64 条完成 64/64：耗时 `67.27 s`，峰值 CUDA `4,744.79 MiB`，prediction SHA-256 `6abe680b68f4110069c1aff6a2b795aae66d67448efd69db2c3aa6f6ab0a14fb`。

同 ID 配对结果：direct B0 为 `37/64 = 57.81%`，M6 为 `27/64 = 42.19%`，delta `-15.63` points，95% CI `[-28.13, -3.13]`，McNemar exact `p=0.0414`。M6 invalid 为 0，B0 为 1，但格式收益不能抵消 10 道净正确率损失。

预注册要求至少 `+3.0` points，实际显著为负。因此 M6 停止，不运行 512，不根据逐题结果改 completion 长度归一化、EOM 处理或 prompt。该失败说明 base model 的生成偏好不能直接当作可靠的判别式选项分数；若未来重启此方向，必须作为新的训练目标而非事后 decoder 调参。
