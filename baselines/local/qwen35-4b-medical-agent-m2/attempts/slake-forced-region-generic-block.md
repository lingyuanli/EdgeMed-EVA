# M2 P2 generic forced-region screen

日期：2026-09-04

状态：`OPERATIONAL GATE BLOCK / EFFICACY NOT RUN`

## 失败证据

- 首次冻结 16 条 run：`runs/qwen35-4b-medical-agent-m2-slake16-forced-region-20260904`
- code commit：`b3734fdf17667ab9d6c9c9c8ab1472ca4890fe52`
- run contract SHA-256：`acd9ef6cd36b2fae60b4cd813f06e8d7f58179a25944ab3c21924992e42c47b1`
- 结果：第 1 条即触发 `RuntimeError`，`completed_total=0`；run manifest `status=failed`、`scientific_result=false`
- 单变量 prompt 优先级修复后的 1 条重试：`runs/qwen35-4b-medical-agent-m2-slake1-forced-region-prompt2-20260904`
- 修复 commit：`433d1e0775f47f1852029722bd3a9bb2c7160dfe`
- 重试 contract SHA-256：`37ff066888711ed000282170de07bc08d19bd15b05395c200acaf4dcf45e7bbd`
- 结果：同样在第 1 条失败，`completed_total=0`；模型加载和 overview 工具均成功

## 最小重现与判定

同一首条问题 `Is the lung healthy?` 在完成 overview 后，即使 controller requirement 已提升到决策指令顶层，通用 decision 仍返回：

```json
{"content":"evidence is sufficient","tool_call":null}
```

可证伪根因是：通用 decision schema 保留合法的 `tool_call=null` 最短出口；当前 Qwen3.5-4B 没有可靠服从强制 region 约束。两个 run 都未进入 reference scoring，因此不能解释为模型正确率或工具增益。

下一步只允许测试 P3 dedicated locator：定位 schema 移除 null/答案出口，同时保持 overview、工具预算、finalizer、解码和样本不变。P3 仍失败时停止 zero-shot crop prompt 迭代。
