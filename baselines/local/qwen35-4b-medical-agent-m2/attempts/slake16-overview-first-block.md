# M2 P1 overview-first operational screen

日期：2026-09-04

状态：`INFERENCE COMPLETED / OPERATIONAL GATE BLOCK`

- code commit：`a5f3362743f597fa9142936ccab30dd57634b38d`
- run：`runs/qwen35-4b-medical-agent-m2-slake16-overview-first-20260904`
- surface：SLAKE answer-blind binary unique-image surface 的前 16 条；inference manifest 不含 reference
- 16/16 inference completed；16/16 `inspect_overview` completed；failed tools 0
- E0 schema/citation/trace：`1.0/1.0/1.0`
- targeted region：`0/16 = 0%`，低于预注册 30% 子门
- 32 model calls；input/output tokens `31,830/5,039`
- peak CUDA allocated/reserved：`3,523.60/3,782 MiB`
- operational analyzer overall：`BLOCK`，退出码 1

所有 16 个 backend decision 在看完 overview 后均返回 `tool_call=null`。因此 P1 没有实际执行 coarse-to-fine zoom，取消 96 条正确率评测，且未在本轮读取 references。下一单变量诊断是 P2 forced-localizer；它只估计定位能力上界，不是部署策略或效果结论。
