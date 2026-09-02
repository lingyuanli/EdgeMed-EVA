# M1 real smoke8：证据状态机失败到 operational closure

日期：2026-09-02

最终状态：`OPERATIONAL PASS / EFFICACY DEFER`

完整设计与命令见 `docs/design/04_implementation/qwen_agent_m1_runtime.md`。

## 保留的三个 8 条 run

1. `runs/qwen35-4b-medical-agent-m1-pmc-smoke8-schemav2-20260902`：commit `dc31336`，在 6/8 后停止。第 7 条 backend 首步输出 `tool_call=null`，无成功视觉证据，控制器正确阻断 final。
2. `runs/qwen35-4b-medical-agent-m1-pmc-smoke8-policy1-20260902`：commit `01a0ebf`，8/8 inference 完成；E0 schema 0.875，citation/trace 1.0；verifier `BLOCK`。唯一失败是 forced overview evidence 带非空 region。
3. `runs/qwen35-4b-medical-agent-m1-pmc-smoke8-policy2-20260902`：commit `6de6909`，8/8 inference、8/8 tool、E0 三项全 1，verifier `PASS`。一次 policy intervention 和一次 canonicalization 均在轨迹中留痕。

最终 run 的 predictions/tool traces/trajectories SHA-256 分别为：

- `c6637cf3703e8ae6adb6c8d0d4dc53ad7c8e65ceb9a44a6c349d87c95cedae2c`
- `e0577b582fa0467d4486f1f9ee98291fbc3f131871b92910a80e10a880f97617`
- `9f950dd7dfc8cac31991808c669b19ac785451c030ff6397807dcf03f8b11c01`

峰值 CUDA allocated/reserved 为 3,562.71/3,946 MiB。E1 为 3/8，只是 operational observation；E2、E3 和医学专家正确性没有相应证据，保持 `DEFER`。
