# 运行与产物契约

## 1. 原则

首版使用单进程/单一状态所有者的轻量运行器，不新建 daemon、队列、服务化 supervisor 或数据库。目标是让每个结果能恢复、能复算、能追责，而不是复制大型研究平台。

## 2. 建议目录

```text
configs/
  baseline/
  train/
  eval/
data/
  manifests/
  overlap_reports/
runs/<run_id>/
  run_manifest.json
  resolved_config.yaml
  events.jsonl
  checkpoints/
  predictions.jsonl
  tool_traces.jsonl
  metrics.json
  slice_metrics.json
  verifier_report.json
  stderr.log
artifacts/
  data_access_receipt.json
  evaluator_receipt.json
  frozen_test_receipt.json
```

代码实现时 `runs/` 和受许可数据应加入 `.gitignore`；小型 manifest、schema 和汇总指标可版本化。

## 3. `run_manifest.json`

```json
{
  "schema_version":"med-agent-run/v1",
  "run_id":"20260831_b1_seed0_xxxxxxxx",
  "stage":"baseline|sft|preference|tool_policy|rl|eval",
  "status":"created|running|completed|failed|stopped",
  "created_at":"ISO-8601",
  "code_commit":"...",
  "dirty_worktree":false,
  "config_sha256":"...",
  "data_manifest_sha256":"...",
  "base_model":"Qwen/Qwen3.5-4B",
  "base_model_revision":"immutable revision",
  "checkpoint_sha256":"...",
  "prompt_version":"...",
  "tool_version":"...",
  "evaluator_version":"...",
  "seed":0,
  "hardware":"1x RTX 4090 24GB",
  "parent_run_id":null
}
```

正式结果要求 clean worktree 或保存完整 source snapshot hash；dirty run 只用于开发，不进入主表。

## 4. Append-only 事件

```json
{"seq":1,"time":"...","event":"run_started","sample_cursor":0}
{"seq":2,"time":"...","event":"checkpoint_saved","path":"...","sha256":"...","sample_cursor":256}
{"seq":3,"time":"...","event":"eval_chunk_completed","range":[0,255],"output_sha256":"..."}
```

运行状态只有 runner 写；监控脚本只读。恢复时验证 run id、代码/配置/数据/父 checkpoint 哈希和最后一个完整 chunk。哈希不一致时创建新 run，不在原 run 上“续写”。

## 5. 逐样本输出

`predictions.jsonl` 每行包含：sample id、输入 manifest 引用、完整模型输出、解析状态、耗时、token、tool trace ids、checkpoint hash 和 evaluator input hash。原始输出不能只保留解析后的答案。

`tool_traces.jsonl` 每次调用保存：请求、预算、输入图像哈希、区域、输出图像哈希、状态、耗时和错误。图像本体可放 content-addressed artifact store。

## 6. Verifier report

```json
{
  "schema_version":"med-agent-verification/v1",
  "run_id":"...",
  "checks":[
    {"name":"manifest_bound","status":"PASS","evidence":"..."},
    {"name":"sample_count","status":"PASS","expected":1000,"actual":1000},
    {"name":"schema_validity","status":"PASS","rate":0.997},
    {"name":"metric_recompute","status":"PASS","metrics_sha256":"..."}
  ],
  "overall":"PASS|DEFER|BLOCK"
}
```

- `PASS`：检查执行并有绑定证据；
- `DEFER`：该检查不适用或需要后续专家验证；
- `BLOCK`：关键绑定、数量或复算失败。

静态 verifier 不能把医学正确性标为 PASS；专家/黄金参考是独立检查。

## 7. Baseline 来源分层

对每个外部 baseline 保存：

1. `source_record`：论文/仓库/模型版本和原始报告分数；
2. `adapter_record`：本项目输入输出适配，不能改核心推理；
3. `reproduction_record`：实际运行和指标；
4. `deviation_record`：任何硬件、prompt、量化或数据差异。

没有 reproduction record 的数值标为 `paper-reported`，不放入 paired significance test。

## 8. 最小实现模块

```text
src/
  schemas.py              # 输入、evidence、tool、run schema
  data/loader.py          # manifest 驱动加载
  agent/state_machine.py  # 唯一流程控制
  tools/crop_zoom.py      # 确定性工具
  models/qwen35.py        # 推理/训练适配
  eval/medcmr.py          # 官方或代理 evaluator wrapper
  eval/evidence.py        # E0-E3 指标
  runtime/runner.py       # run/event/checkpoint 所有者
  verify/run_verifier.py  # 纯函数验证
tests/
  test_schemas.py
  test_crop_zoom.py
  test_resume.py
  test_metric_recompute.py
```

实现顺序：schema → 20 题 loader/evaluator → B0 runner → evidence parser → crop tool → Agent → training。不要在 baseline 可运行前实现 RL。

## 9. 精确恢复测试

在 50 题 smoke 中主动于第 23 题终止，恢复后要求：

- 前 23 题输出哈希不变且不重复；
- 后 27 题完整；
- 合并指标与一次性运行一致；
- 不完整 chunk 被丢弃而非混入；
- config/checkpoint/data 任一哈希改变时拒绝原 run 恢复。
