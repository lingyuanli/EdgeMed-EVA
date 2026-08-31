# Med-CMR Formal Baseline Checklist

## Identity

- baseline id：`qwen35-4b-medcmr-b0`
- route：`reproduce`
- owner stage：`baseline`
- current phase：`analysis`

## Core

- [x] baseline object and route are explicit
- [x] dataset schema、内部 split、图像数量和答案字段已核验
- [ ] metric contract 已完整冻结；open evaluator 不再有关键未知项
- [x] `PLAN.md` 已记录 route、来源、预期产物、接受条件和 fallback
- [ ] 官方 archive 已按固定 revision 下载并通过 SHA-256/数量核验
- [ ] runner/evaluator 已实现并通过测试
- [ ] bounded smoke 已运行一次且产物完整
- [ ] real run 决策已根据 smoke 吞吐和磁盘证据记录
- [ ] 正式运行已完成，预期结果文件与指标齐全
- [ ] 逐样本结果可独立复算且 verification report 完成
- [ ] baseline 已接受、降级或阻塞，并留下 durable note

## Source Audit

- [x] paper source identified
- [x] official repo identified and commit frozen
- [x] official HF dataset identified and revision frozen
- [x] paper read enough to restate MCQ/open metric family
- [x] supplementary generation/scoring prompt 已核验
- [ ] DeepSeek-V3.2-Exp 不可用偏差已获得官方替代/作者确认，或 open 被正式降级
- [ ] 数据内许可/来源边界核验完成

## Runtime

- [x] remote working directory confirmed
- [x] V100 environment and Qwen3.5-4B model smoke verified
- [ ] dataset path and free-space budget confirmed
- [ ] exact run command/config frozen
- [ ] durable log and checkpoint/resume paths verified

## Closeout

- [ ] concise trusted baseline summary written
- [ ] canonical `baselines/local/qwen35-4b-medcmr-b0/json/metric_contract.json` written
- [ ] baseline artifact/report written
- [ ] next stage named explicitly
