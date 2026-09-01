# Med-CMR Formal Baseline Checklist

## Identity

- baseline id：`qwen35-4b-medcmr-b0`
- route：`reproduce`
- owner stage：`baseline`
- current phase：`MCQ verified / optimization next`

## Core

- [x] baseline object and route are explicit
- [x] dataset schema、内部 split、图像数量和答案字段已核验
- [ ] metric contract 已完整冻结；open evaluator 不再有关键未知项
- [x] `PLAN.md` 已记录 route、来源、预期产物、接受条件和 fallback
- [x] 官方 archive 已按固定 revision 下载并通过 SHA-256/数量核验
- [x] runner/evaluator 已实现并通过测试
- [x] bounded smoke 已运行一次且产物完整
- [x] real run 决策已根据 smoke 吞吐和磁盘证据记录
- [x] 正式 MCQ 运行已完成，预期结果文件与指标齐全
- [x] MCQ 逐样本结果可独立复算且 verification report 完成
- [x] baseline 已收口：MCQ `verified_diverged`；Open `operational_but_incomparable`

## Source Audit

- [x] paper source identified
- [x] official repo identified and commit frozen
- [x] official HF dataset identified and revision frozen
- [x] paper read enough to restate MCQ/open metric family
- [x] supplementary generation/scoring prompt 已核验
- [x] DeepSeek-V3.2-Exp 不可用偏差已正式记录，Open 降级为 `operational_but_incomparable`
- [ ] 数据内许可/来源边界核验完成

## Runtime

- [x] remote working directory confirmed
- [x] V100 environment and Qwen3.5-4B model smoke verified
- [x] dataset path and free-space budget confirmed
- [x] exact run command/config frozen
- [x] durable log and checkpoint/resume paths verified

## Closeout

- [x] concise trusted baseline summary written
- [x] canonical `baselines/local/qwen35-4b-medcmr-b0/json/metric_contract.json` written
- [x] baseline artifact/report written
- [x] next stage named explicitly

## Optimization Data Gate

- [x] B1-v2 answer-blind operational smoke passed
- [x] external development manifest schema designed
- [x] provenance/file/exact/near-overlap validator implemented
- [x] validator unit/CLI tests pass in the project environment
- [x] external source datasets selected and immutable revisions/license boundaries recorded
- [x] real external manifests generated: SLAKE 1,053; PMC dev 512; PMC train 1,968 accepted + 32 quarantined; all final gates `passed`
- [x] direct-vs-B1 external development comparison completed with 512/512 paired predictions
- [x] B1 answer line archived after significant -17.7734 point development regression
- [x] single-V100 two-step T1a QLoRA backward/save smoke passed with finite/applied gradients and adapter hash
- [x] saved T1a adapter reload/inference smoke passed (4/4 complete, zero invalid)
- [x] 128-step T1a seed 20260901 completed: +4.1016 points, paired CI lower bound >0
- [x] T1a seeds 20260902/20260903 completed; 3/3 directions and CI lower bounds positive
- [x] seed 20260903 512-step study completed; 128-step incumbent retained
- [x] best-SFT checkpoint frozen before Med-CMR access (seed3/128, hash-bound)
- [ ] best-SFT full Med-CMR MCQ milestone run completed and independently scored
- [ ] golden validation annotation protocol frozen before model scoring
