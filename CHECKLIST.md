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

## Medical Multimodal Agent M0

- [x] 用户提供的 VideoSeek/VideoSpy 目录按不可信参考材料完成行为审计
- [x] 参考源码哈希、缺失 license/test suite 的再分发风险已记录；未 vendor 原包
- [x] `inspect_overview`、`temporal_skim`、`region_inspect` 确定性工具已实现
- [x] per-run 显式工具 allowlist、重复调用拒绝和失败 trace 已实现
- [x] 至少一个成功视觉产物才能 final，decision 与 finalizer 调用已分离
- [x] inference manifest 硬拒绝 reference 字段；scorer-only reference access 已测试
- [x] E0/E1/E2/E3 分层 scorer 与哈希/复算 verifier 已实现
- [x] synthetic closure fresh audit 暴露并修复 `region_inspect` 哈希错误与 verifier 无质量门问题
- [x] run manifest 显式声明 E0 最低阈值和最大失败工具数；缺失质量门时 verifier BLOCK
- [x] repaired synthetic artifact closure 七项 verifier 检查 PASS；原 false-pass 形状被回归测试拒绝；全仓 73 tests 通过
- [x] Qwen3.5-4B backend adapter 与可恢复 batch runner 已实现并通过 mock exact-resume contract
- [x] reference-only finalize-eval 阶段已实现；推理结束前 run 目录不存在 references
- [x] V100 real preflight-1 完成模型/工具路径并由 E0 质量门正确 BLOCK；失败产物保留
- [x] final schema-v2 同样本 answer-blind operational retry 通过 E0 门
- [x] 单 V100、8 条 answer-blind real-model operational smoke 完成；E0 全 1、8/8 工具成功、峰值显存收据已绑定
- [ ] source-diverse 开发集上的 no-tool/tool/forced/oracle/compute-matched 对照完成
- [ ] 工具净收益与 evidence dependency 通过多 seed/干预门
- [x] SLAKE train detection 构建答案隔离的 locator surface；与 validation 图像 SHA 零重叠
- [x] 单 V100 locator QLoRA 2-step save/reload smoke 与 64-step pilot 完成
- [x] locator-64 在 43 张未见图像上通过定位门：mean IoU 0.3299、IoU@0.3 58.14%
- [x] full/oracle-crop/black-crop 答案因果筛选完成并保留失败：crop-only 相对 full `-9.26` token-F1 points
- [x] full+oracle-crop 多视图门通过：相对 full `+7.59`、相对 full+black `+8.37`，两个 paired CI 下界均为正
- [x] locator-64 learned crop 在 validation 达到 `+1.98` token-F1 与 26.06% oracle 增量捕获；标记为 CI 未闭合的 pilot signal
- [x] 45 条零重叠 SLAKE official test 一次性 held-out 完成；定位泛化但答案增益未复现，crop-fusion 路线关闭

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
- [x] best-SFT full Med-CMR MCQ milestone completed and independently scored: 24.3591%, significant -2.8100 point regression
- [x] M1a archived without per-sample test tuning or repeated test evaluation
- [x] M1a transfer-failure campaign and non-Med-CMR stop rules frozen
- [x] SLAKE source-diverse retention comparison complete; M1a improved exact/F1 with paired CIs wholly positive
- [x] PMC option-label invariance comparison complete; M1a significantly less content-consistent than B0
- [ ] golden validation annotation protocol frozen before model scoring
