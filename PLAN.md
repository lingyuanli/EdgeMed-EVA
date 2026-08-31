# Med-CMR Formal Baseline Plan

更新：2026-08-31
当前主阶段：`setup`
目标状态：一条可复算、可比较、绑定来源与运行产物的 Qwen3.5-4B Med-CMR baseline。

## 1. Core Contract

- quest goal：准备正式 Med-CMR Benchmark，然后在单张 V100-SXM2-32GB 上开始跑分。
- 用户核心要求：使用 Qwen3.5-4B；结果可继续支持小模型训练、Agent 和证据评测研究。
- 不可妥协约束：不把代理集、论文报告值、未完成运行或非官方 evaluator 输出称为正式成绩；不把 test 答案用于训练或 prompt 调优。
- chosen route：`reproduce`。
- baseline id：`qwen35-4b-medcmr-b0`。
- variants：`direct_mcq`、`direct_open`；后续 `structured_b1` 不是 B0 的替代结果。
- source paper：Gong et al., Med-CMR, arXiv:2512.00818 / 用户提供 PDF。
- source repo：`https://github.com/LsmnBmnc/Med-CMR`。
- source commit：`808a035ba7ce6831d8c076b8ce1edd68b3604ff8`。
- dataset：`aaassddaadf/Med-CMR`，HF revision `a9b2d6e610c6c5dcf4f3e5aa89c7ec9fd7a05b73`。
- dataset archive：`dataset.zip`，6,895,143,178 bytes，LFS SHA-256 `8495f3b6b2901a095918ab27c600e42da7287a87f6aea490912ba60d27de3775`。
- license：官方仓库和 HF card 均标为 Apache-2.0；数据内嵌来源/再分发边界仍需在解包后核验。
- task：论文报告 16,655 MCQ + 3,998 open-ended；官方 release 实含 16,655 + 3,999；七个复杂推理维度。
- dataset / split：released package 没有 split 字段，作为公开零样本 benchmark test 使用。MCQ JSON 为 16,655 行；open JSON 为 3,999 行，与论文声称的 3,998 相差 1 行。不得自行删除样本；同时报告 released-all 与 paper count mismatch。
- metric contract：
  - MCQ：总体准确率 + SOD/FDD/SU/TP/CR/LTG/MSI 准确率，方向越高越好；
  - Open：Consistency、Coherence、Visual Accuracy、Ground-Truth Correctness 与加权总分，权重 1/1/4/4；
  - Open 官方 judge：DeepSeek-V3.2-Exp；supplement 精确 scoring prompt 已取得，但论文未披露 judge sampling 参数；当前官方 DeepSeek API 已不提供该模型 id，因此 open 评分先降级为 proxy candidate。
- expected command path：`PYTHONPATH=src python -m edgemed_bench.run`；完整参数在 smoke 通过后冻结，正式运行必须使用同一 contract。
- expected outputs：run manifest、resolved config、dataset manifest、predictions JSONL、raw generations、MCQ metrics、open judge records、slice metrics、verification report、durable log。
- acceptance condition：完整 intended split 成功；样本数和哈希一致；指标可从逐样本输出独立复算；无答案泄漏；open evaluator 与论文协议一致或明确降级。
- cheapest fallback：若 open 官方 judge 无法完全复现，先接受 `MCQ-only verified`，将 open 标为 `operational_but_incomparable`，不生成伪官方 open 总分。

## 2. Execution Path

- local working directory：本仓库。
- remote working directory：`/home/ubuntu/EdgeMed-EVA`。
- environment：项目 `.venv`；PyTorch 2.10.0+cu126；固定 Transformers commit；FP16 compute + NF4；eager attention。
- model：`/home/ubuntu/models/Qwen3.5-4B`，已通过真实多模态 smoke。
- required download：官方 `dataset.zip`；固定 revision，顺序下载并校验 LFS SHA-256。当前经本地 `127.0.0.1:7890` 到远端回环 `127.0.0.1:17890` 的 SSH 反向隧道断点续传，仅该下载显式使用代理。
- hardware assumption：1× Tesla V100-SXM2-32GB，10 CPU cores，62GB RAM；2026-08-31 下载中复核 `/home/ubuntu` 可用 101,857,734,656 bytes，覆盖约 6.9GB archive、约 6.9GB 解包和运行产物。
- smoke test：需要。先解包结构 QA，再运行每类至少 2 个样本且总量不超过 20 的无答案可见 smoke。
- smoke command：实现后写入本计划修订日志。
- main run command：实现和 smoke 通过后冻结。
- expected runtime / budget：由 20 样本实测吞吐外推；在吞吐证据前不承诺总时长。
- durable logs：`runs/<run_id>/stderr.log`、`events.jsonl`、`predictions.jsonl`。
- fastest failure signals：字段/图像缺失、模型看到参考答案、解析合法率不足、V100 OOM、judge prompt/模型不匹配。

## 3. Risks And Revision

- 数据包没有 train/dev/test 分离；全量只用于零样本正式 baseline，不能用答案、`visual_description` 或运行结果调 prompt。
- 官方仓库尚无运行代码或 evaluator；需要忠实实现，但实现值不自动等于官方验证。
- DeepSeek-V3.2-Exp 可用性、API alias 和精确 scoring prompt 可能阻塞 open 可比性。
- 6.9GB zip 解压体积未知；先列 central directory 和估算体积，再解压。
- V100 不支持硬件 BF16；不得启用 BF16 或 FlashAttention 2。
- 正式运行不得覆盖远端用户未跟踪的 `plan/`、`requirements.lock.txt`。

## 4. Verification Plan

- required result files：见 metric contract 与 runtime contract。
- required metric keys：`mcq_accuracy_all`、七维 MCQ；open 四维和 `open_weighted_all` 仅在官方 judge 契约通过时启用。
- comparability checks：任务、数据 revision、split、prompt、图像预处理、解析器、judge 模型、权重、seed/temperature、样本覆盖和来源 commit。
- verification verdict：`verified_match` / `verified_close` / `verified_diverged` / `broken`；同时给 feasibility 与 downstream trust。
- downgrade condition：任何 split/evaluator 关键字段未知时不得确认完整 baseline。

## 5. Checklist Link

- checklist path：`CHECKLIST.md`。
- next item：完成 archive 断点续传与 SHA-256 校验，在 V100 环境运行测试和结构化数据准备。

## 6. Revision Log

| Time | What changed | Why | Impact |
|---|---|---|---|
| 2026-08-31 | route 从公开不可得风险转为 official HF reproduce | GitHub README 已链接公开 6.9GB 数据包 | Gate 0 继续，不再以代理集为默认路线 |
| 2026-08-31 | released open count 记为 3,999；open judge 降级 | 数据包比论文多一条，DeepSeek-V3.2-Exp 当前不可用 | MCQ 继续正式复现；open 先生成、后分层评分 |
| 2026-08-31 | 实现 source-pinned 数据准备、无答案推理面、Qwen runner、MCQ scorer 与 exact-resume 校验 | 让正式运行可复算且不向 runner 暴露参考答案 | 等待远端 pytest、数据 SHA 和 bounded smoke 后冻结主命令 |
| 2026-08-31 | Hugging Face 下载改为 SSH 反向代理断点续传 | 远端直连/HF API 不稳定，用户提供本地代理 | 仅改变传输路径，不改变 revision、字节或内容哈希 |
