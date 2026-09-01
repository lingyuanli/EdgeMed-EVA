# Med-CMR Formal Baseline Plan

更新：2026-09-01
当前主阶段：`B1 dev regression archived / M1a backward smoke next`
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
- model：`/home/ubuntu/models/Qwen3.5-4B`，HF revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，已通过真实多模态 smoke；正式 run contract 绑定 source manifest 哈希。
- required download：官方 `dataset.zip`；固定 revision，顺序下载并校验 LFS SHA-256。当前经本地 `127.0.0.1:7890` 到远端回环 `127.0.0.1:17890` 的 SSH 反向隧道断点续传，仅该下载显式使用代理。
- hardware assumption：1× Tesla V100-SXM2-32GB，10 CPU cores，62GB RAM；2026-08-31 下载中复核 `/home/ubuntu` 可用 101,857,734,656 bytes，覆盖约 6.9GB archive、约 6.9GB 解包和运行产物。
- smoke test：需要。先解包结构 QA，再运行每类至少 2 个样本且总量不超过 20 的无答案可见 smoke。
- smoke command：`PYTHONPATH=src .venv/bin/python -m edgemed_bench.run --kind mcq --manifest /home/ubuntu/data/medcmr/release_a9b2d6e6/manifests/mcq.jsonl --data-root /home/ubuntu/data/medcmr/release_a9b2d6e6/raw --model-path /home/ubuntu/models/Qwen3.5-4B --model-source-manifest baselines/local/qwen35-4b-medcmr-b0/source_manifest.json --run-dir runs/qwen35-4b-medcmr-b0-mcq-contract-smoke-20260831T0421Z --sample-id-file runs/_selections/medcmr-mcq-2-per-task.txt --sync-every 1`。
- main run command：`scripts/run-medcmr-b0-mcq-full.sh`；同一 runner、model/data/source contract，去掉 `--sample-id-file`，run dir 固定为 `runs/qwen35-4b-medcmr-b0-mcq-full-20260831T0427Z`，`--sync-every 10`；通过 tmux `medcmr-b0-mcq-full` 持久运行。仅当原 contract 不变且需要精确续跑时设置 `EDGEMED_EXACT_RESUME=1`。
- expected runtime / budget：最终 14 条 contract smoke 推理 17.33 秒（模型加载外），线性点估计约 5.7 小时；考虑图像/token 差异，操作预算 6–8 小时。
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
- next item：在单张 V100 上运行冻结的两步 `m1a-answer-qlora` backward/save smoke；B1-v2 已因外部 development 显著回退归档。

## 6. Revision Log

| Time | What changed | Why | Impact |
|---|---|---|---|
| 2026-08-31 | route 从公开不可得风险转为 official HF reproduce | GitHub README 已链接公开 6.9GB 数据包 | Gate 0 继续，不再以代理集为默认路线 |
| 2026-08-31 | released open count 记为 3,999；open judge 降级 | 数据包比论文多一条，DeepSeek-V3.2-Exp 当前不可用 | MCQ 继续正式复现；open 先生成、后分层评分 |
| 2026-08-31 | 实现 source-pinned 数据准备、无答案推理面、Qwen runner、MCQ scorer 与 exact-resume 校验 | 让正式运行可复算且不向 runner 暴露参考答案 | 等待远端 pytest、数据 SHA 和 bounded smoke 后冻结主命令 |
| 2026-08-31 | Hugging Face 下载改为 SSH 反向代理断点续传 | 远端直连/HF API 不稳定，用户提供本地代理 | 仅改变传输路径，不改变 revision、字节或内容哈希 |
| 2026-08-31 | Qwen3.5-4B 固定到 HF revision `851bf6e8…`，两片权重 SHA-256 与 HF 下载收据一致 | 排除本地目录漂移或损坏 | run contract 新增 model source manifest 哈希 |
| 2026-08-31 | 首次 archive 字节门失败；确认两个被 SSH 中断遗留的 orphan `curl` 与 aria2 并发写同一目标 | 文件多出 238,483,190 bytes 且 ZIP 尾目录损坏；属于传输控制面事故 | 精确终止本任务遗留 PID，坏文件隔离保留，单 writer 从零重下；不改变科学配置、不消耗实验 revision |
| 2026-08-31 | 首次 14 条 MCQ smoke 为 100% invalid parse；固定 Qwen README 与 Jinja 证明默认 thinking 消耗了 16-token final-answer 预算 | 输出均停在思考前缀，不能解释为 0% 模型正确率 | 按模型官方接口设置 `enable_thinking=False` 并写入 contract；Med-CMR prompt、样本、答案隔离与 token 上限不变，重新 smoke |
| 2026-08-31 | non-thinking smoke 中 10/14 为单字母，3/14 以明确 `A)`–`E)` 标签开头，1/14 被 16 tokens 截断；论文仅披露 regex extraction | 严格整行 parser 与任意 16-token cap 造成实现性 parse loss | 冻结行首 option-label regex 和 64-token MCQ 上限；不从正文猜字母、不依据正确率调参，执行最终 contract smoke |
| 2026-08-31 | 最终 64-token contract smoke：14/14 完成，13/14 可解析，17.33 秒，峰值 3,419 MiB；256-token 诊断仍使同一格式失败样本截断 | 继续增加 token 不能解决该模型的 prompt 不遵循 | 保持 64-token contract，将该类输出计 invalid；允许启动全量 16,655 MCQ |
| 2026-08-31 | 全量 MCQ 通过 tmux 启动，run contract `7de0a22e…`；首轮检查 predictions 从 47 增至 61，真实 Python/GPU/log 均移动 | 跨过仅有控制面状态的假运行边界 | 状态保持 `running`；完成前不报告正式准确率，收据见 baseline `execution.md` |
| 2026-09-01 | 全量 MCQ 完成并经独立实现逐样本复算 | 16,655 个唯一样本完整覆盖，退出码 0，运行/来源/预测哈希一致 | MCQ 以 `verified_diverged` 收口：27.1690%（4,525/16,655）；Open 因 exact judge 不可用正式降级为 `operational_but_incomparable` |
| 2026-09-01 | B1 v1/v2 在同一 14 样本 answer-blind smoke 上完成一次预注册格式修复 | v1 仅 8/14 严格 schema；v2 简化为短 observation + answer 后 14/14 严格 JSON，且未访问 references | B1-v2 仅获 operational promotion；准确率评测继续由外部 development 数据 gate 阻塞 |
| 2026-09-01 | SLAKE English validation 真实 manifest 通过两阶段 overlap gate | 一阶段 dHash 对相似胸片产生 278 条候选；24 个唯一图像对经 correlation≥0.98 确认门和逐对审计均为非重复 | 1,053 条跨数据集记录获准使用；B1 MCQ 选择仍等待 PMC-VQA disjoint dev |
| 2026-09-01 | PMC-VQA 512 条冻结开发集完成 direct-vs-B1 成对比较 | B1 512/512 严格 JSON，但准确率 39.8438%，相对 direct 57.6172% 回退 17.7734 点；bootstrap 95% CI `[-22.8516,-12.5000]`，McNemar `p=1.05e-10` | 归档零样本 B1 答案优化线；不消费新的 Med-CMR test 评测，转入独立的 direct-answer QLoRA smoke |

## 7. External Development Gate Contract

- experiment tier: `auxiliary/dev`; no GPU and no benchmark scoring.
- changed surface: external-data admission only; B0/B1 prompts, model, inference, and scorer remain frozen.
- primary output: `edgemed-external-data-gate/v1` report with `status=passed`.
- abandonment condition: license/provenance cannot be established, real files cannot be hash-bound, or overlap findings cannot be resolved without using Med-CMR labels.
- frozen training seed: PMC-VQA v2 at HF revision `b56ae594f794867893143b337b4118a835794647`; synthetic caption-derived MCQ only, with article-level license join.
- frozen primary MCQ development source: 512 deterministic records from PMC-VQA v2 official test; frozen inspection found zero image and PMCID overlap with its train source.
- frozen cross-dataset source: SLAKE official validation at HF revision `a9083ce6c34ac3ffb17671a605962924d8a8f9e9`; English records only and never used for training.
- deferred source: MS-CXR/VinDr-CXR evidence boxes require credentialed PhysioNet access and do not block cycle 1.
- admitted training seed: 1,968 PMC-VQA v2 rows; 32 unresolved dHash candidates remain visible but quarantined.
- admitted primary development: 512 PMC-VQA v2 MCQs, article/image/question-disjoint from the train seed and zero confirmed Med-CMR overlap.
- completed comparison: direct 295/512 (57.6172%); B1 204/512 (39.8438%); paired delta -17.7734 points with 95% bootstrap interval `[-22.8516,-12.5000]`.
- next execution: run the frozen 2-step T1a direct-answer QLoRA backward/save smoke on the single V100; B1 is not its parent objective.
