# MedVidBench

MedVidBench 包含两个 split：

- `validation`：`MedVidU_ECCV2026_TrainVal`，包含 Ground Truth，支持 smoke、lite、full 和本地评测。
- `test`：`MedVidBench-data`，没有 Ground Truth，只支持 full，结果提交到网页 Leaderboard。

## 运行

validation 默认运行 smoke：

```bash
python benchmark/medvidbench/run.py --agent videospy
python benchmark/medvidbench/run.py --agent videospy --mode lite
python benchmark/medvidbench/run.py --agent videospy --mode full
```

smoke 对 11 个 `qa_type` 各取 2 题，共 22 题。lite 各取 10 题，共 110 题。
固定样本位于 `samples.json`，smoke 是 lite 的子集。

test 自动使用 full：

```bash
python benchmark/medvidbench/run.py --agent videospy --split test
```

VideoSpy 使用 MedVidBench 的独立配置：

- `benchmark/medvidbench/config/videospy_general.yml`：本机 General 配置，已被 Git 忽略。
- `benchmark/medvidbench/config/videospy_general.example.yml`：仅供参考的配置模板，不会自动加载。
- `benchmark/medvidbench/config/videospy_prompt.yml`：默认 Prompt 配置。

默认 Prompt 还包含按 `qa_type` 区分的 `TASK_PROMPTS`。VideoSpy 每次只注入当前
问题对应的任务策略，避免不同任务相互干扰。`dense_captioning_*`、
`region_caption_*` 和 `video_summary_*` 会分别使用各自共享的策略。通过
`--prompt` 可以覆盖单个策略；设置 `TASK_PROMPTS: null` 可以关闭任务策略注入。

首次使用时可以复制参考模板并按本机环境修改。私有文件不存在时继续使用全局
VideoSpy General 默认配置，不会读取 `.example.yml`：

```bash
cp benchmark/medvidbench/config/videospy_general.example.yml \
  benchmark/medvidbench/config/videospy_general.yml
```

可以通过 `--config` 和 `--prompt` 传入只包含差异项的 YAML 文件：

```bash
python benchmark/medvidbench/run.py \
  --agent videospy \
  --config custom-general.yml \
  --prompt custom-prompt.yml \
  --mode full
```

优先级依次为全局 VideoSpy 默认配置、MedVidBench 专用配置、显式传入的配置文件、
`--model-name` 和 `--max-steps` 等具体命令行参数。字典按层递归合并，列表和普通值
整体替换。`--prompt` 仅支持 VideoSpy，VideoSeek 的配置行为保持不变。这里的 Agent
配置与下文用于 LLM Judge 的 Evaluation 配置相互独立。

目录规则与 LVBench 一致：

```text
output/<agent>/medvidbench/<mode>_<model>_<timestamp>/
```

## Run 目录

核心 JSON 产物只保留四类：

- `run.json`：运行配置和样本列表，用于安全续跑。
- `records.jsonl`：逐题检查点、Prediction、轨迹路径和运行消耗。validation 还包含 question、Ground Truth 和 `struc_info`，可直接逐题对比。
- `submission.json`：网页提交格式，只包含 `id`、`qa_type`、`prediction`。
- `evaluation.json`：本地评测结果、完整性状态和 LLM 配置，仅在本地评测后生成。

此外还有便于阅读的 `report.md`、日志文件和 `trajectories/`。test 的
`records.jsonl` 不写入 Ground Truth。

## 本地评测

仅 validation 支持本地评测：

```bash
python benchmark/medvidbench/evaluate.py \
  --run-dir output/<agent>/medvidbench/<run_dir_name> \
  --skip-llm-judge
```

`--skip-llm-judge` 跳过 DVC、Video Summary、Region Caption 的 LLM 指标，
但仍要求其余 7 个确定性指标完整。缺依赖、任务异常或指标缺失都会使评测返回失败，
具体原因写入 `evaluation.json` 和 `evaluation.log`。

评测配置默认读取被 Git 忽略的
`benchmark/medvidbench/config/evaluation.yml`：

```yaml
evaluation:
  llm:
    model: gpt-4.1
    api_base: https://example.com/v1
    api_key: <api-key>
```

该文件可以保存本机 Key。文件不存在时会回退到仓库中的
`benchmark/medvidbench/config/evaluation.example.yml` 模板。
配置完成后直接运行：

```bash
python benchmark/medvidbench/evaluate.py \
  --run-dir output/<agent>/medvidbench/<run_dir_name>
```

评测模型配置与 Agent 配置相互独立，Key 不会写入 run 目录。首次评测前安装官方依赖：

```bash
pip install -r data/MedVidBench/MedVidBench-Leaderboard/requirements.txt
```

`report.md` 按网页顺序展示 10 个官方指标：`CVS_acc`、`NAP_acc`、`SA_acc`、
`STG_mIoU`、`TAG_mIoU@0.3`、`TAG_mIoU@0.5`、`DVC_F1`、`DVC_llm`、
`VS_llm`、`RC_llm`。Run Statistics 中的 `Generated` 仅表示样本生成成功，不表示
答案正确。

validation 的本地评测还会计算诊断指标 `NAP_exact_acc`。它对 Next Action 的完整
标签做大小写不敏感和空白规范化后的精确匹配，不使用语义映射，也不替代官方
`NAP_acc`。如果两者不同，`evaluation.json` 和 `report.md` 会保留两个结果并给出
警告。这可以识别 smoke 子集中只有一个类别时，官方语义映射将任意预测计为正确的
情况。

## Prediction 如何解析

官方评测器主要使用正则表达式从模型原始文本提取结构化结果，不使用额外 LLM：

- TAL：正则提取 `start-end`，找不到时再尝试 `start to end` 和时分秒区间，然后转换为浮点时间段。
- STG：正则提取 `timestamp seconds: [x1, y1, x2, y2]`，坐标按逗号切分并转换为浮点数。
- DVC：正则逐行提取 `start-end seconds: caption`，得到开始时间、结束时间和描述文本。
- Skill Assessment：按逗号切分，再用正则提取 `维度名: N/5`。
- CVS Assessment：用三个固定名称的正则提取整数分数。
- Next Action：完整文本先做类别精确匹配，失败后用语义相似度映射到动作类别。
- Region Caption 和 Video Summary：完整文本直接交给官方 LLM 评分。

本地脚本不重复解析 Prediction，也不修改官方计分公式和
`data/MedVidBench/MedVidBench-Leaderboard` 中的官方代码。

## 续跑与提交

使用原目录续跑，已成功完成的题目会跳过：

```bash
python benchmark/medvidbench/run.py \
  --agent videospy \
  --run-dir output/videospy/medvidbench/<run_dir_name>
```

续跑参数必须与 `run.json` 一致。test 续跑还需传入 `--split test`。test 全量完成后，
将 `submission.json` 上传到
[MedVidBench Leaderboard](https://huggingface.co/spaces/UII-AI/MedVidBench-Leaderboard)。
