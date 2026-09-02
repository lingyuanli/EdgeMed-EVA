# LVBench

从仓库根目录运行：

```bash
python benchmark/lvbench/run.py --mode smoke
python benchmark/lvbench/run.py --mode lite
python benchmark/lvbench/run.py --mode full
```

默认运行原始 VideoSeek。使用 `--agent` 切换到优化版 VideoSpy：

```bash
python benchmark/lvbench/run.py --agent videospy --mode smoke
```

`smoke` 按六类能力各运行 2 道题，共 12 道。`lite` 每类运行 10 道，
共 60 道。固定样本记录在 `samples.json`，两种模式不会在运行时重新抽样。

默认数据位置：

- 元数据：`data/LVBench/zai-org-LVBench/video_info.meta.jsonl`
- 视频：`data/LVBench/AIWinter-LVBench/all_videos`

可以使用 `--metadata-path` 和 `--video-dir` 覆盖。VideoSeek 使用
`config/videoseek/`，VideoSpy 使用 `config/videospy/`。模型参数也可以通过
`--model-name`、`--api-base`、`--api-key`、`--max-steps` 等参数覆盖。对于
VideoSpy，这些参数只覆盖 Agent 决策模型，三个视觉工具的模型仍由配置文件控制。

VideoSpy 使用当前 Benchmark 的独立配置：

- `benchmark/lvbench/config/videospy_general.yml`：本机 General 配置，已被 Git 忽略。
- `benchmark/lvbench/config/videospy_general.example.yml`：仅供参考的配置模板，不会自动加载。
- `benchmark/lvbench/config/videospy_prompt.yml`：默认 Prompt 配置。

首次使用时可以复制参考模板并按本机环境修改。私有文件不存在时继续使用全局
VideoSpy General 默认配置，不会读取 `.example.yml`：

```bash
cp benchmark/lvbench/config/videospy_general.example.yml \
  benchmark/lvbench/config/videospy_general.yml
```

可以通过 `--config` 和 `--prompt` 传入只包含差异项的 YAML 文件：

```bash
python benchmark/lvbench/run.py \
  --agent videospy \
  --config custom-general.yml \
  --prompt custom-prompt.yml \
  --mode full
```

优先级依次为全局 VideoSpy 默认配置、LVBench 专用配置、显式传入的配置文件、
`--model-name` 和 `--max-steps` 等具体命令行参数。字典按层递归合并，列表和普通值
整体替换。`--prompt` 仅支持 VideoSpy，VideoSeek 的配置行为保持不变。

每次新运行会写入 `output/<agent>/lvbench/`（例如 `output/videoseek/lvbench/`）。运行目录命名为 `{mode}_{model_name}_{time}`，如 `smoke_openai_qwen3.6-flash_20260816_220736`。主要结果包括：

- `predictions.json`：官方 UID 到选项字母的映射
- `metrics.json`：总体准确率和分类准确率
- `report.md`：按任务类别汇总 Agent、模型、准确率、轮数、token 和耗时
- `records.jsonl`：逐题回答、状态、Agent 轮数、token 消耗和耗时
- `run.log`：运行配置、逐题状态、Agent 详细输出和异常堆栈
- `trajectories/`：逐题 Agent 轨迹

终端会通过 `tqdm` 持续显示题目级进度。断点续跑时 `run.log` 会追加记录，不会
覆盖上一次运行日志。
如需同时查看并保存 Agent 的逐步日志，请显式传入 `--verbose`。
`metrics.json` 的 `run_statistics` 会汇总总轮数、token 和耗时，并给出每题均值。
`report.md` 会在每次运行或续跑结束后重新生成，便于直接查看和比较实验结果。

## 断点续跑

每道题完成后都会立即写入 `records.jsonl`。断网或手动停止进程后，使用原来的
运行目录继续执行：

```bash
python benchmark/lvbench/run.py \
  --mode full \
  --run-dir output/videoseek/lvbench/<run_dir_name>
```

VideoSeek 的 `<run_dir_name>` 形如 `full_<model>_<timestamp>`。VideoSpy 会在末尾追加运行时 Git Commit ID 的前 7 位，形如 `full_<model>_<timestamp>_<commit7>`。目录名可从上次运行的终端输出获取。

恢复时会跳过已成功完成且包含完整统计信息的题目，重新执行失败题目以及中断时
尚未写入结果的题目。`run.log` 会继续追加，最终的 `predictions.json` 和
`metrics.json` 会根据完整记录重新生成。

续跑必须保持原来的 Agent、运行模式、样本列表、数据路径和模型配置。若这些配置
与目录中的 `run.json` 不一致，程序会拒绝续跑，避免把 VideoSeek 和 VideoSpy 或
其他不同实验的结果混在一起。旧的 `run.json` 没有 Agent 字段时按 VideoSeek
兼容处理。

LVBench 数据目录中没有字幕文件，评测默认按无字幕模式执行。
