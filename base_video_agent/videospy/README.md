# VideoSpy

VideoSpy 是基于原生函数调用实现的长视频理解 Agent。每轮模型负责判断是否继续收集视觉证据，工具结果以标准消息形式保存在完整轨迹中，最终答案由独立的无工具模型调用统一生成。

## 工具

- `overview`：对完整视频均匀抽帧，建立全局认识。
- `clip_skim(query, start_time, end_time)`：粗粒度浏览指定视频片段并定位候选时刻。
- `frame_inspect(query, timestep)`：检查指定时间点附近的单个原分辨率视频帧。

没有字幕和视觉证据时，Agent 不允许直接回答。

## 配置

```bash
cp config/videospy/general.example.yaml config/videospy/general.yaml
```

配置文件位于 `config/videospy/`：

- `general.yaml`：Agent、工具和模型配置。
- `prompts.yaml`：系统提示词。

运行时可分别覆盖两个文件：

```bash
videospy-cli \
  --video_path video.mp4 \
  --user_query "What happens in the video?" \
  --config custom-general.yaml \
  --prompt custom-prompts.yaml \
  --verbose
```

字幕是可选参数：

```bash
--subtitle_path video.srt
```

## 输出

结果写入 `output/videospy/demo/`：

- `prediction.json`：最终答案。
- `trajectory.json`：完整消息、工具调用、工具观察及工具内部模型调用。

运行 benchmark 请参阅 [LVBench](../benchmark/lvbench/README.md) 和 [MedVidBench](../benchmark/medvidbench/README.md)。
