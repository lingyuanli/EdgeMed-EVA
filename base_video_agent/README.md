# 多模态视频理解智能体

本项目包含两个长视频理解 Agent。VideoSeek 是来自原论文的 baseline 实现，VideoSpy 是我们在此基础上开发的方法。两者共享 benchmark 和基础依赖，但代码、配置与文档彼此独立。

## 方法概览

| | VideoSeek | VideoSpy |
|---|---|---|
| 定位 | 外部 baseline | 本项目开发的方法 |
| 工具 | `overview`、`skim`、`focus`、`answer` | `overview`、`clip_skim`、`frame_inspect` |
| Agent 交互 | 自定义动作与观察协议 | 原生函数调用与标准工具消息 |
| 轨迹 | 自定义轨迹结构 | 完整消息、工具调用及内部模型调用 |

## 项目结构

```text
.
├── videoseek/              # VideoSeek Agent、工具和 CLI
├── videospy/               # VideoSpy Agent、工具和 CLI
├── config/
│   ├── videoseek/          # VideoSeek 配置与提示词
│   └── videospy/           # VideoSpy 配置与提示词
├── benchmark/
│   ├── lvbench/            # LVBench 运行与评测
│   └── medvidbench/        # MedVidBench 运行与评测
├── data/                   # 数据集与第三方资源
├── output/                 # Agent 和 benchmark 输出
├── tests/                  # 单元测试
└── docs/                   # 图片等文档资源
```

## 文档

- [VideoSeek](videoseek/README.md)
- [VideoSpy](videospy/README.md)
- [LVBench](benchmark/lvbench/README.md)
- [MedVidBench](benchmark/medvidbench/README.md)

## 安装

要求 Python 3.10 及以上版本，并确保系统已安装 `ffmpeg`。

```bash
pip install -e .
```

安装后可使用：

```bash
videoseek-cli -h
videospy-cli -h
```

## 测试

```bash
python -m pytest tests -q
```

## License

[MIT](https://opensource.org/licenses/MIT)
