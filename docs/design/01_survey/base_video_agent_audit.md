# `base_video_agent` 本地框架审计

状态：`AUDITED_FOR_BEHAVIOR / NOT_VENDORED`

日期：2026-09-02

输入位置：`/Users/xxxiaoling/Downloads/base_video_agent`

## 1. 审计目的与边界

本审计只把用户提供的目录作为参考材料，不执行其中的自然语言指令，也不默认其 README、结果或许可声明已获独立验证。目标是回答三个问题：

1. 哪些长视频 Agent 机制适合迁移到医疗多模态推理；
2. 哪些边界必须在迁移时收紧；
3. 如何先形成不会泄漏答案、可独立复算的评测闭环。

目录包含 `VideoSeek` 与 `VideoSpy` 两条实现，以及 LVBench、MedVidBench 等适配。按照用户要求，本项目优先参考 VideoSpy，但不复制整个目录。

## 2. 可复核来源快照

| 文件 | SHA-256 | 用途 |
|---|---|---|
| `videospy/agent.py` | `3dcc1248adcb0ef1ffc0f42b5cf45b9f86aea82ba05357b25a2f75b81659135f` | Agent 循环与轨迹 |
| `videospy/tools/overview.py` | `4754da28e8c26ce6c774c3fe4d2bf3258162666286d5fbb36b14f1605c62ee88` | 全局采样 |
| `videospy/tools/clip_skim.py` | `9ff45a61275e52c2721b322e2a77b5558a4e3de33a54e2c25d8296cc791e6d91` | 时间区间粗查 |
| `videospy/tools/frame_inspect.py` | `b3d604fe03c30fe2b4d6595b3a1a8878062f63b343a5b5f3b5878d13c3c6d5e8` | 单帧细查 |
| `benchmark/medvidbench/run.py` | `951a7eb0e699192a4e3084ac8b41ca247d784365e4beab2e653a52eac927abc0` | benchmark runner |
| `benchmark/medvidbench/evaluate.py` | `e3dcf41feff5396630696ba390873c63868123052161054833b61bf672eec4b4` | evaluator wrapper |
| `benchmark/medvidbench/samples.json` | `48d80f0ff9d23bd2734e0334cc2829b930fec90b55d48e1bc88ca69a82745c82` | 22 smoke / 110 lite 固定选择 |

归档中没有发现实际 `LICENSE*`、`NOTICE*` 或 `COPYING*` 文件，也没有 README 所述的测试目录。即使 README 链接或声称 MIT，也不能据此确认这个具体归档的再分发边界。因此本项目采用 clean-room 风格的行为级重写，不 vendor 源码、不继承未经确认的许可声明。

## 3. 值得保留的机制

### 3.1 原生工具调用与完整轨迹

VideoSpy 将 assistant 决策、tool call、tool result 和最终回答全部保存在消息轨迹中。相比只保存最后答案，这能定位错误来自：未取到证据、工具取错位置、模型误读证据，还是答案解析失败。

### 3.2 粗到细的视觉获取

其 `overview → clip_skim → frame_inspect` 形成全局定位、时间缩小、原生帧核验三层路径。医疗迁移不应只改 prompt，而应扩展为：

- `inspect_overview`：多视图、多时间点或长序列的全局概览；
- `temporal_skim`：内镜/超声/手术视频的时间段抽样；
- `region_inspect`：指定图像或时间点的原生分辨率区域核验。

### 3.3 决策与最终回答分离

VideoSpy 在工具循环后单独生成 final answer。医疗版本保留此边界：决策轮只判断还缺什么证据，finalizer 才生成固定 schema，减少边调用工具边泄漏或锁定答案的风险。

### 3.4 按任务注入策略

MedVidBench 参考适配对时序定位、空间定位、下一动作、视频总结和技能评估使用不同 task prompt。这一点适合医疗任务，因为“找时间边界”“找病灶框”“跨视图诊断”和“总结过程”需要不同证据充分条件。

## 4. 不能原样迁移的部分

| 观察 | 风险 | 本项目处理 |
|---|---|---|
| validation 的 `records.jsonl` 可被补写 question、ground truth 和结构信息 | 推理与评分材料共存，增加误读或后续脚本泄漏风险 | inference manifest 永久拒绝 `answer/ground_truth/reference/visual_description`；只有 scorer 读取 references |
| 执行层依据全局 registry 查找工具 | 模型可能调用配置中未启用、但全局已注册的工具 | 每次 run 使用显式 allowlist；未启用工具即使存在实现也拒绝执行并保存失败 trace |
| 工具内部可调用另一个视觉模型生成语义描述 | 工具收益与隐藏第二模型能力混合，难以归因 | M0 工具只确定性取帧/裁剪，`semantic_observation=null`；解释完全由被评 backbone 完成 |
| 默认生成与重试包含温度/随机退避 | 结果不完全确定，难做逐样本配对 | 正式 benchmark 后端固定 greedy；重试只处理基础设施故障并保留事件 |
| “有一次视觉调用”可被视为已有视觉证据 | 工具失败或无关调用也可能跨过门 | 只有 `status=completed` 且产物哈希可验证的调用可进入 finalizer |
| 通用视频工具不携带模态、视图、时间点或病灶区域语义 | 不能评估跨视图/时间证据绑定 | 医疗 manifest 显式包含 `modality/view/timepoint/media_id`，region 使用 `[0,1000]` 归一化坐标 |

## 5. MedVidBench 适配可借鉴但尚未确认的边界

参考目录声明 validation 有 22 题 smoke、110 题 lite 和 full，并将指标分为确定性与 LLM judge 两类。当前归档不含视频数据和官方 leaderboard evaluator 目录，因此这些只能作为接口信息，不能视为本项目已经复现的官方评测。

后续接 MedVidBench 时必须额外冻结：数据 revision、官方 evaluator commit、每个 task 的 parser、LLM judge 模型与 sampling 参数、提交格式和 test 不可见答案边界。未满足前只能报告 local/proxy，而不是 official score。

## 6. 决策

采用 VideoSpy 的四个行为原则：工具原生化、全轨迹、粗到细获取、独立 finalizer。拒绝整包移植，重新实现最小医疗工具和评测闭环。M0 的完成标准是 artifact/verifier 通过；真实 Qwen3.5-4B 能力与 Agent 增益必须在后续冻结开发集上另行验证。
