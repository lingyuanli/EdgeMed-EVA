# Evidence-Grounded Medical Agent 设计

工作名：`EG-MedAgent-4B`。名称仅用于产物标识，不构成最终论文命名。

## 1. 目标与非目标

目标：用一个 Qwen3.5-4B backbone，在有限轮次内完成视觉证据获取、竞争假设比较和答案生成，并使每个关键主张都能回指到输入图像或明确的文本上下文。

非目标：首版不做多 Agent 辩论、不做自动临床决策、不为每种模态配置专科模型、不把隐藏 CoT 当作评测证据、不允许工具直接给出最终诊断。

## 2. 输入规范

```json
{
  "sample_id": "medcmr_xxx",
  "question_type": "mcq|open",
  "question": "...",
  "options": ["A...", "B..."],
  "images": [
    {
      "image_id": "img_0",
      "path": "...",
      "modality": "CXR|CT|MRI|pathology|...",
      "view": "frontal|lateral|axial|...|unknown",
      "timepoint": "current|prior|unknown"
    }
  ],
  "clinical_context": "...",
  "benchmark_dimension": "hidden_during_inference"
}
```

`benchmark_dimension` 只供分层评测，不能注入推理提示。图像预处理参数必须写入 run manifest；严禁为单个测试样本人工调 crop。

## 3. 状态机

| 状态 | 允许动作 | 输出 | 失败策略 |
|---|---|---|---|
| `INGEST` | 校验输入、绑定视图/时间、生成缩略图 | input receipt | 非法输入直接记录，不猜测 |
| `PLAN_EVIDENCE` | 建立 1–3 个假设和证据需求 | hypothesis table | 格式修复一次 |
| `ACQUIRE` | `inspect_native` 或 `crop_zoom` | evidence packet | 超预算转 `SYNTHESIZE` |
| `UPDATE` | 更新支持/反对关系和置信度 | revised table | 禁止新增未观察证据 |
| `SYNTHESIZE` | 选择答案、引用证据 | draft output | 引用缺失则静态修复一次 |
| `VERIFY_STATIC` | schema、引用、坐标、预算检查 | verifier result | benchmark 模式保留答案并标记失败 |
| `FINAL` | 输出固定 schema | prediction | append-only 保存 |

只有 `ACQUIRE` 状态可以调用视觉工具。最多两轮 acquisition，每轮最多两个 crop；总 crop 数最多四个。

## 4. 工具接口

### 4.1 `inspect_native`

用途：返回图像尺寸、模态元数据、面板/视图标识和统一缩略输入；不做医学分类。

```json
{"tool":"inspect_native","image_id":"img_0"}
```

### 4.2 `crop_zoom`

```json
{
  "tool": "crop_zoom",
  "image_id": "img_0",
  "region_xyxy_1000": [100, 200, 550, 700],
  "target": "区分局灶性实变与胸腔积液边缘",
  "scale": 2.0
}
```

确定性实现：裁剪框 clamp 到图像边界，最小边长阈值，固定插值和输出分辨率，返回工具输入/输出哈希。拒绝空框、重复框和无 `target` 的调用。

### 4.3 候选工具进入条件

RAG、segmentation、detection、DICOM window/level 或 3D slice navigation 只有同时满足以下条件才能加入：

1. B1/M1 错误切片中有稳定、足量失败；
2. 人工或 oracle 工具输出能显著提高该切片；
3. 可获得无测试泄漏的训练/验证数据；
4. 真实工具在 held-out 上的准确率足以保留 oracle 收益；
5. 增加后的逐工具消融显示净收益且成本可接受。

## 5. 模型输出 schema

```json
{
  "sample_id": "medcmr_xxx",
  "hypotheses": [
    {"id":"H1","label":"...","status":"supported|refuted|uncertain"}
  ],
  "evidence": [
    {
      "evidence_id":"E1",
      "image_id":"img_0",
      "view_or_time":"current_frontal",
      "region_xyxy_1000":[120,210,480,620],
      "observation":"...",
      "supports":["H1"],
      "contradicts":[],
      "acquisition":"native|crop",
      "confidence":0.78
    }
  ],
  "answer": "B",
  "answer_text": "...",
  "answer_evidence_ids": ["E1"],
  "confidence": 0.74,
  "insufficient_evidence": false,
  "tool_trace_ids": ["T1"]
}
```

MCQ 的 `answer` 必须严格是一个选项标识；开放题 `answer_text` 必须是短结论。Benchmark 模式中 `insufficient_evidence` 不能代替作答。

## 6. Prompt/控制策略

固定 system contract：

- 只引用当前样本可见信息；
- 将观察事实与解释分开；
- 每个答案至少引用一个 evidence id，若确无视觉依据则引用临床上下文并标记 `region=null`；
- 不确定时降低置信度，而不是创造病灶；
- 工具调用必须说明要区分的假设；
- 输出只含 JSON schema，不输出给 evaluator 的讨好性文本。

实现时分别保存 `system_prompt_version`、`task_template_version` 和 `schema_version`，不将提示内容嵌入模型权重版本名。

## 7. 静态 verifier

首版 verifier 是纯函数，不使用另一个 LLM：

- JSON schema、必填字段和枚举；
- evidence id 唯一且引用存在；
- 坐标范围、框面积和 image id 合法；
- 工具预算与轨迹哈希一致；
- observation 不能为空；
- MCQ 选项合法；
- 置信度属于 `[0,1]`。

它只能证明结构与来源绑定，不能证明医学正确。医学正确性由冻结参考、定位标注、反事实测试和专家抽检评估。

## 8. 失败语义

- 工具失败：保留失败事件，回到已有证据作答；不得静默重试超过一次。
- schema 失败：固定解析器修复一次；仍失败则记 `invalid_output`。
- 多图缺失：记输入故障，不让模型猜缺失视图。
- 超时/OOM：记基础设施失败，不计为科学错误；从样本边界精确恢复。
- 安全模式证据不足：允许弃答；benchmark 模式仍输出最可能答案并保留低置信度。

## 9. 可替换性

backbone、工具和 evaluator 通过版本化接口解耦。第一阶段不得同时更换 backbone、训练数据和 Agent 逻辑；每次迭代只改变一个主要因子，否则无法归因。
