# M4：全局—局部多视图证据融合

## 1. 已冻结的触发证据

在 SLAKE validation 的 43 条 answer-isolated 开发样本上，Qwen3.5-4B 的 oracle-crop 单图并未超过 full-image：token F1 为 `59.69%` 对 `68.95%`，paired delta 为 `-9.26` 点，95% bootstrap CI 为 `[-20.74, 2.25]`。但 oracle-crop 明显超过同尺寸 black-crop：`59.69%` 对 `16.07%`，paired delta 为 `+43.62` 点，95% CI 为 `[26.39, 59.69]`。

因此，M3-S2 的“以 crop 替换全图”路线失败。新假设不是继续提高 locator IoU，而是保留全局上下文，并把局部裁剪作为第二张有标签的证据图输入。

## 2. 单一假设

对同一问题同时输入 `(full_context, local_detail)`，可以保留全局解剖/模态信息，同时放大问题相关区域。若局部图确有增量信息，`full + oracle crop` 应优于尺寸完全一致的 `full + black crop`；若这种融合也无法超过单张 full-image，则不应把 learned locator 接入最终问答链路。

## 3. 冻结设计

- 样本：与 M3-S2 完全相同的 43 条 validation 样本与顺序。
- 模型：同一 hash-bound Qwen3.5-4B base，不加载 locator adapter。
- 解码：`answer_only`、greedy、`max_new_tokens=32`、每图 `max_image_pixels=786432`。
- 输入标签：固定为 `Full context image:` 与 `Localized detail image:`。
- 臂 A：已完成的单张 `full-image`。
- 臂 B：`full + oracle crop`。
- 臂 C：`full + black crop`；第二图尺寸与 oracle crop 逐样本一致。
- 评分：normalized exact、token F1；B-A 与 B-C 均用 10,000 次 paired bootstrap，seed `20260904`。

## 4. 预注册门槛

只有同时满足以下条件才进入 learned-crop：

1. `token_F1(B) - token_F1(A) > 0`；
2. `token_F1(B) - token_F1(C) > 0`，且 95% bootstrap CI 下界大于 0；
3. 三臂 43/43 完整，输入 manifest、predictions、references 均有 SHA-256；
4. 推理输入不包含 answer/reference/ground-truth 字段。

门槛失败时，停止 crop-fusion 路线，转向不依赖空间裁剪的证据表示；不得根据逐题 validation 正误修改框或 prompt，也不得访问 Med-CMR test 调参。

## 5. 结果

三臂均完成 43/43。双图运行来自 commit `28ad772`，每臂峰值 CUDA `4,498.41 MiB`；oracle/black 双图推理分别耗时 `39.31/40.00 s`。

| Arm | Normalized exact | Token F1 | Prediction SHA-256 |
|---|---:|---:|---|
| full-image | 53.49% | 68.95% | `9607c565234eae627668e9ba8591891d5688f6d1b3e904c6128d8a36778cc157` |
| full + black crop | 51.16% | 68.17% | `eba75e430a0329fcc5267a89f2651f017c972d0d955f9671315b47f24422b6b0` |
| full + oracle crop | 60.47% | 76.55% | `50046e16be7a2c846cdf906640159c853310a7cc1ac1ed8ed8e172a108a99589` |

`full + oracle crop` 相对 full-image 的 token F1 为 `+7.59` 点，95% CI `[0.23, 16.74]`；相对 compute-matched `full + black crop` 为 `+8.37` 点，95% CI `[0.47, 18.60]`。两个预注册 efficacy 条件均通过，且黑图双视图与单全图近似持平，排除了“仅增加第二张图/token”足以解释增益。

结论边界：这只是 SLAKE 43 条 answer-isolated development slice 上的 oracle 上界，不是 learned locator 的答案增益，也不是 Med-CMR 分数。允许下一步只替换第二图来源为冻结 locator-64 预测框；其余模型、prompt、解码、scorer 与样本保持不变。
