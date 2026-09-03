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
