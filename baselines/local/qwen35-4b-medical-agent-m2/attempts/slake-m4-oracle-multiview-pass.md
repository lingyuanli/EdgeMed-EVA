# SLAKE M4 oracle multiview gate：PASS

日期：2026-09-04  
性质：external development proxy；不是 Med-CMR official score

## 运行合同

- 数据：冻结的 SLAKE validation locator slice，43 条。
- 模型：hash-bound Qwen3.5-4B base，NF4 / FP16 / eager / greedy / non-thinking。
- 回答：`answer_only`，`max_new_tokens=32`。
- 输入：`Full context image:` + `Localized detail image:`；每图像素上限 786,432。
- 对照：单 full-image；full + 同尺寸 black crop。
- 配对 bootstrap：10,000 次，seed 20260904。
- references SHA-256：`f47b05c2786b3c93e4b61e52102119a84c2d4a2c603e4b5fe582f1cb5e316f0e`。

## 保留的执行失败

commit `0749a1c` 的首个双图运行在 0/43 处因两张图分属不同 data root 而触发 `FileNotFoundError`。未产生 prediction。commit `28ad772` 将原图按原始字节/hash 物化进 answer surface；失败目录未复用，正式运行使用新 `v2` run 与新合同。

## 结果

| Arm | Complete | Exact | Token F1 | Peak CUDA | Predictions SHA-256 |
|---|---:|---:|---:|---:|---|
| full-image | 43/43 | 53.49% | 68.95% | 4,449.19 MiB | `9607c565234eae627668e9ba8591891d5688f6d1b3e904c6128d8a36778cc157` |
| full + black crop | 43/43 | 51.16% | 68.17% | 4,498.41 MiB | `eba75e430a0329fcc5267a89f2651f017c972d0d955f9671315b47f24422b6b0` |
| full + oracle crop | 43/43 | 60.47% | 76.55% | 4,498.41 MiB | `50046e16be7a2c846cdf906640159c853310a7cc1ac1ed8ed8e172a108a99589` |

- oracle multiview - full：token F1 `+7.59` points，95% CI `[0.23, 16.74]`；exact `+6.98` points。
- oracle multiview - black multiview：token F1 `+8.37` points，95% CI `[0.47, 18.60]`；exact `+9.30` points，95% CI `[2.33, 18.60]`。
- black multiview - full：token F1 `-0.78` points，95% CI `[-9.31, 7.36]`。

## 判定

预注册 M4 oracle gate 通过。增益需要局部图的真实内容，且保留全局上下文后才出现；单图 crop 路线仍为失败。下一阶段允许使用冻结的 locator-64 predictions 构建 learned-crop 双图臂。该结果不允许外推为 Med-CMR 提升或“4B 超过商业模型”。
