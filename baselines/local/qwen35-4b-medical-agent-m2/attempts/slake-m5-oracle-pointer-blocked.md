# SLAKE M5 oracle region pointer：BLOCKED

日期：2026-09-04  
性质：external development causal screen；不是 Med-CMR official score

## 合同

- 43 条冻结 SLAKE validation 样本。
- 单张完整原图；不裁剪、不增加图片数量、不修改 answer-only prompt。
- oracle arm 在 GT box 位置画红色矩形；sham arm 使用 sample-id 排序后循环移位的另一条 GT box，保持框分布。
- pointer manifest SHA-256：`3977c242b0eba38c47e90f01a9807c45b39311831658faa5ef0b70339858aab4`。
- sham manifest SHA-256：`7b2252fa6da1b3461e52e9649564515b11131aa74c7ad356af7926b4c2c71ccf`。
- references 未被构建器读取；两臂均 43/43 completed。

## 结果

| Arm | Exact | Token F1 | Predictions SHA-256 |
|---|---:|---:|---|
| full | 53.49% | 68.95% | `9607c565234eae627668e9ba8591891d5688f6d1b3e904c6128d8a36778cc157` |
| sham pointer | 48.84% | 65.24% | `d67b62c77cd69f2bc35bdf7f4525b3be40ac3d66e8c4ddf22e186fd12a959205` |
| oracle pointer | 53.49% | 71.63% | `ae9029463bdb3a9faf02d89267864f7d971ecf3b338a241b418b2a1649cfdcfa` |

- oracle - sham：token F1 `+6.39`，95% CI `[-1.48, 15.81]`；exact `+4.65`。
- oracle - full：token F1 `+2.68`，95% CI `[-6.30, 12.54]`；exact `0.00`。
- sham - full：token F1 `-3.71`，95% CI `[-12.14, 3.81]`。

## 判定

点估计提示正确位置可能比错误位置更好，但预注册主门要求 oracle-sham CI 下界大于 0，实际失败。不得接 learned pointer，也不得继续调颜色、线宽、prompt 或框位置。下一步先确认 Med-CMR 官方 evidence 指标是否直接奖励空间定位，再决定 locator 的价值路径。
